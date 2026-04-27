#!/usr/bin/env python3
"""
assemble_code_shards.py — Phase 2: assemble per-domain chunks into shards.

Reads pre-built .txt.gz chunk files produced by build_code_chunks.py (and the
existing Curated_Corpus_180GB chunks for English), and stitches them into
uniformly-distributed shards.

Target
------
  20 shards × 1 GB uncompressed = 20 GB total

  Domain      Fraction   Per-shard quota
  ─────────────────────────────────────
  English       55 %       563 MB
  C++           15 %       154 MB
  Rust          10 %       102 MB
  Python        10 %       102 MB
  Shell         10 %       102 MB
  ─────────────────────────────────────
  Total        100 %      ~1023 MB ≈ 1 GB

Each shard gets the same domain distribution. A single streaming cursor is
maintained per domain across all shards — data is never repeated between shards.
If a domain is exhausted before all 20 shards are filled, remaining slots for
that domain are left empty (shard will be slightly under 1 GB).

Chunk reading order per domain
-------------------------------
  english : dclm → fineweb_edu
  cpp     : chunks_py_cpp/cpp
  python  : chunks_py_cpp/python
  rust    : rust_stackedu → rust_starcoder → rust_ir
  shell   : shell_stackedu → shell_starcoder

Within each shard, domains are interleaved in round-robin batches of
INTERLEAVE_BATCH lines so BPE sees mixed content throughout.

Output layout
-------------
  SHARDS_DIR/
    shard_00.txt.gz
    .shard_00.done
    shard_01.txt.gz
    .shard_01.done
    ...

Resume
------
  Shards with a matching .done file are skipped.
  A partial shard (missing .done) is removed and rebuilt.
  Cursors always start from shard 0 and fast-forward through completed shards
  so the data stream position is correct before writing resumes.

Usage
-----
  python assemble_code_shards.py              # full run
  python assemble_code_shards.py --dry-run    # validate paths + print plan only
"""

import argparse
import gzip
import logging
import os
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here only
# ══════════════════════════════════════════════════════════════════════════════

GB = 1024 ** 3
MB = 1024 ** 2

NUM_SHARDS       = 20
SHARD_SIZE_BYTES = 1 * GB          # uncompressed target per shard

# Lines written per domain per interleave pass inside each shard.
# 500 lines ≈ a few dozen KB — small enough to mix domains well.
INTERLEAVE_BATCH = 500

# Per-shard byte quotas — must sum to ≤ SHARD_SIZE_BYTES
DOMAIN_QUOTAS = {
    "english": int(0.55 * SHARD_SIZE_BYTES),   # ~563 MB
    "cpp":     int(0.15 * SHARD_SIZE_BYTES),   # ~154 MB
    "rust":    int(0.10 * SHARD_SIZE_BYTES),   # ~102 MB
    "python":  int(0.10 * SHARD_SIZE_BYTES),   # ~102 MB
    "shell":   int(0.10 * SHARD_SIZE_BYTES),   # ~102 MB
}

_ENG    = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_Corpus_180GB/chunks"
_CODE   = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/chunks"
_PY_CPP = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/chunks_py_cpp"

# Directories streamed in the listed order per domain.
DOMAIN_CHUNK_DIRS = {
    "english": [
        os.path.join(_ENG, "dclm"),
        os.path.join(_ENG, "fineweb_edu"),
    ],
    "cpp": [
        os.path.join(_PY_CPP, "cpp"),
    ],
    "python": [
        os.path.join(_PY_CPP, "python"),
    ],
    "rust": [
        os.path.join(_CODE, "rust_stackedu"),
        os.path.join(_CODE, "rust_starcoder"),
        os.path.join(_CODE, "rust_ir"),
    ],
    "shell": [
        os.path.join(_CODE, "shell_stackedu"),
        os.path.join(_CODE, "shell_starcoder"),
    ],
}

OUTPUT_BASE = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus"
SHARDS_DIR  = os.path.join(OUTPUT_BASE, "shards")
LOG_FILE    = os.path.join(OUTPUT_BASE, "assemble_shards.log")


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SHARD PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def shard_path(idx: int) -> str:
    return os.path.join(SHARDS_DIR, f"shard_{idx:02d}.txt.gz")


def shard_done_marker(idx: int) -> str:
    return os.path.join(SHARDS_DIR, f".shard_{idx:02d}.done")


def shard_is_complete(idx: int) -> bool:
    return (
        os.path.exists(shard_done_marker(idx))
        and os.path.exists(shard_path(idx))
        and os.path.getsize(shard_path(idx)) > 0
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK LINE ITERATOR
# ══════════════════════════════════════════════════════════════════════════════

def iter_chunk_dirs(domain: str, dirs: list[str]):
    """
    Yield lines from all chunk_XX.txt.gz files across dirs, in sorted order.

    This is a persistent generator — the caller pulls exactly as many lines
    as needed per shard; the generator resumes from where it left off for
    the next shard. No restarts, no repeated data.
    """
    for d in dirs:
        if not os.path.exists(d):
            logger.warning(f"  [{domain}] Directory not found, skipping: {d}")
            continue

        chunk_files = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(".txt.gz") and not f.startswith(".")
        )

        if not chunk_files:
            logger.warning(f"  [{domain}] No .txt.gz chunk files in: {d}")
            continue

        logger.info(f"  [{domain}] Entering {os.path.basename(d)}/ ({len(chunk_files)} chunks)")

        for cf in chunk_files:
            with gzip.open(cf, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CURSOR
# ══════════════════════════════════════════════════════════════════════════════

class DomainCursor:
    """
    Wraps a per-domain line generator.
    pull(quota_bytes) returns a list of lines totalling ~quota_bytes,
    or fewer if the source is exhausted.
    """
    def __init__(self, domain: str, dirs: list[str]):
        self.domain       = domain
        self.exhausted    = False
        self.total_bytes  = 0          # bytes yielded across all shards
        self._gen         = iter_chunk_dirs(domain, dirs)

    def pull(self, quota_bytes: int) -> tuple[list[str], int]:
        """
        Consume up to quota_bytes from the stream.
        Returns (lines, bytes_consumed).
        """
        if self.exhausted:
            return [], 0

        lines   = []
        pulled  = 0

        try:
            while pulled < quota_bytes:
                line = next(self._gen)
                nb   = len(line.encode("utf-8"))
                lines.append(line)
                pulled          += nb
                self.total_bytes += nb
        except StopIteration:
            if not self.exhausted:
                logger.warning(
                    f"  [{self.domain}] SOURCE EXHAUSTED — "
                    f"{self.total_bytes / GB:.3f} GB consumed total"
                )
            self.exhausted = True

        return lines, pulled


# ══════════════════════════════════════════════════════════════════════════════
# INTERLEAVED WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_interleaved(shard_file, domain_lines: dict[str, list[str]]) -> tuple[int, int]:
    """
    Write lines from all domains in round-robin batches of INTERLEAVE_BATCH.
    Returns (total_lines_written, total_bytes_written).
    """
    total_lines = 0
    total_bytes = 0

    # Build per-domain iterators
    iters  = {d: iter(lines) for d, lines in domain_lines.items()}
    active = list(domain_lines.keys())   # domains still producing lines

    def take_batch(it):
        """Pull up to INTERLEAVE_BATCH lines from iterator; return list."""
        batch = []
        for _ in range(INTERLEAVE_BATCH):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        return batch

    while active:
        still_active = []
        for domain in active:
            batch = take_batch(iters[domain])
            if batch:
                still_active.append(domain)
                for line in batch:
                    shard_file.write(line)
                    total_lines += 1
                    total_bytes += len(line.encode("utf-8"))
        active = still_active

    return total_lines, total_bytes


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_config() -> bool:
    ok = True

    total_quota = sum(DOMAIN_QUOTAS.values())
    logger.info(f"  Quota sum: {total_quota / MB:.0f} MB  (shard target: {SHARD_SIZE_BYTES / MB:.0f} MB)")
    if total_quota > SHARD_SIZE_BYTES:
        logger.error("Quotas exceed shard size — adjust fractions.")
        ok = False

    for domain, dirs in DOMAIN_CHUNK_DIRS.items():
        for d in dirs:
            if not os.path.exists(d):
                logger.error(f"  [{domain}] MISSING: {d}")
                ok = False
            else:
                chunks = [f for f in os.listdir(d) if f.endswith(".txt.gz")]
                sz     = sum(os.path.getsize(os.path.join(d, f)) for f in chunks)
                logger.info(f"  [{domain}]  OK  {d}  ({len(chunks)} chunks, {sz / MB:.0f} MB compressed)")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Code corpus shard assembler — Phase 2")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print plan; do not write any files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 65)
    logger.info("CODE SHARD ASSEMBLER — Phase 2")
    logger.info(f"  Output      : {SHARDS_DIR}")
    logger.info(f"  Num shards  : {NUM_SHARDS}")
    logger.info(f"  Shard size  : {SHARD_SIZE_BYTES / MB:.0f} MB uncompressed")
    logger.info(f"  Total target: {NUM_SHARDS * SHARD_SIZE_BYTES / GB:.0f} GB uncompressed")
    logger.info(f"  Dry run     : {args.dry_run}")
    logger.info("=" * 65)
    logger.info("Per-shard domain quotas:")
    for domain, quota in DOMAIN_QUOTAS.items():
        frac = quota / SHARD_SIZE_BYTES * 100
        logger.info(f"  {domain:<10s}  {quota / MB:>6.0f} MB  ({frac:.0f}%)")
    logger.info(f"  {'TOTAL':<10s}  {sum(DOMAIN_QUOTAS.values()) / MB:>6.0f} MB")
    logger.info("=" * 65)

    logger.info("\nValidating chunk directories ...")
    if not validate_config():
        logger.error("Validation failed — fix errors above and retry.")
        sys.exit(1)
    logger.info("Validation OK.\n")

    if args.dry_run:
        logger.info("DRY RUN complete — no files written.")
        return

    os.makedirs(SHARDS_DIR, exist_ok=True)

    # ── Build persistent cursors — one per domain, shared across all shards ──
    cursors = {
        domain: DomainCursor(domain, DOMAIN_CHUNK_DIRS[domain])
        for domain in DOMAIN_QUOTAS
    }

    t_total = time.time()

    for shard_idx in range(NUM_SHARDS):

        sp   = shard_path(shard_idx)
        done = shard_done_marker(shard_idx)

        if shard_is_complete(shard_idx):
            logger.info(f"[shard_{shard_idx:02d}] SKIP — .done exists (resume)")
            # Fast-forward each cursor by consuming and discarding this shard's quota.
            # Required so the stream position is correct for subsequent shards.
            for domain, cursor in cursors.items():
                cursor.pull(DOMAIN_QUOTAS[domain])
            continue

        # Remove any partial shard from a previous failed run
        if os.path.exists(sp):
            os.remove(sp)
            logger.info(f"[shard_{shard_idx:02d}] Removed incomplete shard from prior run")

        logger.info(f"[shard_{shard_idx:02d}] Assembling ...")
        t0 = time.time()

        # Pull this shard's quota from each domain cursor
        shard_domain_lines = {}
        shard_domain_bytes = {}

        for domain, cursor in cursors.items():
            lines, pulled = cursor.pull(DOMAIN_QUOTAS[domain])
            shard_domain_lines[domain] = lines
            shard_domain_bytes[domain] = pulled

        # Write interleaved to the shard file
        with gzip.open(sp, "wt", encoding="utf-8", compresslevel=3) as sf:
            total_lines, total_bytes = write_interleaved(sf, shard_domain_lines)

        elapsed = time.time() - t0
        compressed_size = os.path.getsize(sp)

        logger.info(
            f"[shard_{shard_idx:02d}] DONE — "
            f"{total_lines:,} lines | "
            f"{total_bytes / MB:.1f} MB uncompressed | "
            f"{compressed_size / MB:.1f} MB on disk | "
            f"{elapsed:.0f}s"
        )
        for domain in DOMAIN_QUOTAS:
            quota  = DOMAIN_QUOTAS[domain]
            actual = shard_domain_bytes[domain]
            pct    = 100 * actual / quota if quota > 0 else 0
            exh    = "  ⚠ EXHAUSTED" if cursors[domain].exhausted else ""
            logger.info(
                f"  {domain:<10s}  {actual / MB:>6.1f} MB  "
                f"(target {quota / MB:.0f} MB, {pct:.0f}%){exh}"
            )

        # Write .done — its presence is the resume signal
        with open(done, "w") as f:
            f.write(f"lines={total_lines}\nbytes={total_bytes}\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_wall = time.time() - t_total
    logger.info("")
    logger.info("=" * 65)
    logger.info("ASSEMBLY COMPLETE")
    logger.info(f"  {NUM_SHARDS} shards in {SHARDS_DIR}")
    logger.info(f"  Wall time: {total_wall:.0f}s")
    logger.info("Domain totals across all shards:")
    for domain, cursor in cursors.items():
        status = "EXHAUSTED" if cursor.exhausted else "ok"
        logger.info(
            f"  {domain:<10s}  {cursor.total_bytes / GB:.3f} GB used  [{status}]"
        )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# Run commands
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Dry run — validate all paths, print plan, write nothing
#    python assemble_code_shards.py --dry-run
#
# 2. Full run
#    tmux new -s code_shards
#    systemd-run --uid=$(id -u) --gid=$(id -g) --scope \
#      -p MemoryMax=40G -p OOMScoreAdjust=-900 \
#      python assemble_code_shards.py 2>&1 | tee -a assemble_shards_stdout.log
#
# 3. Monitor progress (from another pane)
#    watch -n 10 'ls -lh /home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/shards/'
#
# 4. Check which shards are done
#    for i in $(seq -w 0 19); do
#        f=/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/shards/.shard_${i}.done
#        [ -f "$f" ] && echo "✓ shard_$i  $(cat $f | tr '\n' '  ')" || echo "✗ shard_$i"
#    done
#
# 5. Spot-check uncompressed size of one shard
#    zcat /home/sushmetha/.../shards/shard_00.txt.gz | wc -c | awk '{printf "%.2f GB\n", $1/1073741824}'