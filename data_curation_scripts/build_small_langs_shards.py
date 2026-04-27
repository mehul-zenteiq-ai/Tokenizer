#!/usr/bin/env python3
"""
assemble_small_lang_shards.py — Phase 2: assemble small-language chunks into shards.

Two phases run sequentially:

PHASE A — small langs (one shard each, named by lang code)
──────────────────────────────────────────────────────────
  doi, sat, kas, mni, gom, mai
  All data from that lang's chunk dir is written into a single shard.
  Output: shard_doi.txt.gz, shard_sat.txt.gz, ...

PHASE B — large langs mixed proportionally
──────────────────────────────────────────
  Langs: asm (asm/ + asm_beng/), san (san/ + san_deva/), snd, nep
  Proportional weights derived from compressed chunk sizes:
    asm  922 MB  → ~19.5 %
    san 2593 MB  → ~54.9 %
    snd  798 MB  → ~16.9 %
    nep  413 MB  → ~ 8.7 %
  Target shard size: ~1.1 GB uncompressed (≈350 MB compressed at 3× ratio)
  Assembly continues until ALL large-lang cursors are exhausted.
  Output: shard_mixed_00.txt.gz, shard_mixed_01.txt.gz, ...

Chunk dirs expected (produced by build_small_lang_chunks.py)
─────────────────────────────────────────────────────────────
  CHUNKS_DIR/asm/       CHUNKS_DIR/asm_beng/
  CHUNKS_DIR/san/       CHUNKS_DIR/san_deva/
  CHUNKS_DIR/snd/       CHUNKS_DIR/nep/
  CHUNKS_DIR/doi/       CHUNKS_DIR/sat/
  CHUNKS_DIR/kas/       CHUNKS_DIR/mni/
  CHUNKS_DIR/gom/       CHUNKS_DIR/mai/

Output layout
─────────────
  SHARDS_DIR/
    shard_doi.txt.gz          .shard_doi.done
    shard_sat.txt.gz          .shard_sat.done
    shard_kas.txt.gz          .shard_kas.done
    shard_mni.txt.gz          .shard_mni.done
    shard_gom.txt.gz          .shard_gom.done
    shard_mai.txt.gz          .shard_mai.done
    shard_mixed_00.txt.gz     .shard_mixed_00.done
    shard_mixed_01.txt.gz     .shard_mixed_01.done
    ...

Resume
──────
  Any shard whose .done marker exists is skipped.
  Mixed-shard cursors fast-forward through completed shards to maintain
  the correct stream position before resuming.

Usage
─────
  python assemble_small_lang_shards.py              # full run
  python assemble_small_lang_shards.py --dry-run    # validate + print plan only
  python assemble_small_lang_shards.py --phase a    # small-lang shards only
  python assemble_small_lang_shards.py --phase b    # mixed shards only
"""

import argparse
import gzip
import logging
import os
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MB = 1024 ** 2
GB = 1024 ** 3

# Target uncompressed bytes per mixed shard (≈ 350 MB compressed at ~3× ratio)
MIXED_SHARD_TARGET_BYTES = int(1.1 * GB)

# Lines written per domain per interleave pass inside each mixed shard
INTERLEAVE_BATCH = 500

CHUNKS_DIR  = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_small_langs/chunks"
SHARDS_DIR  = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_small_langs/shards"
LOG_FILE    = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_small_langs/assemble_shards.log"

# ── Phase A: small langs ──────────────────────────────────────────────────────
# Each gets one shard file named shard_{code}.txt.gz
SMALL_LANGS = ["doi", "sat", "kas", "mni", "gom", "mai"]

# ── Phase B: large langs ──────────────────────────────────────────────────────
# chunk_dirs: directories streamed in order per domain
# weight    : proportional to compressed chunk size on disk (MB)
#             asm=922, san=2593, snd=798, nep=413  → total 4726 MB
LARGE_LANGS = {
    "asm": {
        "chunk_dirs": [
            os.path.join(CHUNKS_DIR, "asm"),
            os.path.join(CHUNKS_DIR, "asm_beng"),
        ],
        "weight": 922,
    },
    "san": {
        "chunk_dirs": [
            os.path.join(CHUNKS_DIR, "san"),
            os.path.join(CHUNKS_DIR, "san_deva"),
        ],
        "weight": 2593,
    },
    "snd": {
        "chunk_dirs": [
            os.path.join(CHUNKS_DIR, "snd"),
        ],
        "weight": 798,
    },
    "nep": {
        "chunk_dirs": [
            os.path.join(CHUNKS_DIR, "nep"),
        ],
        "weight": 413,
    },
}

# Derive per-shard byte quotas from weights
_total_weight = sum(cfg["weight"] for cfg in LARGE_LANGS.values())
LARGE_LANG_QUOTAS = {
    lang: int(cfg["weight"] / _total_weight * MIXED_SHARD_TARGET_BYTES)
    for lang, cfg in LARGE_LANGS.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
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

def small_shard_path(lang: str) -> str:
    return os.path.join(SHARDS_DIR, f"shard_{lang}.txt.gz")

def small_done_marker(lang: str) -> str:
    return os.path.join(SHARDS_DIR, f".shard_{lang}.done")

def mixed_shard_path(idx: int) -> str:
    return os.path.join(SHARDS_DIR, f"shard_mixed_{idx:02d}.txt.gz")

def mixed_done_marker(idx: int) -> str:
    return os.path.join(SHARDS_DIR, f".shard_mixed_{idx:02d}.done")

def shard_complete(path: str, done: str) -> bool:
    return (
        os.path.exists(done)
        and os.path.exists(path)
        and os.path.getsize(path) > 0
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK LINE ITERATOR
# ══════════════════════════════════════════════════════════════════════════════

def iter_chunk_dirs(label: str, dirs: list[str]):
    """
    Yield lines from all chunk_XX.txt.gz files across dirs, sorted order.
    Persistent generator — resumes from where it left off across shard boundaries.
    """
    for d in dirs:
        if not os.path.exists(d):
            logger.warning(f"  [{label}] Directory not found, skipping: {d}")
            continue

        chunk_files = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(".txt.gz") and not f.startswith(".")
        )

        if not chunk_files:
            logger.warning(f"  [{label}] No .txt.gz chunks in: {d}")
            continue

        logger.info(f"  [{label}] Entering {os.path.basename(d)}/ ({len(chunk_files)} chunks)")

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
    def __init__(self, lang: str, dirs: list[str]):
        self.lang        = lang
        self.exhausted   = False
        self.total_bytes = 0
        self._gen        = iter_chunk_dirs(lang, dirs)

    def pull(self, quota_bytes: int) -> tuple[list[str], int]:
        """Pull up to quota_bytes of lines. Returns (lines, bytes_pulled)."""
        if self.exhausted:
            return [], 0

        lines  = []
        pulled = 0

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
                    f"  [{self.lang}] SOURCE EXHAUSTED — "
                    f"{self.total_bytes / GB:.3f} GB total consumed"
                )
            self.exhausted = True

        return lines, pulled


# ══════════════════════════════════════════════════════════════════════════════
# INTERLEAVED WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_interleaved(shard_file, domain_lines: dict[str, list[str]]) -> tuple[int, int]:
    """
    Write lines from all domains in round-robin batches of INTERLEAVE_BATCH.
    Returns (total_lines, total_bytes).
    """
    total_lines = 0
    total_bytes = 0

    iters  = {lang: iter(lines) for lang, lines in domain_lines.items() if lines}
    active = list(iters.keys())

    while active:
        still_active = []
        for lang in active:
            batch = []
            it    = iters[lang]
            for _ in range(INTERLEAVE_BATCH):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break

            if batch:
                still_active.append(lang)
                for line in batch:
                    shard_file.write(line)
                    total_lines += 1
                    total_bytes += len(line.encode("utf-8"))

        active = still_active

    return total_lines, total_bytes


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — small lang shards (one per lang)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase_a(dry_run: bool):
    logger.info("")
    logger.info("── PHASE A: Small language shards ──────────────────────────")

    for lang in SMALL_LANGS:
        sp   = small_shard_path(lang)
        done = small_done_marker(lang)

        if shard_complete(sp, done):
            logger.info(f"[{lang}] SKIP — shard_{{lang}}.txt.gz already complete")
            continue

        chunk_dir = os.path.join(CHUNKS_DIR, lang)
        if not os.path.exists(chunk_dir):
            logger.error(f"[{lang}] Chunk dir not found: {chunk_dir} — skipping")
            continue

        chunk_files = sorted(
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".txt.gz") and not f.startswith(".")
        )
        if not chunk_files:
            logger.warning(f"[{lang}] No chunk files found — skipping")
            continue

        logger.info(f"[{lang}] Writing shard_{lang}.txt.gz ...")

        if dry_run:
            logger.info(f"[{lang}] DRY RUN — would read {len(chunk_files)} chunk(s)")
            continue

        if os.path.exists(sp):
            os.remove(sp)

        t0          = time.time()
        total_lines = 0
        total_bytes = 0

        with gzip.open(sp, "wt", encoding="utf-8", compresslevel=6) as sf:
            for cf in chunk_files:
                with gzip.open(cf, "rt", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        line += "\n"
                        sf.write(line)
                        total_lines += 1
                        total_bytes += len(line.encode("utf-8"))

        elapsed = time.time() - t0
        logger.info(
            f"[{lang}] DONE — {total_lines:,} lines | "
            f"{total_bytes / MB:.1f} MB uncompressed | "
            f"{os.path.getsize(sp) / MB:.1f} MB on disk | "
            f"{elapsed:.0f}s"
        )

        with open(done, "w") as f:
            f.write(f"lines={total_lines}\nbytes={total_bytes}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B — large lang mixed shards
# ══════════════════════════════════════════════════════════════════════════════

def run_phase_b(dry_run: bool):
    logger.info("")
    logger.info("── PHASE B: Mixed large-language shards ────────────────────")
    logger.info("Per-shard quotas:")
    for lang, quota in LARGE_LANG_QUOTAS.items():
        frac = quota / MIXED_SHARD_TARGET_BYTES * 100
        logger.info(f"  {lang:<6s}  {quota / MB:>6.0f} MB  ({frac:.1f}%)")
    logger.info(f"  {'TOTAL':<6s}  {sum(LARGE_LANG_QUOTAS.values()) / MB:>6.0f} MB")

    if dry_run:
        logger.info("DRY RUN — skipping shard writes")
        return

    os.makedirs(SHARDS_DIR, exist_ok=True)

    # Build persistent cursors — span all mixed shards
    cursors = {
        lang: DomainCursor(lang, cfg["chunk_dirs"])
        for lang, cfg in LARGE_LANGS.items()
    }

    shard_idx = 0
    t_total   = time.time()

    while True:
        sp   = mixed_shard_path(shard_idx)
        done = mixed_done_marker(shard_idx)

        # Check if all cursors are exhausted before attempting this shard
        # (must be checked AFTER fast-forward if resuming)
        if shard_complete(sp, done):
            logger.info(f"[mixed_{shard_idx:02d}] SKIP — already complete (resume)")
            # Fast-forward cursors to maintain correct stream position
            for lang, cursor in cursors.items():
                cursor.pull(LARGE_LANG_QUOTAS[lang])
            shard_idx += 1
            continue

        # Pull this shard's quota from each cursor
        shard_domain_lines = {}
        shard_domain_bytes = {}

        for lang, cursor in cursors.items():
            lines, pulled = cursor.pull(LARGE_LANG_QUOTAS[lang])
            shard_domain_lines[lang] = lines
            shard_domain_bytes[lang] = pulled

        # If every cursor returned nothing, we're done
        total_pulled = sum(shard_domain_bytes.values())
        if total_pulled == 0:
            logger.info("All large-lang cursors exhausted — Phase B complete")
            break

        # Remove any partial shard from a prior failed run
        if os.path.exists(sp):
            os.remove(sp)

        logger.info(f"[mixed_{shard_idx:02d}] Assembling ...")
        t0 = time.time()

        with gzip.open(sp, "wt", encoding="utf-8", compresslevel=6) as sf:
            total_lines, total_bytes = write_interleaved(sf, shard_domain_lines)

        elapsed         = time.time() - t0
        compressed_size = os.path.getsize(sp)

        logger.info(
            f"[mixed_{shard_idx:02d}] DONE — "
            f"{total_lines:,} lines | "
            f"{total_bytes / MB:.1f} MB uncompressed | "
            f"{compressed_size / MB:.1f} MB on disk | "
            f"{elapsed:.0f}s"
        )
        for lang in LARGE_LANGS:
            quota  = LARGE_LANG_QUOTAS[lang]
            actual = shard_domain_bytes[lang]
            pct    = 100 * actual / quota if quota > 0 else 0
            exh    = "  ⚠ EXHAUSTED" if cursors[lang].exhausted else ""
            logger.info(
                f"  {lang:<6s}  {actual / MB:>6.1f} MB  "
                f"(target {quota / MB:.0f} MB, {pct:.0f}%){exh}"
            )

        with open(done, "w") as f:
            f.write(f"lines={total_lines}\nbytes={total_bytes}\n")

        shard_idx += 1

    # Summary
    total_wall = time.time() - t_total
    logger.info("")
    logger.info(f"Phase B: {shard_idx} mixed shards written in {total_wall:.0f}s")
    logger.info("Domain totals:")
    for lang, cursor in cursors.items():
        status = "EXHAUSTED" if cursor.exhausted else "ok"
        logger.info(f"  {lang:<6s}  {cursor.total_bytes / GB:.3f} GB  [{status}]")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    ok = True
    logger.info("Validating chunk directories ...")

    for lang in SMALL_LANGS:
        d = os.path.join(CHUNKS_DIR, lang)
        if not os.path.exists(d):
            logger.error(f"  [small/{lang}] MISSING: {d}")
            ok = False
        else:
            n = len([f for f in os.listdir(d) if f.endswith(".txt.gz")])
            sz = sum(os.path.getsize(os.path.join(d, f))
                     for f in os.listdir(d) if f.endswith(".txt.gz"))
            logger.info(f"  [small/{lang}]  OK  {n} chunk(s), {sz / MB:.0f} MB compressed")

    for lang, cfg in LARGE_LANGS.items():
        for d in cfg["chunk_dirs"]:
            if not os.path.exists(d):
                logger.error(f"  [large/{lang}] MISSING: {d}")
                ok = False
            else:
                n = len([f for f in os.listdir(d) if f.endswith(".txt.gz")])
                sz = sum(os.path.getsize(os.path.join(d, f))
                         for f in os.listdir(d) if f.endswith(".txt.gz"))
                logger.info(f"  [large/{lang}]  OK  {d}  ({n} chunks, {sz / MB:.0f} MB compressed)")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Small language shard assembler — Phase 2")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate and print plan; write nothing.")
    p.add_argument("--phase", choices=["a", "b", "both"], default="both",
                   help="Which phase to run (default: both).")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 65)
    logger.info("SMALL LANGUAGE SHARD ASSEMBLER — Phase 2")
    logger.info(f"  Chunks dir  : {CHUNKS_DIR}")
    logger.info(f"  Shards dir  : {SHARDS_DIR}")
    logger.info(f"  Mixed target: ~{MIXED_SHARD_TARGET_BYTES / MB:.0f} MB uncompressed "
                f"(≈350 MB compressed)")
    logger.info(f"  Phase       : {args.phase}")
    logger.info(f"  Dry run     : {args.dry_run}")
    logger.info("=" * 65)

    if not validate():
        logger.error("Validation failed — fix missing paths and retry.")
        sys.exit(1)
    logger.info("Validation OK.")

    if not args.dry_run:
        os.makedirs(SHARDS_DIR, exist_ok=True)

    t_start = time.time()

    if args.phase in ("a", "both"):
        run_phase_a(dry_run=args.dry_run)

    if args.phase in ("b", "both"):
        run_phase_b(dry_run=args.dry_run)

    logger.info("")
    logger.info("=" * 65)
    logger.info(f"COMPLETE — total wall time: {time.time() - t_start:.0f}s")
    logger.info(f"Shards written to: {SHARDS_DIR}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# Run commands
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Dry run — validate all paths, print plan
#    python assemble_small_lang_shards.py --dry-run
#
# 2. Full run
#    tmux new -s small_shards
#    systemd-run --uid=$(id -u) --gid=$(id -g) --scope \
#      -p MemoryMax=40G -p OOMScoreAdjust=-900 \
#      python assemble_small_lang_shards.py 2>&1 | tee -a small_shards_stdout.log
#
# 3. Phase A only (small lang shards)
#    python assemble_small_lang_shards.py --phase a
#
# 4. Phase B only (mixed large-lang shards)
#    python assemble_small_lang_shards.py --phase b
#
# 5. Check progress
#    ls -lh ~/raw_storage_mount/Tokenizer_Data/Curated_small_langs/shards/
#
# 6. Check which shards are done
#    for f in ~/raw_storage_mount/Tokenizer_Data/Curated_small_langs/shards/.*.done; do
#        echo "✓ $(basename $f .done | sed 's/^\.//')  $(cat $f | tr '\n' '  ')"
#    done
