#!/usr/bin/env python3
"""
build_small_lang_chunks.py — Phase 1: extract small/minority language sources
into per-source chunk files for tokenizer training.

Sources
-------
  sangraha/verified/verified/  — no quality filter, full data except nep:
    asm  (935M),  mai (45M),   san (3.9G),  snd (1.4G),  gom (31M),
    doi  (104K),  mni (4.5M),  sat (191K),  kas (775K)
    nep  (16G)   ← capped at 2 GB extracted text

  sangraha/synthetic/synthetic/  — capped at 2 GB extracted text each:
    asm_Beng  (17G raw)
    san_Deva  (18G raw)

All sources: text column = "text", no quality score filter.

Output layout
-------------
  CHUNKS_DIR/
    asm/
      chunk_00.txt.gz
      ...
      .done
    mai/
    san/
    ...
    nep/
    asm_beng/
    san_deva/

Each chunk is ~85 MB of uncompressed text.
Documents are stripped and written one per line (internal newlines → space).

Resume
------
  Source with .done present → skipped entirely.
  Source with partial chunks but no .done → cleaned and restarted.

Usage
-----
  python build_small_lang_chunks.py                     # all sources
  python build_small_lang_chunks.py --source nep        # single source
  python build_small_lang_chunks.py --dry-run           # stats only
  python build_small_lang_chunks.py --workers 4         # parallel
"""

import argparse
import gzip
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MB = 1024 ** 2
GB = 1024 ** 3

CHUNK_TARGET_BYTES = 85 * MB      # uncompressed bytes per chunk file
MIN_CONTENT_CHARS  = 5            # short threshold — preserves short Indic sentences

_VERIFIED  = "/home/sushmetha/raw_storage_mount/Multilingual_data/sangraha/verified/verified"
_SYNTHETIC = "/home/sushmetha/raw_storage_mount/Multilingual_data/sangraha/synthetic/synthetic"

OUTPUT_BASE = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_small_langs"
CHUNKS_DIR  = os.path.join(OUTPUT_BASE, "chunks")
LOG_FILE    = os.path.join(OUTPUT_BASE, "build_chunks.log")

NO_CAP = None   # sentinel: stream all available data

# ── Source definitions ────────────────────────────────────────────────────────
# cap_bytes: maximum extracted text bytes to consume (None = no cap)
SOURCES = {
    # ── verified, uncapped ────────────────────────────────────────────────
    "asm": {
        "path":      os.path.join(_VERIFIED, "asm"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "mai": {
        "path":      os.path.join(_VERIFIED, "mai"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "san": {
        "path":      os.path.join(_VERIFIED, "san"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "snd": {
        "path":      os.path.join(_VERIFIED, "snd"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "gom": {
        "path":      os.path.join(_VERIFIED, "gom"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "doi": {
        "path":      os.path.join(_VERIFIED, "doi"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "mni": {
        "path":      os.path.join(_VERIFIED, "mni"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "sat": {
        "path":      os.path.join(_VERIFIED, "sat"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "kas": {
        "path":      os.path.join(_VERIFIED, "kas"),
        "text_col":  "text",
        "cap_bytes": NO_CAP,
        "min_chars": MIN_CONTENT_CHARS,
    },
    # ── verified, capped ─────────────────────────────────────────────────
    "nep": {
        "path":      os.path.join(_VERIFIED, "nep"),
        "text_col":  "text",
        "cap_bytes": 2 * GB,
        "min_chars": MIN_CONTENT_CHARS,
    },
    # ── synthetic, capped ────────────────────────────────────────────────
    "asm_beng": {
        "path":      os.path.join(_SYNTHETIC, "asm_Beng"),
        "text_col":  "text",
        "cap_bytes": 2 * GB,
        "min_chars": MIN_CONTENT_CHARS,
    },
    "san_deva": {
        "path":      os.path.join(_SYNTHETIC, "san_Deva"),
        "text_col":  "text",
        "cap_bytes": 2 * GB,
        "min_chars": MIN_CONTENT_CHARS,
    },
}


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
# CHUNK PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def source_chunk_dir(name: str) -> str:
    return os.path.join(CHUNKS_DIR, name)

def chunk_path(name: str, idx: int) -> str:
    return os.path.join(source_chunk_dir(name), f"chunk_{idx:02d}.txt.gz")

def done_marker(name: str) -> str:
    return os.path.join(source_chunk_dir(name), ".done")

def source_is_done(name: str) -> bool:
    return os.path.exists(done_marker(name))

def clean_partial_chunks(name: str):
    d = source_chunk_dir(name)
    if os.path.exists(d):
        removed = 0
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
            removed += 1
        if removed:
            logger.info(f"  [{name}] Cleaned {removed} partial files from previous run")
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET ITERATOR
# ══════════════════════════════════════════════════════════════════════════════

def iter_parquet_dir(name: str, path: str, text_col: str,
                     min_chars: int, cap_bytes: int | None):
    """
    Stream all .parquet files in path, yield cleaned text strings.
    Stops early if cap_bytes of extracted text has been consumed.
    """
    files = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {path}")

    logger.info(
        f"  [{name}] {len(files)} parquet files in {path}"
        + (f"  [cap: {cap_bytes / GB:.1f} GB]" if cap_bytes else "  [no cap]")
    )

    bytes_extracted = 0

    for fpath in files:
        logger.info(f"  [{name}] Reading {os.path.basename(fpath)}")
        pf = pq.ParquetFile(fpath)

        for batch in pf.iter_batches(batch_size=2000, columns=[text_col]):
            texts = batch.to_pydict()[text_col]

            for text in texts:
                if not text:
                    continue
                text = str(text).strip()
                if len(text) < min_chars:
                    continue

                nb = len(text.encode("utf-8"))
                bytes_extracted += nb

                yield text

                # Check cap after yielding — so the caller always gets at least
                # the bytes it asked for rather than being cut short mid-doc
                if cap_bytes is not None and bytes_extracted >= cap_bytes:
                    logger.info(
                        f"  [{name}] Cap reached: "
                        f"{bytes_extracted / GB:.3f} GB extracted "
                        f"(cap {cap_bytes / GB:.1f} GB) — stopping early"
                    )
                    return


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PROCESS ONE SOURCE
# ══════════════════════════════════════════════════════════════════════════════

def process_source(name: str, cfg: dict, dry_run: bool = False):
    if source_is_done(name):
        logger.info(f"[{name}] SKIP — .done exists (resume)")
        return

    cap_str = f"{cfg['cap_bytes'] / GB:.1f} GB cap" if cfg["cap_bytes"] else "no cap"
    logger.info(f"[{name}] Starting  ({cap_str}, chunk_target={CHUNK_TARGET_BYTES / MB:.0f} MB) ...")

    if not dry_run:
        clean_partial_chunks(name)

    total_docs    = 0
    total_bytes   = 0
    chunk_idx     = 0
    chunk_bytes   = 0
    docs_in_chunk = 0
    chunk_summary = []    # (idx, docs, bytes) per finished chunk
    chunk_handle  = None
    t0 = time.time()

    def open_next_chunk():
        p = chunk_path(name, chunk_idx)
        logger.info(f"  [{name}] Opening chunk_{chunk_idx:02d}.txt.gz")
        return gzip.open(p, "wt", encoding="utf-8", compresslevel=6)

    def close_current_chunk():
        chunk_handle.close()
        logger.info(
            f"  [{name}] Closed  chunk_{chunk_idx:02d}.txt.gz  "
            f"({chunk_bytes / MB:.1f} MB uncompressed, {docs_in_chunk:,} docs)"
        )
        chunk_summary.append((chunk_idx, docs_in_chunk, chunk_bytes))

    try:
        if not dry_run:
            chunk_handle = open_next_chunk()

        for doc in iter_parquet_dir(
            name,
            cfg["path"],
            cfg["text_col"],
            cfg["min_chars"],
            cfg["cap_bytes"],
        ):
            line = doc.replace("\n", " ").replace("\r", " ").strip() + "\n"
            nb   = len(line.encode("utf-8"))

            if not dry_run:
                if chunk_bytes >= CHUNK_TARGET_BYTES:
                    close_current_chunk()
                    chunk_idx    += 1
                    chunk_bytes   = 0
                    docs_in_chunk = 0
                    chunk_handle  = open_next_chunk()

                chunk_handle.write(line)
                chunk_bytes   += nb
                docs_in_chunk += 1

            total_bytes += nb
            total_docs  += 1

            if total_docs % 50_000 == 0:
                elapsed = time.time() - t0
                rate    = total_bytes / elapsed / MB if elapsed > 0 else 0
                logger.info(
                    f"  [{name}] {total_docs:,} docs | "
                    f"{total_bytes / GB:.3f} GB | "
                    f"{rate:.1f} MB/s | "
                    f"chunk_{chunk_idx:02d}"
                )

    except Exception as e:
        logger.error(f"[{name}] ERROR: {e}", exc_info=True)
        if chunk_handle:
            try:
                chunk_handle.close()
            except Exception:
                pass
        raise

    # Close final chunk
    if not dry_run and chunk_handle:
        close_current_chunk()

    num_chunks = len(chunk_summary)
    elapsed    = time.time() - t0

    logger.info(
        f"[{name}] {'DRY RUN ' if dry_run else ''}DONE — "
        f"{total_docs:,} docs | "
        f"{total_bytes / GB:.3f} GB uncompressed | "
        f"{num_chunks} chunk(s) | "
        f"{elapsed:.0f}s"
    )

    if not dry_run:
        with open(done_marker(name), "w") as f:
            f.write(
                f"docs={total_docs}\n"
                f"bytes={total_bytes}\n"
                f"chunks={num_chunks}\n"
            )
        logger.info(f"[{name}] .done written")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Small language chunk builder — Phase 1")
    p.add_argument(
        "--source", type=str, default=None,
        help=f"Process a single source. Valid: {list(SOURCES.keys())}"
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (one per source). Default: 1 (sequential)."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Iterate data and report stats; write nothing to disk."
    )
    return p.parse_args()


def _worker(args_tuple):
    name, cfg, dry_run = args_tuple
    setup_logging()
    try:
        process_source(name, cfg, dry_run=dry_run)
        return name, "done"
    except Exception as e:
        return name, f"error: {e}"


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 65)
    logger.info("SMALL LANGUAGE CHUNK BUILDER — Phase 1")
    logger.info(f"  Output     : {CHUNKS_DIR}")
    logger.info(f"  Chunk size : ~{CHUNK_TARGET_BYTES / MB:.0f} MB uncompressed")
    logger.info(f"  Min chars  : {MIN_CONTENT_CHARS}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info("=" * 65)
    logger.info("Sources:")
    for name, cfg in SOURCES.items():
        cap = f"{cfg['cap_bytes'] / GB:.1f} GB cap" if cfg["cap_bytes"] else "no cap"
        logger.info(f"  {name:<12s}  {cap:<14s}  {cfg['path']}")
    logger.info("=" * 65)

    if args.source:
        if args.source not in SOURCES:
            logger.error(f"Unknown source '{args.source}'. Valid: {list(SOURCES.keys())}")
            sys.exit(1)
        sources_to_run = {args.source: SOURCES[args.source]}
    else:
        sources_to_run = SOURCES

    t_start = time.time()

    if args.workers == 1 or len(sources_to_run) == 1:
        for name, cfg in sources_to_run.items():
            try:
                process_source(name, cfg, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"[{name}] FAILED — continuing. Error: {e}")
    else:
        num_workers = min(args.workers, len(sources_to_run))
        logger.info(f"Launching {num_workers} parallel workers ...")
        tasks = [(name, cfg, args.dry_run) for name, cfg in sources_to_run.items()]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                name, status = future.result()
                if status == "done":
                    logger.info(f"[{name}] ✓ finished")
                else:
                    logger.error(f"[{name}] ✗ FAILED — {status}")

    logger.info(f"\nPHASE 1 COMPLETE — wall time: {time.time() - t_start:.0f}s")
    logger.info(f"Chunks written to: {CHUNKS_DIR}")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# Run commands
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Dry run — validate paths, no output
#    python build_small_lang_chunks.py --dry-run
#
# 2. Single source test (start with a tiny one)
#    python build_small_lang_chunks.py --source doi --dry-run
#    python build_small_lang_chunks.py --source doi
#
# 3. Full sequential run (recommended on gcsfuse)
#    tmux new -s small_lang_chunks
#    systemd-run --uid=$(id -u) --gid=$(id -g) --scope \
#      -p MemoryMax=40G -p OOMScoreAdjust=-900 \
#      python build_small_lang_chunks.py 2>&1 | tee -a small_lang_chunks_stdout.log
#
# 4. Parallel run — safe for sources on different gcsfuse paths
#    python build_small_lang_chunks.py --workers 4
#
# 5. Check which sources are done
#    for src in asm mai san snd gom doi mni sat kas nep asm_beng san_deva; do
#        f=/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_small_langs/chunks/$src/.done
#        [ -f "$f" ] && echo "✓ $src  $(cat $f | tr '\n' '  ')" || echo "✗ $src"
#    done