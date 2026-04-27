#!/usr/bin/env python3
"""
build_code_chunks.py — Phase 1: extract code sources into per-source chunk files.

Sources
-------
  stack_edu      (parquet, text col,    int_score >= 3):
    cpp, python, rust_stackedu, shell_stackedu

  starcoderdata  (parquet, content col, max_stars_count >= 5):
    rust_starcoder, shell_starcoder

  starcoder2-data-extras (parquet, content col, no quality filter):
    rust_ir

Output layout
-------------
  CHUNKS_DIR/
    cpp/
      chunk_00.txt.gz
      chunk_01.txt.gz
      ...
      .done                  ← written only after all chunks close cleanly
    python/
      ...
    rust_stackedu/
      ...
    ...

Each chunk is ~CHUNK_TARGET_BYTES of uncompressed text (default 85 MB).
A new chunk file is opened whenever the current one crosses the threshold.
Documents are newline-cleaned (internal \\n → space) and written one per line.

Resume
------
  If a source's .done file exists, that source is skipped entirely.
  If .done is absent but partial chunk files exist, they are cleaned before restart.

Usage
-----
  python build_code_chunks.py                        # all sources, sequential
  python build_code_chunks.py --source cpp           # single source (good for testing)
  python build_code_chunks.py --dry-run              # stats only, no files written
  python build_code_chunks.py --workers 4            # parallel across sources
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
# CONFIG — edit paths here only
# ══════════════════════════════════════════════════════════════════════════════

MB = 1024 ** 2
GB = 1024 ** 3

CHUNK_TARGET_BYTES = 85 * MB        # uncompressed bytes per chunk file

STACK_EDU_BASE  = "/home/sushmetha/raw_storage_mount/code/stack_edu"
STARCODER_BASE  = "/home/sushmetha/raw_storage_mount/code/starcoderdata"
STARCODER2_BASE = "/home/sushmetha/raw_storage_mount/code/starcoder2-data-extras"

OUTPUT_BASE = "/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus"
CHUNKS_DIR  = os.path.join(OUTPUT_BASE, "chunks")
LOG_FILE    = os.path.join(OUTPUT_BASE, "build_chunks.log")

MIN_CONTENT_CHARS = 100

# ── Source definitions ────────────────────────────────────────────────────────
#
# path      : directory containing .parquet files
# text_col  : column that holds the code text
# filters   : {col_name: (operator, value)} — empty dict = no quality filter
#             supported operators: ">=" "<=" ">" "<" "=="
# min_chars : minimum document character length (applied after filters)
#
SOURCES = {
    "cpp": {
        "path":      os.path.join(STACK_EDU_BASE, "Cpp"),
        "text_col":  "text",
        "filters":   {"int_score": (">=", 3)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "python": {
        "path":      os.path.join(STACK_EDU_BASE, "Python"),
        "text_col":  "text",
        "filters":   {"int_score": (">=", 3)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "rust_stackedu": {
        "path":      os.path.join(STACK_EDU_BASE, "Rust"),
        "text_col":  "text",
        "filters":   {"int_score": (">=", 3)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "shell_stackedu": {
        "path":      os.path.join(STACK_EDU_BASE, "Shell"),
        "text_col":  "text",
        "filters":   {"int_score": (">=", 3)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "rust_starcoder": {
        "path":      os.path.join(STARCODER_BASE, "rust"),
        "text_col":  "content",
        "filters":   {"max_stars_count": (">=", 5)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "shell_starcoder": {
        "path":      os.path.join(STARCODER_BASE, "shell"),
        "text_col":  "content",
        "filters":   {"max_stars_count": (">=", 5)},
        "min_chars": MIN_CONTENT_CHARS,
    },
    "rust_ir": {
        "path":      os.path.join(STARCODER2_BASE, "ir_rust"),
        "text_col":  "content",
        "filters":   {},            # no quality filter — take everything
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

def source_chunk_dir(source_name: str) -> str:
    return os.path.join(CHUNKS_DIR, source_name)


def chunk_path(source_name: str, idx: int) -> str:
    return os.path.join(source_chunk_dir(source_name), f"chunk_{idx:02d}.txt.gz")


def done_marker(source_name: str) -> str:
    return os.path.join(source_chunk_dir(source_name), ".done")


def source_is_done(source_name: str) -> bool:
    return os.path.exists(done_marker(source_name))


def clean_partial_chunks(source_name: str):
    """Remove any partial files from a previous incomplete run."""
    d = source_chunk_dir(source_name)
    if os.path.exists(d):
        removed = 0
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
            removed += 1
        if removed:
            logger.info(f"  [{source_name}] Cleaned {removed} partial files from previous run")
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET ITERATOR WITH INLINE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def iter_parquet_dir(source_name: str, path: str, text_col: str,
                     filters: dict, min_chars: int):
    """
    Stream all .parquet files in path, yield cleaned document strings.

    Quality filters are applied per-row using plain Python comparisons.
    Filter columns not present in a file's schema are silently ignored
    for that file (with a one-time warning).
    """
    files = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {path}")

    logger.info(f"  [{source_name}] Found {len(files)} parquet files in {path}")

    op_fns = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b,
        "<":  lambda a, b: a <  b,
        "==": lambda a, b: a == b,
    }

    for fpath in files:
        logger.info(f"  [{source_name}] Reading {os.path.basename(fpath)}")
        pf = pq.ParquetFile(fpath)
        schema_names = set(pf.schema_arrow.names)

        # Determine which filter columns actually exist in this file
        read_cols = [text_col]
        active_filters = {}
        for col, (op, val) in filters.items():
            if col in schema_names:
                read_cols.append(col)
                active_filters[col] = (op_fns[op], val)
            else:
                logger.warning(
                    f"  [{source_name}] Filter column '{col}' not in schema of "
                    f"{os.path.basename(fpath)} — skipping filter for this file"
                )

        for batch in pf.iter_batches(batch_size=2000, columns=read_cols):
            d = batch.to_pydict()
            texts = d[text_col]

            for i, text in enumerate(texts):
                if not text:
                    continue
                text = str(text).strip()
                if len(text) < min_chars:
                    continue

                # Apply all active quality filters
                passed = True
                for col, (cmp_fn, val) in active_filters.items():
                    score = d[col][i]
                    if score is None or not cmp_fn(score, val):
                        passed = False
                        break

                if passed:
                    yield text


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PROCESS ONE SOURCE
# ══════════════════════════════════════════════════════════════════════════════

def process_source(source_name: str, cfg: dict, dry_run: bool = False):
    """
    Read all data for source_name, write to sequentially-numbered chunk files.
    A new chunk is opened whenever the current one exceeds CHUNK_TARGET_BYTES.
    """
    if source_is_done(source_name):
        logger.info(f"[{source_name}] SKIP — .done exists (resume)")
        return

    logger.info(f"[{source_name}] Starting  (chunk_target={CHUNK_TARGET_BYTES / MB:.0f} MB) ...")

    if not dry_run:
        clean_partial_chunks(source_name)

    total_docs        = 0
    total_bytes       = 0
    chunk_idx         = 0
    chunk_bytes       = 0      # uncompressed bytes written to current chunk
    docs_in_chunk     = 0      # docs in current chunk
    chunk_summary     = []     # (idx, docs, bytes) for each finished chunk
    chunk_handle      = None
    t0 = time.time()

    def open_next_chunk():
        p = chunk_path(source_name, chunk_idx)
        logger.info(f"  [{source_name}] Opening chunk_{chunk_idx:02d}.txt.gz")
        return gzip.open(p, "wt", encoding="utf-8", compresslevel=6)

    def close_current_chunk():
        chunk_handle.close()
        logger.info(
            f"  [{source_name}] Closed  chunk_{chunk_idx:02d}.txt.gz  "
            f"({chunk_bytes / MB:.1f} MB uncompressed, {docs_in_chunk:,} docs)"
        )
        chunk_summary.append((chunk_idx, docs_in_chunk, chunk_bytes))

    try:
        if not dry_run:
            chunk_handle = open_next_chunk()

        for doc in iter_parquet_dir(
            source_name,
            cfg["path"],
            cfg["text_col"],
            cfg["filters"],
            cfg["min_chars"],
        ):
            line = doc.replace("\n", " ").replace("\r", " ").strip() + "\n"
            nb   = len(line.encode("utf-8"))

            if not dry_run:
                # Roll over to a new chunk when current one is full
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
                    f"  [{source_name}] {total_docs:,} docs | "
                    f"{total_bytes / GB:.3f} GB | "
                    f"{rate:.1f} MB/s | "
                    f"chunk_{chunk_idx:02d}"
                )

    except Exception as e:
        logger.error(f"[{source_name}] ERROR during iteration: {e}", exc_info=True)
        if chunk_handle:
            try:
                chunk_handle.close()
            except Exception:
                pass
        raise

    # Close the final chunk
    if not dry_run and chunk_handle:
        close_current_chunk()

    num_chunks = len(chunk_summary) if not dry_run else 0
    elapsed    = time.time() - t0

    logger.info(
        f"[{source_name}] {'DRY RUN ' if dry_run else ''}DONE — "
        f"{total_docs:,} docs | "
        f"{total_bytes / GB:.3f} GB uncompressed | "
        f"{num_chunks} chunks | "
        f"{elapsed:.0f}s"
    )

    if not dry_run:
        # Write .done marker — its presence is the resume signal
        with open(done_marker(source_name), "w") as f:
            f.write(
                f"docs={total_docs}\n"
                f"bytes={total_bytes}\n"
                f"chunks={num_chunks}\n"
            )
        logger.info(f"[{source_name}] .done written")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Code corpus chunk builder — Phase 1")
    p.add_argument(
        "--source", type=str, default=None,
        help=f"Process a single source only. Valid: {list(SOURCES.keys())}"
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (one per source). Default: 1 (sequential). "
             "Max useful: number of sources."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Iterate all data and report stats but write nothing to disk."
    )
    return p.parse_args()


def _worker(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    source_name, cfg, dry_run = args_tuple
    setup_logging()   # each subprocess needs its own logging setup
    try:
        process_source(source_name, cfg, dry_run=dry_run)
        return source_name, "done"
    except Exception as e:
        return source_name, f"error: {e}"


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 65)
    logger.info("CODE CHUNK BUILDER — Phase 1")
    logger.info(f"  Output     : {CHUNKS_DIR}")
    logger.info(f"  Chunk size : ~{CHUNK_TARGET_BYTES / MB:.0f} MB uncompressed")
    logger.info(f"  Min chars  : {MIN_CONTENT_CHARS}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info("=" * 65)
    logger.info("Sources:")
    for name, cfg in SOURCES.items():
        fstr = ", ".join(f"{c} {op} {v}" for c, (op, v) in cfg["filters"].items()) or "none"
        logger.info(f"  {name:<20s}  filter: {fstr:<30s}  path: {cfg['path']}")
    logger.info("=" * 65)

    # Select sources to run
    if args.source:
        if args.source not in SOURCES:
            logger.error(f"Unknown source '{args.source}'. Valid: {list(SOURCES.keys())}")
            sys.exit(1)
        sources_to_run = {args.source: SOURCES[args.source]}
    else:
        sources_to_run = SOURCES

    t_start = time.time()

    if args.workers == 1 or len(sources_to_run) == 1:
        # Sequential — simpler, easier to read logs on gcsfuse
        for name, cfg in sources_to_run.items():
            try:
                process_source(name, cfg, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"[{name}] FAILED — continuing to next source. Error: {e}")
    else:
        # Parallel across sources
        num_workers = min(args.workers, len(sources_to_run))
        logger.info(f"Launching {num_workers} parallel workers ...")

        tasks = [(name, cfg, args.dry_run) for name, cfg in sources_to_run.items()]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                name, status = future.result()
                if status == "done":
                    logger.info(f"[{name}] ✓ Worker finished")
                else:
                    logger.error(f"[{name}] ✗ FAILED — {status}")

    logger.info(f"\nPHASE 1 COMPLETE — total wall time: {time.time() - t_start:.0f}s")
    logger.info(f"Chunks written to: {CHUNKS_DIR}")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# Run commands
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Dry run — verify paths and data access, no output written
#    python build_code_chunks.py --dry-run
#
# 2. Single source test before full run (start with a small one)
#    python build_code_chunks.py --source rust_ir --dry-run
#    python build_code_chunks.py --source rust_ir
#
# 3. Full sequential run (recommended on gcsfuse — avoids concurrent write pressure)
#    tmux new -s code_chunks
#    systemd-run --uid=$(id -u) --gid=$(id -g) --scope \
#      -p MemoryMax=60G -p OOMScoreAdjust=-900 \
#      python build_code_chunks.py 2>&1 | tee -a build_chunks_stdout.log
#
# 4. Parallel run (faster, but gcsfuse handles concurrent writes less gracefully)
#    python build_code_chunks.py --workers 4
#
# 5. Check which sources are done
#    for src in cpp python rust_stackedu shell_stackedu rust_starcoder shell_starcoder rust_ir; do
#        f=/home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/chunks/$src/.done
#        [ -f "$f" ] && echo "✓ $src  $(cat $f | tr '\n' '  ')" || echo "✗ $src"
#    done
#
# 6. Inspect a .done file
#    cat /home/sushmetha/raw_storage_mount/Tokenizer_Data/Curated_code_corpus/chunks/cpp/.done
