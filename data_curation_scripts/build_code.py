#!/usr/bin/env python3
"""
build_code_shard.py — Extract code data from Stack Edu + BIGCODE into a
                       single gzip-compressed shard for tokenizer training.

Sources:
  Stack Edu (parquet, text column, int_score >= 3):
    Python, C++ (Cpp), JavaScript, Rust, Shell

  BIGCODE (Arrow IPC, content column, quality-filtered):
    Fortran, HTML

Output:
  shard_code.txt.gz — one document per line, ready for BPE training.

Target allocations (2 GB total):
  Python       500 MB
  C++          400 MB
  Fortran      300 MB
  Rust         250 MB
  JavaScript   250 MB
  HTML         150 MB
  Shell        150 MB

Usage:
  python build_code_shard.py                    # full run
  python build_code_shard.py --dry-run          # stats only, no output
  python build_code_shard.py --seed 42          # reproducible sampling
"""

import argparse
import gzip
import logging
import os
import random
import sys
import time
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GB = 1024 ** 3
MB = 1024 ** 2

OUTPUT_DIR = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount"
    "/Tokenizer_Data/Curated_Corpus_180GB/shards"
)
OUTPUT_FILENAME = "shard_code.txt.gz"

STACK_EDU_BASE = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount/code/stack_edu"
)
BIGCODE_BASE = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/BIGCODE"
)

DEFAULT_SEED = 42
MIN_CONTENT_CHARS = 100     # skip trivially short files

# ── Per-language config ───────────────────────────────────────────────────────
#
# source:         "stack_edu" or "bigcode"
# dir_name:       subdirectory name under the respective base path
# text_col:       column containing the code
# target_bytes:   how many bytes to sample
# filters:        dict of column_name → (operator, value) for quality gates
#
# Stack Edu filter:  int_score >= 3  (same as original build.py)
# BIGCODE filters:   alphanum_fraction >= 0.25  (skip binary/autogen)
#                    avg_line_length <= 200      (skip minified)
#                    max_stars_count >= 5        (prefer popular repos)

LANGUAGES = {
    "python": {
        "source": "stack_edu",
        "dir_name": "Python",
        "text_col": "text",
        "target_bytes": int(500 * MB),
        "filters": {"int_score": (">=", 3)},
    },
    "cpp": {
        "source": "stack_edu",
        "dir_name": "Cpp",
        "text_col": "text",
        "target_bytes": int(400 * MB),
        "filters": {"int_score": (">=", 3)},
    },
    "fortran": {
        "source": "bigcode",
        "dir_name": "fortran",
        "text_col": "content",
        "target_bytes": int(300 * MB),
        "filters": {
            "alphanum_fraction": (">=", 0.25),
            "avg_line_length":   ("<=", 200),
            "max_stars_count":   (">=", 5),
        },
    },
    "rust": {
        "source": "stack_edu",
        "dir_name": "Rust",
        "text_col": "text",
        "target_bytes": int(250 * MB),
        "filters": {"int_score": (">=", 3)},
    },
    "javascript": {
        "source": "stack_edu",
        "dir_name": "JavaScript",
        "text_col": "text",
        "target_bytes": int(250 * MB),
        "filters": {"int_score": (">=", 3)},
    },
    "html": {
        "source": "bigcode",
        "dir_name": "html",
        "text_col": "content",
        "target_bytes": int(150 * MB),
        "filters": {
            "alphanum_fraction": (">=", 0.25),
            "avg_line_length":   ("<=", 200),
            "max_stars_count":   (">=", 5),
        },
    },
    "shell": {
        "source": "stack_edu",
        "dir_name": "Shell",
        "text_col": "text",
        "target_bytes": int(150 * MB),
        "filters": {"int_score": (">=", 3)},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FILE READERS
# ══════════════════════════════════════════════════════════════════════════════

def read_parquet_dir(dir_path: Path):
    """Yield pyarrow Tables from all .parquet files in a directory."""
    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {dir_path}")
    log.info("    Found %d parquet files", len(files))
    for f in files:
        yield pq.read_table(f)


def read_arrow_dir(dir_path: Path):
    """Yield pyarrow Tables from all .arrow files (Arrow IPC stream format)."""
    files = sorted(dir_path.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No .arrow files in {dir_path}")
    log.info("    Found %d arrow files", len(files))
    for f in files:
        with open(f, "rb") as fh:
            reader = ipc.open_stream(fh)
            yield reader.read_all()


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def apply_filters(table, filters: dict):
    """
    Apply column-based filters to a pyarrow Table.

    filters: {column_name: (operator, value)}
      operator is one of: ">=", "<=", ">", "<", "=="

    Returns a filtered pyarrow Table.
    """
    import pyarrow.compute as pc

    mask = None
    for col_name, (op, val) in filters.items():
        if col_name not in table.column_names:
            log.warning("    Filter column '%s' not found — skipping filter", col_name)
            continue

        col = table[col_name]

        if op == ">=":
            col_mask = pc.greater_equal(col, val)
        elif op == "<=":
            col_mask = pc.less_equal(col, val)
        elif op == ">":
            col_mask = pc.greater(col, val)
        elif op == "<":
            col_mask = pc.less(col, val)
        elif op == "==":
            col_mask = pc.equal(col, val)
        else:
            raise ValueError(f"Unknown operator: {op}")

        # Handle nulls: treat null as False (don't keep)
        col_mask = pc.if_else(pc.is_null(col_mask), False, col_mask)

        if mask is None:
            mask = col_mask
        else:
            mask = pc.and_(mask, col_mask)

    if mask is not None:
        return table.filter(mask)
    return table


# ══════════════════════════════════════════════════════════════════════════════
# PER-LANGUAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_language(lang_name: str, config: dict, seed: int) -> list[str]:
    """
    Extract and sample documents for one language.

    Returns list of document strings, shuffled, totaling approximately
    target_bytes of UTF-8 content.
    """
    source = config["source"]
    text_col = config["text_col"]
    target_bytes = config["target_bytes"]
    filters = config["filters"]

    if source == "stack_edu":
        dir_path = STACK_EDU_BASE / config["dir_name"]
        reader = read_parquet_dir(dir_path)
    else:
        dir_path = BIGCODE_BASE / config["dir_name"]
        reader = read_arrow_dir(dir_path)

    log.info("  [%s] Reading from %s (%s)", lang_name, dir_path, source)
    log.info("  [%s] Target: %.0f MB", lang_name, target_bytes / MB)
    log.info("  [%s] Filters: %s", lang_name, filters)

    all_docs = []
    total_read = 0
    total_passed = 0
    total_bytes_available = 0

    for table in reader:
        rows_before = len(table)
        total_read += rows_before

        # Apply quality filters
        table = apply_filters(table, filters)
        total_passed += len(table)

        # Extract text column
        if text_col not in table.column_names:
            log.error("  [%s] Column '%s' not found! Available: %s",
                      lang_name, text_col, table.column_names)
            continue

        col = table[text_col]
        for i in range(len(col)):
            val = col[i].as_py()
            if val is None:
                continue
            text = str(val).strip()
            if len(text) < MIN_CONTENT_CHARS:
                continue
            byte_len = len(text.encode("utf-8"))
            total_bytes_available += byte_len
            all_docs.append((text, byte_len))

        # Early exit if we already have way more than needed (3x buffer)
        if total_bytes_available > target_bytes * 3:
            log.info("  [%s] Collected %.0f MB (3x target) — stopping read early",
                     lang_name, total_bytes_available / MB)
            break

    log.info("  [%s] Read %d rows, %d passed filters, %d docs (≥%d chars)",
             lang_name, total_read, total_passed, len(all_docs), MIN_CONTENT_CHARS)
    log.info("  [%s] Available: %.0f MB", lang_name, total_bytes_available / MB)

    if total_bytes_available == 0:
        log.error("  [%s] NO DATA after filtering!", lang_name)
        return []

    # ── Sampling ──────────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_docs)

    sampled = []
    sampled_bytes = 0

    for text, byte_len in all_docs:
        if sampled_bytes >= target_bytes:
            break
        sampled.append(text)
        sampled_bytes += byte_len

    log.info("  [%s] Sampled %d docs = %.0f MB (target: %.0f MB)",
             lang_name, len(sampled), sampled_bytes / MB, target_bytes / MB)

    if sampled_bytes < target_bytes * 0.5:
        log.warning(
            "  [%s] WARNING: Only got %.0f MB, less than half of target %.0f MB. "
            "Consider relaxing filters.",
            lang_name, sampled_bytes / MB, target_bytes / MB
        )

    return sampled


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build code shard for tokenizer training."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name", type=str, default=OUTPUT_FILENAME,
        help=f"Output filename (default: {OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read and filter data, report stats, but don't write output.",
    )
    parser.add_argument(
        "--lang", type=str, default=None,
        help=f"Process single language only. Valid: {list(LANGUAGES.keys())}",
    )
    args = parser.parse_args()

    output_path = args.output_dir / args.output_name

    log.info("=" * 70)
    log.info("CODE SHARD BUILDER")
    log.info("=" * 70)
    log.info("  Output     : %s", output_path if not args.dry_run else "(dry run)")
    log.info("  Seed       : %d", args.seed)
    log.info("  Languages  : %d", len(LANGUAGES))

    total_target = sum(cfg["target_bytes"] for cfg in LANGUAGES.values())
    log.info("  Total target: %.0f MB (%.2f GB)", total_target / MB, total_target / GB)
    log.info("")

    for lang, cfg in LANGUAGES.items():
        log.info("  %-12s  %-10s  %4.0f MB  from %s/%s",
                 lang, cfg["source"],
                 cfg["target_bytes"] / MB,
                 cfg["source"], cfg["dir_name"])
    log.info("=" * 70)

    # ── Select languages to process ───────────────────────────────────────
    if args.lang:
        if args.lang not in LANGUAGES:
            log.error("Unknown language: %s. Valid: %s",
                      args.lang, list(LANGUAGES.keys()))
            sys.exit(1)
        langs_to_run = {args.lang: LANGUAGES[args.lang]}
    else:
        langs_to_run = LANGUAGES

    # ── Extract all languages ─────────────────────────────────────────────
    t_start = time.perf_counter()
    all_documents = []
    per_lang_stats = {}

    for lang_name, config in langs_to_run.items():
        log.info("")
        log.info("─" * 50)
        t_lang = time.perf_counter()

        try:
            docs = extract_language(lang_name, config, args.seed)
        except FileNotFoundError as e:
            log.error("  [%s] SKIPPED — %s", lang_name, e)
            docs = []
        except Exception as e:
            log.error("  [%s] FAILED — %s", lang_name, e)
            docs = []

        lang_bytes = sum(len(d.encode("utf-8")) for d in docs)
        per_lang_stats[lang_name] = {
            "docs": len(docs),
            "bytes": lang_bytes,
            "time": time.perf_counter() - t_lang,
        }

        all_documents.extend(docs)
        log.info("  [%s] Done in %.1f s", lang_name, per_lang_stats[lang_name]["time"])

    # ── Shuffle all documents together ────────────────────────────────────
    # Interleaving languages prevents BPE from seeing long runs of one
    # language, which could bias early merges.
    log.info("")
    log.info("Shuffling %d documents...", len(all_documents))
    rng = random.Random(args.seed + 1)  # different seed from per-lang sampling
    rng.shuffle(all_documents)

    # ── Write output ──────────────────────────────────────────────────────
    if not args.dry_run and all_documents:
        log.info("")
        log.info("Writing output...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        t_write = time.perf_counter()

        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for i, doc in enumerate(all_documents):
                # Replace newlines within a document with spaces to keep
                # one-doc-per-line format consistent with other shards
                clean = doc.replace("\n", " ").replace("\r", "")
                fh.write(clean + "\n")

                if (i + 1) % 100_000 == 0:
                    log.info("  ... wrote %d / %d docs", i + 1, len(all_documents))

        write_time = time.perf_counter() - t_write
        output_size = output_path.stat().st_size
        log.info("Write complete: %.2f GB compressed (%.1f s)",
                 output_size / GB, write_time)
    elif args.dry_run:
        log.info("")
        log.info("DRY RUN — no output written.")

    # ── Final report ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    total_bytes = sum(s["bytes"] for s in per_lang_stats.values())
    total_docs = sum(s["docs"] for s in per_lang_stats.values())

    log.info("")
    log.info("=" * 70)
    log.info("CODE SHARD REPORT")
    log.info("=" * 70)
    log.info("")
    log.info("  %-12s  %8s  %10s  %10s  %6s",
             "Language", "Docs", "Bytes", "Target", "Hit%")
    log.info("  " + "-" * 56)

    for lang_name in LANGUAGES:
        if lang_name not in per_lang_stats:
            continue
        s = per_lang_stats[lang_name]
        target = LANGUAGES[lang_name]["target_bytes"]
        hit_pct = 100 * s["bytes"] / target if target > 0 else 0
        log.info("  %-12s  %8d  %8.0f MB  %8.0f MB  %5.1f%%",
                 lang_name, s["docs"], s["bytes"] / MB, target / MB, hit_pct)

    log.info("  " + "-" * 56)
    log.info("  %-12s  %8d  %8.0f MB  %8.0f MB",
             "TOTAL", total_docs, total_bytes / MB, total_target / MB)
    log.info("")
    log.info("  Total time: %.1f s", total_time)

    if not args.dry_run and all_documents:
        log.info("  Output: %s", output_path)

    log.info("=" * 70)


if __name__ == "__main__":
    main()


"""
# ── Run commands ─────────────────────────────────────────────────────────

# 1. Test single language first (fast)
python build_code_shard.py --lang fortran --dry-run

# 2. Test all languages (dry run)
python build_code_shard.py --dry-run

# 3. Full run
tmux new -s code_shard
python build_code_shard.py --seed 42

# 4. After output, filter it for unwanted Unicode (same as shard 20):
python filter_shard.py \
    --input /home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/Curated_Corpus_180GB/shards/shard_code.txt.gz \
    --output-dir /home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/Curated_Corpus_180GB/filtered_shards \
    --output-name shard_code_filtered.txt.gz \
    --workers 32

# 5. Train tokenizer on both shards:
#    Point SHARD_DIR to filtered_shards/ which now contains:
#      shard_20_filtered.txt.gz   (10 GB — math, English, Indic, scientific)
#      shard_code_filtered.txt.gz (~2 GB — code)
"""
