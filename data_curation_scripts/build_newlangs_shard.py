#!/usr/bin/env python3
"""
build_newlangs_shard.py — Build multilingual shards from FineWeb2 + Sangraha.

14 languages, streaming parquet reads, progress bars, parallel shard assembly.

Usage:
  python build_newlangs_shard.py --phase 1 --lang sinhala --dry-run
  python build_newlangs_shard.py --phase 1
  python build_newlangs_shard.py --phase 2 --num-shards 2 --workers 16
  python build_newlangs_shard.py --num-shards 2
"""

import argparse
import gzip
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import pyarrow.parquet as pq

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GB = 1024 ** 3
MB = 1024 ** 2

OUTPUT_BASE = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount"
    "/Tokenizer_Data/Curated_Corpus_all_langs"
)
CHUNKS_DIR = OUTPUT_BASE / "chunks"
SHARDS_DIR = OUTPUT_BASE / "shards"

FINEWEB2_BASE = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount/Multilingual_data/fineweb2"
)
SANGRAHA_VERIFIED = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount"
    "/Multilingual_data/sangraha/verified/verified"
)
SANGRAHA_SYNTHETIC = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount"
    "/Multilingual_data/sangraha/synthetic/synthetic"
)

DEFAULT_SEED    = 42
DEFAULT_WORKERS = 16
MIN_CHARS       = 100
BATCH_SIZE      = 2000    # rows per parquet batch (matches build.py)
LANG_SCORE_THRESHOLD = 0.65
ABUNDANCE_FACTOR     = 3.0

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGES = {
    "punjabi": {
        "source": "sangraha",
        "verified_path": SANGRAHA_VERIFIED / "pan",
        "synthetic_path": SANGRAHA_SYNTHETIC / "pan_Guru",
        "text_col": "text",
        "target_bytes": int(372 * MB),
        "has_lang_score": False,
    },
    "odia": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "ory" / "train",
        "text_col": "text",
        "target_bytes": int(372 * MB),
        "has_lang_score": True,
    },
    "russian": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "rus" / "train",
        "text_col": "text",
        "target_bytes": int(298 * MB),
        "has_lang_score": True,
    },
    "french": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "fra" / "train",
        "text_col": "text",
        "target_bytes": int(298 * MB),
        "has_lang_score": True,
    },
    "german": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "deu" / "train",
        "text_col": "text",
        "target_bytes": int(298 * MB),
        "has_lang_score": True,
    },
    "mandarin": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "cmn" / "train",
        "text_col": "text",
        "target_bytes": int(150 * MB),
        "has_lang_score": True,
    },
    "japanese": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "jpn" / "train",
        "text_col": "text",
        "target_bytes": int(150 * MB),
        "has_lang_score": True,
    },
    "korean": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "kor" / "train",
        "text_col": "text",
        "target_bytes": int(150 * MB),
        "has_lang_score": True,
    },
    "nepali": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "npi" / "train",
        "text_col": "text",
        "target_bytes": int(99 * MB),
        "has_lang_score": True,
    },
    "urdu": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "urd" / "train",
        "text_col": "text",
        "target_bytes": int(99 * MB),
        "has_lang_score": True,
    },
    "pashto": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "pbt" / "train",
        "text_col": "text",
        "target_bytes": int(99 * MB),
        "has_lang_score": True,
    },
    "dari": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "fas" / "train",
        "text_col": "text",
        "target_bytes": int(99 * MB),
        "has_lang_score": True,
    },
    "burmese": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "mya" / "train",
        "text_col": "text",
        "target_bytes": int(71 * MB),
        "has_lang_score": True,
    },
    "sinhala": {
        "source": "fineweb2",
        "path": FINEWEB2_BASE / "sin" / "train",
        "text_col": "text",
        "target_bytes": int(71 * MB),
        "has_lang_score": True,
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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_parquet_files(dir_path: Path) -> list[Path]:
    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {dir_path}")
    return files


def estimate_available_bytes(dir_path: Path) -> int:
    return sum(f.stat().st_size for f in dir_path.glob("*.parquet"))


def chunk_path(lang_name: str) -> Path:
    return CHUNKS_DIR / lang_name / "chunk.txt.gz"


def chunk_done_marker(lang_name: str) -> Path:
    return CHUNKS_DIR / lang_name / ".done"


def lang_is_complete(lang_name: str) -> bool:
    return chunk_done_marker(lang_name).exists() and chunk_path(lang_name).exists()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING PARQUET ITERATOR (matches build.py pattern)
# ══════════════════════════════════════════════════════════════════════════════

def iter_parquet_file(
    filepath: Path,
    text_col: str,
    quality_filter: Optional[Tuple[str, float]] = None,
) -> Iterator[str]:
    """Stream documents from a single parquet file in small batches."""
    cols = [text_col]
    if quality_filter:
        score_col, _ = quality_filter
        schema = pq.read_schema(str(filepath))
        if score_col in schema.names:
            cols.append(score_col)
        else:
            quality_filter = None

    pf = pq.ParquetFile(str(filepath))
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        d = batch.to_pydict()
        texts = d[text_col]
        scores = d.get(quality_filter[0]) if quality_filter else None

        for i, text in enumerate(texts):
            if not text:
                continue
            if quality_filter and scores:
                s = scores[i]
                if s is None or s < quality_filter[1]:
                    continue
            yield str(text)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: EXTRACT + FILTER + SAMPLE → CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def extract_fineweb2(lang_name: str, config: dict, seed: int, dry_run: bool) -> dict:
    """Extract from FineWeb2 with streaming reads and progress bar."""
    dir_path = config["path"]
    text_col = config["text_col"]
    target_bytes = config["target_bytes"]
    has_lang_score = config["has_lang_score"]

    files = get_parquet_files(dir_path)
    available_bytes = estimate_available_bytes(dir_path)
    is_abundant = available_bytes > (target_bytes * ABUNDANCE_FACTOR)
    use_quality_filter = has_lang_score and is_abundant

    quality_filter = ("language_score", LANG_SCORE_THRESHOLD) if use_quality_filter else None

    log.info("  [%s] Source: FineWeb2 — %s", lang_name, dir_path)
    log.info("  [%s] Files: %d, Available: ~%.1f GB", lang_name, len(files), available_bytes / GB)
    log.info("  [%s] Target: %.0f MB", lang_name, target_bytes / MB)
    log.info("  [%s] Quality filter: %s",
             lang_name,
             f"language_score >= {LANG_SCORE_THRESHOLD}" if use_quality_filter else "NONE (data <= 3x target)")

    all_docs = []
    total_bytes_collected = 0
    total_read = 0
    collect_limit = target_bytes * 3

    bar = tqdm(
        desc=f"  {lang_name:14s}",
        unit=" docs",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    for file_idx, fpath in enumerate(files):
        bar.set_postfix(
            file=f"{file_idx+1}/{len(files)}",
            MB=f"{total_bytes_collected/MB:.0f}/{target_bytes/MB:.0f}",
        )

        for text in iter_parquet_file(fpath, text_col, quality_filter):
            total_read += 1

            if len(text) < MIN_CHARS:
                continue

            byte_len = len(text.encode("utf-8"))
            total_bytes_collected += byte_len
            all_docs.append((text, byte_len))
            bar.update(1)

        if total_bytes_collected >= collect_limit:
            log.info("  [%s] Collected %.0f MB (3x target) at file %d/%d — stopping",
                     lang_name, total_bytes_collected / MB, file_idx + 1, len(files))
            break

    bar.close()

    log.info("  [%s] Scanned %d rows -> %d docs (>=%d chars) = %.0f MB",
             lang_name, total_read, len(all_docs), MIN_CHARS, total_bytes_collected / MB)

    # Sample to target
    rng = random.Random(seed)
    rng.shuffle(all_docs)

    sampled = []
    sampled_bytes = 0
    for text, byte_len in all_docs:
        if sampled_bytes >= target_bytes:
            break
        sampled.append(text)
        sampled_bytes += byte_len

    log.info("  [%s] Sampled %d docs = %.0f MB (target: %.0f MB, hit: %.0f%%)",
             lang_name, len(sampled), sampled_bytes / MB, target_bytes / MB,
             100 * sampled_bytes / target_bytes if target_bytes else 0)

    if sampled_bytes < target_bytes * 0.5:
        log.warning("  [%s] WARNING: Only %.0f MB — less than half of %.0f MB target!",
                     lang_name, sampled_bytes / MB, target_bytes / MB)

    # Write chunk
    if not dry_run and sampled:
        out_path = chunk_path(lang_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("  [%s] Writing chunk...", lang_name)
        with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for doc in tqdm(sampled, desc=f"  {lang_name:14s} write", unit=" docs"):
                clean = doc.replace("\n", " ").replace("\r", "")
                fh.write(clean + "\n")

        with open(chunk_done_marker(lang_name), "w") as f:
            f.write(f"docs={len(sampled)}\nbytes={sampled_bytes}\n")

        compressed = out_path.stat().st_size
        log.info("  [%s] Chunk: %.0f MB uncompressed, %.0f MB compressed -> %s",
                 lang_name, sampled_bytes / MB, compressed / MB, out_path)

    return {"docs": len(sampled), "bytes": sampled_bytes}


def extract_sangraha(lang_name: str, config: dict, seed: int, dry_run: bool) -> dict:
    """Extract from Sangraha (verified + synthetic, 50-50 split)."""
    verified_path = config["verified_path"]
    synthetic_path = config["synthetic_path"]
    text_col = config["text_col"]
    target_bytes = config["target_bytes"]
    half_target = target_bytes // 2

    log.info("  [%s] Source: Sangraha (50/50 verified + synthetic)", lang_name)
    log.info("  [%s] Target: %.0f MB (%.0f MB each half)", lang_name, target_bytes / MB, half_target / MB)

    rng = random.Random(seed)

    def collect_from_dir(dir_path: Path, label: str, limit: int) -> list:
        docs = []
        bytes_collected = 0

        try:
            files = get_parquet_files(dir_path)
        except FileNotFoundError:
            log.warning("  [%s] No parquet files in %s — skipping %s", lang_name, dir_path, label)
            return docs

        bar = tqdm(desc=f"  {lang_name:14s} {label}", unit=" docs", dynamic_ncols=True)

        for fpath in files:
            for text in iter_parquet_file(fpath, text_col):
                if len(text) < MIN_CHARS:
                    continue
                byte_len = len(text.encode("utf-8"))
                bytes_collected += byte_len
                docs.append((text, byte_len))
                bar.update(1)
                bar.set_postfix(MB=f"{bytes_collected/MB:.0f}/{limit/MB:.0f}")

            if bytes_collected >= limit * 3:
                break

        bar.close()
        log.info("  [%s] %s: %d docs, %.0f MB", lang_name, label, len(docs), bytes_collected / MB)
        return docs

    verified_docs = collect_from_dir(verified_path, "verified", half_target)
    synthetic_docs = collect_from_dir(synthetic_path, "synthetic", half_target)

    rng.shuffle(verified_docs)
    rng.shuffle(synthetic_docs)

    sampled = []
    sampled_bytes = 0

    for source_docs, source_target in [(verified_docs, half_target), (synthetic_docs, half_target)]:
        source_bytes = 0
        for text, byte_len in source_docs:
            if source_bytes >= source_target:
                break
            sampled.append(text)
            sampled_bytes += byte_len
            source_bytes += byte_len

    rng.shuffle(sampled)

    log.info("  [%s] Sampled %d docs = %.0f MB (target: %.0f MB, hit: %.0f%%)",
             lang_name, len(sampled), sampled_bytes / MB, target_bytes / MB,
             100 * sampled_bytes / target_bytes if target_bytes else 0)

    if not dry_run and sampled:
        out_path = chunk_path(lang_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("  [%s] Writing chunk...", lang_name)
        with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for doc in tqdm(sampled, desc=f"  {lang_name:14s} write", unit=" docs"):
                clean = doc.replace("\n", " ").replace("\r", "")
                fh.write(clean + "\n")

        with open(chunk_done_marker(lang_name), "w") as f:
            f.write(f"docs={len(sampled)}\nbytes={sampled_bytes}\n")

        compressed = out_path.stat().st_size
        log.info("  [%s] Chunk: %.0f MB uncompressed, %.0f MB compressed -> %s",
                 lang_name, sampled_bytes / MB, compressed / MB, out_path)

    return {"docs": len(sampled), "bytes": sampled_bytes}


def _process_one_language(args):
    """Worker function for parallel Phase 1. Must be top-level for pickling."""
    lang_name, config, seed, dry_run, parallel = args
    t0 = time.perf_counter()

    # In parallel mode, disable tqdm to avoid garbled output
    if parallel:
        import builtins
        # Monkey-patch tqdm to be silent in worker processes
        import tqdm as tqdm_module
        original_tqdm = tqdm_module.tqdm
        tqdm_module.tqdm = lambda *a, **kw: (kw.pop('disable', None), original_tqdm(*a, disable=True, **kw))[1]

    try:
        if config["source"] == "sangraha":
            result = extract_sangraha(lang_name, config, seed, dry_run)
        else:
            result = extract_fineweb2(lang_name, config, seed, dry_run)
    except Exception as e:
        log.error("  [%s] FAILED — %s", lang_name, e)
        result = {"docs": 0, "bytes": 0}

    elapsed = time.perf_counter() - t0
    return lang_name, result["docs"], result["bytes"], elapsed


def run_phase1(langs_to_run: dict, seed: int, dry_run: bool, workers: int):
    log.info("=" * 70)
    log.info("PHASE 1 — EXTRACT + FILTER + SAMPLE -> CHUNKS (workers=%d)", workers)
    log.info("=" * 70)

    stats = {}

    # Handle already-complete languages first
    to_process = {}
    for lang_name, config in langs_to_run.items():
        if not dry_run and lang_is_complete(lang_name):
            log.info("  [%s] SKIP — chunk already exists", lang_name)
            with open(chunk_done_marker(lang_name)) as f:
                s = dict(line.strip().split("=") for line in f if "=" in line)
            stats[lang_name] = {"docs": int(s.get("docs", 0)), "bytes": int(s.get("bytes", 0)), "time": 0}
        else:
            to_process[lang_name] = config

    if not to_process:
        log.info("  All languages already complete!")
    elif workers <= 1 or len(to_process) == 1:
        # Sequential mode — tqdm progress bars work normally
        for lang_name, config in to_process.items():
            log.info("")
            log.info("-" * 50)
            t0 = time.perf_counter()

            try:
                if config["source"] == "sangraha":
                    result = extract_sangraha(lang_name, config, seed, dry_run)
                else:
                    result = extract_fineweb2(lang_name, config, seed, dry_run)
            except Exception as e:
                log.error("  [%s] FAILED — %s", lang_name, e, exc_info=True)
                result = {"docs": 0, "bytes": 0}

            elapsed = time.perf_counter() - t0
            stats[lang_name] = {"docs": result["docs"], "bytes": result["bytes"], "time": elapsed}
            log.info("  [%s] Done in %.1f s", lang_name, elapsed)
    else:
        # Parallel mode — disable per-language tqdm, use overall progress bar
        log.info("")
        log.info("  Processing %d languages with %d parallel workers...",
                 len(to_process), min(workers, len(to_process)))
        log.info("  (per-language progress bars disabled in parallel mode)")
        log.info("")

        tasks = [
            (lang_name, config, seed, dry_run, True)
            for lang_name, config in to_process.items()
        ]

        actual_workers = min(workers, len(tasks))

        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            for lang_name, docs, bytes_out, elapsed in pool.map(_process_one_language, tasks):
                stats[lang_name] = {"docs": docs, "bytes": bytes_out, "time": elapsed}
                target = LANGUAGES[lang_name]["target_bytes"]
                hit = 100 * bytes_out / target if target else 0
                log.info("  ✓ %-14s  %6d docs  %6.0f MB / %3.0f MB  (%5.1f%%)  %.0fs",
                         lang_name, docs, bytes_out / MB, target / MB, hit, elapsed)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 1 SUMMARY")
    log.info("=" * 70)
    log.info("  %-14s  %8s  %10s  %10s  %6s",
             "Language", "Docs", "Sampled", "Target", "Hit%")
    log.info("  " + "-" * 58)

    total_docs = 0
    total_bytes = 0
    for lang_name in LANGUAGES:
        if lang_name not in stats:
            continue
        s = stats[lang_name]
        target = LANGUAGES[lang_name]["target_bytes"]
        hit = 100 * s["bytes"] / target if target else 0
        log.info("  %-14s  %8d  %8.0f MB  %8.0f MB  %5.1f%%",
                 lang_name, s["docs"], s["bytes"] / MB, target / MB, hit)
        total_docs += s["docs"]
        total_bytes += s["bytes"]

    total_target = sum(LANGUAGES[l]["target_bytes"] for l in stats)
    log.info("  " + "-" * 58)
    log.info("  %-14s  %8d  %8.0f MB  %8.0f MB",
             "TOTAL", total_docs, total_bytes / MB, total_target / MB)
    log.info("=" * 70)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: ASSEMBLE CHUNKS -> SHARDS
# ══════════════════════════════════════════════════════════════════════════════

def _write_shard(args):
    """Worker: write one shard .txt.gz from a list of doc strings."""
    shard_idx, shard_path_str, docs = args
    shard_path = Path(shard_path_str)
    shard_bytes = 0

    with gzip.open(shard_path, "wt", encoding="utf-8", compresslevel=6) as fh:
        for doc in docs:
            fh.write(doc + "\n")
            shard_bytes += len(doc.encode("utf-8"))

    compressed = shard_path.stat().st_size
    return shard_idx, len(docs), shard_bytes, compressed


def run_phase2(num_shards: int, seed: int, workers: int):
    log.info("=" * 70)
    log.info("PHASE 2 — ASSEMBLE CHUNKS -> %d SHARDS (workers=%d)", num_shards, workers)
    log.info("=" * 70)

    missing = [l for l in LANGUAGES if not lang_is_complete(l)]
    if missing:
        log.error("Missing chunks: %s", missing)
        log.error("Run phase 1 first.")
        sys.exit(1)

    # Read all chunks
    log.info("Reading chunks...")
    all_docs = []

    for lang_name in LANGUAGES:
        cpath = chunk_path(lang_name)
        count = 0
        with gzip.open(cpath, "rt", encoding="utf-8") as fh:
            for line in tqdm(fh, desc=f"  {lang_name:14s}", unit=" docs", dynamic_ncols=True):
                line = line.rstrip("\n")
                if line:
                    all_docs.append(line)
                    count += 1
        log.info("  %-14s  %8d docs", lang_name, count)

    log.info("Total: %d docs", len(all_docs))

    # Shuffle
    log.info("Shuffling %d docs...", len(all_docs))
    rng = random.Random(seed + 100)
    rng.shuffle(all_docs)

    # Split into per-shard doc lists
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    docs_per_shard = len(all_docs) // num_shards
    remainder = len(all_docs) % num_shards

    shard_tasks = []
    offset = 0
    for i in range(num_shards):
        count = docs_per_shard + (1 if i < remainder else 0)
        shard_file = SHARDS_DIR / f"shard_newlang_{i:02d}.txt.gz"
        shard_tasks.append((i, str(shard_file), all_docs[offset:offset + count]))
        offset += count

    del all_docs

    # Write shards in parallel
    actual_workers = min(workers, num_shards)
    log.info("Writing %d shards with %d workers...", num_shards, actual_workers)
    t_write = time.perf_counter()

    if actual_workers <= 1:
        results = [_write_shard(t) for t in tqdm(shard_tasks, desc="  shards", unit=" shard")]
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            results = list(tqdm(
                pool.map(_write_shard, shard_tasks),
                total=num_shards,
                desc="  shards",
                unit=" shard",
            ))

    for shard_idx, doc_count, shard_bytes, compressed in sorted(results):
        log.info("  shard_%02d: %d docs | %.0f MB uncompressed | %.0f MB compressed",
                 shard_idx, doc_count, shard_bytes / MB, compressed / MB)

    log.info("Write complete in %.1f s", time.perf_counter() - t_write)
    log.info("Output: %s", SHARDS_DIR)
    log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build multilingual shards for tokenizer training."
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 or 2 (default: both)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of output shards (default: 1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed (default: {DEFAULT_SEED})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers for phase 2 (default: {DEFAULT_WORKERS})")
    parser.add_argument("--lang", type=str, default=None,
                        help=f"Single language (phase 1 only). Valid: {list(LANGUAGES.keys())}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stats only, don't write chunks.")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("NEW LANGUAGES SHARD BUILDER")
    log.info("=" * 70)
    log.info("  Output   : %s", OUTPUT_BASE)
    log.info("  Shards   : %d", args.num_shards)
    log.info("  Workers  : %d", args.workers)
    log.info("  Seed     : %d", args.seed)
    log.info("  Languages: %d", len(LANGUAGES))

    total_target = sum(c["target_bytes"] for c in LANGUAGES.values())
    log.info("  Target   : %.0f MB (%.2f GB)", total_target / MB, total_target / GB)
    log.info("")

    for lang, cfg in LANGUAGES.items():
        log.info("  %-14s  %4.0f MB  %s", lang, cfg["target_bytes"] / MB, cfg["source"])
    log.info("=" * 70)

    run_p1 = args.phase in (None, 1)
    run_p2 = args.phase in (None, 2) and args.lang is None and not args.dry_run

    if run_p1:
        if args.lang:
            if args.lang not in LANGUAGES:
                log.error("Unknown: %s. Valid: %s", args.lang, list(LANGUAGES.keys()))
                sys.exit(1)
            langs_to_run = {args.lang: LANGUAGES[args.lang]}
        else:
            langs_to_run = LANGUAGES
        run_phase1(langs_to_run, args.seed, args.dry_run, args.workers)

    if run_p2:
        run_phase2(args.num_shards, args.seed, args.workers)

    log.info("\nDONE")


if __name__ == "__main__":
    main()


"""
# ── Run commands ─────────────────────────────────────────────────────────

# 1. Test single language
python build_newlangs_shard.py --phase 1 --lang sinhala --dry-run

# 2. Phase 1: all languages
tmux new -s newlangs
python build_newlangs_shard.py --phase 1

# 3. Phase 2: assemble
python build_newlangs_shard.py --phase 2 --num-shards 2 --workers 16

# 4. Or both at once
python build_newlangs_shard.py --num-shards 2 --workers 16

# 5. Copy existing shards into same directory:
#    cp .../filtered_shards/shard_20_filtered.txt.gz .../Curated_Corpus_all_langs/shards/
#    cp .../filtered_shards/shard_code_filtered.txt.gz .../Curated_Corpus_all_langs/shards/

# NOTE: Do NOT run filter_shard.py on the new languages shard!
"""
