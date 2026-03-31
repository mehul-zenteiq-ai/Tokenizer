#!/usr/bin/env python3
"""
estimate_english_tokens.py — Estimate token counts for Wikipedia, DCLM, FineWeb-Edu.

Approach per source:
  1. List all files
  2. Sample N files spread across the list
  3. For each sampled file, count words in all rows → tokens = words × 1.35
  4. Compute (tokens / file_size_bytes) ratio from samples
  5. Sum total bytes across ALL files → extrapolate total tokens

This avoids reading every file while giving a reliable estimate.

Usage:
    python estimate_english_tokens.py 2>&1 | tee english_token_estimate.log
"""

import glob
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

try:
    import zstandard as zstd
except ImportError:
    print("FATAL: pip install zstandard")
    sys.exit(1)

BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
MAX_WORKERS = 48
SAMPLES_PER_SOURCE = 10  # number of files to fully read per source
TOKEN_MULTIPLIER = 1.35


# ──────────────────────────────────────────────────────────────
# Word counting helpers
# ──────────────────────────────────────────────────────────────

def count_words_parquet(filepath: str, text_col: str) -> tuple[int, int]:
    """
    Read entire parquet file, count total words across all rows.
    Returns (total_words, file_size_bytes).
    """
    file_size = os.path.getsize(filepath)
    total_words = 0
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(batch_size=5000, columns=[text_col]):
        for val in batch.column(text_col):
            text = val.as_py()
            if text:
                total_words += len(text.split())
    return total_words, file_size


def count_words_jsonl_zst(filepath: str, text_key: str) -> tuple[int, int]:
    """
    Read entire jsonl.zst file, count total words.
    Returns (total_words, file_size_bytes).
    """
    file_size = os.path.getsize(filepath)
    total_words = 0
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            import io
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_key, "")
                    if text:
                        total_words += len(text.split())
                except json.JSONDecodeError:
                    continue
    return total_words, file_size


def _process_sample(args):
    """Worker function for parallel sampling."""
    source_name, filepath, fmt, text_col_or_key = args
    try:
        t0 = time.time()
        if fmt == "parquet":
            words, fsize = count_words_parquet(filepath, text_col_or_key)
        elif fmt == "jsonl_zst":
            words, fsize = count_words_jsonl_zst(filepath, text_col_or_key)
        else:
            return source_name, filepath, 0, 0, 0, "unknown format"
        elapsed = time.time() - t0
        tokens = int(words * TOKEN_MULTIPLIER)
        return source_name, os.path.basename(filepath), tokens, fsize, elapsed, None
    except Exception as e:
        return source_name, os.path.basename(filepath), 0, 0, 0, str(e)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  ENGLISH TOKEN ESTIMATION")
    print("=" * 80)

    # ── 1. Discover all files ──────────────────────────────
    sources = {
        "Wikipedia": {
            "glob": f"{BASE}/English/wikipedia/*.parquet",
            "fmt": "parquet",
            "col": "content",
        },
        "DCLM": {
            "glob": f"{BASE}/English/dclm_baseline_4T/global-shard_02_of_10/local-shard_*_of_10/*.jsonl.zst",
            "fmt": "jsonl_zst",
            "col": "text",
        },
        "FineWeb-Edu": {
            "glob": f"{BASE}/English/fineweb_edu/data/CC-MAIN-*/*.parquet",
            "fmt": "parquet",
            "col": "text",
        },
    }

    all_files = {}
    all_sizes = {}
    for name, cfg in sources.items():
        files = sorted(glob.glob(cfg["glob"]))
        all_files[name] = files
        total_bytes = 0
        for f in files:
            try:
                total_bytes += os.path.getsize(f)
            except OSError:
                pass
        all_sizes[name] = total_bytes
        print(f"\n  {name}:")
        print(f"    Files:      {len(files):,}")
        print(f"    Total size: {total_bytes / (1024**3):,.1f} GB")

    # ── 2. Select sample files ─────────────────────────────
    sample_tasks = []
    for name, cfg in sources.items():
        files = all_files[name]
        n = min(SAMPLES_PER_SOURCE, len(files))
        if n == 0:
            continue
        # Spread samples evenly across the file list
        if n >= len(files):
            indices = list(range(len(files)))
        else:
            step = len(files) / n
            indices = [int(i * step) for i in range(n)]
        for idx in indices:
            sample_tasks.append((name, files[idx], cfg["fmt"], cfg["col"]))

    print(f"\n  Sampling {len(sample_tasks)} files across {len(sources)} sources...")
    print(f"  Using {MAX_WORKERS} workers\n")

    # ── 3. Process samples in parallel ─────────────────────
    results_by_source = {name: [] for name in sources}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_sample, task): task for task in sample_tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            src_name, fname, tokens, fsize, elapsed, error = future.result()
            if error:
                print(f"    [{done}/{len(sample_tasks)}] ❌ {src_name}/{fname}: {error}")
            else:
                results_by_source[src_name].append((tokens, fsize, elapsed))
                tok_b = tokens / 1e9
                ratio = tokens / fsize if fsize > 0 else 0
                print(f"    [{done}/{len(sample_tasks)}] ✅ {src_name}/{fname}: "
                      f"{tok_b:.3f}B tokens, {fsize/(1024**2):.0f}MB, "
                      f"ratio={ratio:.2f} tok/byte, {elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n  Sampling completed in {total_elapsed:.0f}s")

    # ── 4. Extrapolate totals ──────────────────────────────
    print("\n" + "=" * 80)
    print("  EXTRAPOLATED TOKEN COUNTS")
    print("=" * 80)

    grand_total = 0
    source_estimates = {}

    for name in sources:
        samples = results_by_source[name]
        if not samples:
            print(f"\n  {name}: NO SAMPLES — cannot estimate")
            source_estimates[name] = 0
            continue

        total_sample_tokens = sum(s[0] for s in samples)
        total_sample_bytes = sum(s[1] for s in samples)

        if total_sample_bytes == 0:
            print(f"\n  {name}: zero sample bytes — cannot estimate")
            source_estimates[name] = 0
            continue

        # tokens-per-byte ratio from samples
        ratio = total_sample_tokens / total_sample_bytes
        total_bytes = all_sizes[name]
        estimated_tokens = int(ratio * total_bytes)
        estimated_B = estimated_tokens / 1e9

        source_estimates[name] = estimated_tokens
        grand_total += estimated_tokens

        print(f"\n  {name}:")
        print(f"    Sampled files:      {len(samples)}")
        print(f"    Sample tokens:      {total_sample_tokens:,}")
        print(f"    Sample bytes:       {total_sample_bytes:,}")
        print(f"    Tokens/byte ratio:  {ratio:.4f}")
        print(f"    Total disk bytes:   {total_bytes:,} ({total_bytes/(1024**3):.1f} GB)")
        print(f"    *** Estimated tokens: {estimated_B:.2f}B ***")

    # ── 5. FineWeb-Edu quota calculation ───────────────────
    print("\n" + "=" * 80)
    print("  ENGLISH BUDGET CALCULATION")
    print("=" * 80)

    wiki_B = source_estimates.get("Wikipedia", 0) / 1e9
    dclm_B = source_estimates.get("DCLM", 0) / 1e9
    fineweb_available_B = source_estimates.get("FineWeb-Edu", 0) / 1e9
    target_B = 300.0

    print(f"\n  Wikipedia (take all):       ~{wiki_B:.1f}B")
    print(f"  DCLM (take all):           ~{dclm_B:.1f}B")
    print(f"  FineWeb-Edu (available):   ~{fineweb_available_B:.1f}B")
    print(f"  ─────────────────────────────────")
    print(f"  Total available:           ~{wiki_B + dclm_B + fineweb_available_B:.1f}B")
    print(f"  Target:                     {target_B:.0f}B")

    gap_B = target_B - wiki_B - dclm_B
    if gap_B <= 0:
        print(f"\n  ✅ Wikipedia + DCLM alone exceed 300B! FineWeb-Edu quota = 0")
        print(f"     (surplus: {-gap_B:.1f}B)")
    elif gap_B > fineweb_available_B:
        print(f"\n  ⚠️  FineWeb-Edu needed: {gap_B:.1f}B but only {fineweb_available_B:.1f}B available")
        print(f"     English total will be: ~{wiki_B + dclm_B + fineweb_available_B:.1f}B")
        print(f"     Shortfall: ~{gap_B - fineweb_available_B:.1f}B")
    else:
        print(f"\n  ✅ FineWeb-Edu quota to fill 300B: ~{gap_B:.1f}B")
        pct = (gap_B / fineweb_available_B) * 100
        print(f"     That's {pct:.0f}% of available FineWeb-Edu data")

    # Estimated shard counts
    print(f"\n  Estimated shard counts (100M tokens each):")
    print(f"    Wikipedia:   {int(wiki_B * 10):,} shards")
    print(f"    DCLM:        {int(dclm_B * 10):,} shards")
    fineweb_quota_B = min(gap_B, fineweb_available_B) if gap_B > 0 else 0
    print(f"    FineWeb-Edu: {int(fineweb_quota_B * 10):,} shards")
    print(f"    Total:       {int((wiki_B + dclm_B + fineweb_quota_B) * 10):,} shards")

    print("\n" + "=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()