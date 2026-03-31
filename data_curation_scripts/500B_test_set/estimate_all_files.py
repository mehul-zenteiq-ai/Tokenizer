#!/usr/bin/env python3
"""
estimate_all_tokens.py — Estimate token counts for EVERY source in the 500B spec.

For each source:
  1. List all files, compute total disk size
  2. Sample N files (spread across the file list)
  3. Fully read each sampled file, count words → tokens via heuristic
  4. Compute tokens/byte ratio from samples
  5. Extrapolate total tokens from total disk size

Token heuristics:
  - Latin:  words × 1.35
  - CJK (Mandarin, Japanese):  chars / 1.5

Usage:
    python estimate_all_tokens.py 2>&1 | tee all_token_estimates.log
"""

import glob
import gzip
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

# ── Force unbuffered stdout so tee/pipe shows output immediately ──
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

try:
    import zstandard as zstd
except ImportError:
    print("FATAL: pip install zstandard --break-system-packages")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
MAX_WORKERS = 64
SAMPLES_PER_SOURCE = 5
TOKEN_MULT = 1.35
TOKEN_DIV_CJK = 1.5


# ═══════════════════════════════════════════════════════════════
# Source definition
# ═══════════════════════════════════════════════════════════════

@dataclass
class InputSource:
    name: str
    domain: str
    glob_pattern: str
    fmt: str            # parquet | arrow_stream | jsonl_zst | jsonl_gz | json_gz | txt_xz
    text_col: str = ""
    text_key: str = ""
    special: str = ""
    is_single_file: bool = False
    is_cjk: bool = False
    quota: int = -1


def build_all_sources() -> list[InputSource]:
    S = []

    # ── English ────────────────────────────────────────────
    S.append(InputSource(
        name="English/wikipedia", domain="English",
        glob_pattern=f"{BASE}/English/wikipedia/*.parquet",
        fmt="parquet", text_col="content", quota=-1,
    ))
    S.append(InputSource(
        name="English/dclm", domain="English",
        glob_pattern=f"{BASE}/English/dclm_baseline_4T/global-shard_02_of_10/local-shard_*_of_10/*.jsonl.zst",
        fmt="jsonl_zst", text_key="text", quota=100_000_000_000,
    ))
    S.append(InputSource(
        name="English/fineweb_edu", domain="English",
        glob_pattern=f"{BASE}/English/fineweb_edu/data/CC-MAIN-*/*.parquet",
        fmt="parquet", text_col="text", quota=196_000_000_000,
    ))

    # ── Math ───────────────────────────────────────────────
    mp_base = f"{BASE}/math/MathPile/train"
    for sub in ["arXiv", "commoncrawl", "textbooks", "wikipedia"]:
        S.append(InputSource(
            name=f"Math/MathPile/{sub}", domain="Math",
            glob_pattern=f"{mp_base}/{sub}/*.jsonl.gz",
            fmt="jsonl_gz", text_key="text", quota=40_000_000_000,
        ))
    S.append(InputSource(
        name="Math/MathPile/stackexchange", domain="Math",
        glob_pattern=f"{mp_base}/stackexchange/*.jsonl.gz",
        fmt="jsonl_gz", special="stackexchange", quota=40_000_000_000,
    ))
    S.append(InputSource(
        name="Math/finemath4+", domain="Math",
        glob_pattern=f"{BASE}/math/finemath4+/finemath-4plus/*.parquet",
        fmt="parquet", text_col="text", quota=80_000_000_000,
    ))
    S.append(InputSource(
        name="Math/finemath3+", domain="Math",
        glob_pattern=f"{BASE}/math/finemath3+/finemath-3plus/*.parquet",
        fmt="parquet", text_col="text", quota=80_000_000_000,
    ))

    # ── Code ───────────────────────────────────────────────
    code_langs = [
        ("Python",     "python",     "python",     "Python"),
        ("Cpp",        "cpp",        "c++",        "Cpp"),
        ("JavaScript", "javascript", "javascript", "JavaScript"),
        ("Java",       None,         "java",       "Java"),
        ("HTML",       "html",       "html",       None),
        ("Fortran",    "fortran",    "fortran",    None),
        ("Rust",       "rust",       "rust",       None),
        ("C",          None,         "c",          "C"),
        ("TypeScript", None,         "typescript", "TypeScript"),
        ("SQL",        None,         "sql",        "SQL"),
        ("Shell",      None,         "shell",      "Shell"),
    ]
    for lang_out, sc_dir, bc_dir, se_dir in code_langs:
        if sc_dir:
            S.append(InputSource(
                name=f"Code/{lang_out}/starcoderdata", domain="Code",
                glob_pattern=f"{BASE}/code/starcoderdata/{sc_dir}/*.parquet",
                fmt="parquet", text_col="content",
            ))
        if bc_dir:
            S.append(InputSource(
                name=f"Code/{lang_out}/BIGCODE", domain="Code",
                glob_pattern=f"{BASE}/Tokenizer_Data/BIGCODE/{bc_dir}/*.arrow",
                fmt="arrow_stream", text_col="content",
            ))
        if se_dir:
            S.append(InputSource(
                name=f"Code/{lang_out}/stack_edu", domain="Code",
                glob_pattern=f"{BASE}/code/stack_edu/{se_dir}/*.parquet",
                fmt="parquet", text_col="text",
            ))

    # ── Scientific ─────────────────────────────────────────
    S.append(InputSource(
        name="Scientific/peS2o", domain="Scientific",
        glob_pattern=f"{BASE}/Scientific/peS2o/data/v2/*.json.gz",
        fmt="json_gz", text_key="text",
    ))
    S.append(InputSource(
        name="Scientific/arxiv", domain="Scientific",
        glob_pattern=f"{BASE}/Tokenizer_data/final_arxiv_arnav/*.jsonl.zst",
        fmt="jsonl_zst", text_key="text",
    ))

    # ── Multilingual / Sangraha ────────────────────────────
    sg = f"{BASE}/Multilingual_data/sangraha/unverified/unverified"
    for lang_name, code in [
        ("Hindi", "hin"), ("Bengali", "ben"), ("Urdu", "urd"),
        ("Tamil", "tam"), ("Marathi", "mar"), ("Gujarati", "guj"),
        ("Malayalam", "mal"), ("Kannada", "kan"), ("Punjabi", "pan"),
        ("Odia", "ori"),
    ]:
        S.append(InputSource(
            name=f"Multilingual/{lang_name}/sangraha", domain="Multilingual",
            glob_pattern=f"{sg}/{code}/*.parquet",
            fmt="parquet", text_col="text", quota=6_818_181_818,
        ))

    # ── Multilingual / CC100 ───────────────────────────────
    cc = f"{BASE}/Multilingual_data/cc100"
    for lang_name, subfolder, filename, cjk in [
        ("Russian",  "rus", "ru.txt.xz",      False),
        ("French",   "fra", "fr.txt.xz",      False),
        ("German",   "deu", "de.txt.xz",      False),
        ("Mandarin", "cmn", "zh-Hans.txt.xz",  True),
        ("Japanese", "jpn", "ja.txt.xz",       True),
        ("Korean",   "kor", "ko.txt.xz",      False),
        ("Farsi",    "fas", "fa.txt.xz",      False),
        ("Nepali",   "npi", "ne.txt.xz",      False),
        ("Sinhala",  "sin", "si.txt.xz",      False),
        ("Pashto",   "pus", "ps.txt.xz",      False),
        ("Burmese",  "mya", "my.txt.xz",      False),
    ]:
        S.append(InputSource(
            name=f"Multilingual/{lang_name}/cc100", domain="Multilingual",
            glob_pattern=f"{cc}/{subfolder}/{filename}",
            fmt="txt_xz", is_single_file=True, is_cjk=cjk,
            quota=6_818_181_818,
        ))

    return S


# ═══════════════════════════════════════════════════════════════
# Token counting
# ═══════════════════════════════════════════════════════════════

def count_tokens_text(text: str, is_cjk: bool) -> int:
    if not text:
        return 0
    if is_cjk:
        return int(len(text) / TOKEN_DIV_CJK)
    return int(len(text.split()) * TOKEN_MULT)


# ═══════════════════════════════════════════════════════════════
# Per-file token counters (each reads one file completely)
# ═══════════════════════════════════════════════════════════════

def count_file_parquet(filepath, text_col, is_cjk):
    fsize = os.path.getsize(filepath)
    total = 0
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(batch_size=5000, columns=[text_col]):
        col = batch.column(text_col)
        for i in range(len(col)):
            val = col[i].as_py()
            if val:
                total += count_tokens_text(val, is_cjk)
    return total, fsize


def count_file_arrow_stream(filepath, text_col, is_cjk):
    fsize = os.path.getsize(filepath)
    total = 0
    with open(filepath, "rb") as f:
        reader = ipc.open_stream(f)
        try:
            while True:
                batch = reader.read_next_batch()
                col = batch.column(text_col)
                for i in range(len(col)):
                    val = col[i].as_py()
                    if val:
                        total += count_tokens_text(val, is_cjk)
        except StopIteration:
            pass
    return total, fsize


def count_file_jsonl_zst(filepath, text_key, is_cjk):
    fsize = os.path.getsize(filepath)
    total = 0
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_key, "")
                    if text:
                        total += count_tokens_text(text, is_cjk)
                except (json.JSONDecodeError, KeyError):
                    continue
    return total, fsize


def count_file_jsonl_gz(filepath, text_key, special, is_cjk):
    fsize = os.path.getsize(filepath)
    total = 0
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if special == "stackexchange":
                    parts = []
                    q = obj.get("question", "")
                    if q:
                        parts.append(q)
                    for ans in obj.get("answers", []):
                        if isinstance(ans, dict):
                            body = ans.get("Body", "")
                            if body:
                                parts.append(body)
                    text = "\n\n".join(parts)
                else:
                    text = obj.get(text_key, "")
                if text:
                    total += count_tokens_text(text, is_cjk)
            except (json.JSONDecodeError, KeyError):
                continue
    return total, fsize


def count_file_json_gz(filepath, text_key, is_cjk):
    return count_file_jsonl_gz(filepath, text_key, "", is_cjk)


def count_file_txt_xz(filepath, is_cjk):
    fsize = os.path.getsize(filepath)
    total = 0
    proc = subprocess.Popen(
        ["xz", "-dc", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )
    try:
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            total += count_tokens_text(line, is_cjk)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    return total, fsize


# ═══════════════════════════════════════════════════════════════
# Single-file worker — called by the pool for EACH file
# ═══════════════════════════════════════════════════════════════

def count_single_file(args: tuple) -> dict:
    """
    Worker function: count tokens in a single file.
    This is the unit of parallelism — one file = one task.
    Must be a TOP-LEVEL function (picklable).
    """
    source_name, filepath, fmt, text_col, text_key, special, is_cjk, file_idx, total_files = args

    pid = os.getpid()
    basename = os.path.basename(filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"  [PID {pid}] START  {source_name}  file {file_idx+1}/{total_files}  "
          f"{basename}  ({fsize_mb:.0f} MB)", flush=True)

    t0 = time.time()
    result = {
        "source_name": source_name,
        "filepath": filepath,
        "basename": basename,
        "tokens": 0,
        "file_bytes": 0,
        "elapsed": 0,
        "error": None,
    }

    try:
        if fmt == "parquet":
            tok, fb = count_file_parquet(filepath, text_col, is_cjk)
        elif fmt == "arrow_stream":
            tok, fb = count_file_arrow_stream(filepath, text_col, is_cjk)
        elif fmt == "jsonl_zst":
            tok, fb = count_file_jsonl_zst(filepath, text_key, is_cjk)
        elif fmt == "jsonl_gz":
            tok, fb = count_file_jsonl_gz(filepath, text_key, special, is_cjk)
        elif fmt == "json_gz":
            tok, fb = count_file_json_gz(filepath, text_key, is_cjk)
        elif fmt == "txt_xz":
            tok, fb = count_file_txt_xz(filepath, is_cjk)
        else:
            result["error"] = f"Unknown format: {fmt}"
            return result

        elapsed = time.time() - t0
        result["tokens"] = tok
        result["file_bytes"] = fb
        result["elapsed"] = elapsed

        tok_B = tok / 1e9
        rate = tok / elapsed / 1e6 if elapsed > 0 else 0
        print(f"  [PID {pid}] DONE   {source_name}  file {file_idx+1}/{total_files}  "
              f"{basename}  → {tok_B:.3f}B tokens  {elapsed:.1f}s  {rate:.0f}M tok/s",
              flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        result["error"] = f"{e}"
        print(f"  [PID {pid}] ERROR  {source_name}  {basename}  → {e}", flush=True)

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 120, flush=True)
    print("  COMPREHENSIVE TOKEN ESTIMATION — ALL SOURCES", flush=True)
    print(f"  Base: {BASE}", flush=True)
    print(f"  Workers: {MAX_WORKERS}, Samples/source: {SAMPLES_PER_SOURCE}", flush=True)
    print("=" * 120, flush=True)
    print(flush=True)

    sources = build_all_sources()
    print(f"Registered {len(sources)} input sources.", flush=True)

    # ═══════════════════════════════════════════════════════
    # Phase 1: Discover files & build task list
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*120}", flush=True)
    print("  PHASE 1: Discovering files and selecting samples...", flush=True)
    print(f"{'─'*120}", flush=True)

    source_meta = {}   # name → {files, total_bytes, sample_indices}
    all_tasks = []     # flat list of (args_tuple) for the pool

    for src in sources:
        # Resolve files
        if src.is_single_file:
            files = [src.glob_pattern] if os.path.isfile(src.glob_pattern) else []
        else:
            files = sorted(glob.glob(src.glob_pattern))

        total_bytes = sum(os.path.getsize(f) for f in files)

        # Select sample indices
        n_samples = min(SAMPLES_PER_SOURCE, len(files))
        if n_samples == 0:
            sample_indices = []
        elif n_samples >= len(files):
            sample_indices = list(range(len(files)))
        else:
            step = len(files) / n_samples
            sample_indices = [int(i * step) for i in range(n_samples)]

        source_meta[src.name] = {
            "files": files,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "sample_indices": sample_indices,
            "domain": src.domain,
            "quota": src.quota,
            "is_cjk": src.is_cjk,
        }

        disk_gb = total_bytes / (1024**3)
        status = "OK" if files else "NO FILES"
        print(f"  {status:>8s}  {src.name:<45s}  files={len(files):>5d}  "
              f"disk={disk_gb:>7.1f} GB  sampling {len(sample_indices)} files",
              flush=True)

        # Build tasks for each sampled file
        for local_idx, file_idx in enumerate(sample_indices):
            filepath = files[file_idx]
            task = (
                src.name,
                filepath,
                src.fmt,
                src.text_col,
                src.text_key,
                src.special,
                src.is_cjk,
                local_idx,
                len(sample_indices),
            )
            all_tasks.append(task)

    print(f"\n  Total file-level tasks: {len(all_tasks)}", flush=True)
    print(f"  Launching {MAX_WORKERS} worker processes...\n", flush=True)

    # ═══════════════════════════════════════════════════════
    # Phase 2: Count tokens in parallel (file-level parallelism)
    # ═══════════════════════════════════════════════════════
    print(f"{'─'*120}", flush=True)
    print("  PHASE 2: Counting tokens (one task per file)...", flush=True)
    print(f"{'─'*120}\n", flush=True)

    t0 = time.time()
    file_results = []

    # Use multiprocessing.Pool with fork context for guaranteed OS processes
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        # imap_unordered gives us results as they complete
        for i, result in enumerate(pool.imap_unordered(count_single_file, all_tasks)):
            file_results.append(result)
            # Progress counter every 10 files
            if (i + 1) % 10 == 0 or (i + 1) == len(all_tasks):
                elapsed = time.time() - t0
                print(f"\n  >>> Progress: {i+1}/{len(all_tasks)} files done, "
                      f"{elapsed:.0f}s elapsed <<<\n", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n  All {len(all_tasks)} files processed in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f}m)\n", flush=True)

    # ═══════════════════════════════════════════════════════
    # Phase 3: Aggregate results per source and extrapolate
    # ═══════════════════════════════════════════════════════
    print(f"{'─'*120}", flush=True)
    print("  PHASE 3: Aggregating and extrapolating...", flush=True)
    print(f"{'─'*120}\n", flush=True)

    # Group file results by source name
    source_samples = {}  # name → list of {tokens, file_bytes}
    for fr in file_results:
        sn = fr["source_name"]
        if sn not in source_samples:
            source_samples[sn] = []
        if fr["error"] is None:
            source_samples[sn].append({"tokens": fr["tokens"], "file_bytes": fr["file_bytes"]})

    # Compute estimates
    source_estimates = {}  # name → {estimated_tokens, tokens_per_byte, ...}

    print(f"  {'Source':<45s}  {'Files':>6s}  {'Disk GB':>8s}  {'Est Tokens':>12s}  "
          f"{'Quota':>12s}  {'Will Take':>12s}  {'tok/byte':>8s}", flush=True)
    print("  " + "─" * 115, flush=True)

    for src in sources:
        meta = source_meta[src.name]
        samples = source_samples.get(src.name, [])

        sample_tokens = sum(s["tokens"] for s in samples)
        sample_bytes = sum(s["file_bytes"] for s in samples)

        if sample_bytes > 0 and samples:
            ratio = sample_tokens / sample_bytes
            estimated = int(ratio * meta["total_bytes"])
        else:
            ratio = 0
            estimated = 0

        quota = meta["quota"]
        will_take = min(estimated, quota) if quota > 0 else estimated

        source_estimates[src.name] = {
            "domain": meta["domain"],
            "file_count": meta["file_count"],
            "total_bytes": meta["total_bytes"],
            "sample_tokens": sample_tokens,
            "sample_bytes": sample_bytes,
            "tokens_per_byte": ratio,
            "estimated_tokens": estimated,
            "quota": quota,
            "will_take": will_take,
        }

        est_B = estimated / 1e9
        disk_gb = meta["total_bytes"] / (1024**3)
        quota_str = f"{quota/1e9:.2f}B" if quota > 0 else "take all"
        take_B = will_take / 1e9

        print(f"  {src.name:<45s}  {meta['file_count']:>6d}  {disk_gb:>7.1f}G  "
              f"{est_B:>10.2f}B  {quota_str:>12s}  {take_B:>10.2f}B  {ratio:>8.4f}",
              flush=True)

    # ═══════════════════════════════════════════════════════
    # Aggregation by output folder
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*120}", flush=True)
    print("  AGGREGATED BY OUTPUT FOLDER", flush=True)
    print(f"{'='*120}\n", flush=True)

    folder_data = {}
    for name, est in source_estimates.items():
        domain = est["domain"]
        parts = name.split("/")
        if domain == "Code":
            folder = f"Code/{parts[1]}"
        elif name.startswith("Math/MathPile"):
            folder = "Math/MathPile"
        elif domain == "Multilingual":
            folder = f"Multilingual/{parts[1]}"
        else:
            folder = name

        if folder not in folder_data:
            folder_data[folder] = {"available": 0, "quota": est["quota"], "domain": domain}
        folder_data[folder]["available"] += est["estimated_tokens"]
        if est["quota"] > 0:
            folder_data[folder]["quota"] = est["quota"]

    print(f"  {'Output Folder':<35s}  {'Available':>12s}  {'Quota':>12s}  "
          f"{'Will Take':>12s}  {'Shards':>8s}", flush=True)
    print("  " + "─" * 85, flush=True)

    grand_available = 0
    grand_take = 0
    grand_shards = 0
    domain_summary = {}

    for folder in sorted(folder_data.keys()):
        d = folder_data[folder]
        avail = d["available"]
        quota = d["quota"]
        domain = d["domain"]
        will_take = min(avail, quota) if quota > 0 else avail
        shards = int(will_take / 100_000_000) if will_take > 0 else 0

        avail_B = avail / 1e9
        quota_str = f"{quota/1e9:.2f}B" if quota > 0 else "take all"
        take_B = will_take / 1e9

        print(f"  {folder:<35s}  {avail_B:>10.2f}B  {quota_str:>12s}  "
              f"{take_B:>10.2f}B  {shards:>8,}", flush=True)

        grand_available += avail
        grand_take += will_take
        grand_shards += shards

        if domain not in domain_summary:
            domain_summary[domain] = {"available": 0, "take": 0, "shards": 0}
        domain_summary[domain]["available"] += avail
        domain_summary[domain]["take"] += will_take
        domain_summary[domain]["shards"] += shards

    # ═══════════════════════════════════════════════════════
    # Domain summary
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*120}", flush=True)
    print("  DOMAIN SUMMARY", flush=True)
    print(f"{'='*120}\n", flush=True)
    print(f"  {'Domain':<15s}  {'Available':>12s}  {'Will Take':>12s}  {'Shards':>8s}",
          flush=True)
    print("  " + "─" * 55, flush=True)

    for domain in ["English", "Math", "Code", "Scientific", "Multilingual"]:
        if domain in domain_summary:
            ds = domain_summary[domain]
            print(f"  {domain:<15s}  {ds['available']/1e9:>10.2f}B  "
                  f"{ds['take']/1e9:>10.2f}B  {ds['shards']:>8,}", flush=True)

    print(f"  {'─'*55}", flush=True)
    print(f"  {'TOTAL':<15s}  {grand_available/1e9:>10.2f}B  "
          f"{grand_take/1e9:>10.2f}B  {grand_shards:>8,}", flush=True)

    # ═══════════════════════════════════════════════════════
    # Spec comparison
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*120}", flush=True)
    print("  SPEC vs REALITY", flush=True)
    print(f"{'='*120}\n", flush=True)
    spec_avail = {
        "English": 300, "Math": 44, "Code": 80,
        "Scientific": 23, "Multilingual": 62,
    }
    print(f"  {'Domain':<15s}  {'Spec Said':>12s}  {'Actual':>12s}  {'Delta':>12s}",
          flush=True)
    print("  " + "─" * 55, flush=True)
    for domain in ["English", "Math", "Code", "Scientific", "Multilingual"]:
        spec = spec_avail.get(domain, 0)
        actual = domain_summary.get(domain, {}).get("available", 0) / 1e9
        delta = actual - spec
        sign = "+" if delta >= 0 else ""
        print(f"  {domain:<15s}  {spec:>10.1f}B  {actual:>10.1f}B  "
              f"{sign}{delta:>10.1f}B", flush=True)

    total_spec = sum(spec_avail.values())
    total_actual = grand_available / 1e9
    delta = total_actual - total_spec
    sign = "+" if delta >= 0 else ""
    print(f"  {'─'*55}", flush=True)
    print(f"  {'TOTAL':<15s}  {total_spec:>10.1f}B  {total_actual:>10.1f}B  "
          f"{sign}{delta:>10.1f}B", flush=True)

    # ═══════════════════════════════════════════════════════
    # Failures
    # ═══════════════════════════════════════════════════════
    failed_files = [fr for fr in file_results if fr["error"] is not None]
    no_file_sources = [s.name for s in sources if source_meta[s.name]["file_count"] == 0]

    if no_file_sources:
        print(f"\n  ❌ Sources with NO FILES:", flush=True)
        for n in no_file_sources:
            print(f"     {n}", flush=True)

    if failed_files:
        print(f"\n  ⚠️  {len(failed_files)} individual file(s) had errors:", flush=True)
        for fr in failed_files:
            print(f"     {fr['source_name']} / {fr['basename']}: {fr['error']}", flush=True)

    print(f"\n  Total estimation time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)",
          flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()