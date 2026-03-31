#!/usr/bin/env python3
"""
eval_tokenizer.py — Evaluate a tokenizer on the curated ~557B dataset.

Metrics per shard (100M tokens):
  - Fertility:   num_tokens / num_words  (chars for CJK)
  - NSL:         num_tokens / num_chars
  - Bytes/token: total_utf8_bytes / num_tokens

Architecture:
  - Single tokenizer loaded once, shared across 64 threads
  - One thread per source, shards processed sequentially within each thread
  - HF fast tokenizer uses Rust encode_batch (releases GIL → true parallelism)
  - Resume via tracking completed shard filenames in master.csv
  - Checkpoint markers at 1B-token boundaries

Output:
  {EVAL_BASE}/{tokenizer_name}/{domain}/{source}/master.csv

Usage:
    python eval_tokenizer.py \\
        --tokenizer-name my_tok \\
        --tokenizer-type auto \\
        --tokenizer-path /path/to/tokenizer.json \\
        --workers 64 \\
        --log eval.log
"""

import argparse
import csv
import glob
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import pyarrow.parquet as pq
from tqdm import tqdm

# Lazy imports for tokenizer loading (avoid crash if not installed)
# transformers, tiktoken imported inside load_tokenizer()


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
DATA_DIR = f"{BASE}/final_tokenizer_test_data"
EVAL_BASE = f"{BASE}/final_tokenizer_test_data_eval/tokenizer_results"

ENCODE_BATCH_SIZE = 1000              # docs per encode_batch call
CHECKPOINT_INTERVAL = 1_000_000_000   # 1B tokens between checkpoint markers

# ── Domain selection — set True to evaluate, False to skip ──
EVAL_DOMAINS = {
    "Math":         False,
    "Code":         True,
    "Scientific":   False,
    "English":      False,
    "Multilingual": False,
}


# ═══════════════════════════════════════════════════════════════
# Source registry — maps domain → list of eval targets
# ═══════════════════════════════════════════════════════════════

ALL_SOURCES = {
    "English": [
        {"name": "wikipedia",   "subfolder": "English/wikipedia",   "is_cjk": False},
        {"name": "dclm",        "subfolder": "English/dclm",        "is_cjk": False},
        {"name": "fineweb_edu", "subfolder": "English/fineweb_edu", "is_cjk": False},
    ],
    "Math": [
        {"name": "MathPile",   "subfolder": "Math/MathPile",   "is_cjk": False},
        {"name": "finemath4+", "subfolder": "Math/finemath4+", "is_cjk": False},
        {"name": "finemath3+", "subfolder": "Math/finemath3+", "is_cjk": False},
    ],
    "Code": [
        {"name": "Python",     "subfolder": "Code/Python",     "is_cjk": False},
        {"name": "Cpp",        "subfolder": "Code/Cpp",        "is_cjk": False},
        {"name": "JavaScript", "subfolder": "Code/JavaScript", "is_cjk": False},
        {"name": "Java",       "subfolder": "Code/Java",       "is_cjk": False},
        {"name": "HTML",       "subfolder": "Code/HTML",       "is_cjk": False},
        {"name": "Fortran",    "subfolder": "Code/Fortran",    "is_cjk": False},
        {"name": "Rust",       "subfolder": "Code/Rust",       "is_cjk": False},
        {"name": "C",          "subfolder": "Code/C",          "is_cjk": False},
        {"name": "TypeScript", "subfolder": "Code/TypeScript", "is_cjk": False},
        {"name": "SQL",        "subfolder": "Code/SQL",        "is_cjk": False},
        {"name": "Shell",      "subfolder": "Code/Shell",      "is_cjk": False},
    ],
    "Scientific": [
        {"name": "peS2o", "subfolder": "Scientific/peS2o", "is_cjk": False},
        {"name": "arxiv", "subfolder": "Scientific/arxiv", "is_cjk": False},
    ],
    "Multilingual": [
        {"name": "Hindi",     "subfolder": "Multilingual/Hindi",     "is_cjk": False},
        {"name": "Bengali",   "subfolder": "Multilingual/Bengali",   "is_cjk": False},
        {"name": "Urdu",      "subfolder": "Multilingual/Urdu",      "is_cjk": False},
        {"name": "Tamil",     "subfolder": "Multilingual/Tamil",     "is_cjk": False},
        {"name": "Marathi",   "subfolder": "Multilingual/Marathi",   "is_cjk": False},
        {"name": "Gujarati",  "subfolder": "Multilingual/Gujarati",  "is_cjk": False},
        {"name": "Malayalam", "subfolder": "Multilingual/Malayalam", "is_cjk": False},
        {"name": "Kannada",   "subfolder": "Multilingual/Kannada",   "is_cjk": False},
        {"name": "Punjabi",   "subfolder": "Multilingual/Punjabi",   "is_cjk": False},
        {"name": "Odia",      "subfolder": "Multilingual/Odia",      "is_cjk": False},
        {"name": "Russian",   "subfolder": "Multilingual/Russian",   "is_cjk": False},
        {"name": "French",    "subfolder": "Multilingual/French",    "is_cjk": False},
        {"name": "German",    "subfolder": "Multilingual/German",    "is_cjk": False},
        {"name": "Mandarin",  "subfolder": "Multilingual/Mandarin",  "is_cjk": True},
        {"name": "Japanese",  "subfolder": "Multilingual/Japanese",  "is_cjk": True},
        {"name": "Korean",    "subfolder": "Multilingual/Korean",    "is_cjk": False},
        {"name": "Farsi",     "subfolder": "Multilingual/Farsi",     "is_cjk": False},
        {"name": "Nepali",    "subfolder": "Multilingual/Nepali",    "is_cjk": False},
        {"name": "Sinhala",   "subfolder": "Multilingual/Sinhala",   "is_cjk": False},
        {"name": "Pashto",    "subfolder": "Multilingual/Pashto",    "is_cjk": False},
        {"name": "Burmese",   "subfolder": "Multilingual/Burmese",   "is_cjk": False},
    ],
}

# CSV header
CSV_FIELDS = [
    "shard_file", "num_docs", "num_words", "num_chars", "num_bytes",
    "num_tokens", "fertility", "nsl", "bytes_per_token", "cumulative_tokens",
]


# ═══════════════════════════════════════════════════════════════
# Logging — thread-safe, writes to both stdout and optional file
# ═══════════════════════════════════════════════════════════════

_log_lock = threading.Lock()
_log_file_handle = None


def log(msg: str):
    """Thread-safe log: writes via tqdm.write (preserves progress bar)."""
    tqdm.write(msg)
    if _log_file_handle:
        with _log_lock:
            _log_file_handle.write(msg + "\n")
            _log_file_handle.flush()


# ═══════════════════════════════════════════════════════════════
# Tokenizer loading (adapted from user's codebase)
# ═══════════════════════════════════════════════════════════════

def _is_valid_tokenizers_json(json_path: str) -> bool:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            head = fh.read(4096)
        return '"model"' in head and '"added_tokens"' in head
    except Exception:
        return False


def load_tokenizer(name: str, tok_type: str, path: str):
    """
    Load a tokenizer and return (tokenizer_object, vocab_size, effective_type).
    effective_type is "hf" or "tiktoken".
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if tok_type == "hf":
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tok, getattr(tok, "vocab_size", len(tok.get_vocab())), "hf"

    elif tok_type == "tiktoken":
        import tiktoken
        tok = tiktoken.get_encoding(path)
        return tok, tok.n_vocab, "tiktoken"

    elif tok_type == "fast":
        tok = PreTrainedTokenizerFast(tokenizer_file=path)
        return tok, tok.vocab_size or len(tok.get_vocab()), "hf"

    elif tok_type == "auto":
        parent_dir = os.path.dirname(path)
        # 1. Try AutoTokenizer from directory
        if _is_valid_tokenizers_json(path):
            try:
                tok = AutoTokenizer.from_pretrained(parent_dir, trust_remote_code=True)
                return tok, getattr(tok, "vocab_size", len(tok.get_vocab())), "hf"
            except Exception:
                pass
        # 2. Try PreTrainedTokenizerFast directly
        try:
            tok = PreTrainedTokenizerFast(tokenizer_file=path)
            return tok, tok.vocab_size or len(tok.get_vocab()), "hf"
        except Exception:
            pass
        # 3. Raw tokenizers lib
        from tokenizers import Tokenizer as _RawTokenizer
        raw = _RawTokenizer.from_file(path)
        tok = PreTrainedTokenizerFast(tokenizer_object=raw)
        return tok, tok.vocab_size or len(tok.get_vocab()), "hf"

    elif tok_type == "hf_fast":
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        return tok, getattr(tok, "vocab_size", len(tok.get_vocab())), "hf"

    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")


def create_encoder(tokenizer, tok_type: str):
    """
    Return a function: encode_fn(texts: list[str]) -> list[int]
    that returns token counts for each text.

    For HF fast tokenizers, uses the Rust backend's encode_batch
    which releases the GIL for true thread parallelism.
    """
    if tok_type == "tiktoken":
        def encode_fn(texts):
            return [len(ids) for ids in tokenizer.encode_ordinary_batch(texts)]
        return encode_fn

    # HF tokenizer — try fast backend first
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        def encode_fn(texts):
            encodings = backend.encode_batch(texts, add_special_tokens=False)
            return [len(enc.ids) for enc in encodings]
        return encode_fn

    # Slow tokenizer fallback (sequential, holds GIL)
    log("  ⚠️  No fast backend found — falling back to sequential encode (slow)")

    def encode_fn(texts):
        return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return encode_fn


# ═══════════════════════════════════════════════════════════════
# Shard evaluation
# ═══════════════════════════════════════════════════════════════

def eval_shard(encode_fn, shard_path: str, is_cjk: bool) -> dict:
    """
    Evaluate a single shard. Returns metrics dict.
    Reads parquet in batches and encodes in chunks.
    """
    pf = pq.ParquetFile(shard_path)

    total_docs = 0
    total_words = 0     # whitespace words for Latin, char count for CJK
    total_chars = 0
    total_bytes = 0
    total_tokens = 0

    for batch in pf.iter_batches(batch_size=ENCODE_BATCH_SIZE, columns=["text"]):
        texts_raw = batch.column("text").to_pylist()
        texts = [t for t in texts_raw if t is not None and len(t) > 0]

        if not texts:
            continue

        # Encode batch → token counts
        token_counts = encode_fn(texts)

        # Accumulate metrics
        for text, n_tok in zip(texts, token_counts):
            total_docs += 1
            total_tokens += n_tok
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            if is_cjk:
                total_words += len(text)       # char-based fertility for CJK
            else:
                total_words += len(text.split())

    # Compute ratios (guard against division by zero)
    fertility = total_tokens / total_words if total_words > 0 else 0.0
    nsl = total_tokens / total_chars if total_chars > 0 else 0.0
    bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0.0

    return {
        "shard_file": os.path.basename(shard_path),
        "num_docs": total_docs,
        "num_words": total_words,
        "num_chars": total_chars,
        "num_bytes": total_bytes,
        "num_tokens": total_tokens,
        "fertility": round(fertility, 6),
        "nsl": round(nsl, 6),
        "bytes_per_token": round(bytes_per_token, 6),
    }


# ═══════════════════════════════════════════════════════════════
# Resume helpers
# ═══════════════════════════════════════════════════════════════

def get_completed_shards(csv_path: str) -> tuple[set, int]:
    """
    Read existing master.csv to find completed shard filenames
    and the last cumulative_tokens value.
    Returns (set_of_completed_filenames, last_cumulative_tokens).
    """
    if not os.path.isfile(csv_path):
        return set(), 0
    completed = set()
    last_cum = 0
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["shard_file"])
                last_cum = int(row["cumulative_tokens"])
    except Exception:
        return set(), 0
    return completed, last_cum


def write_checkpoint(out_dir: str, cumulative_tokens: int, last_checkpoint: int) -> int:
    """
    Write a checkpoint marker file if we crossed a 1B boundary.
    Returns the updated last_checkpoint value.
    """
    current_cp = (cumulative_tokens // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
    if current_cp > last_checkpoint and current_cp > 0:
        marker = os.path.join(out_dir, f"checkpoint_{current_cp // 1_000_000_000}B.marker")
        with open(marker, "w") as f:
            f.write(f"cumulative_tokens={cumulative_tokens}\nshards_so_far={cumulative_tokens // 100_000_000}\n")
        return current_cp
    return last_checkpoint


# ═══════════════════════════════════════════════════════════════
# Source worker — runs in a thread
# ═══════════════════════════════════════════════════════════════

def process_source(encode_fn, source_info: dict, tokenizer_name: str, pbar) -> dict:
    """
    Evaluate all shards for one source. Runs in a thread.
    Writes results to master.csv incrementally with resume support.
    """
    name = source_info["name"]
    subfolder = source_info["subfolder"]
    is_cjk = source_info["is_cjk"]
    domain = subfolder.split("/")[0]

    result = {
        "name": f"{domain}/{name}",
        "status": "UNKNOWN",
        "shards_total": 0,
        "shards_evaluated": 0,
        "shards_skipped": 0,
        "total_tokens": 0,
        "error": None,
    }

    try:
        # ── Discover shards ────────────────────────────────
        shard_dir = os.path.join(DATA_DIR, subfolder)
        if not os.path.isdir(shard_dir):
            result["status"] = "NO_DATA"
            log(f"  ⚠️  {domain}/{name}: no data directory found")
            return result

        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
        if not shard_files:
            result["status"] = "NO_DATA"
            log(f"  ⚠️  {domain}/{name}: no shard files found")
            return result

        result["shards_total"] = len(shard_files)

        # ── Setup output CSV ───────────────────────────────
        out_dir = os.path.join(EVAL_BASE, tokenizer_name, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "master.csv")

        # ── Resume check ───────────────────────────────────
        completed, cumulative_tokens = get_completed_shards(csv_path)
        last_checkpoint = (cumulative_tokens // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL

        if len(completed) >= len(shard_files):
            result["status"] = "SKIPPED"
            result["shards_skipped"] = len(shard_files)
            result["shards_evaluated"] = 0
            result["total_tokens"] = cumulative_tokens
            log(f"  ⏭️  {domain}/{name}: all {len(shard_files)} shards already done "
                f"({cumulative_tokens/1e9:.2f}B tokens)")
            pbar.update(len(shard_files))
            return result

        if completed:
            log(f"  🔄 {domain}/{name}: resuming — {len(completed)}/{len(shard_files)} "
                f"shards done, {cumulative_tokens/1e9:.2f}B tokens so far")
            pbar.update(len(completed))

        # ── Open CSV for appending ─────────────────────────
        write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        # ── Process shards sequentially ────────────────────
        t0 = time.time()
        evaluated = 0

        for shard_path in shard_files:
            shard_basename = os.path.basename(shard_path)

            # Skip already-completed shards
            if shard_basename in completed:
                continue

            try:
                metrics = eval_shard(encode_fn, shard_path, is_cjk)
            except Exception as e:
                log(f"  ⚠️  {domain}/{name}/{shard_basename}: ERROR — {e}")
                pbar.update(1)
                continue

            cumulative_tokens += metrics["num_tokens"]
            metrics["cumulative_tokens"] = cumulative_tokens

            # Write row to CSV immediately (resume-safe)
            writer.writerow(metrics)
            csv_file.flush()

            # Checkpoint marker
            last_checkpoint = write_checkpoint(out_dir, cumulative_tokens, last_checkpoint)

            evaluated += 1
            pbar.update(1)

        csv_file.close()

        elapsed = time.time() - t0
        rate = cumulative_tokens / elapsed / 1e6 if elapsed > 0 else 0
        result["status"] = "OK"
        result["shards_evaluated"] = evaluated
        result["shards_skipped"] = len(completed)
        result["total_tokens"] = cumulative_tokens
        log(f"  ✅ {domain}/{name}: {evaluated} new + {len(completed)} resumed = "
            f"{len(shard_files)} shards, {cumulative_tokens/1e9:.2f}B tokens, "
            f"{elapsed/60:.1f}m, {rate:.0f}M tok/s")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        log(f"  ❌ {domain}/{name}: FAILED — {e}\n{traceback.format_exc()}")

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer on curated dataset")
    parser.add_argument("--tokenizer-name", required=True, help="Name for output folder")
    parser.add_argument("--tokenizer-type", required=True,
                        choices=["hf", "auto", "fast", "tiktoken", "hf_fast"],
                        help="Tokenizer type")
    parser.add_argument("--tokenizer-path", required=True, help="Path or HF model name")
    parser.add_argument("--workers", type=int, default=64, help="Number of threads")
    parser.add_argument("--log", default=None, help="Path to log file (optional)")
    args = parser.parse_args()

    # ── Setup log file ─────────────────────────────────────
    global _log_file_handle
    if args.log:
        _log_file_handle = open(args.log, "a")

    log("=" * 100)
    log("  TOKENIZER EVALUATION")
    log(f"  Tokenizer: {args.tokenizer_name} ({args.tokenizer_type}: {args.tokenizer_path})")
    log(f"  Data:      {DATA_DIR}")
    log(f"  Output:    {EVAL_BASE}/{args.tokenizer_name}/")
    log(f"  Workers:   {args.workers}")
    log("=" * 100)
    log("")

    # ── Print enabled domains ──────────────────────────────
    enabled = [d for d, v in EVAL_DOMAINS.items() if v]
    disabled = [d for d, v in EVAL_DOMAINS.items() if not v]
    log(f"  Enabled domains:  {', '.join(enabled)}")
    log(f"  Disabled domains: {', '.join(disabled)}")
    log("")

    # ── Load tokenizer ─────────────────────────────────────
    log("Loading tokenizer...")
    t0 = time.time()
    tokenizer, vocab_size, effective_type = load_tokenizer(
        args.tokenizer_name, args.tokenizer_type, args.tokenizer_path
    )
    elapsed = time.time() - t0
    log(f"  Loaded: vocab_size={vocab_size:,}, type={effective_type}, time={elapsed:.1f}s")

    # ── Create encoder function ────────────────────────────
    encode_fn = create_encoder(tokenizer, effective_type)
    log(f"  Encoder ready (backend: "
        f"{'Rust encode_batch' if getattr(tokenizer, 'backend_tokenizer', None) else 'sequential'})")
    log("")

    # ── Build task list ────────────────────────────────────
    tasks = []
    total_shards = 0

    for domain, sources in ALL_SOURCES.items():
        if not EVAL_DOMAINS.get(domain, False):
            continue
        for src in sources:
            shard_dir = os.path.join(DATA_DIR, src["subfolder"])
            shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
            n = len(shard_files)
            total_shards += n
            tasks.append(src)
            log(f"  {domain}/{src['name']:<30s}  {n:>5d} shards")

    log(f"\n  Total: {len(tasks)} sources, {total_shards:,} shards")
    log("")

    # ── Launch threads with tqdm ───────────────────────────
    log("Starting evaluation...\n")
    results = []
    t0_global = time.time()

    pbar = tqdm(
        total=total_shards,
        desc="Evaluating",
        unit="shard",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_source, encode_fn, src, args.tokenizer_name, pbar): src
            for src in tasks
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                domain = src["subfolder"].split("/")[0]
                log(f"  ❌ {domain}/{src['name']}: THREAD ERROR — {e}")
                results.append({
                    "name": f"{domain}/{src['name']}",
                    "status": "FAIL",
                    "shards_total": 0,
                    "shards_evaluated": 0,
                    "shards_skipped": 0,
                    "total_tokens": 0,
                    "error": str(e),
                })

    pbar.close()
    total_elapsed = time.time() - t0_global

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    log("")
    log("=" * 100)
    log("  EVALUATION SUMMARY")
    log("=" * 100)

    ok = [r for r in results if r["status"] == "OK"]
    skipped = [r for r in results if r["status"] == "SKIPPED"]
    failed = [r for r in results if r["status"] == "FAIL"]
    no_data = [r for r in results if r["status"] == "NO_DATA"]

    total_evaluated = sum(r.get("shards_evaluated", 0) for r in results)
    total_skipped_shards = sum(r.get("shards_skipped", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    log(f"\n  Tokenizer:        {args.tokenizer_name} (vocab={vocab_size:,})")
    log(f"  ✅ Completed:      {len(ok)} sources")
    log(f"  ⏭️  Fully skipped:  {len(skipped)} sources (all shards already done)")
    log(f"  ❌ Failed:         {len(failed)} sources")
    if no_data:
        log(f"  ⚠️  No data:       {len(no_data)} sources")
    log(f"\n  Shards evaluated:  {total_evaluated:,} (new)")
    log(f"  Shards skipped:    {total_skipped_shards:,} (resumed)")
    log(f"  Total tokens:      {total_tokens/1e9:.2f}B")
    log(f"  Wall time:         {total_elapsed/3600:.1f}h ({total_elapsed/60:.0f}m)")

    # Per-domain breakdown
    log(f"\n  Per-domain breakdown:")
    log(f"  {'Domain':<15s}  {'Sources':>8s}  {'Shards':>8s}  {'Tokens':>12s}")
    log(f"  {'─'*50}")
    domain_stats = {}
    for r in results:
        domain = r["name"].split("/")[0]
        if domain not in domain_stats:
            domain_stats[domain] = {"sources": 0, "shards": 0, "tokens": 0}
        domain_stats[domain]["sources"] += 1
        domain_stats[domain]["shards"] += r.get("shards_evaluated", 0) + r.get("shards_skipped", 0)
        domain_stats[domain]["tokens"] += r.get("total_tokens", 0)
    for domain in ["English", "Math", "Code", "Scientific", "Multilingual"]:
        if domain in domain_stats:
            ds = domain_stats[domain]
            log(f"  {domain:<15s}  {ds['sources']:>8d}  {ds['shards']:>8d}  "
                f"{ds['tokens']/1e9:>10.2f}B")

    # Per-source detail
    log(f"\n  Per-source detail:")
    for r in sorted(results, key=lambda x: x["name"]):
        icon = {"OK": "✅", "SKIPPED": "⏭️ ", "FAIL": "❌", "NO_DATA": "⚠️ "}.get(
            r["status"], "?")
        tok_b = r.get("total_tokens", 0) / 1e9
        ev = r.get("shards_evaluated", 0)
        sk = r.get("shards_skipped", 0)
        log(f"    {icon} {r['name']:<40s}  {tok_b:>8.2f}B  "
            f"eval={ev:>5d}  skip={sk:>5d}  [{r['status']}]")

    if failed:
        log(f"\n  ─── FAILURES ───")
        for r in failed:
            log(f"  ❌ {r['name']}: {r.get('error', '?')}")

    log("")
    log("=" * 100)
    output_path = os.path.join(EVAL_BASE, args.tokenizer_name)
    log(f"  Results at: {output_path}")
    if failed:
        log(f"  ⚠️  {len(failed)} source(s) failed. Fix and re-run (resume-safe).")
    else:
        log("  ✅ All sources evaluated successfully!")
    log("=" * 100)

    if _log_file_handle:
        _log_file_handle.close()


if __name__ == "__main__":
    main()