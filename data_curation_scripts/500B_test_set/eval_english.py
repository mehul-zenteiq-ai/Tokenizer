#!/usr/bin/env python3
"""
eval_english.py — Evaluate a tokenizer on English data only.

Evaluates 44 (wikipedia) + 500 (dclm) + 500 (fineweb_edu) = 1,044 shards.
Uses shard-level parallelism: every shard is an independent task in the thread pool.

Resume: each completed shard writes a small JSON result file. On restart,
shards with existing result files are skipped. Final assembly into master.csv
happens only after all shards are done.

Usage:
    python eval_english.py \\
        --tokenizer-name superbpe_dedup \\
        --tokenizer-type auto \\
        --tokenizer-path /path/to/tokenizer.json \\
        --workers 128 \\
        --log eval_english.log
"""

import argparse
import csv
import glob
import json
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


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
DATA_DIR = f"{BASE}/final_tokenizer_test_data"
EVAL_BASE = f"{BASE}/final_tokenizer_test_data_eval/tokenizer_results"

ENCODE_BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 1_000_000_000   # 1B tokens

# English shard limits
WIKI_MAX_SHARDS     = None   # None = take all (44 shards)
DCLM_MAX_SHARDS     = 500
FINEWEB_MAX_SHARDS  = 500

# Source definitions
ENGLISH_SOURCES = [
    {
        "name": "wikipedia",
        "subfolder": "English/wikipedia",
        "max_shards": WIKI_MAX_SHARDS,
    },
    {
        "name": "dclm",
        "subfolder": "English/dclm",
        "max_shards": DCLM_MAX_SHARDS,
    },
    {
        "name": "fineweb_edu",
        "subfolder": "English/fineweb_edu",
        "max_shards": FINEWEB_MAX_SHARDS,
    },
]

CSV_FIELDS = [
    "shard_file", "num_docs", "num_words", "num_chars", "num_bytes",
    "num_tokens", "fertility", "nsl", "bytes_per_token", "cumulative_tokens",
]


# ═══════════════════════════════════════════════════════════════
# Thread-safe logging
# ═══════════════════════════════════════════════════════════════

_log_lock = threading.Lock()
_log_file_handle = None


def log(msg: str):
    tqdm.write(msg)
    if _log_file_handle:
        with _log_lock:
            _log_file_handle.write(msg + "\n")
            _log_file_handle.flush()


# ═══════════════════════════════════════════════════════════════
# Tokenizer loading
# ═══════════════════════════════════════════════════════════════

def _is_valid_tokenizers_json(json_path: str) -> bool:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            head = fh.read(4096)
        return '"model"' in head and '"added_tokens"' in head
    except Exception:
        return False


def load_tokenizer(tok_type: str, path: str):
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
        if _is_valid_tokenizers_json(path):
            try:
                tok = AutoTokenizer.from_pretrained(parent_dir, trust_remote_code=True)
                return tok, getattr(tok, "vocab_size", len(tok.get_vocab())), "hf"
            except Exception:
                pass
        try:
            tok = PreTrainedTokenizerFast(tokenizer_file=path)
            return tok, tok.vocab_size or len(tok.get_vocab()), "hf"
        except Exception:
            pass
        from tokenizers import Tokenizer as _RawTokenizer
        raw = _RawTokenizer.from_file(path)
        tok = PreTrainedTokenizerFast(tokenizer_object=raw)
        return tok, tok.vocab_size or len(tok.get_vocab()), "hf"

    elif tok_type == "hf_fast":
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        return tok, getattr(tok, "vocab_size", len(tok.get_vocab())), "hf"

    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")


def create_encoder(tokenizer, tok_type: str):
    if tok_type == "tiktoken":
        def encode_fn(texts):
            return [len(ids) for ids in tokenizer.encode_ordinary_batch(texts)]
        return encode_fn

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        def encode_fn(texts):
            encodings = backend.encode_batch(texts, add_special_tokens=False)
            return [len(enc.ids) for enc in encodings]
        return encode_fn

    log("  ⚠️  No fast backend — falling back to sequential encode (slow)")
    def encode_fn(texts):
        return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return encode_fn


# ═══════════════════════════════════════════════════════════════
# Shard evaluation
# ═══════════════════════════════════════════════════════════════

def eval_shard(encode_fn, shard_path: str) -> dict:
    """Evaluate a single shard. English is never CJK, so word-based fertility."""
    pf = pq.ParquetFile(shard_path)

    total_docs = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    total_tokens = 0

    for batch in pf.iter_batches(batch_size=ENCODE_BATCH_SIZE, columns=["text"]):
        texts_raw = batch.column("text").to_pylist()
        texts = [t for t in texts_raw if t is not None and len(t) > 0]
        if not texts:
            continue

        token_counts = encode_fn(texts)

        for text, n_tok in zip(texts, token_counts):
            total_docs += 1
            total_tokens += n_tok
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            total_words += len(text.split())

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
# Temp result file helpers (for resume)
# ═══════════════════════════════════════════════════════════════

def temp_result_path(temp_dir: str, shard_basename: str) -> str:
    """Path to the per-shard temp result JSON file."""
    return os.path.join(temp_dir, shard_basename.replace(".parquet", ".json"))


def save_temp_result(path: str, result: dict):
    """Write shard result to a temp JSON file atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.rename(tmp, path)


def load_temp_result(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# Worker — called per shard from thread pool
# ═══════════════════════════════════════════════════════════════

def process_shard(args: tuple) -> dict:
    """
    Evaluate a single shard and write a temp result file.
    Returns the result dict (or loaded from temp if already done).
    """
    encode_fn, shard_path, source_name, temp_dir = args
    shard_basename = os.path.basename(shard_path)
    t_path = temp_result_path(temp_dir, shard_basename)

    # Resume: if temp result already exists, return it
    if os.path.isfile(t_path):
        try:
            return load_temp_result(t_path)
        except Exception:
            pass  # corrupt file, re-evaluate

    try:
        result = eval_shard(encode_fn, shard_path)
        result["source_name"] = source_name
        save_temp_result(t_path, result)
        return result
    except Exception as e:
        log(f"  ⚠️  {source_name}/{shard_basename}: ERROR — {e}")
        return {
            "shard_file": shard_basename,
            "source_name": source_name,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════
# Assembly: temp results → sorted master.csv + checkpoints
# ═══════════════════════════════════════════════════════════════

def assemble_csv(source_name: str, temp_dir: str, out_dir: str):
    """
    Read all temp JSON files for a source, sort by shard name,
    compute cumulative_tokens, write master.csv and checkpoint markers.
    """
    # Read all temp results
    results = []
    for fname in os.listdir(temp_dir):
        if fname.endswith(".json"):
            path = os.path.join(temp_dir, fname)
            try:
                r = load_temp_result(path)
                if "error" not in r:
                    results.append(r)
            except Exception:
                continue

    if not results:
        log(f"  ⚠️  {source_name}: no valid results to assemble")
        return 0, 0

    # Sort by shard filename (shard_00000 < shard_00001 < ...)
    results.sort(key=lambda r: r["shard_file"])

    # Compute cumulative tokens
    cumulative = 0
    for r in results:
        cumulative += r["num_tokens"]
        r["cumulative_tokens"] = cumulative

    # Write master.csv
    csv_path = os.path.join(out_dir, "master.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_FIELDS})

    # Write checkpoint markers
    last_checkpoint = 0
    for r in results:
        cum = r["cumulative_tokens"]
        current_cp = (cum // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
        if current_cp > last_checkpoint and current_cp > 0:
            marker = os.path.join(out_dir, f"checkpoint_{current_cp // 1_000_000_000}B.marker")
            with open(marker, "w") as f:
                f.write(f"cumulative_tokens={cum}\n")
            last_checkpoint = current_cp

    return len(results), cumulative


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer on English data")
    parser.add_argument("--tokenizer-name", required=True, help="Name for output folder")
    parser.add_argument("--tokenizer-type", required=True,
                        choices=["hf", "auto", "fast", "tiktoken", "hf_fast"])
    parser.add_argument("--tokenizer-path", required=True, help="Path or HF model name")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--log", default=None, help="Log file path")
    args = parser.parse_args()

    # ── Log file ───────────────────────────────────────────
    global _log_file_handle
    if args.log:
        _log_file_handle = open(args.log, "a")

    log("=" * 100)
    log("  ENGLISH TOKENIZER EVALUATION")
    log(f"  Tokenizer: {args.tokenizer_name} ({args.tokenizer_type}: {args.tokenizer_path})")
    log(f"  Data:      {DATA_DIR}")
    log(f"  Output:    {EVAL_BASE}/{args.tokenizer_name}/English/")
    log(f"  Workers:   {args.workers}")
    log(f"  Shards:    wiki=all, dclm=first {DCLM_MAX_SHARDS}, fineweb=first {FINEWEB_MAX_SHARDS}")
    log("=" * 100)
    log("")

    # ── Load tokenizer ─────────────────────────────────────
    log("Loading tokenizer...")
    t0 = time.time()
    tokenizer, vocab_size, effective_type = load_tokenizer(args.tokenizer_type, args.tokenizer_path)
    elapsed = time.time() - t0
    log(f"  Loaded: vocab_size={vocab_size:,}, type={effective_type}, time={elapsed:.1f}s")

    encode_fn = create_encoder(tokenizer, effective_type)
    backend_name = "Rust encode_batch" if getattr(tokenizer, "backend_tokenizer", None) else "sequential"
    log(f"  Encoder: {backend_name}")
    log("")

    # ── Build task list ────────────────────────────────────
    all_tasks = []
    source_info = {}  # source_name → {out_dir, temp_dir, total_shards}

    for src in ENGLISH_SOURCES:
        name = src["name"]
        subfolder = src["subfolder"]
        max_shards = src["max_shards"]

        shard_dir = os.path.join(DATA_DIR, subfolder)
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))

        if max_shards is not None:
            shard_files = shard_files[:max_shards]

        # Output and temp directories
        out_dir = os.path.join(EVAL_BASE, args.tokenizer_name, "English", name)
        temp_dir = os.path.join(out_dir, "_temp")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        source_info[name] = {
            "out_dir": out_dir,
            "temp_dir": temp_dir,
            "total_shards": len(shard_files),
        }

        # Count already-done shards (for resume display)
        already_done = sum(
            1 for sf in shard_files
            if os.path.isfile(temp_result_path(temp_dir, os.path.basename(sf)))
        )

        log(f"  {name:<15s}  {len(shard_files):>5d} shards  "
            f"({already_done} already done, {len(shard_files) - already_done} remaining)")

        for shard_path in shard_files:
            all_tasks.append((encode_fn, shard_path, name, temp_dir))

    total_shards = len(all_tasks)

    # Count total already done across all sources
    total_already_done = sum(
        1 for task in all_tasks
        if os.path.isfile(temp_result_path(task[3], os.path.basename(task[1])))
    )
    total_remaining = total_shards - total_already_done

    log(f"\n  Total: {total_shards:,} shards ({total_already_done:,} done, "
        f"{total_remaining:,} remaining)")
    log("")

    if total_remaining == 0:
        log("All shards already evaluated! Assembling CSVs...")
    else:
        log(f"Starting evaluation with {args.workers} threads...\n")

    # ── Run thread pool ────────────────────────────────────
    t0_global = time.time()
    completed = 0
    errors = 0

    # Note: already-done shards are still submitted but return instantly
    # (loading temp JSON file). The bar will jump quickly for those.
    pbar = tqdm(
        total=total_shards,
        desc="English eval",
        unit="shard",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_shard, task): task for task in all_tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                if "error" in result:
                    errors += 1
                else:
                    completed += 1
            except Exception as e:
                errors += 1
                log(f"  ❌ Thread error: {e}")

            pbar.update(1)

    pbar.close()
    eval_elapsed = time.time() - t0_global

    log(f"\n  Evaluation phase done: {completed:,} OK, {errors} errors, "
        f"{eval_elapsed/60:.1f}m wall time")

    # ── Assemble master CSVs ───────────────────────────────
    log("\nAssembling master.csv files...\n")

    grand_shards = 0
    grand_tokens = 0

    for src in ENGLISH_SOURCES:
        name = src["name"]
        info = source_info[name]

        n_shards, total_tokens = assemble_csv(name, info["temp_dir"], info["out_dir"])
        grand_shards += n_shards
        grand_tokens += total_tokens

        log(f"  ✅ English/{name}: {n_shards:,} shards, {total_tokens/1e9:.2f}B tokens → "
            f"{os.path.join(info['out_dir'], 'master.csv')}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    total_elapsed = time.time() - t0_global

    log("")
    log("=" * 100)
    log("  ENGLISH EVALUATION SUMMARY")
    log("=" * 100)
    log(f"\n  Tokenizer:    {args.tokenizer_name} (vocab={vocab_size:,})")
    log(f"  Total shards: {grand_shards:,}")
    log(f"  Total tokens: {grand_tokens/1e9:.2f}B")
    log(f"  Errors:       {errors}")
    log(f"  Wall time:    {total_elapsed/3600:.1f}h ({total_elapsed/60:.0f}m)")

    log(f"\n  Per-source:")
    log(f"  {'Source':<15s}  {'Shards':>8s}  {'Tokens':>12s}")
    log(f"  {'─'*40}")
    for src in ENGLISH_SOURCES:
        name = src["name"]
        info = source_info[name]
        csv_path = os.path.join(info["out_dir"], "master.csv")
        if os.path.isfile(csv_path):
            # Read final row to get cumulative tokens
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                n = len(rows)
                tok = int(rows[-1]["cumulative_tokens"]) if rows else 0
            log(f"  {name:<15s}  {n:>8d}  {tok/1e9:>10.2f}B")

    log(f"\n  Output: {EVAL_BASE}/{args.tokenizer_name}/English/")
    log("")
    log("=" * 100)
    if errors > 0:
        log(f"  ⚠️  {errors} shard(s) had errors. Re-run to retry (resume-safe).")
    else:
        log("  ✅ All English shards evaluated successfully!")
    log("=" * 100)

    if _log_file_handle:
        _log_file_handle.close()


if __name__ == "__main__":
    main()
