#!/usr/bin/env python3
"""
verify_all_sources.py  —  Smoke-test every data source in the 500B curation spec.

For each source this script:
  1. Expands the glob pattern and counts matching files.
  2. Opens the FIRST file in the correct format (parquet / arrow-ipc-stream /
     jsonl.zst / jsonl.gz / json.gz / txt.xz).
  3. Reads a small batch and checks for the expected text column / JSON key.
  4. Extracts a sample text snippet (first 200 chars) to prove readability.
  5. For MathPile/stackexchange, verifies the special key layout (question + answers).
  6. Reports OK / FAIL / WARN per source.

Parallelism: 64 workers via ProcessPoolExecutor.

Usage:
    python verify_all_sources.py 2>&1 | tee verify_sources.log
"""

import glob
import gzip
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
OUT_DIR = f"{BASE}/smoke_test_data"
MAX_WORKERS = 64

# How many files to sample beyond the first (for multi-file sources)
EXTRA_FILES_TO_SAMPLE = 2  # check first + 2 more spread across the list


# ──────────────────────────────────────────────────────────────
# Source registry  —  one entry per logical data source
# ──────────────────────────────────────────────────────────────
@dataclass
class Source:
    """Describes one data source to verify."""
    name: str                       # human-readable label
    domain: str                     # English / Math / Code / Scientific / Multilingual
    output_folder: str              # subfolder under OUT_DIR (e.g. "English/wikipedia")
    glob_pattern: str               # file glob (relative patterns are joined to BASE)
    fmt: str                        # parquet | arrow_stream | jsonl_zst | jsonl_gz | json_gz | txt_xz
    text_col: Optional[str] = None  # column name for parquet / arrow
    text_key: Optional[str] = None  # JSON key for jsonl / json
    special: Optional[str] = None   # "stackexchange" for MathPile SE
    notes: str = ""


def build_source_registry() -> list[Source]:
    """Build the complete list of sources from the spec."""
    sources = []

    # ── English ────────────────────────────────────────────
    sources.append(Source(
        name="English/wikipedia",
        domain="English",
        output_folder="English/wikipedia",
        glob_pattern=f"{BASE}/English/wikipedia/*.parquet",
        fmt="parquet",
        text_col="content",
    ))
    sources.append(Source(
        name="English/dclm",
        domain="English",
        output_folder="English/dclm",
        glob_pattern=f"{BASE}/English/dclm_baseline_4T/global-shard_02_of_10/local-shard_*_of_10/*.jsonl.zst",
        fmt="jsonl_zst",
        text_key="text",
        notes="All local shards 0-9 within global-shard_02",
    ))
    sources.append(Source(
        name="English/fineweb_edu",
        domain="English",
        output_folder="English/fineweb_edu",
        # CRITICAL: explicit CC-MAIN-* glob, NOT recursive **
        glob_pattern=f"{BASE}/English/fineweb_edu/data/CC-MAIN-*/*.parquet",
        fmt="parquet",
        text_col="text",
        notes="Explicit glob — no recursive ** on gcsfuse",
    ))

    # ── Math ───────────────────────────────────────────────
    mathpile_base = f"{BASE}/math/MathPile/train"
    for subsrc in ["arXiv", "commoncrawl", "textbooks", "wikipedia"]:
        sources.append(Source(
            name=f"Math/MathPile/{subsrc}",
            domain="Math",
            output_folder="Math/MathPile",
            glob_pattern=f"{mathpile_base}/{subsrc}/*.jsonl.gz",
            fmt="jsonl_gz",
            text_key="text",
        ))
    sources.append(Source(
        name="Math/MathPile/stackexchange",
        domain="Math",
        output_folder="Math/MathPile",
        glob_pattern=f"{mathpile_base}/stackexchange/*.jsonl.gz",
        fmt="jsonl_gz",
        text_key=None,  # special handling
        special="stackexchange",
        notes="No 'text' key — must concat question + answers[i]['Body']",
    ))
    sources.append(Source(
        name="Math/finemath4+",
        domain="Math",
        output_folder="Math/finemath4+",
        glob_pattern=f"{BASE}/math/finemath4+/finemath-4plus/*.parquet",
        fmt="parquet",
        text_col="text",
    ))
    sources.append(Source(
        name="Math/finemath3+",
        domain="Math",
        output_folder="Math/finemath3+",
        glob_pattern=f"{BASE}/math/finemath3+/finemath-3plus/*.parquet",
        fmt="parquet",
        text_col="text",
        notes="Explicit glob — no recursive ** on gcsfuse",
    ))

    # ── Code ───────────────────────────────────────────────
    # (language, starcoderdata_dir, bigcode_dir, stackedu_dir)
    # None means source doesn't exist for that language
    code_langs = [
        ("Python",     "python",     "Python",     "Python"),
        ("Cpp",        "cpp",        "Cpp",        "Cpp"),
        ("JavaScript", "javascript", "JavaScript", "JavaScript"),
        ("Java",       None,         None,         "Java"),
        ("HTML",       "html",       "html",       None),
        ("Fortran",    "fortran",    "fortran",    None),
        ("Rust",       "rust",       None,         None),
        ("C",          None,         None,         "C"),
        ("TypeScript", None,         None,         "TypeScript"),
        ("SQL",        None,         None,         "SQL"),
        ("Shell",      None,         None,         "Shell"),
    ]
    for lang, sc_dir, bc_dir, se_dir in code_langs:
        if sc_dir:
            sources.append(Source(
                name=f"Code/{lang}/starcoderdata",
                domain="Code",
                output_folder=f"Code/{lang}",
                glob_pattern=f"{BASE}/code/starcoderdata/{sc_dir}/*.parquet",
                fmt="parquet",
                text_col="content",
            ))
        if bc_dir:
            sources.append(Source(
                name=f"Code/{lang}/BIGCODE",
                domain="Code",
                output_folder=f"Code/{lang}",
                glob_pattern=f"{BASE}/Tokenizer_Data/BIGCODE/{bc_dir}/*.arrow",
                fmt="arrow_stream",
                text_col="content",
                notes="IPC stream format — must use pa.ipc.open_stream()",
            ))
        if se_dir:
            sources.append(Source(
                name=f"Code/{lang}/stack_edu",
                domain="Code",
                output_folder=f"Code/{lang}",
                glob_pattern=f"{BASE}/code/stack_edu/{se_dir}/*.parquet",
                fmt="parquet",
                text_col="text",
            ))

    # ── Scientific ─────────────────────────────────────────
    sources.append(Source(
        name="Scientific/peS2o",
        domain="Scientific",
        output_folder="Scientific/peS2o",
        glob_pattern=f"{BASE}/Scientific/peS2o/data/v2/*.json.gz",
        fmt="json_gz",
        text_key="text",
    ))
    sources.append(Source(
        name="Scientific/arxiv",
        domain="Scientific",
        output_folder="Scientific/arxiv",
        glob_pattern=f"{BASE}/Tokenizer_data/final_arxiv_arnav/*.jsonl.zst",
        fmt="jsonl_zst",
        text_key="text",
    ))

    # ── Multilingual / Sangraha (Indic) ────────────────────
    sangraha_base = f"{BASE}/Multilingual_data/sangraha/unverified/unverified"
    sangraha_langs = [
        ("Hindi", "hin"), ("Bengali", "ben"), ("Urdu", "urd"),
        ("Tamil", "tam"), ("Marathi", "mar"), ("Gujarati", "guj"),
        ("Malayalam", "mal"), ("Kannada", "kan"), ("Punjabi", "pan"),
        ("Odia", "ori"),
    ]
    for lang_name, code in sangraha_langs:
        sources.append(Source(
            name=f"Multilingual/{lang_name}/sangraha",
            domain="Multilingual",
            output_folder=f"Multilingual/{lang_name}",
            glob_pattern=f"{sangraha_base}/{code}/*.parquet",
            fmt="parquet",
            text_col="text",
        ))

    # ── Multilingual / CC100 (foreign + neighbour) ─────────
    cc100_base = f"{BASE}/Multilingual_data/cc100"
    cc100_langs = [
        ("Russian",  "rus", "ru.txt.xz"),
        ("French",   "fra", "fr.txt.xz"),
        ("German",   "deu", "de.txt.xz"),
        ("Mandarin", "cmn", "zh-Hans.txt.xz"),
        ("Japanese", "jpn", "ja.txt.xz"),
        ("Korean",   "kor", "ko.txt.xz"),
        ("Farsi",    "fas", "fa.txt.xz"),
        ("Nepali",   "npi", "ne.txt.xz"),
        ("Sinhala",  "sin", "si.txt.xz"),
        ("Pashto",   "pus", "ps.txt.xz"),
        ("Burmese",  "mya", "my.txt.xz"),
    ]
    for lang_name, subfolder, filename in cc100_langs:
        sources.append(Source(
            name=f"Multilingual/{lang_name}/cc100",
            domain="Multilingual",
            output_folder=f"Multilingual/{lang_name}",
            # Single file — exact path, not a glob
            glob_pattern=f"{cc100_base}/{subfolder}/{filename}",
            fmt="txt_xz",
            notes="One sentence per line; decompress via subprocess xz -dc",
        ))

    return sources


# ──────────────────────────────────────────────────────────────
# Per-format readers  —  each returns (columns_or_keys, sample_text, row_count_in_batch)
# ──────────────────────────────────────────────────────────────

def _read_parquet_sample(filepath: str, text_col: str):
    """Read a small batch from a parquet file."""
    pf = pq.ParquetFile(filepath)
    schema_names = pf.schema_arrow.names
    # Read first row group, but only 5 rows
    batch = next(pf.iter_batches(batch_size=5, columns=[text_col] if text_col in schema_names else None))
    table = pa.table({col: batch.column(col) for col in batch.schema.names})
    n = len(table)
    if text_col in schema_names:
        sample = table[text_col][0].as_py()
    else:
        sample = None
    return schema_names, sample, n, pf.metadata.num_rows


def _read_arrow_stream_sample(filepath: str, text_col: str):
    """Read a small batch from an Arrow IPC stream file."""
    with open(filepath, "rb") as f:
        reader = ipc.open_stream(f)
        schema_names = reader.schema.names
        batch = reader.read_next_batch()
        n = len(batch)
        if text_col in schema_names:
            sample = batch.column(text_col)[0].as_py()
        else:
            sample = None
    return schema_names, sample, n, None  # total rows unknown without reading all


def _read_jsonl_zst_sample(filepath: str, text_key: str | None, special: str | None):
    """Read first few lines from a zstandard-compressed JSONL file."""
    if not HAS_ZSTD:
        raise ImportError("zstandard package not installed — pip install zstandard")
    dctx = zstd.ZstdDecompressor()
    keys_found = set()
    sample = None
    n = 0
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while n < 5:
                chunk = reader.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf and n < 5:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    keys_found.update(obj.keys())
                    if sample is None:
                        if special == "stackexchange":
                            sample = _extract_stackexchange(obj)
                        elif text_key and text_key in obj:
                            sample = obj[text_key]
                    n += 1
    return sorted(keys_found), sample, n, None


def _read_jsonl_gz_sample(filepath: str, text_key: str | None, special: str | None):
    """Read first few lines from a gzip-compressed JSONL file."""
    keys_found = set()
    sample = None
    n = 0
    with gzip.open(filepath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys_found.update(obj.keys())
            if sample is None:
                if special == "stackexchange":
                    sample = _extract_stackexchange(obj)
                elif text_key and text_key in obj:
                    sample = obj[text_key]
            n += 1
            if n >= 5:
                break
    return sorted(keys_found), sample, n, None


def _read_json_gz_sample(filepath: str, text_key: str):
    """Read first few lines from a gzip-compressed JSON-lines file (peS2o)."""
    # Same as jsonl_gz in practice
    return _read_jsonl_gz_sample(filepath, text_key, None)


def _read_txt_xz_sample(filepath: str):
    """Read first ~20 lines from an xz-compressed text file via subprocess."""
    proc = subprocess.Popen(
        ["xz", "-dc", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = []
    try:
        for raw_line in proc.stdout:
            lines.append(raw_line.decode("utf-8", errors="replace").rstrip("\n"))
            if len(lines) >= 20:
                break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    sample = "\n".join(lines[:20]) if lines else None
    return ["(plain text lines)"], sample, len(lines), None


def _extract_stackexchange(obj: dict) -> str:
    """
    MathPile stackexchange: concatenate question + answers.
    Expected keys: 'question', 'answers' (list of dicts with 'Body').
    """
    parts = []
    # Try common key names for the question body
    for qkey in ["question", "Question", "title", "Title", "Body", "body"]:
        if qkey in obj and isinstance(obj[qkey], str):
            parts.append(obj[qkey])
            break
    # answers
    answers = obj.get("answers", obj.get("Answers", []))
    if isinstance(answers, list):
        for ans in answers[:3]:  # sample first 3 answers
            if isinstance(ans, dict):
                body = ans.get("Body", ans.get("body", ans.get("text", "")))
                if body:
                    parts.append(str(body))
    return "\n\n".join(parts) if parts else f"[RAW KEYS: {sorted(obj.keys())}]"


# ──────────────────────────────────────────────────────────────
# Main verification worker
# ──────────────────────────────────────────────────────────────

def verify_source(src: Source) -> dict:
    """
    Verify a single source. Returns a result dict with status and details.
    """
    result = {
        "name": src.name,
        "domain": src.domain,
        "output_folder": src.output_folder,
        "status": "UNKNOWN",
        "file_count": 0,
        "columns_or_keys": [],
        "sample_text_snippet": "",
        "total_rows_first_file": None,
        "errors": [],
        "warnings": [],
        "notes": src.notes,
    }

    try:
        # ── Step 1: Glob / file existence ──────────────────
        if src.fmt == "txt_xz":
            # Single file, not a glob
            if os.path.isfile(src.glob_pattern):
                files = [src.glob_pattern]
            else:
                files = []
        else:
            files = sorted(glob.glob(src.glob_pattern))

        result["file_count"] = len(files)

        if len(files) == 0:
            result["status"] = "FAIL"
            result["errors"].append(f"No files matched glob: {src.glob_pattern}")
            # Check if parent dir exists
            parent = os.path.dirname(src.glob_pattern.split("*")[0].rstrip("/"))
            if not os.path.isdir(parent):
                result["errors"].append(f"Parent directory does not exist: {parent}")
            else:
                result["errors"].append(f"Parent exists but is empty or glob wrong. Contents: {os.listdir(parent)[:10]}")
            return result

        # ── Step 2: Read first file ────────────────────────
        first_file = files[0]
        result["first_file"] = first_file

        if src.fmt == "parquet":
            cols, sample, n, total_rows = _read_parquet_sample(first_file, src.text_col)
            result["columns_or_keys"] = list(cols)
            result["total_rows_first_file"] = total_rows
            if src.text_col not in cols:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Expected column '{src.text_col}' not found. Available: {cols}"
                )
                return result

        elif src.fmt == "arrow_stream":
            cols, sample, n, total_rows = _read_arrow_stream_sample(first_file, src.text_col)
            result["columns_or_keys"] = list(cols)
            if src.text_col not in cols:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Expected column '{src.text_col}' not found. Available: {cols}"
                )
                return result

        elif src.fmt == "jsonl_zst":
            cols, sample, n, total_rows = _read_jsonl_zst_sample(
                first_file, src.text_key, src.special
            )
            result["columns_or_keys"] = cols
            if src.text_key and src.text_key not in cols and src.special is None:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Expected key '{src.text_key}' not found. Keys: {cols}"
                )
                return result

        elif src.fmt in ("jsonl_gz", "json_gz"):
            cols, sample, n, total_rows = _read_jsonl_gz_sample(
                first_file, src.text_key, src.special
            )
            result["columns_or_keys"] = cols
            if src.text_key and src.text_key not in cols and src.special is None:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Expected key '{src.text_key}' not found. Keys: {cols}"
                )
                return result

        elif src.fmt == "txt_xz":
            cols, sample, n, total_rows = _read_txt_xz_sample(first_file)
            result["columns_or_keys"] = cols

        else:
            result["status"] = "FAIL"
            result["errors"].append(f"Unknown format: {src.fmt}")
            return result

        # ── Step 3: Validate sample text ───────────────────
        if sample is None or (isinstance(sample, str) and len(sample.strip()) == 0):
            result["status"] = "WARN"
            result["warnings"].append("First record has empty/null text")
            result["sample_text_snippet"] = "(empty)"
        else:
            snippet = str(sample)[:300].replace("\n", "\\n")
            result["sample_text_snippet"] = snippet

        # ── Step 4: Spot-check additional files (spread) ──
        if len(files) > 1:
            # Pick files spread across the list
            indices = [len(files) // 3, 2 * len(files) // 3]
            for idx in indices:
                if idx >= len(files) or idx == 0:
                    continue
                extra_file = files[idx]
                try:
                    if src.fmt == "parquet":
                        pf = pq.ParquetFile(extra_file)
                        batch = next(pf.iter_batches(batch_size=1))
                    elif src.fmt == "arrow_stream":
                        with open(extra_file, "rb") as f:
                            reader = ipc.open_stream(f)
                            _ = reader.read_next_batch()
                    elif src.fmt == "jsonl_zst":
                        dctx = zstd.ZstdDecompressor()
                        with open(extra_file, "rb") as fh:
                            with dctx.stream_reader(fh) as reader:
                                chunk = reader.read(4096)
                                if chunk:
                                    _ = json.loads(chunk.split(b"\n")[0])
                    elif src.fmt in ("jsonl_gz", "json_gz"):
                        with gzip.open(extra_file, "rt") as fh:
                            _ = json.loads(fh.readline())
                except Exception as e:
                    result["warnings"].append(
                        f"Spot-check failed on file #{idx} ({os.path.basename(extra_file)}): {e}"
                    )

        # ── Step 5: Output dir writability ─────────────────
        out_path = os.path.join(OUT_DIR, src.output_folder)
        try:
            os.makedirs(out_path, exist_ok=True)
            test_file = os.path.join(out_path, ".write_test")
            with open(test_file, "w") as tf:
                tf.write("ok")
            os.remove(test_file)
        except Exception as e:
            result["warnings"].append(f"Output dir not writable: {out_path} — {e}")

        # ── Final status ───────────────────────────────────
        if result["errors"]:
            result["status"] = "FAIL"
        elif result["warnings"]:
            result["status"] = "WARN"
        else:
            result["status"] = "OK"

    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Unhandled exception: {e}\n{traceback.format_exc()}")

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  500B TOKENIZER EVAL DATA — SOURCE VERIFICATION")
    print(f"  Base path:  {BASE}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Workers:    {MAX_WORKERS}")
    print("=" * 80)
    print()

    # Check prerequisites
    if not os.path.isdir(BASE):
        print(f"FATAL: Base path does not exist: {BASE}")
        sys.exit(1)
    if not HAS_ZSTD:
        print("FATAL: 'zstandard' package not installed. Run: pip install zstandard")
        sys.exit(1)

    sources = build_source_registry()
    print(f"Registered {len(sources)} sources to verify.\n")

    # Run in parallel
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(verify_source, src): src for src in sources}
        for future in as_completed(futures):
            src = futures[future]
            try:
                res = future.result()
                results.append(res)
                status_icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(res["status"], "?")
                print(f"  {status_icon} {res['name']:45s}  files={res['file_count']:>6d}  {res['status']}")
                if res["errors"]:
                    for err in res["errors"]:
                        print(f"       ERROR: {err}")
                if res["warnings"]:
                    for w in res["warnings"]:
                        print(f"       WARN:  {w}")
            except Exception as e:
                print(f"  ❌ {src.name:45s}  EXECUTOR ERROR: {e}")
                results.append({"name": src.name, "status": "FAIL", "errors": [str(e)]})

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    ok = [r for r in results if r["status"] == "OK"]
    warn = [r for r in results if r["status"] == "WARN"]
    fail = [r for r in results if r["status"] == "FAIL"]

    print(f"  ✅ OK:   {len(ok)}")
    print(f"  ⚠️  WARN: {len(warn)}")
    print(f"  ❌ FAIL: {len(fail)}")
    print(f"  Total:   {len(results)} sources verified in {elapsed:.1f}s")
    print()

    # Group by domain for token availability overview
    domain_files = {}
    for r in results:
        d = r.get("domain", "?")
        domain_files.setdefault(d, 0)
        domain_files[d] += r.get("file_count", 0)
    print("  Files per domain:")
    for domain, count in sorted(domain_files.items()):
        print(f"    {domain:15s}: {count:>8,d} files")
    print()

    # Print failures in detail
    if fail:
        print("─" * 80)
        print("  FAILURES (need fixing before curation):")
        print("─" * 80)
        for r in sorted(fail, key=lambda x: x["name"]):
            print(f"\n  ❌ {r['name']}")
            for e in r.get("errors", []):
                print(f"     → {e}")
        print()

    # Print sample texts for OK sources
    print("─" * 80)
    print("  SAMPLE TEXTS (first 300 chars per source):")
    print("─" * 80)
    for r in sorted(results, key=lambda x: x["name"]):
        if r["status"] in ("OK", "WARN"):
            snippet = r.get("sample_text_snippet", "")
            cols = r.get("columns_or_keys", [])
            total = r.get("total_rows_first_file", "?")
            print(f"\n  [{r['status']}] {r['name']}")
            print(f"       files={r.get('file_count', '?')}, cols/keys={cols}, rows_in_first_file={total}")
            print(f"       text: {snippet[:200]}")

    print()
    print("=" * 80)
    if fail:
        print(f"  ⚠️  {len(fail)} source(s) FAILED — fix before running curation.")
    else:
        print("  ✅ All sources verified — ready for curation.")
    print("=" * 80)


if __name__ == "__main__":
    main()