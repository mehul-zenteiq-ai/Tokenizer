#!/usr/bin/env python3
"""
curate_final.py — Curate ~557B token evaluation dataset for tokenizer benchmarking.

Architecture:
  - 36 output folders, one worker process each (via multiprocessing.Pool)
  - Each worker streams input files sequentially, writes 100M-token shards
  - Shards are written atomically (.parquet.tmp → rename)
  - Resume via .done marker files; partial runs resume from next shard index
  - Line-buffered stdout for live progress even when piped to tee

Output: /home/sushmetha_zenteiq_com/raw_storage_mount/final_tokenizer_test_data/

Usage:
    python curate_final.py 2>&1 | tee curation.log
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
from dataclasses import dataclass, field
from typing import Optional

# ── Force line-buffered stdout/stderr for live output through tee ──
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
OUT_DIR = f"{BASE}/final_tokenizer_test_data"
MAX_WORKERS = 64

TOKENS_PER_SHARD = 100_000_000        # 100M tokens per shard
PARQUET_BATCH_SIZE = 2000              # rows per parquet iter_batches call
CC100_WORDS_PER_ROW = 500             # group CC100 sentences into ~500-word docs
TOKEN_MULT = 1.35                     # Latin: words × 1.35
TOKEN_DIV_CJK = 1.5                   # CJK: chars / 1.5
LOG_INTERVAL_SECS = 30                # progress log every N seconds

# ── Quotas (from measured data) ────────────────────────────
# -1 = take all availablex
DCLM_QUOTA       = 100_000_000_000    # 100B
FINEWEB_QUOTA    = 196_000_000_000    # 196B (to fill 300B English total)
UNLIMITED        = -1


# ═══════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════

@dataclass
class InputStream:
    """One set of files from a single source dataset."""
    glob_pattern: str
    fmt: str               # parquet | arrow_stream | jsonl_zst | jsonl_gz | json_gz | txt_xz
    text_col: str = ""     # column name for parquet / arrow
    text_key: str = ""     # JSON key for jsonl / json
    special: str = ""      # "stackexchange" for MathPile SE
    is_single_file: bool = False


@dataclass
class Source:
    """One output folder pulling from one or more input streams."""
    name: str              # human label (e.g. "English/wikipedia")
    output_folder: str     # relative to OUT_DIR
    token_quota: int       # max tokens; -1 = unlimited
    is_cjk: bool           # True for Mandarin/Japanese
    inputs: list           # list[InputStream], read in order


# ═══════════════════════════════════════════════════════════════
# Source registry — VERIFIED paths only
# ═══════════════════════════════════════════════════════════════

def build_sources() -> list[Source]:
    S = []

    # ── English ────────────────────────────────────────────

    S.append(Source(
        name="English/wikipedia",
        output_folder="English/wikipedia",
        token_quota=UNLIMITED,  # take all ~4.35B
        is_cjk=False,
        inputs=[InputStream(
            glob_pattern=f"{BASE}/English/wikipedia/*.parquet",
            fmt="parquet", text_col="content",
        )],
    ))

    S.append(Source(
        name="English/dclm",
        output_folder="English/dclm",
        token_quota=DCLM_QUOTA,  # 100B
        is_cjk=False,
        inputs=[InputStream(
            # VERIFIED: local-shard naming has NO zero-padding (local-shard_0_of_10)
            # Glob local-shard_*_of_10 matches all 10 → 2790 files
            glob_pattern=f"{BASE}/English/dclm_baseline_4T/global-shard_02_of_10/local-shard_*_of_10/*.jsonl.zst",
            fmt="jsonl_zst", text_key="text",
        )],
    ))

    S.append(Source(
        name="English/fineweb_edu",
        output_folder="English/fineweb_edu",
        token_quota=FINEWEB_QUOTA,  # 196B
        is_cjk=False,
        inputs=[InputStream(
            # VERIFIED: explicit CC-MAIN-* glob, NOT recursive ** (gcsfuse bug)
            glob_pattern=f"{BASE}/English/fineweb_edu/data/CC-MAIN-*/*.parquet",
            fmt="parquet", text_col="text",
        )],
    ))

    # ── Math ───────────────────────────────────────────────

    mp_base = f"{BASE}/math/MathPile/train"
    S.append(Source(
        name="Math/MathPile",
        output_folder="Math/MathPile",
        token_quota=UNLIMITED,  # take all ~1.2B
        is_cjk=False,
        inputs=[
            InputStream(glob_pattern=f"{mp_base}/arXiv/*.jsonl.gz",       fmt="jsonl_gz", text_key="text"),
            InputStream(glob_pattern=f"{mp_base}/commoncrawl/*.jsonl.gz", fmt="jsonl_gz", text_key="text"),
            InputStream(glob_pattern=f"{mp_base}/textbooks/*.jsonl.gz",   fmt="jsonl_gz", text_key="text"),
            InputStream(glob_pattern=f"{mp_base}/wikipedia/*.jsonl.gz",   fmt="jsonl_gz", text_key="text"),
            # VERIFIED: question is a dict with "Body" key, answers is list of dicts with "Body"
            InputStream(glob_pattern=f"{mp_base}/stackexchange/*.jsonl.gz", fmt="jsonl_gz",
                        special="stackexchange"),
        ],
    ))

    S.append(Source(
        name="Math/finemath4+",
        output_folder="Math/finemath4+",
        token_quota=UNLIMITED,  # take all ~7.9B
        is_cjk=False,
        inputs=[InputStream(
            glob_pattern=f"{BASE}/math/finemath4+/finemath-4plus/*.parquet",
            fmt="parquet", text_col="text",
        )],
    ))

    S.append(Source(
        name="Math/finemath3+",
        output_folder="Math/finemath3+",
        token_quota=UNLIMITED,  # take all ~26.6B
        is_cjk=False,
        inputs=[InputStream(
            # VERIFIED: explicit glob, NOT recursive (gcsfuse bug)
            glob_pattern=f"{BASE}/math/finemath3+/finemath-3plus/*.parquet",
            fmt="parquet", text_col="text",
        )],
    ))

    # ── Code ───────────────────────────────────────────────
    # VERIFIED BIGCODE paths: all lowercase
    # VERIFIED empty (removed): java, rust, typescript, sql have 0 arrow files

    code_langs = [
        # (output, starcoderdata_dir, bigcode_dir, stackedu_dir)
        ("Python",     "python",     "python",     "Python"),
        ("Cpp",        "cpp",        "c++",        "Cpp"),
        ("JavaScript", "javascript", "javascript", "JavaScript"),
        ("Java",       None,         None,         "Java"),       # BIGCODE/java has 0 arrow files
        ("HTML",       "html",       "html",       None),
        ("Fortran",    "fortran",    "fortran",    None),
        ("Rust",       "rust",       None,         None),         # BIGCODE/rust has 0 arrow files
        ("C",          None,         "c",          "C"),
        ("TypeScript", None,         None,         "TypeScript"), # BIGCODE/typescript has 0 arrow files
        ("SQL",        None,         None,         "SQL"),        # BIGCODE/sql has 0 arrow files
        ("Shell",      None,         "shell",      "Shell"),
    ]

    for lang_out, sc_dir, bc_dir, se_dir in code_langs:
        inputs = []
        if sc_dir:
            inputs.append(InputStream(
                glob_pattern=f"{BASE}/code/starcoderdata/{sc_dir}/*.parquet",
                fmt="parquet", text_col="content",
            ))
        if bc_dir:
            inputs.append(InputStream(
                # VERIFIED: IPC stream format, must use pa.ipc.open_stream()
                glob_pattern=f"{BASE}/Tokenizer_Data/BIGCODE/{bc_dir}/*.arrow",
                fmt="arrow_stream", text_col="content",
            ))
        if se_dir:
            inputs.append(InputStream(
                glob_pattern=f"{BASE}/code/stack_edu/{se_dir}/*.parquet",
                fmt="parquet", text_col="text",
            ))
        S.append(Source(
            name=f"Code/{lang_out}",
            output_folder=f"Code/{lang_out}",
            token_quota=UNLIMITED,
            is_cjk=False,
            inputs=inputs,
        ))

    # ── Scientific ─────────────────────────────────────────

    S.append(Source(
        name="Scientific/peS2o",
        output_folder="Scientific/peS2o",
        token_quota=UNLIMITED,  # take all ~11.6B
        is_cjk=False,
        inputs=[InputStream(
            glob_pattern=f"{BASE}/Scientific/peS2o/data/v2/*.json.gz",
            fmt="json_gz", text_key="text",
        )],
    ))

    S.append(Source(
        name="Scientific/arxiv",
        output_folder="Scientific/arxiv",
        token_quota=UNLIMITED,  # take all ~8.4B
        is_cjk=False,
        inputs=[InputStream(
            glob_pattern=f"{BASE}/Tokenizer_data/final_arxiv_arnav/*.jsonl.zst",
            fmt="jsonl_zst", text_key="text",
        )],
    ))

    # ── Multilingual / Sangraha (Indic) — NO CAP ──────────
    sg = f"{BASE}/Multilingual_data/sangraha/unverified/unverified"
    # VERIFIED: flat structure, no nested subdirectories
    for lang_name, code in [
        ("Hindi", "hin"), ("Bengali", "ben"), ("Urdu", "urd"),
        ("Tamil", "tam"), ("Marathi", "mar"), ("Gujarati", "guj"),
        ("Malayalam", "mal"), ("Kannada", "kan"), ("Punjabi", "pan"),
        ("Odia", "ori"),
    ]:
        S.append(Source(
            name=f"Multilingual/{lang_name}",
            output_folder=f"Multilingual/{lang_name}",
            token_quota=UNLIMITED,  # no cap — take all
            is_cjk=False,
            inputs=[InputStream(
                glob_pattern=f"{sg}/{code}/*.parquet",
                fmt="parquet", text_col="text",
            )],
        ))

    # ── Multilingual / CC100 — NO CAP ─────────────────────
    cc = f"{BASE}/Multilingual_data/cc100"
    # VERIFIED: xz -dc streaming works
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
        S.append(Source(
            name=f"Multilingual/{lang_name}",
            output_folder=f"Multilingual/{lang_name}",
            token_quota=UNLIMITED,  # no cap — take all
            is_cjk=cjk,
            inputs=[InputStream(
                glob_pattern=f"{cc}/{subfolder}/{filename}",
                fmt="txt_xz", is_single_file=True,
            )],
        ))

    return S


# ═══════════════════════════════════════════════════════════════
# Token counting
# ═══════════════════════════════════════════════════════════════

def count_tokens(text: str, is_cjk: bool) -> int:
    if not text:
        return 0
    if is_cjk:
        return int(len(text) / TOKEN_DIV_CJK)
    return int(len(text.split()) * TOKEN_MULT)


# ═══════════════════════════════════════════════════════════════
# Text iterators — one per format
# ═══════════════════════════════════════════════════════════════

def iter_parquet_texts(filepath: str, text_col: str):
    """Yield text strings from parquet using batched reads."""
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=[text_col]):
        col = batch.column(text_col)
        for i in range(len(col)):
            val = col[i].as_py()
            if val:
                yield val


def iter_arrow_stream_texts(filepath: str, text_col: str):
    """Yield text strings from Arrow IPC stream file (BIGCODE)."""
    with open(filepath, "rb") as f:
        reader = ipc.open_stream(f)
        try:
            while True:
                batch = reader.read_next_batch()
                col = batch.column(text_col)
                for i in range(len(col)):
                    val = col[i].as_py()
                    if val:
                        yield val
        except StopIteration:
            pass


def iter_jsonl_zst_texts(filepath: str, text_key: str):
    """Yield text strings from zstandard-compressed JSONL."""
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
                        yield text
                except (json.JSONDecodeError, KeyError):
                    continue


def iter_jsonl_gz_texts(filepath: str, text_key: str):
    """Yield text strings from gzip-compressed JSONL."""
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError):
                continue


def iter_jsonl_gz_stackexchange(filepath: str):
    """
    Yield text from MathPile stackexchange JSONL.
    VERIFIED structure:
      - question: dict with "Body" (str), "Title" (str), etc.
      - answers: list of dicts, each with "Body" (str, capital B)
    Reconstruction: question["Body"] + "\\n\\n" + "\\n\\n".join(ans["Body"])
    """
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parts = []
                # Question body — VERIFIED: question is a dict, "Body" key
                question = obj.get("question", {})
                if isinstance(question, dict):
                    body = question.get("Body", "")
                    if body:
                        parts.append(body)
                # Answers — VERIFIED: list of dicts, each has "Body" (capital B)
                answers = obj.get("answers", [])
                if isinstance(answers, list):
                    for ans in answers:
                        if isinstance(ans, dict):
                            body = ans.get("Body", "")
                            if body:
                                parts.append(body)
                if parts:
                    yield "\n\n".join(parts)
            except (json.JSONDecodeError, KeyError):
                continue


def iter_json_gz_texts(filepath: str, text_key: str):
    """Yield text from gzip-compressed JSON-lines (peS2o)."""
    yield from iter_jsonl_gz_texts(filepath, text_key)


def iter_txt_xz_grouped(filepath: str, words_per_row: int = CC100_WORDS_PER_ROW):
    """
    Yield grouped text from xz-compressed text (CC100).
    Groups ~500 words of consecutive sentences into one document.
    VERIFIED: uses subprocess xz -dc (not Python lzma which loads full file into RAM).
    """
    proc = subprocess.Popen(
        ["xz", "-dc", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )
    try:
        current_lines = []
        current_words = 0
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            wc = len(line.split())
            current_lines.append(line)
            current_words += wc
            if current_words >= words_per_row:
                yield "\n".join(current_lines)
                current_lines = []
                current_words = 0
        if current_lines:
            yield "\n".join(current_lines)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


# ═══════════════════════════════════════════════════════════════
# Dispatch: iterate texts from an InputStream
# ═══════════════════════════════════════════════════════════════

def iter_input_texts(inp: InputStream, source_name: str):
    """Yield text strings from all files in an InputStream, in sorted order."""
    if inp.is_single_file:
        if os.path.isfile(inp.glob_pattern):
            files = [inp.glob_pattern]
        else:
            print(f"      [{source_name}] WARN: file not found: {inp.glob_pattern}", flush=True)
            return
    else:
        files = sorted(glob.glob(inp.glob_pattern))
        if not files:
            print(f"      [{source_name}] WARN: no files matched: {inp.glob_pattern}", flush=True)
            return

    print(f"      [{source_name}] Reading {len(files)} files from {inp.fmt}: "
          f"{os.path.basename(files[0])} ... {os.path.basename(files[-1])}", flush=True)

    for file_idx, filepath in enumerate(files):
        try:
            if inp.fmt == "parquet":
                yield from iter_parquet_texts(filepath, inp.text_col)
            elif inp.fmt == "arrow_stream":
                yield from iter_arrow_stream_texts(filepath, inp.text_col)
            elif inp.fmt == "jsonl_zst":
                yield from iter_jsonl_zst_texts(filepath, inp.text_key)
            elif inp.fmt == "jsonl_gz":
                if inp.special == "stackexchange":
                    yield from iter_jsonl_gz_stackexchange(filepath)
                else:
                    yield from iter_jsonl_gz_texts(filepath, inp.text_key)
            elif inp.fmt == "json_gz":
                yield from iter_json_gz_texts(filepath, inp.text_key)
            elif inp.fmt == "txt_xz":
                yield from iter_txt_xz_grouped(filepath)
            else:
                print(f"      [{source_name}] ERROR: unknown format: {inp.fmt}", flush=True)
                return
        except Exception as e:
            print(f"      [{source_name}] ERROR reading {os.path.basename(filepath)}: {e}", flush=True)
            continue


# ═══════════════════════════════════════════════════════════════
# Shard writer
# ═══════════════════════════════════════════════════════════════

def write_shard(out_dir: str, shard_idx: int, texts: list[str]) -> str:
    """Write a list of texts as a single-column parquet shard, atomically."""
    shard_name = f"shard_{shard_idx:05d}.parquet"
    tmp_path = os.path.join(out_dir, shard_name + ".tmp")
    final_path = os.path.join(out_dir, shard_name)
    table = pa.table({"text": pa.array(texts, type=pa.string())})
    pq.write_table(table, tmp_path, compression="snappy")
    os.rename(tmp_path, final_path)
    return final_path


def write_done_marker(out_dir: str, shard_count: int, token_count: int):
    """Write .done marker for a completed source."""
    done_path = os.path.join(out_dir, ".done")
    with open(done_path, "w") as f:
        f.write(f"shards={shard_count}\ntokens={token_count}\n")


# ═══════════════════════════════════════════════════════════════
# Resume helpers
# ═══════════════════════════════════════════════════════════════

def read_done_marker(out_dir: str) -> Optional[tuple[int, int]]:
    """Read .done marker. Returns (shards, tokens) or None."""
    done_path = os.path.join(out_dir, ".done")
    if not os.path.isfile(done_path):
        return None
    try:
        with open(done_path) as f:
            content = f.read()
        shards = int(content.split("shards=")[1].split("\n")[0])
        tokens = int(content.split("tokens=")[1].split("\n")[0])
        return shards, tokens
    except Exception:
        return None


def count_existing_shards(out_dir: str) -> int:
    """Count existing .parquet shard files (for resume)."""
    if not os.path.isdir(out_dir):
        return 0
    return len([f for f in os.listdir(out_dir)
                if f.startswith("shard_") and f.endswith(".parquet")])


def cleanup_tmp_files(out_dir: str):
    """Remove leftover .parquet.tmp files from crashed runs."""
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.endswith(".parquet.tmp"):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════
# Worker — TOP-LEVEL function (must be picklable for multiprocessing)
# ═══════════════════════════════════════════════════════════════

def process_source(src: Source) -> dict:
    """Process one source: stream texts, write shards, respect quota."""
    pid = os.getpid()
    result = {
        "name": src.name,
        "status": "UNKNOWN",
        "shards_written": 0,
        "tokens_written": 0,
        "skipped": False,
        "resumed": False,
        "errors": [],
    }

    out_dir = os.path.join(OUT_DIR, src.output_folder)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # ── Check if already done ──────────────────────────
        done = read_done_marker(out_dir)
        if done is not None:
            result["status"] = "SKIPPED"
            result["skipped"] = True
            result["shards_written"] = done[0]
            result["tokens_written"] = done[1]
            print(f"  [PID {pid}] ⏭️  {src.name:40s}  DONE already "
                  f"({done[0]} shards, {done[1]/1e9:.2f}B tokens)", flush=True)
            return result

        # ── Resume from existing shards ────────────────────
        cleanup_tmp_files(out_dir)
        existing_shards = count_existing_shards(out_dir)
        shard_idx = existing_shards
        tokens_written = existing_shards * TOKENS_PER_SHARD  # approx per spec

        if existing_shards > 0:
            result["resumed"] = True
            print(f"  [PID {pid}] 🔄 {src.name:40s}  RESUMING from shard {existing_shards} "
                  f"(~{tokens_written/1e9:.1f}B tokens)", flush=True)

        # ── Check if quota already met ─────────────────────
        if src.token_quota > 0 and tokens_written >= src.token_quota:
            write_done_marker(out_dir, shard_idx, tokens_written)
            result["status"] = "OK"
            result["shards_written"] = shard_idx
            result["tokens_written"] = tokens_written
            print(f"  [PID {pid}] ✅ {src.name:40s}  quota met by existing shards", flush=True)
            return result

        # ── Stream and write ───────────────────────────────
        print(f"  [PID {pid}] 🚀 {src.name:40s}  STARTING "
              f"(quota={'unlimited' if src.token_quota < 0 else f'{src.token_quota/1e9:.0f}B'})",
              flush=True)

        shard_texts = []
        shard_tokens = 0
        total_tokens = tokens_written
        total_shards = shard_idx
        t0 = time.time()
        last_log = t0
        quota_hit = False

        for inp_idx, inp in enumerate(src.inputs):
            if quota_hit:
                break

            for text in iter_input_texts(inp, src.name):
                tok = count_tokens(text, src.is_cjk)
                if tok == 0:
                    continue

                shard_texts.append(text)
                shard_tokens += tok

                # Flush shard at 100M tokens
                if shard_tokens >= TOKENS_PER_SHARD:
                    write_shard(out_dir, total_shards, shard_texts)
                    total_tokens += shard_tokens
                    total_shards += 1
                    shard_texts = []
                    shard_tokens = 0

                    # Progress log
                    now = time.time()
                    if now - last_log >= LOG_INTERVAL_SECS:
                        elapsed = now - t0
                        new_tokens = total_tokens - tokens_written
                        rate = new_tokens / elapsed / 1e6 if elapsed > 0 else 0
                        if src.token_quota > 0:
                            pct = total_tokens / src.token_quota * 100
                            print(f"  [PID {pid}] 📝 {src.name:40s}  shard={total_shards:>5d}  "
                                  f"tokens={total_tokens/1e9:.2f}B  "
                                  f"quota={pct:.0f}%  rate={rate:.0f}M/s  "
                                  f"elapsed={elapsed/60:.0f}m", flush=True)
                        else:
                            print(f"  [PID {pid}] 📝 {src.name:40s}  shard={total_shards:>5d}  "
                                  f"tokens={total_tokens/1e9:.2f}B  "
                                  f"rate={rate:.0f}M/s  elapsed={elapsed/60:.0f}m", flush=True)
                        last_log = now

                    # Quota check
                    if src.token_quota > 0 and total_tokens >= src.token_quota:
                        quota_hit = True
                        break

        # ── Final partial shard ────────────────────────────
        if shard_texts:
            write_shard(out_dir, total_shards, shard_texts)
            total_tokens += shard_tokens
            total_shards += 1

        # ── Write .done ────────────────────────────────────
        write_done_marker(out_dir, total_shards, total_tokens)

        elapsed = time.time() - t0
        new_tokens = total_tokens - tokens_written
        rate = new_tokens / elapsed / 1e6 if elapsed > 0 else 0
        result["status"] = "OK"
        result["shards_written"] = total_shards
        result["tokens_written"] = total_tokens
        print(f"  [PID {pid}] ✅ {src.name:40s}  COMPLETE  shards={total_shards:>5d}  "
              f"tokens={total_tokens/1e9:.2f}B  time={elapsed/60:.1f}m  "
              f"rate={rate:.0f}M/s", flush=True)

    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"{e}\n{traceback.format_exc()}")
        print(f"  [PID {pid}] ❌ {src.name:40s}  FAILED: {e}", flush=True)

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 100, flush=True)
    print("  557B TOKENIZER EVAL DATA — FINAL CURATION", flush=True)
    print(f"  Output:  {OUT_DIR}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)
    print(f"  Shards:  {TOKENS_PER_SHARD/1e6:.0f}M tokens each", flush=True)
    print("=" * 100, flush=True)
    print(flush=True)

    sources = build_sources()
    print(f"Registered {len(sources)} output folders to process.\n", flush=True)

    # ── List all sources ───────────────────────────────────
    for s in sources:
        quota_str = f"{s.token_quota/1e9:.0f}B" if s.token_quota > 0 else "take all"
        n_inputs = len(s.inputs)
        print(f"  {s.name:<40s}  inputs={n_inputs}  quota={quota_str}", flush=True)
    print(flush=True)

    # ── Pre-create output directories (avoid gcsfuse race) ──
    for src in sources:
        out_dir = os.path.join(OUT_DIR, src.output_folder)
        os.makedirs(out_dir, exist_ok=True)
    print("Output directories created.\n", flush=True)

    # ── Scan for already-done sources ──────────────────────
    done_tokens = 0
    done_count = 0
    for src in sources:
        out_dir = os.path.join(OUT_DIR, src.output_folder)
        marker = read_done_marker(out_dir)
        if marker:
            done_count += 1
            done_tokens += marker[1]
    if done_count > 0:
        print(f"Found {done_count} already-completed sources ({done_tokens/1e9:.1f}B tokens). "
              f"They will be skipped.\n", flush=True)

    # ── Launch workers ─────────────────────────────────────
    t0 = time.time()
    results = []

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for result in pool.imap_unordered(process_source, sources):
            results.append(result)
            # Print running tally
            completed = len(results)
            total = len(sources)
            total_tok = sum(r["tokens_written"] for r in results)
            total_shd = sum(r["shards_written"] for r in results)
            elapsed = time.time() - t0
            print(f"\n  >>> [{completed}/{total} sources done] "
                  f"tokens so far: {total_tok/1e9:.2f}B, "
                  f"shards: {total_shd:,}, "
                  f"elapsed: {elapsed/60:.0f}m <<<\n", flush=True)

    total_elapsed = time.time() - t0

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 100, flush=True)
    print("  CURATION SUMMARY", flush=True)
    print("=" * 100, flush=True)

    ok = [r for r in results if r["status"] == "OK"]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if r["status"] == "FAIL"]

    total_shards = sum(r["shards_written"] for r in results)
    total_tokens = sum(r["tokens_written"] for r in results)

    print(f"\n  ✅ Completed:  {len(ok)}", flush=True)
    print(f"  ⏭️  Skipped:    {len(skipped)} (already done)", flush=True)
    print(f"  ❌ Failed:     {len(failed)}", flush=True)
    print(f"\n  Total shards:  {total_shards:,}", flush=True)
    print(f"  Total tokens:  {total_tokens/1e9:.2f}B", flush=True)
    print(f"  Wall time:     {total_elapsed/3600:.1f}h ({total_elapsed/60:.0f}m)", flush=True)

    # Per-domain breakdown
    print(f"\n  Per-domain breakdown:", flush=True)
    domain_tokens = {}
    domain_shards = {}
    for r in results:
        domain = r["name"].split("/")[0]
        domain_tokens[domain] = domain_tokens.get(domain, 0) + r["tokens_written"]
        domain_shards[domain] = domain_shards.get(domain, 0) + r["shards_written"]
    print(f"  {'Domain':<15s}  {'Tokens':>12s}  {'Shards':>8s}", flush=True)
    print(f"  {'─'*40}", flush=True)
    for domain in ["English", "Math", "Code", "Scientific", "Multilingual"]:
        tok = domain_tokens.get(domain, 0)
        shd = domain_shards.get(domain, 0)
        print(f"  {domain:<15s}  {tok/1e9:>10.2f}B  {shd:>8,}", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  {'TOTAL':<15s}  {total_tokens/1e9:>10.2f}B  {total_shards:>8,}", flush=True)

    # Per-source detail
    print(f"\n  Per-source detail:", flush=True)
    for r in sorted(results, key=lambda x: x["name"]):
        status = r["status"]
        name = r["name"]
        tok_b = r["tokens_written"] / 1e9
        shards = r["shards_written"]
        icon = {"OK": "✅", "SKIPPED": "⏭️ ", "FAIL": "❌"}.get(status, "?")
        print(f"    {icon} {name:<40s}  {tok_b:>8.2f}B  {shards:>6,} shards  [{status}]",
              flush=True)

    # Failures
    if failed:
        print(f"\n  ─── FAILURES ───", flush=True)
        for r in failed:
            print(f"\n  ❌ {r['name']}", flush=True)
            for e in r.get("errors", []):
                print(f"     → {e[:300]}", flush=True)

    print(flush=True)
    print("=" * 100, flush=True)
    if failed:
        print(f"  ⚠️  {len(failed)} source(s) FAILED. Fix and re-run (resume-safe).", flush=True)
    else:
        print("  ✅ All sources completed successfully!", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()