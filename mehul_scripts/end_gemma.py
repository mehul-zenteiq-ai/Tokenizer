#!/usr/bin/env python3
"""
ablation6_superbpe_bloom_v3_dedup_single.py
============================================
SINGLE-PHASE version of ablation6_superbpe_bloom_v3_dedup.py

Produces IDENTICAL output to the original Phase 2:
  - Pre-tokenizer : ByteLevel(add_prefix_space=False, use_regex=False)
  - Decoder       : ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)
  - Vocab target  : 262,144 (TOTAL_VOCAB_SIZE)
  - Dedup         : rolling hash-set (identical to original)

Since Phase 2 trains from raw bytes regardless of Phase 1 input,
collapsing into a single phase produces byte-identical merge tables.

─────────────────────────────────────────────────────────────────────────────
Architecture
─────────────────────────────────────────────────────────────────────────────

Pre-tokenizer:
  ByteLevel(add_prefix_space=False, use_regex=False)

Decoder:
  ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)

─────────────────────────────────────────────────────────────────────────────
Special tokens  (UNCHANGED from original)
─────────────────────────────────────────────────────────────────────────────

IDs  0 –   3   Foundational     : <pad> <eos> <bos> <unk>
IDs  4 –   7   Utility          : <mask> [multimodal] [@BOS@] <|image_soft_token|>
IDs  8 –  12   Chat             : system user assistant turn delimiters
IDs 13 –  24   Tool use         : tools tool_call arg/response tokens
IDs 25 –  27   Reasoning        : <think> </think> <|nothink|>
IDs 28 –  52   Language tags    : 25 languages (9 original + 16 new)
IDs 53 –  55   FIM              : fim_prefix fim_middle fim_suffix
IDs 56 – 8247  Reserved         : 8192 slots

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

  # Environment check
  python ablation6_superbpe_bloom_v3_dedup_single.py --check-only

  # Full training
  python ablation6_superbpe_bloom_v3_dedup_single.py \
      --shards-dir /path/to/shards \
      --output /path/to/output \
      --seed 42 \
      --max-chars 3347 \
      --validate
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 0.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextlib.contextmanager
def stage(name: str):
    log.info("")
    log.info("▶ STAGE: %s", name)
    log.info("─" * 65)
    t = time.perf_counter()
    yield
    log.info("✓ %s  (%.1f s)", name, time.perf_counter() - t)
    log.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_NAME       = "ablation6_superbpe_bloom_v3_dedup_single"
TOTAL_VOCAB_SIZE    = 262_144
MIN_FREQUENCY       = 2
NUM_THREADS         = 64
MODEL_MAX_LENGTH    = 2048

DEFAULT_OUTPUT              = Path("./ablation6_out")
DEFAULT_SHARD_DIR           = Path("/path/to/shards")
DEFAULT_NUM_SHARDS: Optional[int] = None
DEFAULT_SEED                = 42
DEFAULT_MAX_CHARS           = 3347   # p90 of dataset

# ── Dedup ─────────────────────────────────────────────────────────────────────
DEDUP_ENABLED   = True
DEDUP_MAX_CACHE = 100000000   # rolling hash-set cap (~400 MB for 5M hashes)

os.environ["RAYON_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPECIAL TOKEN DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

FOUNDATIONAL = [
    "<pad>",    # 0
    "<eos>",    # 1
    "<bos>",    # 2
    "<unk>",    # 3
]

UTILITY = [
    "<mask>",                # 4
    "[multimodal]",          # 5
    "[@BOS@]",               # 6
    "<|image_soft_token|>",  # 7
]

CHAT = [
    "<|system|>",         # 8
    "<|user|>",           # 9
    "<|assistant|>",      # 10
    "<|start_of_turn|>",  # 11
    "<|end_of_turn|>",    # 12
]

TOOL_USE = [
    "<tools>",           # 13
    "</tools>",          # 14
    "<tool_call>",       # 15
    "</tool_call>",      # 16
    "<arg_key>",         # 17
    "</arg_key>",        # 18
    "<arg_value>",       # 19
    "</arg_value>",      # 20
    "<tool_response>",   # 21
    "</tool_response>",  # 22
    "<|tool_declare|>",  # 23
    "<|observation|>",   # 24
]

REASONING = [
    "<think>",       # 25
    "</think>",      # 26
    "<|nothink|>",   # 27
]

LANGUAGE_TAGS = [
    "<lang_en>",    # 28  English
    "<lang_hi>",    # 29  Hindi
    "<lang_ta>",    # 30  Tamil
    "<lang_te>",    # 31  Telugu
    "<lang_kn>",    # 32  Kannada
    "<lang_mar>",   # 33  Marathi
    "<lang_guj>",   # 34  Gujarati
    "<lang_ben>",   # 35  Bengali
    "<lang_mly>",   # 36  Malayalam
    "<lang_pan>",   # 37  Punjabi
    "<lang_odi>",   # 38  Odia
    "<lang_sin>",   # 39  Sinhala
    "<lang_nep>",   # 40  Nepali
    "<lang_bng>",   # 41  Bangla BD
    "<lang_urd>",   # 42  Urdu
    "<lang_psh>",   # 43  Pashto
    "<lang_dar>",   # 44  Dari
    "<lang_bur>",   # 45  Burmese
    "<lang_dzo>",   # 46  Dzongkha
    "<lang_rus>",   # 47  Russian
    "<lang_fra>",   # 48  French
    "<lang_deu>",   # 49  German
    "<lang_zho>",   # 50  Mandarin
    "<lang_jpn>",   # 51  Japanese
    "<lang_kor>",   # 52  Korean
]

FIM = [
    "<|fim_prefix|>",   # 53
    "<|fim_middle|>",   # 54
    "<|fim_suffix|>",   # 55
]

NUM_RESERVED = 8_192
RESERVED     = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]

SPECIAL_TOKENS = (
    FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING
    + LANGUAGE_TAGS + FIM + RESERVED
)

assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), \
    "DUPLICATE SPECIAL TOKENS FOUND"

PAD_TOKEN           = "<pad>"
EOS_TOKEN           = "<eos>"
BOS_TOKEN           = "<bos>"
UNK_TOKEN           = "<unk>"
SYSTEM_TOKEN        = "<|system|>"
USER_TOKEN          = "<|user|>"
ASSISTANT_TOKEN     = "<|assistant|>"
START_OF_TURN_TOKEN = "<|start_of_turn|>"
END_OF_TURN_TOKEN   = "<|end_of_turn|>"
THINK_TOKEN         = "<think>"
END_THINK_TOKEN     = "</think>"
NOTHINK_TOKEN       = "<|nothink|>"
TOOL_CALL_TOKEN     = "<tool_call>"
END_TOOL_CALL_TOKEN = "</tool_call>"
OBSERVATION_TOKEN   = "<|observation|>"
FIM_PREFIX_TOKEN    = "<|fim_prefix|>"
FIM_MIDDLE_TOKEN    = "<|fim_middle|>"
FIM_SUFFIX_TOKEN    = "<|fim_suffix|>"
STOP_TOKENS         = [EOS_TOKEN, END_OF_TURN_TOKEN]

_N_SPECIAL  = len(SPECIAL_TOKENS)
_N_RESERVED = NUM_RESERVED


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIXED SCIENTIFIC VOCABULARY — non-alpha symbols only (ByteLevel safe)
# ══════════════════════════════════════════════════════════════════════════════

FIXED_SCI_VOCAB = [
    "=", "+", "-", "*", "/", "^", "!",
    "<", ">",
    "\u2264", "\u2265", "\u2260",
    "\u221a", "\u2211", "\u220f",
    "\u2202", "\u2207",
    "\u2207\u00b2", "\u2207\u00b7", "\u2207\u00d7",
    "\u222b", "\u222e",
    "\u2202\u209c", "\u2202\u2093", "\u2202\u1d67", "\u2202_z",
    "\u2202\u209c\u209c", "\u2202\u2093\u2093", "\u2202\u1d67\u1d67", "\u2202_zz",
    "\u2202_xy", "\u2202_xz", "\u2202_yz",
]

_N_SCI = len(FIXED_SCI_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  WHITESPACE TOKENS
# ══════════════════════════════════════════════════════════════════════════════

WHITESPACE_TOKENS = [
    "\n",
    "\n\n",
    "\r\n",
    "\t",
    "\v",
    "\f",
]
_N_WHITESPACE = len(WHITESPACE_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DEDUP MIN CHARS (used by corpus_iterator)
# ══════════════════════════════════════════════════════════════════════════════

dedup_min_chars = 40


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RESOURCE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    def __init__(self, interval_sec=30.0, ram_warn_pct=80.0,
                 ram_critical_pct=88.0, log_prefix="  [monitor]"):
        self.interval  = interval_sec
        self.ram_warn  = ram_warn_pct
        self.ram_crit  = ram_critical_pct
        self.prefix    = log_prefix
        self._stop     = threading.Event()
        self._thread   = None
        self._peak_rss = 0.0
        self._peak_ram = 0.0

    @staticmethod
    def _read_meminfo():
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        info[p[0].rstrip(":")] = int(p[1])
        except (OSError, ValueError):
            pass
        return info

    @staticmethod
    def _read_rss_kb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except (OSError, ValueError):
            pass
        return 0

    @staticmethod
    def _read_load():
        try:
            with open("/proc/loadavg") as f:
                p = f.read().split()
                return float(p[0]), float(p[1]), float(p[2])
        except Exception:
            return (0.0, 0.0, 0.0)

    def _snap(self):
        mem    = self._read_meminfo()
        tot    = mem.get("MemTotal", 1)
        avail  = mem.get("MemAvailable", mem.get("MemFree", 0))
        used   = tot - avail
        swap   = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
        cache  = mem.get("Buffers", 0) + mem.get("Cached", 0)
        rss    = self._read_rss_kb()
        l1, l5, l15 = self._read_load()
        ncpu   = os.cpu_count() or 1
        G      = 1 / (1024 * 1024)
        pct    = (used / tot) * 100 if tot else 0
        rss_gb = rss * G
        self._peak_rss = max(self._peak_rss, rss_gb)
        self._peak_ram = max(self._peak_ram, pct)
        warn = " ⚠⚠ CRITICAL" if pct >= self.ram_crit else (
               " ⚠ WARN"      if pct >= self.ram_warn  else "")
        return (f"{self.prefix} {time.strftime('%H:%M:%S')}  "
                f"RAM {used*G:.1f}/{tot*G:.1f} GB ({pct:.1f}%{warn})  "
                f"Avail {avail*G:.1f} GB  Cache {cache*G:.1f} GB  "
                f"Swap {swap*G:.1f} GB  RSS {rss_gb:.1f} GB  "
                f"Load {l1:.0f}/{l5:.0f}/{l15:.0f} (of {ncpu} cores)")

    def _run(self):
        while not self._stop.wait(self.interval):
            print(self._snap(), flush=True)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def log_now(self, label=""):
        if label:
            print(f"{self.prefix} [{label}]")
        print(self._snap(), flush=True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {
            "peak_process_rss_gb": round(self._peak_rss, 2),
            "peak_system_ram_pct": round(self._peak_ram, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SHARD DISCOVERY + SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def discover_shards(shard_dir: Path) -> list[Path]:
    shards = sorted(shard_dir.glob("shard_*.txt.gz"))
    if not shards:
        shards = sorted(shard_dir.glob("shard_*.txt"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.txt.gz (or shard_*.txt) found in {shard_dir}"
        )
    return shards


def select_shards(
    shard_dir  : Path,
    num_shards : Optional[int] = None,
    seed       : int            = DEFAULT_SEED,
) -> list[Path]:
    available = discover_shards(shard_dir)
    if num_shards is None or num_shards >= len(available):
        if num_shards is not None and num_shards > len(available):
            log.warning("Requested %d shards but only %d available — using all.",
                        num_shards, len(available))
        selected = available
    else:
        rng      = random.Random(seed)
        selected = sorted(rng.sample(available, num_shards))

    log.info("Shard selection (seed=%d):", seed)
    log.info("  Available : %d", len(available))
    log.info("  Selected  : %d", len(selected))
    for p in selected:
        log.info("    %-45s  %.2f GB", p.name, p.stat().st_size / 1e9)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_line(raw: str) -> str:
    return raw.rstrip('\n\r')


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CORPUS ITERATOR  — dedup (identical to original)
#
#  Rolling hash-set dedup design:
#    - One set spans ALL shards → catches cross-shard duplicates.
#    - hash(line) is fast and sufficient for exact-match boilerplate removal.
#    - When set exceeds DEDUP_MAX_CACHE it is cleared (rolling window).
#      Lines repeated > 5M apart may be seen twice; lines repeated thousands
#      of times (the actual problem) are almost always within one window.
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB  = 500
MIN_DOC_CHARS = 5


def corpus_iterator(
    shard_dir      : Path,
    num_shards     : Optional[int] = None,
    seed           : int            = DEFAULT_SEED,
    dry_run        : bool           = False,
    max_line_chars : Optional[int]  = None,
    dedup          : bool           = DEDUP_ENABLED,
    dedup_max_cache: int            = DEDUP_MAX_CACHE,
) -> Iterator[str]:
    """
    Stream lines from selected shards.
    max_line_chars: memory optimization. p90 = 3347 chars.
    dedup: rolling hash-set deduplication. Eliminates exact-duplicate lines
           (e.g. repeated boilerplate) from dominating merge budget.
    """
    paths          = select_shards(shard_dir, num_shards, seed)
    total_bytes    = 0
    total_yielded  = 0
    total_skipped  = 0
    long_skipped   = 0
    dedup_skipped  = 0
    last_log_bytes = 0
    t_start        = time.perf_counter()
    seen_hashes: set[int] = set()

    if dedup:
        log.info("  dedup: ENABLED  |  cache_size=%d  (~%.0f MB RAM)",
                 dedup_max_cache, dedup_max_cache * 28 / 1_048_576)
    if max_line_chars is not None:
        log.info("  max_line_chars=%d  (p90 threshold — keeps ~90%% of lines)",
                 max_line_chars)

    pbar = tqdm(desc="Lines", unit=" lines", smoothing=0.1,
                disable=dry_run, mininterval=5.0)

    for shard_idx, path in enumerate(paths):
        shard_bytes   = 0
        shard_yielded = 0
        t_shard       = time.perf_counter()
        pbar.set_postfix(shard=f"{path.name} [{shard_idx+1}/{len(paths)}]")

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if len(line) < MIN_DOC_CHARS:
                    total_skipped += 1
                    continue
                if max_line_chars is not None and len(line) > max_line_chars:
                    long_skipped += 1
                    continue
                # ── rolling dedup ─────────────────────────────────────────
                if dedup:
                    if len(line) >= dedup_min_chars:
                        h = hash(line)
                        if h in seen_hashes:
                            dedup_skipped += 1
                            continue
                        seen_hashes.add(h)
                        if len(seen_hashes) > dedup_max_cache:
                            seen_hashes.clear()   # rolling window — bounded RAM
                # ──────────────────────────────────────────────────────────
                processed      = preprocess_line(raw_line)
                byte_len       = len(processed.encode("utf-8"))
                total_bytes   += byte_len
                shard_bytes   += byte_len
                total_yielded += 1
                shard_yielded += 1
                if not dry_run:
                    yield processed
                pbar.update(1)
                if total_bytes - last_log_bytes >= LOG_EVERY_MB * 1_048_576:
                    elapsed = time.perf_counter() - t_start
                    log.info(
                        "  progress  [%s]  total=%.1f GB  lines=%d  "
                        "dedup_skipped=%d  long_skipped=%d  speed=%.1f MB/s",
                        path.name, total_bytes / 1_073_741_824,
                        total_yielded, dedup_skipped, long_skipped,
                        (total_bytes / 1_048_576) / elapsed if elapsed else 0,
                    )
                    last_log_bytes = total_bytes

        log.info("  %-40s  %.2f GB  %d lines  %.1f s",
                 path.name, shard_bytes / 1_073_741_824,
                 shard_yielded, time.perf_counter() - t_shard)

    pbar.close()
    log.info(
        "Corpus done.  %.2f GB | %d yielded | %d short-skipped | "
        "%d long-skipped (>%s chars) | %d dedup-skipped | %.1f s total",
        total_bytes / 1_073_741_824, total_yielded, total_skipped,
        long_skipped, str(max_line_chars) if max_line_chars else "∞",
        dedup_skipped, time.perf_counter() - t_start,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10. ADDED TOKEN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_added_tokens():
    from tokenizers import AddedToken

    special_objs = [
        AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS
    ]
    sci_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=False, lstrip=False, rstrip=False)
        for t in FIXED_SCI_VOCAB
    ]
    ws_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=False, lstrip=False, rstrip=False)
        for t in WHITESPACE_TOKENS
    ]

    all_tokens = special_objs + sci_objs + ws_objs
    seen, deduped = set(), []
    for tok in all_tokens:
        if tok.content not in seen:
            seen.add(tok.content)
            deduped.append(tok)

    log.info(
        "  Added tokens: %d special + %d sci (non-alpha) "
        "+ %d whitespace  =  %d total",
        _N_SPECIAL, _N_SCI, _N_WHITESPACE, len(deduped),
    )
    log.info(
        "  Special breakdown: %d foundational + %d utility + %d chat + "
        "%d tool + %d reasoning + %d lang (%d extended) + %d fim + %d reserved",
        len(FOUNDATIONAL), len(UTILITY), len(CHAT), len(TOOL_USE),
        len(REASONING), len(LANGUAGE_TAGS), len(LANGUAGE_TAGS) - 9,
        len(FIM), NUM_RESERVED,
    )
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# 11. POST-SAVE JSON PATCH
# ══════════════════════════════════════════════════════════════════════════════

def _patch_tokenizer_json(tok_path: Path) -> None:
    with open(tok_path, encoding="utf-8") as f:
        state = json.load(f)

    changes = []

    target_decoder = {
        "type"             : "ByteLevel",
        "add_prefix_space" : False,
        "trim_offsets"     : False,
        "use_regex"        : False,
    }
    if state.get("decoder") != target_decoder:
        changes.append(f"decoder: {state.get('decoder')} → ByteLevel(...)")
        state["decoder"] = target_decoder

    pre_tok = state.get("pre_tokenizer")
    if pre_tok is None:
        state["pre_tokenizer"] = {
            "type"             : "ByteLevel",
            "add_prefix_space" : False,
            "trim_offsets"     : False,
            "use_regex"        : False,
        }
        changes.append("pre_tokenizer: null → ByteLevel(use_regex=False)")
    elif pre_tok.get("type") == "ByteLevel":
        if pre_tok.get("use_regex", True):
            pre_tok["use_regex"] = False
            changes.append("pre_tokenizer ByteLevel use_regex: True → False")
        pre_tok["add_prefix_space"] = False
    elif pre_tok.get("type") == "Sequence":
        for item in pre_tok.get("pretokenizers", []):
            if item.get("type") == "ByteLevel":
                if item.get("use_regex", True):
                    item["use_regex"] = False
                    changes.append("pre_tokenizer Sequence>ByteLevel use_regex → False")
                item["add_prefix_space"] = False

    sci_set  = set(FIXED_SCI_VOCAB)
    ws_set   = set(WHITESPACE_TOKENS)
    spec_set = set(SPECIAL_TOKENS)
    n_spec = n_sci = n_ws = 0

    for entry in state.get("added_tokens", []):
        content = entry.get("content", "")
        if content in spec_set:
            if not entry.get("special", False):
                entry["special"] = True
                n_spec += 1
            entry["normalized"] = False
        if content in sci_set:
            entry["special"]    = False
            entry["normalized"] = False
            n_sci += 1
        if content in ws_set:
            entry["special"]    = False
            entry["normalized"] = False
            n_ws += 1

    existing = {e["content"] for e in state.get("added_tokens", [])}
    vocab    = state.get("model", {}).get("vocab", {})
    injected = 0
    for ws in WHITESPACE_TOKENS:
        if ws not in existing and ws in vocab:
            state["added_tokens"].append({
                "id"         : vocab[ws],
                "content"    : ws,
                "single_word": False,
                "lstrip"     : False,
                "rstrip"     : False,
                "normalized" : False,
                "special"    : False,
            })
            injected += 1

    if changes:
        log.info("  [patch] changes applied: %s", "; ".join(changes))
    else:
        log.info("  [patch] no decoder/pre_tokenizer changes needed (already correct)")
    log.info("  [patch] special re-asserted=%d  sci fixed=%d  "
             "ws fixed=%d  ws injected=%d",
             n_spec, n_sci, n_ws, injected)

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("  [patch] Written → %s  (%d bytes)", tok_path, tok_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# 12. BUILD TOKENIZER — Single phase: ByteLevel BPE → 262,144
# ══════════════════════════════════════════════════════════════════════════════

def build_tokenizer():
    """
    Build tokenizer identical to original Phase 2 config:
      pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
      decoder       = ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)
      vocab_size    = TOTAL_VOCAB_SIZE (262,144)
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder(
        add_prefix_space = False,
        use_regex        = False,
        trim_offsets     = False,
    )

    deduped = _build_added_tokens()
    trainer = BpeTrainer(
        vocab_size     = TOTAL_VOCAB_SIZE,
        min_frequency  = MIN_FREQUENCY,
        special_tokens = deduped,
        initial_alphabet  = ByteLevel.alphabet(),
        show_progress  = True,
    )

    log.info("Tokenizer configured (single phase):")
    log.info("  pre_tokenizer : ByteLevel(add_prefix_space=False, use_regex=False)")
    log.info("  decoder       : ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)")
    log.info("  vocab_size    : %d", TOTAL_VOCAB_SIZE)
    log.info("  min_frequency : %d", MIN_FREQUENCY)
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 13. TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def train_tokenizer(tokenizer, trainer, shard_dir, num_shards, seed,
                    dry_run=False, max_line_chars=None):
    if dry_run:
        log.info("DRY RUN — no training.")
        consumed = 0
        for _ in corpus_iterator(shard_dir, num_shards, seed,
                                  dry_run=True, max_line_chars=max_line_chars):
            consumed += 1
        log.info("Lines that would be yielded: %d", consumed)
        return tokenizer

    log.info("=" * 65)
    log.info("TRAINING  |  target=%d  |  threads=%d  |  max_chars=%s",
             TOTAL_VOCAB_SIZE, NUM_THREADS,
             str(max_line_chars) if max_line_chars else "∞")
    log.info("=" * 65)

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator = corpus_iterator(shard_dir, num_shards, seed,
                                   max_line_chars=max_line_chars),
        trainer  = trainer,
        length   = None,
    )
    elapsed = time.perf_counter() - t0
    actual  = tokenizer.get_vocab_size()
    log.info("Training complete in %.1f min. Vocab: %d", elapsed / 60, actual)
    if actual < TOTAL_VOCAB_SIZE * 0.95:
        log.warning(
            "Vocab (%d) is well below target (%d). "
            "Add more shards or check corpus size.",
            actual, TOTAL_VOCAB_SIZE,
        )
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 14. SAVE + PATCH + HF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_tokenizer(tokenizer, output_dir: Path, meta: dict):
    save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    tok_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    log.info("Saved raw tokenizer.json (%d bytes)", tok_path.stat().st_size)

    log.info("Applying safety patches...")
    _patch_tokenizer_json(tok_path)

    vocab = tokenizer.get_vocab()
    sv    = dict(sorted(vocab.items(), key=lambda x: x[1]))
    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(sv, f, indent=0, ensure_ascii=False)
    log.info("Saved vocab.json (%d tokens)", len(sv))

    with open(tok_path, encoding="utf-8") as f:
        td = json.load(f)
    merges = td.get("model", {}).get("merges", [])
    with open(save_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges:
            f.write((" ".join(m) if isinstance(m, list) else str(m)) + "\n")
    log.info("Saved merges.txt (%d merges)", len(merges))

    cfg = {
        "tokenizer_class"             : "PreTrainedTokenizerFast",
        "model_type"                  : None,
        "bos_token"                   : BOS_TOKEN,
        "eos_token"                   : EOS_TOKEN,
        "unk_token"                   : None,
        "pad_token"                   : PAD_TOKEN,
        "model_max_length"            : MODEL_MAX_LENGTH,
        "clean_up_tokenization_spaces": False,
        "add_prefix_space"            : False,
        "added_tokens_decoder"        : {},
        "additional_special_tokens"   : (
            UTILITY + CHAT + TOOL_USE + REASONING + LANGUAGE_TAGS + FIM
        ),
    }

    for t in SPECIAL_TOKENS:
        if t in vocab:
            cfg["added_tokens_decoder"][str(vocab[t])] = {
                "content": t, "lstrip": False, "normalized": False,
                "rstrip": False, "single_word": False, "special": True,
            }
    for t in FIXED_SCI_VOCAB:
        if t in vocab:
            cfg["added_tokens_decoder"][str(vocab[t])] = {
                "content": t, "lstrip": False, "normalized": False,
                "rstrip": False, "single_word": False, "special": False,
            }
    for t in WHITESPACE_TOKENS:
        if t in vocab:
            cfg["added_tokens_decoder"][str(vocab[t])] = {
                "content": t, "lstrip": False, "normalized": False,
                "rstrip": False, "single_word": False, "special": False,
            }

    with open(save_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    stm = {
        "bos_token" : BOS_TOKEN,
        "eos_token" : EOS_TOKEN,
        "unk_token" : None,
        "pad_token" : PAD_TOKEN,
        # intentionally removed "additional_special_tokens" here to bypass the TPU loader bug
    }
    with open(save_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(stm, f, indent=2, ensure_ascii=False)

    with open(save_dir / "ablation_metadata.json", "w") as f:
        json.dump({
            **meta,
            "vocab_size_actual"   : len(sv),
            "merges"              : len(merges),
            "special_tokens_count": _N_SPECIAL,
            "reserved_slots"      : NUM_RESERVED,
            "language_tags"       : LANGUAGE_TAGS,
            "num_language_tags"   : len(LANGUAGE_TAGS),
            "dedup_enabled"       : DEDUP_ENABLED,
            "dedup_max_cache"     : DEDUP_MAX_CACHE,
        }, f, indent=2)

    log.info("All files saved → %s", save_dir)
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# 15. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tokenizer(output_dir: Path):
    load_dir = output_dir
    log.info("=" * 65)
    log.info("VALIDATION — %s", load_dir)

    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(str(load_dir))
        hf  = True
        log.info("Loaded via PreTrainedTokenizerFast")
    except Exception as e:
        log.warning("PreTrainedTokenizerFast failed (%s) — using raw Tokenizer", e)
        from tokenizers import Tokenizer as _Tok
        tok = _Tok.from_file(str(load_dir / "tokenizer.json"))
        hf  = False

    vocab = tok.get_vocab()
    log.info("Vocab size: %d", len(vocab))

    log.info("")
    log.info("Special token ID checks:")
    non_reserved = (FOUNDATIONAL + UTILITY + CHAT + TOOL_USE
                    + REASONING + LANGUAGE_TAGS + FIM)
    id_pass = True
    for expected_id, tok_str in enumerate(non_reserved):
        actual_id = vocab.get(tok_str, -1)
        ok = (actual_id == expected_id)
        if not ok:
            id_pass = False
            log.warning("  ID MISMATCH: %r  expected=%d  actual=%d",
                        tok_str, expected_id, actual_id)
    if id_pass:
        log.info("  All %d non-reserved special tokens have correct IDs ✓",
                 len(non_reserved))

    log.info("")
    log.info("Language tag presence (%d total):", len(LANGUAGE_TAGS))
    missing_tags = []
    for lt in LANGUAGE_TAGS:
        in_v = lt in vocab
        if not in_v:
            missing_tags.append(lt)
        log.info("  %-18s  %s  id=%s", lt,
                 "IN VOCAB ✓" if in_v else "NOT IN VOCAB ✗",
                 vocab.get(lt, "—"))
    if missing_tags:
        log.warning("  MISSING LANGUAGE TAGS: %s", missing_tags)

    log.info("")
    log.info("Whitespace token presence:")
    for ws in WHITESPACE_TOKENS:
        in_v = ws in vocab
        log.info("  %-6r → %s  id=%s", ws,
                 "IN VOCAB ✓" if in_v else "NOT IN VOCAB ✗",
                 vocab.get(ws, "—"))

    tests = [
        ("LaTeX",            r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"),
        ("ASCII math",       "d/dt(T) = alpha * laplacian(T)"),
        ("Unicode math",     "∂u/∂t = α∇²u"),
        ("Numbers",          "k = 3.14159 and T = 1e-5"),
        ("Hindi",            "नमस्ते दुनिया"),
        ("Tamil",            "போகிறார்கள்"),
        ("Telugu",           "నమస్కారం"),
        ("Kannada",          "ನಮಸ್ಕಾರ"),
        ("Malayalam",        "നമസ്കാരം"),
        ("Bengali",          "আমার সোনার বাংলা"),
        ("Gujarati",         "ગુજરાતી ભાષા"),
        ("Marathi",          "मराठी भाषा"),
        ("Punjabi",          "ਪੰਜਾਬੀ ਭਾਸ਼ਾ"),
        ("Odia",             "ଓଡ଼ିଆ ଭାଷା"),
        ("Nepali",           "नेपाली भाषा"),
        ("Sinhala",          "සිංහල භාෂාව"),
        ("Bangla BD",        "বাংলাদেশের বাংলা"),
        ("Urdu",             "اردو زبان"),
        ("Pashto",           "پښتو ژبه"),
        ("Dari",             "زبان دری"),
        ("Burmese",          "မြန်မာဘာသာ"),
        ("Dzongkha",         "རྫོང་ཁ་སྐད།"),
        ("Russian",          "Привет мир"),
        ("French",           "Bonjour le monde avec des caractères spéciaux"),
        ("German",           "Guten Morgen mit Umlauten: ü ö ä ß"),
        ("Mandarin",         "你好世界，这是中文测试"),
        ("Japanese",         "こんにちは世界、日本語テスト"),
        ("Korean",           "안녕하세요 세계"),
        ("Code indent",      "    def foo():"),
        ("Code 8sp",         "        return x + 1"),
        ("Newline",          "def foo():\n    return x"),
        ("Multiline",        "x = 1\ny = 2\nz = x + y"),
        ("Tab indent",       "\tdef bar():\n\t\treturn 42"),
        ("Mixed code",       "a  =  b  +  c"),
        ("<think> tag",      "<think>reasoning here</think>"),
        ("FIM",              "<|fim_prefix|>def f():<|fim_suffix|>    pass<|fim_middle|>"),
        ("Lang tag Hindi",   "<lang_hi>नमस्ते दुनिया</lang_hi>"),
        ("Lang tag Tamil",   "<lang_ta>தமிழ் உரை</lang_ta>"),
        ("Lang tag Russian", "<lang_rus>Привет мир</lang_rus>"),
        ("Lang tag Mandarin","<lang_zho>你好世界</lang_zho>"),
        ("Lang tag Japanese","<lang_jpn>こんにちは</lang_jpn>"),
        ("Lang tag Korean",  "<lang_kor>안녕하세요</lang_kor>"),
        ("Lang tag Urdu",    "<lang_urd>اردو زبان</lang_urd>"),
        ("Lang tag Burmese", "<lang_bur>မြန်မာဘာသာ</lang_bur>"),
        ("Tool call",        "<tool_call><arg_key>fn</arg_key></tool_call>"),
        ("Mixed math+Tamil", "The heat equation ∂u/∂t = α∇²u holds for நமஸ்தே"),
        ("Mixed JP",         "私はPythonを使う"),
        ("Mixed FR+math",    "l'équation différentielle ∂u/∂t = α∇²u"),
    ]

    log.info("")
    log.info("Round-trip encode/decode tests — ByteLevel must be exact:")
    log.info("  (No .strip() applied — any mismatch is a real bug)")
    all_pass = True
    for label, t in tests:
        ids = tok.encode(t) if hf else tok.encode(t).ids
        dec = (tok.decode(ids, skip_special_tokens=False)
               if hf else tok.decode(ids, skip_special_tokens=False))
        match = (t == dec)
        if not match:
            all_pass = False
        log.info("  [%-22s] %3d toks | %s | %r",
                 label, len(ids), "OK " if match else "ERR", t[:55])
        if not match:
            log.warning("    MISMATCH: in =%r", t[:60])
            log.warning("             out=%r", dec[:60])

    log.info("")
    log.info("Superword token checks (expected cross-boundary merges):")
    for sw in [" the", " equation", " return", "def ", "\n    ",
               "partial", " 的", " を", " и"]:
        log.info("  %-28r  %s", sw, "IN VOCAB ✓" if sw in vocab else "not yet")

    log.info("")
    if all_pass:
        log.info("ALL PASSED ✓ — ByteLevel decode is lossless across all 25 languages")
    else:
        log.warning("SOME FAILED ✗ — see mismatches above")
    log.info("=" * 65)
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# 16. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(shards_dir, output_dir, num_shards, seed, dry_run, validate,
        monitor_interval, max_chars):

    log.info("=" * 65)
    log.info("  %s", ABLATION_NAME.upper())
    log.info("=" * 65)
    log.info("  Strategy         : Single-phase ByteLevel BPE")
    log.info("  Dedup            : ENABLED  (rolling hash-set, cache=%d)", DEDUP_MAX_CACHE)
    log.info("  Pre-tokenizer    : ByteLevel(add_prefix_space=False, use_regex=False)")
    log.info("  Decoder          : ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)")
    log.info("  Special tokens   : %d  (%d reserved)", _N_SPECIAL, NUM_RESERVED)
    log.info("  Language tags    : %d  (9 original + %d extended)",
             len(LANGUAGE_TAGS), len(LANGUAGE_TAGS) - 9)
    log.info("  Sci vocab        : %d  (non-alpha symbols only)", _N_SCI)
    log.info("  Shards           : %s", num_shards or "all")
    log.info("  Vocab target     : %d", TOTAL_VOCAB_SIZE)
    log.info("  Max chars/line   : %s  (p90=3347 recommended)",
             str(max_chars) if max_chars else "∞")
    log.info("  Shards dir       : %s", shards_dir)
    log.info("  Seed             : %d", seed)
    log.info("  Output           : %s", output_dir)
    log.info("=" * 65)

    monitor = ResourceMonitor(interval_sec=monitor_interval)
    monitor.start()
    monitor.log_now("BASELINE")
    t_total = time.perf_counter()

    meta = {
        "ablation"            : ABLATION_NAME,
        "pretokenizer"        : "ByteLevel(add_prefix_space=False, use_regex=False)",
        "decoder"             : "ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)",
        "bpe_level"           : "byte-level",
        "total_vocab"         : TOTAL_VOCAB_SIZE,
        "min_frequency"       : MIN_FREQUENCY,
        "num_shards"          : num_shards,
        "max_chars"           : max_chars,
        "seed"                : seed,
        "threads"             : NUM_THREADS,
        "special_tokens"      : _N_SPECIAL,
        "reserved_slots"      : NUM_RESERVED,
        "language_tags"       : LANGUAGE_TAGS,
        "num_languages"       : len(LANGUAGE_TAGS),
        "dedup_enabled"       : DEDUP_ENABLED,
        "dedup_max_cache"     : DEDUP_MAX_CACHE,
        "notes": [
            "Single-phase ByteLevel BPE — identical to original Phase 2 output",
            "ByteLevel BPE: no BYTE_FALLBACK tokens (handled internally)",
            "No alpha sci tokens: Ġ-encoding makes them unmatchable at inference",
            "ByteLevel(use_regex=False) — no GPT-2 style regex splitting",
        ],
    }

    with stage("Build tokenizer"):
        tok, trainer = build_tokenizer()

    with stage("Train"):
        tok = train_tokenizer(tok, trainer, shards_dir, num_shards, seed,
                              dry_run, max_line_chars=max_chars)

    if dry_run:
        monitor.stop()
        return

    res     = monitor.stop()
    elapsed = time.perf_counter() - t_total

    with stage("Save + patch"):
        save_tokenizer(tok, output_dir,
                       {**meta,
                        "time_total_sec": round(elapsed, 1),
                        "resources"     : res})

    if validate:
        with stage("Validate"):
            validate_tokenizer(output_dir)

    log.info("=" * 65)
    log.info("  DONE  |  %.2f h  |  Peak RSS %.1f GB  |  %s",
             elapsed / 3600, res["peak_process_rss_gb"], output_dir)
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 17. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=(
            "Single-phase ByteLevel BPE tokenizer trainer.\n"
            "Produces output identical to ablation6_superbpe_bloom_v3_dedup Phase 2.\n"
            "Dedup: rolling hash-set eliminates exact-duplicate lines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
── Full training ────────────────────────────────────────────────────────────
  python ablation6_superbpe_bloom_v3_dedup_single.py \
      --shards-dir /path/to/shards \
      --output /path/to/output \
      --seed 42 \
      --max-chars 3347 \
      --validate

── Max-chars guidance ───────────────────────────────────────────────────────
  p50:   88 chars  (keep 50%)
  p75:  600 chars  (keep 75%)
  p90: 3347 chars  (keep 90% — RECOMMENDED)
  p95: 6581 chars  (keep 95%)
        """,
    )
    p.add_argument("--shards-dir",         type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument("--output",             type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num-shards",         type=int,  default=DEFAULT_NUM_SHARDS)
    p.add_argument("--seed",               type=int,  default=DEFAULT_SEED)
    p.add_argument("--monitor-interval",   type=float, default=30.0)
    p.add_argument("--dry-run",            action="store_true")
    p.add_argument("--validate",           action="store_true")
    p.add_argument("--max-chars",          type=int,  default=DEFAULT_MAX_CHARS,
                   help="Skip lines >N chars (default 3347 = p90). 0=disable.")
    p.add_argument("--check-only",         action="store_true")
    args = p.parse_args()

    max_chars = args.max_chars if args.max_chars > 0 else None

    if args.check_only:
        log.info("=" * 65)
        log.info("Single-Phase ByteLevel BPE — Environment Check")
        log.info("=" * 65)
        log.info("  Dedup            : ENABLED  (cache=%d, ~%.0f MB RAM)",
                 DEDUP_MAX_CACHE, DEDUP_MAX_CACHE * 28 / 1_048_576)
        log.info("  Special tokens   : %d", _N_SPECIAL)
        log.info("    Foundational   : %d  %s", len(FOUNDATIONAL), FOUNDATIONAL)
        log.info("    Utility        : %d  %s", len(UTILITY), UTILITY)
        log.info("    Chat           : %d  %s", len(CHAT), CHAT)
        log.info("    Tool use       : %d  %s", len(TOOL_USE), TOOL_USE)
        log.info("    Reasoning      : %d  %s", len(REASONING), REASONING)
        log.info("    Language tags  : %d  %s", len(LANGUAGE_TAGS), LANGUAGE_TAGS)
        log.info("    FIM            : %d  %s", len(FIM), FIM)
        log.info("    Reserved       : %d  (IDs %d–%d)",
                 NUM_RESERVED,
                 len(FOUNDATIONAL+UTILITY+CHAT+TOOL_USE+REASONING+LANGUAGE_TAGS+FIM),
                 _N_SPECIAL - 1)
        log.info("")
        log.info("  Pre-tokenizer    : ByteLevel(add_prefix_space=False, use_regex=False)")
        log.info("  Decoder          : ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)")
        log.info("  Vocab target     : %d", TOTAL_VOCAB_SIZE)
        try:
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            log.info("  HF tokenizers    : OK")
        except Exception as e:
            log.error("  HF tokenizers    : IMPORT FAILED — %s", e)
        log.info("")
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATES!"
        log.info("  Duplicate check  : OK (%d unique special tokens)", _N_SPECIAL)
        log.info("  Sci vocab        : %d non-alpha symbols", _N_SCI)
        log.info("  Whitespace toks  : %s", WHITESPACE_TOKENS)
        log.info("=" * 65)
        return

    run(
        shards_dir        = args.shards_dir,
        output_dir        = args.output,
        num_shards        = args.num_shards,
        seed              = args.seed,
        dry_run           = args.dry_run,
        validate          = args.validate,
        monitor_interval  = args.monitor_interval,
        max_chars         = max_chars,
    )


if __name__ == "__main__":
    main()
