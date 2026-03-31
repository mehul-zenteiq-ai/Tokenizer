#!/usr/bin/env python3
"""
superbpe_bytelevel_dedup.py
============================
SuperBPE  |  BYTE-LEVEL BPE  |  BRAHMAI_PATTERN Phase 1  |  WITH line-level dedup

Key differences from superbpe_with_dedup.py (char-level):
  - Phase 1 uses ByteLevel pretokenizer + ByteLevel.alphabet() as initial alphabet
    (256 raw bytes, GPT-2 / Mistral style).
  - No byte_fallback=True, no <0xXX> tokens — all bytes are native symbols.
  - Decoder is ByteLevelDecoder (not a Sequence of char decoders).
  - Phase 2 removes ByteLevel pretokenizer → SuperBPE merges across Ġ-encoded spaces.
  - Phase 2 uses train_from_iterator (safe with byte-level: alphabet=256, pair table=65K).
  - Phase 2 max_line_chars=1000 keeps per-line token sequences short → low RAM.

Algorithm (matches SuperBPE paper exactly):
  Phase 1: ByteLevel pretokenizer  +  BPE  →  subwords within word-boundaries
  Phase 2: pretokenizer = None     +  BPE  →  superwords across whitespace
                                               e.g.  "Ġby Ġthe Ġway" → single token

Why train_from_iterator is safe here (unlike the old ID-space approach):
  - Byte-level alphabet = 256 symbols → pair table = 256² = 65,536 entries (~50 MB)
  - max_line_chars=1000 → short lines → small word-frequency table
  - The OOM was caused by the 200K token-ID alphabet (200K² = 40B pairs), not iterator

Space encoding (GPT-2 style):
  " hello"  →  "Ġhello"          (leading space encoded as Ġ = \u0120)
  "hello"   →  "hello"            (no prefix)

Usage
-----
  # Check config
  python superbpe_bytelevel_dedup.py --check-only

  # Phase 1 only
  python superbpe_bytelevel_dedup.py \\
      --shards-dir /path/to/shards \\
      --output ./out --phase1-only --validate

  # Phase 2 only
  python superbpe_bytelevel_dedup.py \\
      --shards-dir /path/to/shards \\
      --output ./out \\
      --phase2-only \\
      --phase1-checkpoint ./out/phase1_checkpoint \\
      --validate

  # Full pipeline
  python superbpe_bytelevel_dedup.py \\
      --shards-dir /path/to/shards \\
      --num-shards 5 --seed 42 \\
      --output ./out --validate
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
from typing import Iterator, List, Optional

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

SCRIPT_NAME         = "superbpe_bytelevel_dedup"
TOTAL_VOCAB_SIZE    = 262_144
PHASE1_VOCAB_SIZE   = 200_000
MIN_FREQUENCY       = 2
NUM_THREADS         = 64
MODEL_MAX_LENGTH    = 8_192

DEFAULT_OUTPUT              = Path("./superbpe_bytelevel_out")
DEFAULT_SHARD_DIR           = Path("/path/to/shards")
DEFAULT_NUM_SHARDS: Optional[int] = None
DEFAULT_SEED                = 42
DEFAULT_PHASE1_VOCAB        = PHASE1_VOCAB_SIZE

# ── Phase 2 max line chars ─────────────────────────────────────────────────────
# 1000 chars keeps per-line token sequences short.
# With byte-level alphabet (256 symbols), pair table = 256² = 65K entries (~50 MB).
# train_from_iterator is safe at this setting.
DEFAULT_PHASE2_MAX_CHARS    = 1_000

DEFAULT_PHASE2_NUM_SHARDS   = None

# ── Dedup ─────────────────────────────────────────────────────────────────────
DEDUP_ENABLED    = True
DEDUP_MAX_CACHE  = 5_000_000   # rolling hash-set cap

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
    "<mask>",                 # 4
    "[multimodal]",           # 5
    "[@BOS@]",                # 6
    "<|image_soft_token|>",   # 7
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
    "<lang_hin>",   # 28
    "<lang_tam>",   # 29
    "<lang_tel>",   # 30
    "<lang_kan>",   # 31
    "<lang_mal>",   # 32
    "<lang_mar>",   # 33
    "<lang_guj>",   # 34
    "<lang_ben>",   # 35
    "<lang_pan>",   # 36
    "<lang_ory>",   # 37
    "<lang_urd>",   # 38
    "<lang_npi>",   # 39
    "<lang_pus>",   # 40
    "<lang_sin>",   # 41
    "<lang_mya>",   # 42
    "<lang_fas>",   # 43
    "<lang_bod>",   # 44
    "<lang_dzo>",   # 45
    "<lang_eng>",   # 46
    "<lang_deu>",   # 47
    "<lang_fra>",   # 48
    "<lang_rus>",   # 49
    "<lang_cmn>",   # 50
    "<lang_jpn>",   # 51
    "<lang_kor>",   # 52
]

assert len(LANGUAGE_TAGS) == 25, f"Expected 25 LANGUAGE_TAGS, got {len(LANGUAGE_TAGS)}"

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

assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATE SPECIAL TOKENS"

# ── Named references ──────────────────────────────────────────────────────────
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
# 3.  FIXED SCIENTIFIC VOCABULARY
#     NOTE: No <0xXX> byte-fallback tokens — byte-level handles all bytes natively.
# ══════════════════════════════════════════════════════════════════════════════

FIXED_SCI_VOCAB = [
    "=", "+", "-", "*", "/", "^", "√", "∑", "∏", "!",
    "<", ">", "≤", "≥", "≠",
    "∂ₜ", "∂ₓ", "∂ᵧ", "∂_z",
    "∂ₜₜ", "∂ₓₓ", "∂ᵧᵧ", "∂_zz",
    "∂_xy", "∂_xz", "∂_yz",
    "∇", "∇²", "∇·", "∇×",
    "∫", "∮", "∂",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "log", "log2", "log10", "sqrt", "abs", "sign",
    "sinh", "cosh", "tanh", "relu", "sigmoid",
    "dot", "inner", "outer", "cross", "tr", "det", "inv",
    "sym", "skew", "dev", "grad", "div", "curl",
    "dx", "ds", "dS", "jump", "avg",
    "TestFunction", "TrialFunction", "Coefficient", "Constant",
    "Lagrange", "DG", "BDM", "RT", "CR",
    "N1curl", "N2curl", "N1div", "N2div", "Bubble", "Quadrature",
    "interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron",
    "dirichlet", "neumann", "periodic", "robin", "cauchy", "none_bc",
    "dim_M", "dim_L", "dim_T", "dim_I", "dim_θ", "dim_N", "dim_J",
    "atom_C", "atom_N", "atom_O", "atom_H", "atom_S", "atom_P",
    "atom_F", "atom_Cl", "atom_Br", "atom_I",
    "bond_single", "bond_double", "bond_triple", "bond_aromatic",
    "ring_open", "ring_close",
]
_N_SCI = len(FIXED_SCI_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  WHITESPACE TOKENS
# ══════════════════════════════════════════════════════════════════════════════

WHITESPACE_TOKENS = ["\n", "\n\n", "\r\n", "\t", "\v", "\f"]
_N_WHITESPACE = len(WHITESPACE_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BRAHMAI PRETOKENIZER PATTERN
#     Used in Phase 1 ONLY. Phase 2 sets pretokenizer=None (SuperBPE).
# ══════════════════════════════════════════════════════════════════════════════

BRAHMAI_PATTERN = (
    r"[ ]?\\[a-zA-Z]+\*?"
    r"|\\[^a-zA-Z\s]"
    r"|[ ]?[{}\[\]^_$&#%~]"
    r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[ ]?\p{Devanagari}+"
    r"|[ ]?\p{Tamil}+"
    r"|[ ]?\p{Telugu}+"
    r"|[ ]?\p{Kannada}+"
    r"|[ ]?\p{Malayalam}+"
    r"|[ ]?\p{Gujarati}+"
    r"|[ ]?\p{Bengali}+"
    r"|[ ]?\p{Gurmukhi}+"
    r"|[ ]?\p{Oriya}+"
    r"|[ ]?\p{Greek}+"
    r"|[\u2200-\u22FF\u2100-\u214F\u27C0-\u27EF\u2980-\u29FF\u2A00-\u2AFF]+"
    r"|[ ]?\p{Latin}+"
    r"|[ ]?\p{L}+"
    r"|[ ]?\p{N}+"
    r"|[ ]?[^\s\p{L}\p{N}\\{}\[\]^_$&#%~]+"
    r"|\r?\n+"
    r"|\s+"
)

V3_FALLBACK_PATTERN = r"[ \t]*[^ \t\n]+|[ \t]*\n"


def get_pretokenizer_pattern():
    try:
        import regex as _regex
        _regex.compile(r"\p{Tamil}+")
        return BRAHMAI_PATTERN, "BRAHMAI"
    except (ImportError, Exception):
        log.warning("'regex' unavailable — falling back to V3_FALLBACK. "
                    "Install: pip install regex --break-system-packages")
        return V3_FALLBACK_PATTERN, "V3_FALLBACK"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RESOURCE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    def __init__(self, interval_sec=30.0):
        self._interval = interval_sec
        self._stop     = threading.Event()
        self._thread   = None
        self._peak_rss = 0.0
        self._peak_pct = 0.0

    @staticmethod
    def _meminfo():
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        info[p[0].rstrip(":")] = int(p[1])
        except Exception:
            pass
        return info

    @staticmethod
    def _rss_kb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def _snap(self):
        m     = self._meminfo()
        tot   = m.get("MemTotal", 1)
        avail = m.get("MemAvailable", m.get("MemFree", 0))
        used  = tot - avail
        swap  = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
        rss   = self._rss_kb()
        G     = 1 / 1_048_576
        pct   = used / tot * 100
        self._peak_rss = max(self._peak_rss, rss * G)
        self._peak_pct = max(self._peak_pct, pct)
        warn  = " ⚠⚠CRIT" if pct >= 88 else (" ⚠WARN" if pct >= 80 else "")
        return (f"  [monitor] RAM {used*G:.1f}/{tot*G:.1f} GB "
                f"({pct:.1f}%{warn})  Swap {swap*G:.1f} GB  RSS {rss*G:.1f} GB")

    def _run(self):
        while not self._stop.wait(self._interval):
            print(self._snap(), flush=True)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def log_now(self, label=""):
        if label:
            log.info("[monitor] %s", label)
        print(self._snap(), flush=True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {"peak_rss_gb": round(self._peak_rss, 2),
                "peak_ram_pct": round(self._peak_pct, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SHARD DISCOVERY + SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def discover_shards(shard_dir: Path) -> List[Path]:
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
) -> List[Path]:
    available = discover_shards(shard_dir)
    if num_shards is None or num_shards >= len(available):
        selected = available
    else:
        rng      = random.Random(seed)
        selected = sorted(rng.sample(available, num_shards))
    log.info("Shards: %d available → %d selected", len(available), len(selected))
    for p in selected:
        log.info("  %-50s  %.2f GB", p.name, p.stat().st_size / 1e9)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CORPUS ITERATOR
#     Shared by Phase 1 and Phase 2.
#     Phase 2 passes max_line_chars=1000 to keep lines short.
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB  = 500
MIN_DOC_CHARS = 20


def corpus_iterator(
    shard_dir      : Path,
    num_shards     : Optional[int] = None,
    seed           : int            = DEFAULT_SEED,
    max_line_chars : Optional[int]  = None,
    dedup          : bool           = DEDUP_ENABLED,
    dedup_max_cache: int            = DEDUP_MAX_CACHE,
) -> Iterator[str]:
    """
    Streaming corpus iterator with optional dedup and line length cap.

    Phase 1: max_line_chars=None  (no limit)
    Phase 2: max_line_chars=1000  (short lines → small word-freq table → low RAM)
    """
    paths         = select_shards(shard_dir, num_shards, seed)
    total_bytes   = 0
    total_yielded = 0
    total_short   = 0
    total_long    = 0
    total_dedup   = 0
    last_log      = 0
    t0            = time.perf_counter()
    seen          : set = set()

    if max_line_chars:
        log.info("  max_line_chars=%d  (lines longer than this are skipped)",
                 max_line_chars)
    log.info("  dedup=%s  cache_size=%d", dedup, dedup_max_cache)

    pbar = tqdm(desc="lines", unit=" lines", smoothing=0.1, mininterval=5.0)

    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n\r")

                if len(line) < MIN_DOC_CHARS:
                    total_short += 1
                    continue
                if max_line_chars and len(line) > max_line_chars:
                    total_long += 1
                    continue
                if dedup:
                    h = hash(line)
                    if h in seen:
                        total_dedup += 1
                        continue
                    seen.add(h)
                    if len(seen) > dedup_max_cache:
                        seen.clear()

                b = len(line.encode("utf-8", errors="replace"))
                total_bytes   += b
                total_yielded += 1
                pbar.update(1)
                yield line

                if total_bytes - last_log >= LOG_EVERY_MB * 1_048_576:
                    elapsed = time.perf_counter() - t0
                    log.info(
                        "  %.1f GB | %d lines | dedup_skip=%d | long_skip=%d | %.1f MB/s",
                        total_bytes / 1_073_741_824, total_yielded,
                        total_dedup, total_long,
                        (total_bytes / 1_048_576) / elapsed if elapsed else 0)
                    last_log = total_bytes

    pbar.close()
    log.info(
        "Corpus done: %.2f GB | %d lines | short=%d | long=%d | dedup=%d | %.1f s",
        total_bytes / 1_073_741_824, total_yielded,
        total_short, total_long, total_dedup,
        time.perf_counter() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ADDED TOKEN BUILDER
#     NOTE: NO <0xXX> byte-fallback tokens.
#     ByteLevel BPE handles all 256 byte values natively via initial_alphabet.
# ══════════════════════════════════════════════════════════════════════════════

def _is_alpha_token(content: str) -> bool:
    return content.replace("_", "").isalpha()


def _build_added_tokens():
    from tokenizers import AddedToken

    special_objs = [
        AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS
    ]
    sci_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=_is_alpha_token(t), lstrip=False, rstrip=False)
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

    n_alpha = sum(1 for t in FIXED_SCI_VOCAB if _is_alpha_token(t))
    log.info("Added tokens: %d special + %d sci (%d alpha→single_word) "
             "+ %d whitespace = %d total",
             len(special_objs), _N_SCI, n_alpha, _N_WHITESPACE, len(deduped))
    log.info("NOTE: No <0xXX> tokens — byte-level handles all bytes natively")
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# 10. POST-SAVE JSON PATCH
# ══════════════════════════════════════════════════════════════════════════════

def _patch_tokenizer_json(tok_path: Path) -> None:
    with open(tok_path, encoding="utf-8") as f:
        state = json.load(f)

    # Ensure decoder is ByteLevel
    current_decoder = state.get("decoder")
    if not (isinstance(current_decoder, dict)
            and current_decoder.get("type") == "ByteLevel"):
        state["decoder"] = {"type": "ByteLevel", "add_prefix_space": False,
                             "trim_offsets": True, "use_regex": False}
        log.info("  [patch] decoder set to ByteLevel (was: %s)", current_decoder)

    sci_set = set(FIXED_SCI_VOCAB)
    ws_set  = set(WHITESPACE_TOKENS)
    n_sci = n_ws = n_sword = 0

    for entry in state.get("added_tokens", []):
        content = entry.get("content", "")
        if content in sci_set:
            entry["special"]     = False
            entry["single_word"] = _is_alpha_token(content)
            n_sci += 1
            if _is_alpha_token(content):
                n_sword += 1
        if content in ws_set:
            entry["special"]    = False
            entry["normalized"] = False
            n_ws += 1

    vocab    = state.get("model", {}).get("vocab", {})
    existing = {e["content"] for e in state.get("added_tokens", [])}
    injected = 0
    for ws in WHITESPACE_TOKENS:
        if ws not in existing and ws in vocab:
            state["added_tokens"].append({
                "id": vocab[ws], "content": ws,
                "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": False,
            })
            injected += 1

    log.info("  [patch] sci=%d (single_word=%d) | ws=%d | injected=%d",
             n_sci, n_sword, n_ws, injected)

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("  [patch] saved → %s (%d bytes)", tok_path, tok_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# 11. PHASE 1 — BYTE-LEVEL BPE + BRAHMAI_PATTERN
#
#  ByteLevel pretokenizer encodes spaces as Ġ (U+0120) before splitting,
#  then BRAHMAI_PATTERN splits on script/word boundaries.
#  initial_alphabet=ByteLevel.alphabet() seeds the vocab with all 256 bytes.
# ══════════════════════════════════════════════════════════════════════════════

def build_phase1_tokenizer(phase1_vocab_size: int, pattern: str, pattern_name: str):
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.pre_tokenizer = Sequence([
        ByteLevel(add_prefix_space=False),
        Split(pattern=Regex(pattern), behavior="isolated", invert=True),
    ])

    deduped = _build_added_tokens()
    trainer = BpeTrainer(
        vocab_size       = phase1_vocab_size,
        min_frequency    = MIN_FREQUENCY,
        special_tokens   = deduped,
        show_progress    = True,
        initial_alphabet = ByteLevel.alphabet(),
    )

    log.info("Phase 1: BYTE-LEVEL + %s | vocab=%d", pattern_name, phase1_vocab_size)
    log.info("  Pretokenizer: ByteLevel → %s (chained)", pattern_name)
    log.info("  initial_alphabet: ByteLevel.alphabet() = 256 byte symbols")
    log.info("  No unk_token, no byte_fallback, no <0xXX> tokens needed")
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 12. PHASE 2 — SuperBPE: remove pretokenizer entirely
#
#  Core SuperBPE operation:
#    Phase 1 pretokenizer = ByteLevel + BRAHMAI_PATTERN
#    Phase 2 pretokenizer = None  → BPE merges across Ġ-encoded spaces
#
#  Why train_from_iterator is safe here:
#    - Alphabet = 256 bytes → pair table = 256² = 65,536 entries (~50 MB, tiny)
#    - max_line_chars=1000 → word-frequency table stays small
#    - The previous OOM was from 200K token-ID alphabet (200K² = 40B pairs)
#      NOT from train_from_iterator itself.
# ══════════════════════════════════════════════════════════════════════════════

def build_phase2_tokenizer(phase1_checkpoint_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.trainers import BpeTrainer

    tok_path = phase1_checkpoint_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {tok_path}\n"
            "Run Phase 1 first (--phase1-only)."
        )

    log.info("Loading Phase 1 checkpoint: %s", tok_path)
    with open(tok_path, encoding="utf-8") as f:
        p1_state = json.load(f)

    p1_vocab_size   = len(p1_state.get("model", {}).get("vocab", {}))
    p1_merges       = len(p1_state.get("model", {}).get("merges", []))
    superword_slots = TOTAL_VOCAB_SIZE - p1_vocab_size
    log.info("  Phase 1 vocab=%d  merges=%d", p1_vocab_size, p1_merges)
    log.info("  Superword slots: %d", superword_slots)

    # ── THE SUPERBPE STEP ─────────────────────────────────────────────────────
    # Remove pretokenizer → BPE merges freely across Ġ-encoded word boundaries.
    # Decoder (ByteLevel) is preserved from Phase 1 state unchanged.
    p2_state                  = dict(p1_state)
    p2_state["pre_tokenizer"] = None

    tokenizer = Tokenizer.from_str(json.dumps(p2_state))
    deduped   = _build_added_tokens()
    trainer   = BpeTrainer(
        vocab_size    = TOTAL_VOCAB_SIZE,
        min_frequency = MIN_FREQUENCY,
        special_tokens= deduped,
        show_progress = True,
        # Do NOT pass initial_alphabet — extending Phase 1 vocab, not restarting
    )

    log.info("Phase 2: pretokenizer=None (SuperBPE) | target=%d", TOTAL_VOCAB_SIZE)
    log.info("  Alphabet: 256 bytes → pair table = 65,536 entries (~50 MB)")
    log.info("  train_from_iterator: SAFE at this alphabet size")
    return tokenizer, trainer, p1_vocab_size


# ══════════════════════════════════════════════════════════════════════════════
# 13. TRAINING RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def train_phase1(tokenizer, trainer, shard_dir, num_shards, seed):
    """Phase 1: no line length limit, iterator-based."""
    log.info("Training Phase 1 (byte-level BPE, no line limit) ...")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(shard_dir, num_shards, seed,
                                  max_line_chars=None),
        trainer=trainer,
        length=None,
    )
    elapsed = time.perf_counter() - t0
    actual  = tokenizer.get_vocab_size()
    log.info("Phase 1 done in %.1f min. Vocab: %d / %d",
             elapsed / 60, actual, trainer.vocab_size)
    if actual < trainer.vocab_size * 0.95:
        log.warning("Phase 1 vocab (%d) < 95%% of target — add more shards.", actual)
    return tokenizer


def train_phase2(tokenizer, trainer, shard_dir, num_shards, seed, max_line_chars):
    """
    Phase 2 SuperBPE: train_from_iterator with max_line_chars=1000.

    Memory analysis:
      - Alphabet = 256 bytes → pair table = 256² = 65,536 entries (~50 MB)
      - max_line_chars=1000 → each line ≤ ~250 byte-level tokens
      - Word-freq table: unique short sequences → manageable RAM
      - Rust buffers the word-freq table only → plateaus after corpus load
      - Expected peak RSS: 30–80 GB (vs 500+ GB with ID-space approach)
    """
    log.info("Training Phase 2 SuperBPE (train_from_iterator) ...")
    log.info("  max_line_chars : %d", max_line_chars)
    log.info("  Alphabet size  : 256 bytes → pair table = 65,536 entries (~50 MB)")
    log.info("  Expected RAM   : 30–80 GB plateau (safe)")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(shard_dir, num_shards, seed,
                                  max_line_chars=max_line_chars),
        trainer=trainer,
        length=None,
    )
    elapsed = time.perf_counter() - t0
    actual  = tokenizer.get_vocab_size()
    log.info("Phase 2 done in %.1f min. Vocab: %d / %d",
             elapsed / 60, actual, trainer.vocab_size)
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 14. SAVE + PATCH + HF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_tokenizer(tokenizer, save_dir: Path, meta: dict) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    tok_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    log.info("Saved tokenizer.json (%d bytes)", tok_path.stat().st_size)

    log.info("Applying patches (ByteLevel decoder + sci/ws flags)...")
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
        "mask_token"                  : "<mask>",
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
                "rstrip": False, "single_word": _is_alpha_token(t), "special": False,
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
        "mask_token": {"content": "<mask>", "lstrip": False, "normalized": False,
                       "rstrip": False, "single_word": False},
        "additional_special_tokens": UTILITY + CHAT + TOOL_USE + REASONING
                                     + LANGUAGE_TAGS + FIM,
    }
    with open(save_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(stm, f, indent=2, ensure_ascii=False)

    with open(save_dir / "training_metadata.json", "w") as f:
        json.dump({**meta, "vocab_size_actual": len(sv),
                   "merges_total": len(merges)}, f, indent=2)

    log.info("All files saved → %s", save_dir)
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# 15. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tokenizer(load_dir: Path, label: str = "final") -> None:
    log.info("=" * 65)
    log.info("VALIDATION [%s]  %s", label, load_dir)

    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(str(load_dir))
        hf  = True
        log.info("Loaded via PreTrainedTokenizerFast")
    except Exception as e:
        log.warning("HF load failed (%s) — using raw tokenizer", e)
        from tokenizers import Tokenizer as _Tok
        tok = _Tok.from_file(str(load_dir / "tokenizer.json"))
        hf  = False

    vocab = tok.get_vocab()
    log.info("Vocab size: %d", len(vocab))

    byte_tokens_present = sum(
        1 for i in range(256) if f"<0x{i:02X}>" in vocab
    )
    if byte_tokens_present == 0:
        log.info("  Byte-fallback <0xXX> tokens: NONE ✓ (correct for byte-level)")
    else:
        log.warning("  Unexpected <0xXX> tokens present: %d", byte_tokens_present)

    log.info("Special token IDs:")
    expected = FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING + LANGUAGE_TAGS + FIM
    mismatches = 0
    for eid, tstr in enumerate(expected):
        aid = vocab.get(tstr, -1)
        if aid != eid:
            log.warning("  MISMATCH: %r  expected=%d  actual=%d", tstr, eid, aid)
            mismatches += 1
    if mismatches == 0:
        log.info("  All %d special token IDs correct ✓", len(expected))

    log.info("Stop tokens:")
    for st in STOP_TOKENS:
        sid = vocab.get(st, -1)
        log.info("  %-25r  id=%d  %s", st, sid, "✓" if sid >= 0 else "✗ MISSING")

    tests = [
        ("LaTeX",        r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"),
        ("ASCII math",   "d/dt(T) = alpha * laplacian(T)"),
        ("Unicode math", "∂u/∂t = α∇²u"),
        ("Hindi",        "नमस्ते दुनिया"),
        ("Tamil",        "போகிறார்கள்"),
        ("Telugu",       "నమస్కారం"),
        ("Kannada",      "ನಮಸ್ಕಾರ"),
        ("Code",         "    def foo():\n        return x + 1"),
        ("Newlines",     "x = 1\ny = 2\nz = x + y"),
        ("sin-safety",   "single sinusoidal"),
        ("tr-safety",    "tr(A) is the trace"),
        ("Think tag",    "<think>reasoning here</think>"),
        ("Nothink tag",  "<|nothink|>answer directly"),
        ("FIM",          "<|fim_prefix|>def f():<|fim_suffix|>    pass<|fim_middle|>"),
        ("Lang hin",     "<lang_hin>नमस्ते</lang_hin>"),
        ("Lang cmn",     "<lang_cmn>你好世界</lang_cmn>"),
        ("Tool call",    "<tool_call><arg_key>fn</arg_key></tool_call>"),
        ("Stop eos",     "<eos>"),
        ("Stop eot",     "<|end_of_turn|>"),
        ("Space prefix", " hello world"),
        ("No prefix",    "hello world"),
        ("Mixed space",  "the quick brown fox"),
    ]

    log.info("Round-trip tests:")
    all_ok = True
    for lbl, text in tests:
        ids = tok.encode(text) if hf else tok.encode(text).ids
        # Always skip_special_tokens=False so special tokens round-trip correctly
        dec = tok.decode(ids, skip_special_tokens=False)
        ok  = (text == dec)
        if not ok:
            all_ok = False
        log.info("  [%-18s] %3d toks  %s",
                 lbl, len(ids), "✓" if ok else f"✗  got={dec[:60]!r}")

    if label == "final":
        log.info("Superword checks (byte-level, Ġ-encoded spaces):")
        for sw in ["Ġthe", "Ġof", "Ġin", "Ġto", "Ġand",
                   "Ġby", "Ġfor", "Ġis", "Ġit", "Ġwith"]:
            log.info("  %-20r  %s", sw,
                     "IN VOCAB ✓" if sw in vocab else "not yet (normal for small corpus)")

    log.info("VALIDATION %s", "ALL PASSED ✓" if all_ok else "SOME FAILED ✗")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 16. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    shards_dir        : Path,
    output_dir        : Path,
    num_shards        : Optional[int],
    seed              : int,
    validate          : bool,
    monitor_interval  : float,
    phase1_vocab_size : int,
    phase1_only       : bool,
    phase2_only       : bool,
    phase1_checkpoint : Optional[Path],
    phase2_num_shards : Optional[int],
    phase2_max_chars  : int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase1_checkpoint is None:
        phase1_checkpoint = output_dir / "phase1_checkpoint"

    p2_shards = phase2_num_shards if phase2_num_shards is not None else num_shards
    pattern, pattern_name = get_pretokenizer_pattern()

    log.info("=" * 65)
    log.info("  %s", SCRIPT_NAME.upper())
    log.info("=" * 65)
    log.info("  BPE type         : BYTE-LEVEL (GPT-2 style, Ġ-encoded spaces)")
    log.info("  Phase 1 preток   : ByteLevel + %s", pattern_name)
    log.info("  Phase 2 preток   : None (SuperBPE — merges across whitespace)")
    log.info("  Phase 2 training : train_from_iterator (safe: alphabet=256)")
    log.info("  Phase 2 max_chars: %d", phase2_max_chars)
    log.info("  Byte fallback    : DISABLED (initial_alphabet covers all 256 bytes)")
    log.info("  <0xXX> tokens    : NONE")
    log.info("  Dedup            : %s (cache=%d)", DEDUP_ENABLED, DEDUP_MAX_CACHE)
    log.info("  Phase 1 vocab    : %d", phase1_vocab_size)
    log.info("  Superword slots  : %d", TOTAL_VOCAB_SIZE - phase1_vocab_size)
    log.info("  Total vocab      : %d", TOTAL_VOCAB_SIZE)
    log.info("  P1 shards        : %s", num_shards or "all")
    log.info("  P2 shards        : %s", p2_shards or "all")
    log.info("  Output           : %s", output_dir)
    log.info("=" * 65)

    monitor = ResourceMonitor(interval_sec=monitor_interval)
    monitor.start()
    monitor.log_now("BASELINE")
    t_total = time.perf_counter()

    meta_base = {
        "script"            : SCRIPT_NAME,
        "bpe_type"          : "byte-level",
        "phase1_pattern"    : pattern_name,
        "phase2_pattern"    : "none (SuperBPE)",
        "phase2_training"   : "train_from_iterator",
        "phase1_vocab"      : phase1_vocab_size,
        "total_vocab_target": TOTAL_VOCAB_SIZE,
        "superword_slots"   : TOTAL_VOCAB_SIZE - phase1_vocab_size,
        "dedup_enabled"     : DEDUP_ENABLED,
        "dedup_max_cache"   : DEDUP_MAX_CACHE,
        "seed"              : seed,
        "p2_max_chars"      : phase2_max_chars,
        "p1_num_shards"     : num_shards,
        "p2_num_shards"     : p2_shards,
        "byte_fallback"     : False,
        "initial_alphabet"  : "ByteLevel.alphabet() — 256 symbols",
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not phase2_only:
        with stage("Phase 1: Build byte-level tokenizer"):
            tok1, tr1 = build_phase1_tokenizer(
                phase1_vocab_size, pattern, pattern_name)

        with stage("Phase 1: Train"):
            tok1 = train_phase1(tok1, tr1, shards_dir, num_shards, seed)

        with stage("Phase 1: Save checkpoint"):
            save_tokenizer(tok1, phase1_checkpoint,
                           {**meta_base, "phase": "phase1"})

        if validate:
            with stage("Phase 1: Validate"):
                validate_tokenizer(phase1_checkpoint, label="phase1")

        if phase1_only:
            res = monitor.stop()
            log.info("Phase 1 complete. %.2f h  RSS %.1f GB",
                     (time.perf_counter() - t_total) / 3600, res["peak_rss_gb"])
            return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    with stage("Phase 2: Build SuperBPE tokenizer (pretokenizer=None)"):
        tok2, tr2, p1_vocab_size = build_phase2_tokenizer(phase1_checkpoint)

    with stage(f"Phase 2: Train SuperBPE via train_from_iterator "
               f"(max_chars={phase2_max_chars})"):
        tok2 = train_phase2(tok2, tr2, shards_dir, p2_shards,
                            seed, phase2_max_chars)

    with stage("Phase 2: Save final tokenizer"):
        elapsed = time.perf_counter() - t_total
        res     = monitor.stop()
        save_tokenizer(
            tok2, output_dir,
            {**meta_base,
             "phase"          : "final",
             "p1_vocab_actual": p1_vocab_size,
             "time_total_sec" : round(elapsed, 1),
             "resources"      : res},
        )

    if validate:
        with stage("Final: Validate"):
            validate_tokenizer(output_dir, label="final")

    log.info("=" * 65)
    log.info("  DONE  |  %.2f h  |  Peak RSS %.1f GB  |  %s",
             elapsed / 3600, res["peak_rss_gb"], output_dir)
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 17. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="SuperBPE — byte-level + BRAHMAI_PATTERN + train_from_iterator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python superbpe_bytelevel_dedup.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_bytelevel_out --validate

  # Phase 1 only
  python superbpe_bytelevel_dedup.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_bytelevel_out --phase1-only --validate

  # Phase 2 only (reuse existing Phase 1 checkpoint)
  python superbpe_bytelevel_dedup.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_bytelevel_out \\
      --phase2-only \\
      --phase1-checkpoint ~/superbpe_bytelevel_out/phase1_checkpoint \\
      --validate

  # Check config
  python superbpe_bytelevel_dedup.py --check-only
        """,
    )
    p.add_argument("--shards-dir",        type=Path,  default=DEFAULT_SHARD_DIR)
    p.add_argument("--output",            type=Path,  default=DEFAULT_OUTPUT)
    p.add_argument("--num-shards",        type=int,   default=DEFAULT_NUM_SHARDS)
    p.add_argument("--seed",              type=int,   default=DEFAULT_SEED)
    p.add_argument("--monitor-interval",  type=float, default=30.0)
    p.add_argument("--validate",          action="store_true")
    p.add_argument("--phase1-vocab",      type=int,   default=DEFAULT_PHASE1_VOCAB)
    p.add_argument("--phase1-only",       action="store_true")
    p.add_argument("--phase2-only",       action="store_true")
    p.add_argument("--phase1-checkpoint", type=Path,  default=None)
    p.add_argument("--phase2-shards",     type=int,   default=DEFAULT_PHASE2_NUM_SHARDS)
    p.add_argument("--phase2-max-chars",  type=int,   default=DEFAULT_PHASE2_MAX_CHARS,
                   help=f"Skip lines longer than N chars in Phase 2 "
                        f"(default {DEFAULT_PHASE2_MAX_CHARS}). 0 = no limit.")
    p.add_argument("--check-only",        action="store_true")

    args = p.parse_args()

    if args.phase1_only and args.phase2_only:
        p.error("--phase1-only and --phase2-only are mutually exclusive.")

    phase2_max_chars = args.phase2_max_chars if args.phase2_max_chars > 0 else None

    if args.check_only:
        log.info("=" * 65)
        log.info("%s — Check Mode", SCRIPT_NAME)
        log.info("=" * 65)
        log.info("  BPE type          : BYTE-LEVEL")
        log.info("  Phase 1 preток    : ByteLevel + BRAHMAI_PATTERN (chained)")
        log.info("  Phase 2 preток    : None (SuperBPE)")
        log.info("  Phase 2 training  : train_from_iterator")
        log.info("  Phase 2 max_chars : %d", DEFAULT_PHASE2_MAX_CHARS)
        log.info("  initial_alphabet  : ByteLevel.alphabet() = 256 bytes")
        log.info("  byte_fallback     : False")
        log.info("  <0xXX> tokens     : NONE")
        log.info("  Dedup             : %s (cache=%d)", DEDUP_ENABLED, DEDUP_MAX_CACHE)
        log.info("  Special tokens    : %d (%d reserved)", _N_SPECIAL, NUM_RESERVED)
        log.info("  RAYON threads     : %d", NUM_THREADS)
        log.info("  CPU count         : %s", os.cpu_count())
        pattern, pname = get_pretokenizer_pattern()
        log.info("  Active pattern    : %s", pname)
        try:
            from tokenizers.pre_tokenizers import ByteLevel as BL
            alpha = BL.alphabet()
            log.info("  ByteLevel.alphabet() size: %d symbols", len(alpha))
        except Exception as e:
            log.error("  ByteLevel.alphabet() error: %s", e)
        try:
            from tokenizers import Regex
            Regex(BRAHMAI_PATTERN)
            log.info("  BRAHMAI_PATTERN   : OK")
        except Exception as e:
            log.error("  BRAHMAI_PATTERN   : INVALID — %s", e)
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATES FOUND"
        log.info("  Duplicate check   : OK")
        log.info("=" * 65)
        return

    run(
        shards_dir        = args.shards_dir,
        output_dir        = args.output,
        num_shards        = args.num_shards,
        seed              = args.seed,
        validate          = args.validate,
        monitor_interval  = args.monitor_interval,
        phase1_vocab_size = args.phase1_vocab,
        phase1_only       = args.phase1_only,
        phase2_only       = args.phase2_only,
        phase1_checkpoint = args.phase1_checkpoint,
        phase2_num_shards = args.phase2_shards,
        phase2_max_chars  = phase2_max_chars,
    )


if __name__ == "__main__":
    main()