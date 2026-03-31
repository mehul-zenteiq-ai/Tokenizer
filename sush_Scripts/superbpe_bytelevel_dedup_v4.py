#!/usr/bin/env python3
"""
superbpe_bytelevel_dedup_v4.py
================================
SuperBPE  |  BYTE-LEVEL BPE  |  BRAHMAI+VERBOSE PATTERN  |  WITH line-level dedup

This is the definitive version. It is built on the v1 architecture and absorbs
every confirmed improvement from v2 and v3, plus one new fix derived from the
PATTERN_VERBOSE analysis.

WHAT V4 IS (relative to v1)
=============================

[v4-A]  SINGLE-DIGIT NUMBER RULE  (new — the key Indic fix)
    v1/v2/v3 all used \\p{N}{1,3}: numbers are pretokenized into 1-3-digit chunks.
    BPE must then spend ~1,890 Phase 1 merges building 2-digit and 3-digit number
    tokens from individual digits.  Those merges cannot go to Indic syllables.
    With 2% Indic in the corpus, every merge counts.

    v4 uses [ ]?[0-9] (single digit, from PATTERN_VERBOSE):
      - Each ASCII digit becomes its own pretokenized piece.
      - BPE cannot merge digits in Phase 1 (piece boundaries block it).
      - Digits 0-9 are already in ByteLevel.alphabet() — zero merge cost.
      - The freed ~1,890 merges go directly to Devanagari/Tamil/Bengali syllables.
      - Phase 2 (no pretokenizer) freely learns frequent numbers like "1947",
        "2024", "128" as cross-word superwords if they are corpus-frequent.
      - Indic-script digits (Devanagari ०-९, Tamil ௦-௯, etc.) are unaffected —
        they are captured by the \\p{Script}+ branch which appears earlier in the
        alternation and takes priority.

    Budget impact: ~270 additional Indic syllable tokens in Phase 1 vocab.

[v4-B]  CORRECTED PRETOKENIZER ORDER  (from v3)
    v1: ByteLevel(use_regex=True) → Split(BRAHMAI_PATTERN)
         ByteLevel encodes spaces to Ġ FIRST. Then BRAHMAI runs on Ġ-encoded text.
         The [ ]? prefix in every BRAHMAI branch never matches because raw spaces
         are already gone. Every Indic word gets a bare Ġ prepended as a separate
         piece rather than being captured with its leading space. -43% efficiency.

    v4: Split(BRAHMAI_PATTERN) → ByteLevel(use_regex=False)
         Split runs on raw text: [ ]?\\p{Devanagari}+ correctly captures " क".
         ByteLevel then only byte-encodes the resulting pieces — no word-split.

[v4-C]  FIXED CODE VOCABULARY  (from v3)
    v1 had no code operators. Code tokens like ==, !=, ->, => were absent.
    v4 seeds them as AddedTokens (operators and dunders only — no long method-chain
    bindings which create merge barriers).

[v4-D]  EXPANDED FIXED_SCI_VOCAB  (user-specified, extends v1's ~80 tokens)
    Adds Greek letter LaTeX commands (\\alpha, \\beta, ...) and structural LaTeX
    bigrams (_{, ^{, ^2, ...) and core physics units.
    NO cmd+brace forms (\\frac{, \\begin{equation}) — those create merge barriers.

[v4-E]  PHASE 2 Ġ PRE-ENCODING + RUN COLLAPSE  (from v3)
    v1 Phase 2 trained on raw text → tokens like "the way" (literal space).
    v4 pre-encodes with GPT-2 byte mapping → tokens like "theĠway" (Ġ-consistent).
    Ġ-runs of 3+ collapse to ĠĠ to prevent indented code lines from producing
    tokens like "list:ĠĠĠĠĠĠĠĠĠ".

[v4-F]  TWO-BUCKET ROLLING DEDUP  (from v3)
    v1 used a single cache that cliff-cleared at 5M lines.
    v4 uses two alternating buckets — coverage stays ≥50% at all times.

[v4-G]  BOILERPLATE FILTER  (from v3)
    Drops homework-site / cookie-banner / navigation-noise lines from Phase 2.

[v4-H]  HF-COMPATIBLE tokenizer_config.json  (critical bug fix from v3)
    v1 was missing "extra_special_tokens": {}.
    transformers ≥4.40 calls .keys() on that field and crashes if it's a list or
    absent. v4 always writes it as an empty dict.

[v4-I]  NUM_RESERVED = 1,024  (from v3)
    v1 wasted 8,192 IDs on reserved tokens. Reducing to 1,024 frees 7,168 IDs
    for actual vocabulary — all absorbed into Phase 1 BPE merge budget.

[v4-J]  INFERENCE BRIDGE  (from v3)
    Phase 2 training uses pretokenizer=None (cross-word merges).
    Saved inference JSON restores ByteLevel(use_regex=False) so raw Unicode input
    is byte-encoded before hitting the BPE model. Without this, spaces and Indic
    chars are silently dropped at inference.

WHAT IS UNCHANGED FROM V1
==========================
  - PHASE1_VOCAB_SIZE = 200,000
  - TOTAL_VOCAB_SIZE  = 262,144
  - MIN_FREQUENCY_P1  = 2
  - ByteLevel byte_fallback = False, initial_alphabet = ByteLevel.alphabet()
  - BRAHMAI_PATTERN structure (all script branches kept, only \\p{N}{1,3} → [0-9])
  - Two-phase SuperBPE algorithm (Phase 1: script-aware subwords,
    Phase 2: cross-word superwords)
  - All special token definitions and IDs (FOUNDATIONAL, UTILITY, CHAT, etc.)
  - FIXED_SCI_VOCAB original ~80 base tokens (unchanged)
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import logging
import os
import random
import re as _re
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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

SCRIPT_NAME        = "superbpe_bytelevel_dedup_v4"
TOTAL_VOCAB_SIZE   = 262_144
PHASE1_VOCAB_SIZE  = 200_000          # same as v1
MIN_FREQUENCY_P1   = 2
MIN_FREQUENCY_P2   = 3                # v3 value; boilerplate filter handles quality
NUM_THREADS        = 64
MODEL_MAX_LENGTH   = 8_192

DEFAULT_OUTPUT             = Path("./superbpe_v4_out")
DEFAULT_SHARD_DIR          = Path("/path/to/shards")
DEFAULT_NUM_SHARDS: Optional[int] = None
DEFAULT_SEED               = 42
DEFAULT_PHASE1_VOCAB       = PHASE1_VOCAB_SIZE
DEFAULT_PHASE2_MAX_CHARS   = 1_000
DEFAULT_PHASE2_NUM_SHARDS  = None

# [v4-F] Two-bucket rolling dedup
DEDUP_ENABLED      = True
DEDUP_BUCKET_SIZE  = 3_000_000

os.environ["RAYON_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPECIAL TOKEN DEFINITIONS  (identical to v1 / v3)
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
    "<|system|>",        # 8
    "<|user|>",          # 9
    "<|assistant|>",     # 10
    "<|start_of_turn|>", # 11
    "<|end_of_turn|>",   # 12
]

TOOL_USE = [
    "<tools>",          # 13
    "</tools>",         # 14
    "<tool_call>",      # 15
    "</tool_call>",     # 16
    "<arg_key>",        # 17
    "</arg_key>",       # 18
    "<arg_value>",      # 19
    "</arg_value>",     # 20
    "<tool_response>",  # 21
    "</tool_response>", # 22
    "<|tool_declare|>", # 23
    "<|observation|>",  # 24
]

REASONING = [
    "<think>",      # 25
    "</think>",     # 26
    "<|nothink|>",  # 27
]

LANGUAGE_TAGS = [
    "<lang_hin>", "<lang_tam>", "<lang_tel>", "<lang_kan>", "<lang_mal>",
    "<lang_mar>", "<lang_guj>", "<lang_ben>", "<lang_pan>", "<lang_ory>",
    "<lang_urd>", "<lang_npi>", "<lang_pus>", "<lang_sin>", "<lang_mya>",
    "<lang_fas>", "<lang_bod>", "<lang_dzo>", "<lang_eng>", "<lang_deu>",
    "<lang_fra>", "<lang_rus>", "<lang_cmn>", "<lang_jpn>", "<lang_kor>",
]
assert len(LANGUAGE_TAGS) == 25

FIM = [
    "<|fim_prefix|>",  # 53
    "<|fim_middle|>",  # 54
    "<|fim_suffix|>",  # 55
]

# [v4-I] 1,024 reserved (v1 had 8,192 — frees 7,168 BPE merge slots)
NUM_RESERVED = 1_024
RESERVED     = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]

SPECIAL_TOKENS = (
    FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING
    + LANGUAGE_TAGS + FIM + RESERVED
)
assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATE SPECIAL TOKENS"

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
FIM_PREFIX_TOKEN    = "<|fim_prefix|>"
FIM_MIDDLE_TOKEN    = "<|fim_middle|>"
FIM_SUFFIX_TOKEN    = "<|fim_suffix|>"
STOP_TOKENS         = [EOS_TOKEN, END_OF_TURN_TOKEN]

_N_SPECIAL  = len(SPECIAL_TOKENS)
_N_RESERVED = NUM_RESERVED


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIXED SCIENTIFIC VOCABULARY  [v4-D]
#
#  v1's original ~80 base tokens are preserved exactly.
#  v3 additions: Greek letter LaTeX commands + structural bigrams + physics units.
#  NOT included: cmd+brace forms (\\frac{, \\begin{equation}) — these create
#  merge barriers preventing BPE from learning longer LaTeX expressions.
# ══════════════════════════════════════════════════════════════════════════════

FIXED_SCI_VOCAB = [

    # ── ORIGINAL V1 BASE (unchanged) ─────────────────────────────────────────
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

    # ── NECESSARY LATEX ATOMS: Greek letters & operators  [v4-D new] ─────────
    # Bare commands only — NO cmd+brace forms.
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta",
    "\\eta", "\\theta", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi",
    "\\pi", "\\rho", "\\sigma", "\\tau", "\\phi", "\\chi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Sigma", "\\Phi", "\\Psi", "\\Omega",
    "\\partial", "\\nabla", "\\infty", "\\times", "\\pm", "\\cdot", "\\neq",
    "\\approx", "\\equiv", "\\propto", "\\leq", "\\geq", "\\to", "\\leftarrow",
    "\\rightarrow", "\\Leftrightarrow", "\\forall", "\\exists", "\\in", "\\subset",

    # ── STRUCTURAL LATEX BIGRAMS  [v4-D new] ─────────────────────────────────
    # Extremely frequent; safe to seed — BPE can still merge further.
    "_{", "^{", "}{", "^2", "_i", "_n", "_j", "_k", "$$", "^{-1}",

    # ── CORE PHYSICS UNITS & CONSTANTS  [v4-D new] ───────────────────────────
    "eV", "nm", "μm", "kg", "m/s", "Pa", "Hz", "k_B", "\\hbar", "c_0",
]

_N_SCI = len(FIXED_SCI_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIXED CODE VOCABULARY  [v4-C]
#
#  v1 had no code operators. Adding them here.
#  RULE: Operators and short structural tokens only.
#  NO long method-chain bindings (np.array(, torch.tensor(, etc.) — those
#  create merge barriers for function-argument patterns.
# ══════════════════════════════════════════════════════════════════════════════

FIXED_CODE_VOCAB = [

    # ── Comparison operators ──────────────────────────────────────────────────
    "==", "!=", ">=", "<=", "===", "!==", "<=>",

    # ── Arithmetic / augmented-assignment operators ───────────────────────────
    "**", "//", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "**=", "//=",
    "&=", "|=", "^=", ">>=", "<<=",
    ">>", "<<",

    # ── Arrow, membership, structural operators ───────────────────────────────
    "->", "=>", "::", "..", "...", "??", "?.", "?..",

    # ── Comment delimiters ────────────────────────────────────────────────────
    "/*", "*/", "/**", "///", "//!",

    # ── Python: dunder / magic method names ──────────────────────────────────
    "__init__", "__main__", "__name__", "__class__", "__repr__", "__str__",
    "__len__", "__iter__", "__next__", "__call__", "__getitem__", "__setitem__",
    "__enter__", "__exit__", "__del__", "__new__", "__doc__",
    "__slots__", "__all__", "__version__", "__file__", "__module__",
    "__annotations__", "__dict__", "__bases__", "__mro__",

    # ── Python decorators ─────────────────────────────────────────────────────
    "@property", "@staticmethod", "@classmethod",
    "@abstractmethod", "@dataclass",

    # ── C / C++: preprocessor directives ─────────────────────────────────────
    "#include", "#define", "#ifndef", "#ifdef", "#endif",
    "#pragma", "#elif",

    # ── C++ / Rust: namespace and generic openers ─────────────────────────────
    "std::", "nullptr", "NULL",
    "template<", "vector<", "pair<",
    "Result<", "Option<", "Vec<", "Box<", "Arc<",

    # ── Rust: macros ──────────────────────────────────────────────────────────
    "println!(", "eprintln!(", "format!(", "panic!(", "todo!(", "assert!(",
    "assert_eq!(", "assert_ne!(", "vec![",
    "#[derive(", "#[allow(", "#[cfg(", "#[test]", "#[inline]",

    # ── JavaScript / TypeScript: structural ───────────────────────────────────
    "JSON.parse(", "JSON.stringify(",
    "Object.keys(", "Object.values(", "Object.entries(",
    "Array.from(", "Array.isArray(",
]

_N_CODE = len(FIXED_CODE_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DIGIT TOKENS  [v4-A]
#
#  EMPTY — no digit AddedTokens.
#  Digits 0-9 are in ByteLevel.alphabet() from day 1.
#  The [0-9] pretokenizer rule isolates digits, freeing Phase 1 merge budget
#  for Indic syllables. Phase 2 (no pretokenizer) learns frequent number
#  patterns freely from corpus frequency.
# ══════════════════════════════════════════════════════════════════════════════

DIGIT_TOKENS: list = []
_N_DIGITS = 0


# ══════════════════════════════════════════════════════════════════════════════
# 6.  WHITESPACE TOKENS  (expanded from v1's 6 tokens)
# ══════════════════════════════════════════════════════════════════════════════

WHITESPACE_TOKENS = [
    "\n", "\n\n", "\n\n\n", "\r\n",        # newlines
    "\t", "\v", "\f",                       # tab / control chars
    "  ", "    ", "        ",               # 2/4/8-space indents
    "\n  ", "\n    ", "\n        ",         # newline + indent combos
    "\t\t",                                 # double-tab
]
_N_WHITESPACE = len(WHITESPACE_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BRAHMAI + VERBOSE PRETOKENIZER PATTERN  [v4-A]
#
#  Based on BRAHMAI_PATTERN from v1, with one change:
#    OLD: |[ ]?\p{N}{1,3}    (groups 1-3 ASCII digits, costs ~1890 Phase 1 merges)
#    NEW: |[ ]?[0-9]          (isolates each ASCII digit, zero Phase 1 merge cost)
#
#  Indic-script digits (Devanagari ०-९, Tamil ௦-௯, etc.) are unaffected:
#  they are captured by their respective \p{Script}+ branch which appears
#  earlier in the alternation and takes priority.
#
#  All other BRAHMAI branches are preserved exactly as in v1.
# ══════════════════════════════════════════════════════════════════════════════

BRAHMAI_PATTERN = (
    r"[ ]?\\[a-zA-Z]+\*?"            # LaTeX commands: \frac, \alpha*, etc.
    r"|\\[^a-zA-Z\s]"                # LaTeX non-letter escapes: \{, \|, etc.
    r"|[ ]?[{}\[\]^_$&#%~]"          # LaTeX structural chars
    r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)" # English contractions
    r"|[ ]?\p{Devanagari}+"          # Hindi, Marathi, Sanskrit, Nepali
    r"|[ ]?\p{Tamil}+"               # Tamil
    r"|[ ]?\p{Telugu}+"              # Telugu
    r"|[ ]?\p{Kannada}+"             # Kannada
    r"|[ ]?\p{Malayalam}+"           # Malayalam
    r"|[ ]?\p{Gujarati}+"            # Gujarati
    r"|[ ]?\p{Bengali}+"             # Bengali, Assamese
    r"|[ ]?\p{Gurmukhi}+"            # Punjabi (Gurmukhi)
    r"|[ ]?\p{Oriya}+"               # Odia / Oriya
    r"|[ ]?\p{Greek}+"               # Greek letters
    r"|[\u2200-\u22FF\u2100-\u214F\u27C0-\u27EF\u2980-\u29FF\u2A00-\u2AFF]+"  # Math Unicode
    r"|[ ]?\p{Latin}+"               # Latin-script words
    r"|[ ]?\p{L}+"                   # Remaining Unicode letters (CJK, Arabic, etc.)
    r"|[ ]?[0-9]"                    # [v4-A] SINGLE digit — was \p{N}{1,3}
    r"|[ ]?[^\s\p{L}\p{N}\\{}\[\]^_$&#%~]+"  # Operators & remaining punctuation
    r"|\r?\n+"                       # Newlines
    r"|\s+"                          # Remaining whitespace
)

V3_FALLBACK_PATTERN = r"[ \t]*[^ \t\n]+|[ \t]*\n"


def get_pretokenizer_pattern() -> Tuple[str, str]:
    try:
        import regex as _regex
        _regex.compile(r"\p{Tamil}+")
        return BRAHMAI_PATTERN, "BRAHMAI_v4"
    except (ImportError, Exception):
        log.warning("'regex' unavailable — falling back to V3_FALLBACK. "
                    "Install: pip install regex --break-system-packages")
        return V3_FALLBACK_PATTERN, "V3_FALLBACK"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GPT-2 BYTE ENCODER  [v4-E]
#     Used to Ġ-pre-encode Phase 2 corpus lines for space consistency.
#     Ġ-run collapse prevents indent-noise superwords like "list:ĠĠĠĠĠĠĠĠĠ".
# ══════════════════════════════════════════════════════════════════════════════

def _build_byte_encoder() -> dict:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER: dict = _build_byte_encoder()
_GSPACE_RUN = _re.compile(r"Ġ{3,}")   # collapse indent runs


def encode_for_phase2(text: str) -> str:
    """
    GPT-2 byte-level encode + Ġ-run collapse.

    space → Ġ, then 3+ consecutive Ġ collapse to ĠĠ.

    Examples:
        "by the way"   → "byĠtheĠway"
        "        x:"   → "ĠĠx:"   (was "ĠĠĠĠĠĠĠĠx:" before collapse)
    """
    encoded = "".join(_BYTE_ENCODER[b] for b in text.encode("utf-8"))
    return _GSPACE_RUN.sub("ĠĠ", encoded)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  BOILERPLATE PATTERNS  [v4-G]
#     Phase 2 only — drops low-quality corpus lines from superword slots.
# ══════════════════════════════════════════════════════════════════════════════

_BOILERPLATE_PATTERNS = [
    _re.compile(r"this solution is locked",                       _re.IGNORECASE),
    _re.compile(r"to view this solution",                         _re.IGNORECASE),
    _re.compile(r"you need to log in",                            _re.IGNORECASE),
    _re.compile(r"please sign up",                                _re.IGNORECASE),
    _re.compile(r"sign up or log in",                             _re.IGNORECASE),
    _re.compile(r"upgrade to premium",                            _re.IGNORECASE),
    _re.compile(r"community treasure hunt",                       _re.IGNORECASE),
    _re.compile(r"find the treasures in matlab central",          _re.IGNORECASE),
    _re.compile(r"discover what matlab",                          _re.IGNORECASE),
    _re.compile(r"brainly\.",                                     _re.IGNORECASE),
    _re.compile(r"chegg\.com",                                    _re.IGNORECASE),
    _re.compile(r"coursehero\.com",                               _re.IGNORECASE),
    _re.compile(r"we use cookies",                                _re.IGNORECASE),
    _re.compile(r"accept all cookies",                            _re.IGNORECASE),
    _re.compile(r"cookie policy",                                 _re.IGNORECASE),
    _re.compile(r"all rights reserved",                           _re.IGNORECASE),
    _re.compile(r"click here to (read|view|download)",            _re.IGNORECASE),
    _re.compile(
        r"(home|about|contact|privacy|terms)\s*\|\s*(home|about|contact|privacy|terms)",
        _re.IGNORECASE,
    ),
    _re.compile(r"(\b\w+\b)(?:\s+\1){7,}",                       _re.IGNORECASE),
    _re.compile(r"\d{15,}"),               # long numeric IDs (account numbers etc.)
    _re.compile(r"https?://\S{60,}"),      # very long URLs (minified content)
    _re.compile(r"(\w)\1{8,}"),            # keyboard spam (aaaaaaaaaa)
]

_MIN_DOC_CHARS = 20


def _is_boilerplate(line: str) -> bool:
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(line):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 10. RESOURCE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    def __init__(self, interval_sec: float = 30.0):
        self._interval = interval_sec
        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss = 0.0
        self._peak_pct = 0.0

    @staticmethod
    def _meminfo() -> dict:
        info: dict = {}
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
    def _rss_kb() -> int:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def _snap(self) -> str:
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

    def log_now(self, label: str = ""):
        if label:
            log.info("[monitor] %s", label)
        print(self._snap(), flush=True)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {"peak_rss_gb": round(self._peak_rss, 2),
                "peak_ram_pct": round(self._peak_pct, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# 11. SHARD DISCOVERY + SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def discover_shards(shard_dir: Path) -> List[Path]:
    shards = sorted(shard_dir.glob("shard_*.txt.gz"))
    if not shards:
        shards = sorted(shard_dir.glob("shard_*.txt"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.txt.gz (or shard_*.txt) found in {shard_dir}")
    return shards


def select_shards(
    shard_dir:           Path,
    num_shards:          Optional[int] = None,
    seed:                int = DEFAULT_SEED,
    oversample_patterns: Optional[dict] = None,
) -> List[Path]:
    """
    Select shards with optional oversampling.

    oversample_patterns: dict mapping filename substring → repeat count.
    E.g. {"newlang": 5} repeats any shard with "newlang" in its name 5×.
    This is the recommended way to compensate for under-represented Indic/CJK shards.

    Example:
        select_shards(shard_dir, oversample_patterns={"newlang": 5})
    """
    available = discover_shards(shard_dir)

    if num_shards is not None and num_shards < len(available):
        rng       = random.Random(seed)
        available = sorted(rng.sample(available, num_shards))

    selected = list(available)

    if oversample_patterns:
        extra = []
        for path in available:
            for pattern, count in oversample_patterns.items():
                if pattern in path.name:
                    extra.extend([path] * (count - 1))
        selected = sorted(selected + extra)

    log.info("Shards: %d available → %d selected (after oversampling)",
             len(available), len(selected))
    for p in sorted(set(selected)):
        repeats = selected.count(p)
        log.info("  %-50s  %.2f GB  ×%d", p.name, p.stat().st_size / 1e9, repeats)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 12. CORPUS ITERATOR  [v4-E, v4-F, v4-G]
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB = 500


def _two_bucket_dedup_check(
    seen_a: set, seen_b: set, h: int, bucket_size: int
) -> Tuple[bool, set, set]:
    """
    [v4-F] Two-bucket rolling dedup.
    When bucket_a fills: rotate (old_a → b, new empty → a).
    Coverage stays ≥50% at all times (vs 0% cliff with single clear).
    """
    if h in seen_a or h in seen_b:
        return True, seen_a, seen_b
    seen_a.add(h)
    if len(seen_a) >= bucket_size:
        seen_a, seen_b = set(), seen_a
    return False, seen_a, seen_b


def corpus_iterator(
    shard_paths:        List[Path],
    max_line_chars:     Optional[int] = None,
    dedup:              bool = DEDUP_ENABLED,
    bucket_size:        int  = DEDUP_BUCKET_SIZE,
    phase2_encode:      bool = False,
    filter_boilerplate: bool = False,
) -> Iterator[str]:
    """
    Streaming corpus iterator.

    Phase 1: max_line_chars=None, phase2_encode=False, filter_boilerplate=False
    Phase 2: max_line_chars=1000, phase2_encode=True,  filter_boilerplate=True
    """
    total_bytes   = 0
    total_yielded = 0
    total_short   = 0
    total_long    = 0
    total_dedup   = 0
    total_bplate  = 0
    last_log      = 0
    t0            = time.perf_counter()

    seen_a: set = set()
    seen_b: set = set()

    log.info("  max_line_chars=%s  dedup=%s  bucket_size=%d  "
             "phase2_encode=%s  filter_boilerplate=%s",
             max_line_chars, dedup, bucket_size, phase2_encode, filter_boilerplate)

    pbar = tqdm(desc="lines", unit=" lines", smoothing=0.1, mininterval=5.0)

    for path in shard_paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n\r")

                if len(line) < _MIN_DOC_CHARS:
                    total_short += 1
                    continue
                if max_line_chars and len(line) > max_line_chars:
                    total_long += 1
                    continue
                if filter_boilerplate and _is_boilerplate(line):
                    total_bplate += 1
                    continue
                if dedup:
                    h = hash(line)
                    is_dup, seen_a, seen_b = _two_bucket_dedup_check(
                        seen_a, seen_b, h, bucket_size)
                    if is_dup:
                        total_dedup += 1
                        continue

                out_line = encode_for_phase2(line) if phase2_encode else line

                b = len(line.encode("utf-8", errors="replace"))
                total_bytes   += b
                total_yielded += 1
                pbar.update(1)
                yield out_line

                if total_bytes - last_log >= LOG_EVERY_MB * 1_048_576:
                    elapsed = time.perf_counter() - t0
                    log.info(
                        "  %.1f GB | %d lines | dedup=%d | long=%d | bplate=%d | %.1f MB/s",
                        total_bytes / 1_073_741_824, total_yielded,
                        total_dedup, total_long, total_bplate,
                        (total_bytes / 1_048_576) / elapsed if elapsed else 0)
                    last_log = total_bytes

    pbar.close()
    log.info(
        "Corpus done: %.2f GB | %d lines | short=%d | long=%d | dedup=%d | bplate=%d | %.1f s",
        total_bytes / 1_073_741_824, total_yielded,
        total_short, total_long, total_dedup, total_bplate,
        time.perf_counter() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# 13. ADDED TOKEN BUILDER
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

    code_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=False, lstrip=False, rstrip=False)
        for t in FIXED_CODE_VOCAB
    ]

    ws_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=False, lstrip=False, rstrip=False)
        for t in WHITESPACE_TOKENS
    ]

    # DIGIT_TOKENS is empty — no digit AddedTokens [v4-A]
    all_tokens = special_objs + sci_objs + code_objs + ws_objs
    seen, deduped = set(), []
    for tok in all_tokens:
        if tok.content not in seen:
            seen.add(tok.content)
            deduped.append(tok)

    n_alpha = sum(1 for t in FIXED_SCI_VOCAB if _is_alpha_token(t))
    log.info(
        "Added tokens: %d special + %d sci + %d code + 0 digit + %d ws = %d (deduped: %d)",
        len(special_objs), _N_SCI, _N_CODE, _N_WHITESPACE,
        len(special_objs) + _N_SCI + _N_CODE + _N_WHITESPACE,
        len(deduped))
    log.info("  No digit AddedTokens — [0-9] pretokenizer handles numbers [v4-A]")
    log.info("  No <0xXX> tokens — byte-level handles all bytes natively")
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# 14. POST-SAVE JSON PATCH  [v4-H, v4-J]
# ══════════════════════════════════════════════════════════════════════════════

def _patch_tokenizer_json(tok_path: Path, is_phase2: bool = False) -> None:
    with open(tok_path, encoding="utf-8") as f:
        state = json.load(f)

    # Ensure ByteLevel decoder
    current_decoder = state.get("decoder")
    if not (isinstance(current_decoder, dict)
            and current_decoder.get("type") == "ByteLevel"):
        state["decoder"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        }
        log.info("  [patch] decoder → ByteLevel (was: %s)", current_decoder)

    # [v4-J] Inference bridge: restore ByteLevel(use_regex=False) for Phase 2
    # Phase 2 BPE vocab was built from Ġ-encoded text (encode_for_phase2).
    # At inference, raw input must be Ġ-encoded before BPE sees it.
    # use_regex=False: only byte-encodes, no GPT-2 word-split → cross-word
    # superword tokens CAN be found. use_regex=True would split first and block them.
    if is_phase2:
        state["pre_tokenizer"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        log.info("  [patch] pre_tokenizer → ByteLevel(use_regex=False) [v4-J inference bridge]")

    sci_set  = set(FIXED_SCI_VOCAB)
    code_set = set(FIXED_CODE_VOCAB)
    ws_set   = set(WHITESPACE_TOKENS)
    n_sci = n_code = n_ws = n_sword = 0

    for entry in state.get("added_tokens", []):
        content = entry.get("content", "")
        if content in sci_set:
            entry["special"]     = False
            entry["single_word"] = _is_alpha_token(content)
            n_sci += 1
            if _is_alpha_token(content):
                n_sword += 1
        if content in code_set:
            entry["special"]     = False
            entry["single_word"] = False
            n_code += 1
        if content in ws_set:
            entry["special"]    = False
            entry["normalized"] = False
            n_ws += 1

    # Inject whitespace tokens that are in vocab but not in added_tokens list
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

    log.info("  [patch] sci=%d (sw=%d) | code=%d | ws=%d | injected=%d",
             n_sci, n_sword, n_code, n_ws, injected)

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("  [patch] saved → %s  (%d bytes)", tok_path, tok_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# 15. PHASE 1  —  BYTE-LEVEL BPE + BRAHMAI_v4 PATTERN  [v4-B]
#
#  Pretokenizer order: Split FIRST → ByteLevel(use_regex=False) SECOND.
#
#  v1 bug: ByteLevel ran first, encoding spaces to Ġ, then BRAHMAI's [ ]?
#  prefix never matched (no raw spaces left). Every Indic word got a bare
#  standalone Ġ piece instead of " क" being captured as one unit.
#  v4 fix: Split sees raw ASCII, correctly captures " क" as one piece, then
#  ByteLevel byte-encodes the results without any further word-splitting.
# ══════════════════════════════════════════════════════════════════════════════

def build_phase1_tokenizer(phase1_vocab_size: int, pattern: str, pattern_name: str):
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())
    tokenizer.decoder = ByteLevelDecoder()

    # CRITICAL ORDER: Split FIRST (raw text), ByteLevel SECOND (encode only).
    # use_regex=False on ByteLevel: disables GPT-2 word-split, only byte-encodes.
    tokenizer.pre_tokenizer = Sequence([
        Split(pattern=Regex(pattern), behavior="isolated", invert=True),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    deduped = _build_added_tokens()
    trainer = BpeTrainer(
        vocab_size       = phase1_vocab_size,
        min_frequency    = MIN_FREQUENCY_P1,
        special_tokens   = deduped,
        show_progress    = True,
        initial_alphabet = ByteLevel.alphabet(),
    )

    log.info("Phase 1: BYTE-LEVEL + %s | vocab=%d | min_freq=%d",
             pattern_name, phase1_vocab_size, MIN_FREQUENCY_P1)
    log.info("  Pretokenizer : Split(%s) → ByteLevel(use_regex=False)  [v4-B: Split-first]",
             pattern_name)
    log.info("  Number rule  : [0-9] single digit  [v4-A: zero digit merge cost]")
    log.info("  SCI tokens   : %d (v1 base + Greek atoms + bigrams)  [v4-D]", _N_SCI)
    log.info("  CODE tokens  : %d (operators + dunders)  [v4-C]", _N_CODE)
    log.info("  DIGIT tokens : 0  [v4-A]")
    log.info("  initial_alphabet: ByteLevel.alphabet() = 256 byte symbols")
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 16. PHASE 2  —  SuperBPE: pretokenizer=None for training  [v4-E, v4-J]
#
#  Training: no pretokenizer → BPE merges freely across Ġ-encoded spaces.
#  Inference: ByteLevel(use_regex=False) restored by _patch_tokenizer_json.
#  Corpus:    encode_for_phase2() applied before feeding to BPE trainer.
# ══════════════════════════════════════════════════════════════════════════════

def build_phase2_tokenizer(phase1_checkpoint_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.trainers import BpeTrainer

    tok_path = phase1_checkpoint_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {tok_path}\n"
            "Run Phase 1 first (--phase1-only).")

    log.info("Loading Phase 1 checkpoint: %s", tok_path)
    with open(tok_path, encoding="utf-8") as f:
        p1_state = json.load(f)

    p1_vocab_size   = len(p1_state.get("model", {}).get("vocab", {}))
    p1_merges       = len(p1_state.get("model", {}).get("merges", []))
    superword_slots = TOTAL_VOCAB_SIZE - p1_vocab_size
    log.info("  Phase 1 vocab=%d  merges=%d  superword_slots=%d",
             p1_vocab_size, p1_merges, superword_slots)

    p2_state                  = dict(p1_state)
    p2_state["pre_tokenizer"] = None  # training: cross-word BPE merges

    tokenizer = Tokenizer.from_str(json.dumps(p2_state))
    deduped   = _build_added_tokens()
    trainer   = BpeTrainer(
        vocab_size    = TOTAL_VOCAB_SIZE,
        min_frequency = MIN_FREQUENCY_P2,
        special_tokens= deduped,
        show_progress = True,
    )

    log.info("Phase 2: pretokenizer=None (training) | min_freq=%d", MIN_FREQUENCY_P2)
    log.info("  Inference bridge: ByteLevel(use_regex=False) set by post-save patch  [v4-J]")
    log.info("  Corpus Ġ-encode: ON with run-collapse  [v4-E]")
    log.info("  Boilerplate filter: ON  [v4-G]")
    return tokenizer, trainer, p1_vocab_size


# ══════════════════════════════════════════════════════════════════════════════
# 17. TRAINING RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def train_phase1(tokenizer, trainer,
                 shard_paths: List[Path]):
    log.info("Training Phase 1 (byte-level BPE, no line limit) ...")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(
            shard_paths,
            max_line_chars=None,
            phase2_encode=False,
            filter_boilerplate=False,
        ),
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


def train_phase2(tokenizer, trainer,
                 shard_paths: List[Path],
                 max_line_chars: int):
    log.info("Training Phase 2 SuperBPE ...")
    log.info("  max_line_chars : %d", max_line_chars)
    log.info("  min_frequency  : %d", MIN_FREQUENCY_P2)
    log.info("  Ġ-encode+collapse: ON")
    log.info("  Boilerplate    : ON")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(
            shard_paths,
            max_line_chars=max_line_chars,
            phase2_encode=True,
            filter_boilerplate=True,
        ),
        trainer=trainer,
        length=None,
    )
    elapsed = time.perf_counter() - t0
    actual  = tokenizer.get_vocab_size()
    log.info("Phase 2 done in %.1f min. Vocab: %d / %d",
             elapsed / 60, actual, trainer.vocab_size)
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 18. SAVE + PATCH + HF EXPORT  [v4-H]
# ══════════════════════════════════════════════════════════════════════════════

def save_tokenizer(tokenizer, save_dir: Path, meta: dict,
                   is_phase2: bool = False) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    tok_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    log.info("Saved tokenizer.json (%d bytes)", tok_path.stat().st_size)

    log.info("Applying patches ...")
    _patch_tokenizer_json(tok_path, is_phase2=is_phase2)

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

    # [v4-H] HF-compatible tokenizer_config.json
    # "extra_special_tokens": {} is REQUIRED — transformers ≥4.40 calls .keys() on it.
    # If absent or a list, AutoTokenizer.from_pretrained raises AttributeError.
    cfg: dict = {
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
        # transformers >=4.40 calls _set_model_specific_special_tokens(special_tokens=
        # self.extra_special_tokens) then does special_tokens.keys().
        # Must be a dict, never a list.
        "extra_special_tokens"        : {},
    }

    all_added = {
        **{t: {"special": True,  "single_word": False} for t in SPECIAL_TOKENS},
        **{t: {"special": False, "single_word": _is_alpha_token(t)}
           for t in FIXED_SCI_VOCAB},
        **{t: {"special": False, "single_word": False} for t in FIXED_CODE_VOCAB},
        **{t: {"special": False, "single_word": False} for t in WHITESPACE_TOKENS},
    }
    for tok_str, flags in all_added.items():
        if tok_str in vocab:
            cfg["added_tokens_decoder"][str(vocab[tok_str])] = {
                "content": tok_str, "lstrip": False,
                "normalized": False, "rstrip": False,
                **flags,
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
        json.dump({**meta,
                   "vocab_size_actual": len(sv),
                   "merges_total": len(merges)}, f, indent=2)

    log.info("All files saved → %s", save_dir)
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# 19. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tokenizer(load_dir: Path, label: str = "final") -> None:
    log.info("=" * 65)
    log.info("VALIDATION [%s]  %s", label, load_dir)

    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(str(load_dir))
        hf  = True
        log.info("Loaded via PreTrainedTokenizerFast ✓  [v4-H verified]")
    except Exception as e:
        log.warning("HF load failed (%s) — using raw tokenizer", e)
        from tokenizers import Tokenizer as _Tok
        tok = _Tok.from_file(str(load_dir / "tokenizer.json"))
        hf  = False

    vocab = tok.get_vocab()
    log.info("Vocab size: %d", len(vocab))

    # Byte-fallback tokens should not exist
    byte_tokens = sum(1 for i in range(256) if f"<0x{i:02X}>" in vocab)
    log.info("  <0xXX> tokens: %d  %s",
             byte_tokens, "✓ none (correct)" if byte_tokens == 0 else "⚠ unexpected")

    # Special token IDs
    log.info("Special token ID check:")
    expected = FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING + LANGUAGE_TAGS + FIM
    mismatches = 0
    for eid, tstr in enumerate(expected):
        aid = vocab.get(tstr, -1)
        if aid != eid:
            log.warning("  MISMATCH %r  expected=%d  actual=%d", tstr, eid, aid)
            mismatches += 1
    log.info("  %d IDs correct %s",
             len(expected) - mismatches, "✓" if mismatches == 0 else f"✗ ({mismatches} bad)")

    # [v4-A] No forced digit AddedTokens
    log.info("Digit check [v4-A]:")
    at_boundary = len(SPECIAL_TOKENS) + _N_SCI + _N_CODE + _N_WHITESPACE
    forced_2d = sum(
        1 for i in range(10, 100)
        if f"{i}" in vocab and vocab[f"{i}"] < at_boundary
    )
    log.info("  Forced 2-digit AddedTokens: %d  %s",
             forced_2d, "✓ none" if forced_2d == 0 else "⚠ unexpected")
    for d in ["0", "1", "9", "42", "128", "1947", "2024"]:
        log.info("  %-8r  id=%d", d, vocab.get(d, -1))

    # [v4-C] Operator coverage
    ops = ["==", "!=", ">=", "<=", "//", "/*", "->", "=>", "++", "--", "**",
           "===", "!==", "::", "..", "??"]
    log.info("Code operator coverage [v4-C]:")
    missing = [op for op in ops if op not in vocab]
    log.info("  %d / %d present%s",
             len(ops) - len(missing), len(ops),
             f"  MISSING: {missing}" if missing else " ✓")

    # [v4-D] LaTeX atom coverage
    log.info("LaTeX atom coverage [v4-D]:")
    for la in ["\\alpha", "\\nabla", "\\partial", "\\frac", "\\begin{equation}",
               "_{", "^{", "^2", "$$"]:
        log.info("  %-30r  id=%d", la, vocab.get(la, -1))

    # Round-trip tests
    rt_tests = [
        ("LaTeX frac",     r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"),
        ("LaTeX subscript","$x_{i+1} = x_i^2 + c$"),
        ("LaTeX env",      r"\begin{equation} E = mc^2 \end{equation}"),
        ("ASCII math",     "d/dt(T) = alpha * laplacian(T)"),
        ("Unicode math",   "∂u/∂t = α∇²u"),
        ("Hindi",          "नमस्ते दुनिया, यह परीक्षण है।"),
        ("Hindi w/ year",  "सन् 1947 में भारत स्वतंत्र हुआ"),
        ("Tamil",          "போகிறார்கள்"),
        ("Telugu",         "నమస్కారం"),
        ("Kannada",        "ನಮಸ್ಕಾರ"),
        ("Bengali",        "বাংলাদেশ"),
        ("Gujarati",       "ભારત"),
        ("Punjabi",        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"),
        ("Odia",           "ନମସ୍କାର"),
        ("Arabic",         "مرحبا بالعالم"),
        ("Chinese",        "你好世界"),
        ("Japanese",       "こんにちは世界"),
        ("Korean",         "안녕하세요 세계"),
        ("Code Python",    "    def fib(n: int) -> int:\n        return n"),
        ("Code ops",       "if x == y and a != b and c >= 0:"),
        ("Code C++",       "std::vector<int> v; v.push_back(42);"),
        ("Code arrow",     "const fn = (x) => x * 2;"),
        ("Code comment",   "// single line\n/* block comment */"),
        ("Rust macro",     'fn main() { println!("{}", Ok(42).unwrap()); }'),
        ("Preprocessor",   "#include <stdio.h>\n#define MAX 100"),
        ("Numbers 1-3d",   "batch=128 lr=0.001 epoch=512"),
        ("Numbers 5d+",    "12345 and 1947 and 2024"),
        ("Think tag",      "<think>step by step</think>"),
        ("FIM",            "<|fim_prefix|>def f():<|fim_suffix|>    pass<|fim_middle|>"),
        ("Tool call",      "<tool_call><arg_key>fn</arg_key></tool_call>"),
        ("Space prefix",   " hello world"),
        ("No prefix",      "hello world"),
    ]

    log.info("Round-trip tests:")
    all_ok = True
    for lbl, text in rt_tests:
        ids = tok.encode(text) if hf else tok.encode(text).ids
        dec = tok.decode(ids, skip_special_tokens=False)
        ok  = (text == dec)
        if not ok:
            all_ok = False
        log.info("  [%-22s] %3d toks  %s",
                 lbl, len(ids), "✓" if ok else f"✗  got={dec[:60]!r}")

    if label == "final":
        log.info("Fertility report (lower = better):")
        samples = {
            "English":    "The quick brown fox jumps over the lazy dog.",
            "Python":     "    def fib(n):\n        if n<=1: return n\n"
                          "        return fib(n-1)+fib(n-2)",
            "LaTeX":      r"\frac{\partial u}{\partial t}=\alpha\nabla^2 u",
            "Hindi":      "नमस्ते दुनिया, यह एक परीक्षण है।",
            "Hindi+nums": "सन् 1947 और 2024 में क्रमशः 150 और 200 लोग थे",
            "Arabic":     "مرحبا بالعالم هذا اختبار",
            "Numbers":    "batch=128 lr=0.001 epochs=10000 seed=42",
        }
        for name, sample in samples.items():
            ids = tok.encode(sample) if hf else tok.encode(sample).ids
            log.info("  %-18s  %3d chars → %3d toks  (%.3f tok/char)",
                     name, len(sample), len(ids), len(ids) / len(sample))

        log.info("Superword spot check:")
        for sw in ["ĠbyĠtheĠway", "ĠofĠthe", "ĠinĠthe", "ĠtoĠthe"]:
            log.info("  %-25r  %s", sw, "IN VOCAB ✓" if sw in vocab else "not yet")

    log.info("VALIDATION %s", "ALL PASSED ✓" if all_ok else "SOME FAILED ✗")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 20. VOCABULARY BUDGET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def log_vocab_summary(phase1_vocab_size: int):
    fixed     = _N_SPECIAL + _N_SCI + _N_CODE + _N_WHITESPACE
    bpe_slots = phase1_vocab_size - fixed - 256
    sw_slots  = TOTAL_VOCAB_SIZE - phase1_vocab_size

    log.info("━" * 65)
    log.info("VOCABULARY BUDGET  (v4)")
    log.info("━" * 65)
    log.info("  Special tokens    : %6d  (reserved=%d)  [v4-I: was 8192]",
             _N_SPECIAL, _N_RESERVED)
    log.info("  FIXED_SCI_VOCAB   : %6d  (v1 base + Greek atoms + bigrams)  [v4-D]",
             _N_SCI)
    log.info("  FIXED_CODE_VOCAB  : %6d  (operators + dunders)  [v4-C]", _N_CODE)
    log.info("  DIGIT_TOKENS      :      0  (freed to Indic budget)  [v4-A]")
    log.info("  WHITESPACE_TOKENS : %6d", _N_WHITESPACE)
    log.info("  ─────────────────────────────────────────")
    log.info("  Total fixed       : %6d", fixed)
    log.info("  ByteLevel.alphabet:    256  (initial_alphabet)")
    log.info("  Phase 1 BPE merges: ~%5d  (vocab=%d − fixed − 256)",
             bpe_slots, phase1_vocab_size)
    log.info("  Phase 2 superwords: %6d", sw_slots)
    log.info("  Total vocab target: %6d", TOTAL_VOCAB_SIZE)
    log.info("━" * 65)
    log.info("Phase 1 BPE budget comparison across versions:")
    log.info("  v1:  200k, ~8,400 fixed  → ~191,300 merges (no code, \\p{N}+ digit)")
    log.info("  v2:  180k, ~2,900 fixed  → ~176,800 merges (1100 digit tokens wasted)")
    log.info("  v3:  200k, ~1,390 fixed  → ~198,354 merges (\\p{N}{1,3} digit)")
    log.info("  v4:  200k, ~%d fixed  → ~%d merges ([0-9] single digit)  ← BEST",
             fixed, bpe_slots)
    log.info("━" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 21. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    shards_dir           : Path,
    output_dir           : Path,
    num_shards           : Optional[int],
    seed                 : int,
    validate             : bool,
    monitor_interval     : float,
    phase1_vocab_size    : int,
    phase1_only          : bool,
    phase2_only          : bool,
    phase1_checkpoint    : Optional[Path],
    phase2_num_shards    : Optional[int],
    phase2_max_chars     : int,
    oversample_newlang   : int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase1_checkpoint is None:
        phase1_checkpoint = output_dir / "phase1_checkpoint"

    pattern, pattern_name = get_pretokenizer_pattern()

    # Build oversample map (key = substring of shard filename)
    oversample = {"newlang": oversample_newlang} if oversample_newlang > 1 else None

    # Select shards for Phase 1 and Phase 2
    p1_shards = select_shards(shards_dir, num_shards, seed,
                               oversample_patterns=oversample)
    p2_shards = (select_shards(shards_dir, phase2_num_shards, seed,
                               oversample_patterns=oversample)
                 if phase2_num_shards is not None
                 else p1_shards)

    log.info("=" * 65)
    log.info("  %s", SCRIPT_NAME.upper())
    log.info("=" * 65)
    log.info("  Phase 1 vocab    : %d  (same as v1)", phase1_vocab_size)
    log.info("  Phase 1 preток   : Split(%s) → ByteLevel(use_regex=False)  [v4-B]",
             pattern_name)
    log.info("  Phase 1 min_freq : %d", MIN_FREQUENCY_P1)
    log.info("  Number rule      : [0-9] single digit  [v4-A: ~1890 merges freed]")
    log.info("  SCI vocab        : %d atoms  [v4-D]", _N_SCI)
    log.info("  CODE vocab       : %d operators  [v4-C]", _N_CODE)
    log.info("  Digit tokens     : NONE  [v4-A]")
    log.info("  Phase 2 preток   : None (training) / ByteLevel(use_regex=False) (infer)")
    log.info("  Phase 2 min_freq : %d", MIN_FREQUENCY_P2)
    log.info("  Phase 2 Ġ-encode : ON + run-collapse  [v4-E]")
    log.info("  Phase 2 boilerplt: ON  [v4-G]")
    log.info("  Phase 2 max_chars: %d", phase2_max_chars)
    log.info("  Dedup            : two-bucket rolling  [v4-F]")
    log.info("  Oversample       : newlang ×%d", oversample_newlang)
    log.info("  NUM_RESERVED     : %d  [v4-I: was 8192]", NUM_RESERVED)
    log.info("  HF compat        : extra_special_tokens={}  [v4-H]")
    log.info("  Superword slots  : %d", TOTAL_VOCAB_SIZE - phase1_vocab_size)
    log.info("  Total vocab      : %d", TOTAL_VOCAB_SIZE)
    log.info("  Output           : %s", output_dir)
    log.info("=" * 65)

    log_vocab_summary(phase1_vocab_size)

    monitor = ResourceMonitor(interval_sec=monitor_interval)
    monitor.start()
    monitor.log_now("BASELINE")
    t_total = time.perf_counter()

    meta_base = {
        "script"              : SCRIPT_NAME,
        "bpe_type"            : "byte-level",
        "phase1_pattern"      : pattern_name,
        "phase1_vocab"        : phase1_vocab_size,
        "total_vocab_target"  : TOTAL_VOCAB_SIZE,
        "superword_slots"     : TOTAL_VOCAB_SIZE - phase1_vocab_size,
        "min_freq_p1"         : MIN_FREQUENCY_P1,
        "min_freq_p2"         : MIN_FREQUENCY_P2,
        "number_rule"         : "single-digit [0-9]",
        "dedup_strategy"      : "two-bucket-rolling",
        "dedup_bucket_size"   : DEDUP_BUCKET_SIZE,
        "digit_tokens"        : "none",
        "n_sci_vocab"         : _N_SCI,
        "n_code_vocab"        : _N_CODE,
        "num_reserved"        : NUM_RESERVED,
        "phase2_g_encode"     : True,
        "phase2_g_run_collapse": True,
        "phase2_boilerplate"  : True,
        "oversample_newlang"  : oversample_newlang,
        "seed"                : seed,
        "p2_max_chars"        : phase2_max_chars,
        "byte_fallback"       : False,
        "v4_fixes"            : [
            "A:single-digit-[0-9]",
            "B:split-first-pretokenizer",
            "C:code-operators",
            "D:sci-atoms-greek-bigrams",
            "E:phase2-g-encode-run-collapse",
            "F:two-bucket-dedup",
            "G:boilerplate-filter",
            "H:hf-compat-extra_special_tokens",
            "I:reserved-1024",
            "J:inference-bridge",
        ],
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not phase2_only:
        with stage("Phase 1: Build tokenizer"):
            tok1, tr1 = build_phase1_tokenizer(
                phase1_vocab_size, pattern, pattern_name)

        with stage("Phase 1: Train"):
            tok1 = train_phase1(tok1, tr1, p1_shards)

        with stage("Phase 1: Save checkpoint"):
            save_tokenizer(tok1, phase1_checkpoint,
                           {**meta_base, "phase": "phase1"},
                           is_phase2=False)

        if validate:
            with stage("Phase 1: Validate"):
                validate_tokenizer(phase1_checkpoint, label="phase1")

        if phase1_only:
            res = monitor.stop()
            log.info("Phase 1 complete. %.2f h  RSS %.1f GB",
                     (time.perf_counter() - t_total) / 3600, res["peak_rss_gb"])
            return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    with stage("Phase 2: Build SuperBPE (pretokenizer=None)"):
        tok2, tr2, p1_vocab_size = build_phase2_tokenizer(phase1_checkpoint)

    with stage(f"Phase 2: Train (max_chars={phase2_max_chars}, "
               f"min_freq={MIN_FREQUENCY_P2})"):
        tok2 = train_phase2(tok2, tr2, p2_shards, phase2_max_chars)

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
            is_phase2=True,
        )

    if validate:
        with stage("Final: Validate"):
            validate_tokenizer(output_dir, label="final")

    log.info("=" * 65)
    log.info("  DONE  |  %.2f h  |  Peak RSS %.1f GB  |  %s",
             elapsed / 3600, res["peak_rss_gb"], output_dir)
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 22. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=(
            "SuperBPE v4 — v1 architecture + single-digit [0-9] pattern "
            "+ code/sci atoms + all v3 data/HF fixes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with 4× Indic shard oversampling
  python superbpe_bytelevel_dedup_v4.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_v4_out \\
      --oversample-newlang 4 \\
      --validate

  # Phase 1 only
  python superbpe_bytelevel_dedup_v4.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_v4_out \\
      --phase1-only --validate

  # Phase 2 only
  python superbpe_bytelevel_dedup_v4.py \\
      --shards-dir ~/data/shards \\
      --output ~/superbpe_v4_out \\
      --phase2-only \\
      --phase1-checkpoint ~/superbpe_v4_out/phase1_checkpoint \\
      --validate

  # Check config without training
  python superbpe_bytelevel_dedup_v4.py --check-only
        """,
    )
    p.add_argument("--shards-dir",           type=Path,  default=DEFAULT_SHARD_DIR)
    p.add_argument("--output",               type=Path,  default=DEFAULT_OUTPUT)
    p.add_argument("--num-shards",           type=int,   default=DEFAULT_NUM_SHARDS)
    p.add_argument("--seed",                 type=int,   default=DEFAULT_SEED)
    p.add_argument("--monitor-interval",     type=float, default=30.0)
    p.add_argument("--validate",             action="store_true")
    p.add_argument("--phase1-vocab",         type=int,   default=DEFAULT_PHASE1_VOCAB)
    p.add_argument("--phase1-only",          action="store_true")
    p.add_argument("--phase2-only",          action="store_true")
    p.add_argument("--phase1-checkpoint",    type=Path,  default=None)
    p.add_argument("--phase2-shards",        type=int,   default=DEFAULT_PHASE2_NUM_SHARDS)
    p.add_argument("--phase2-max-chars",     type=int,   default=DEFAULT_PHASE2_MAX_CHARS,
                   help=f"Max chars per Phase 2 line (default {DEFAULT_PHASE2_MAX_CHARS}; 0=no limit).")
    p.add_argument("--oversample-newlang",   type=int,   default=1,
                   help="Repeat shards containing 'newlang' in filename N times "
                        "(e.g. --oversample-newlang 4 repeats your Indic/foreign shard 4×). "
                        "Default 1 = no oversampling.")
    p.add_argument("--check-only",           action="store_true")

    args = p.parse_args()

    if args.phase1_only and args.phase2_only:
        p.error("--phase1-only and --phase2-only are mutually exclusive.")

    phase2_max_chars = args.phase2_max_chars if args.phase2_max_chars > 0 else None

    if args.check_only:
        log.info("=" * 65)
        log.info("%s — Check Mode", SCRIPT_NAME)
        log.info("=" * 65)
        log.info("  Phase 1 vocab     : %d", DEFAULT_PHASE1_VOCAB)
        log.info("  Number rule       : [0-9] single digit  [v4-A]")
        log.info("  Phase 2 min_freq  : %d", MIN_FREQUENCY_P2)
        log.info("  FIXED_SCI_VOCAB   : %d  [v4-D]", _N_SCI)
        log.info("  FIXED_CODE_VOCAB  : %d  [v4-C]", _N_CODE)
        log.info("  DIGIT_TOKENS      : 0  [v4-A]")
        log.info("  WHITESPACE_TOKENS : %d", _N_WHITESPACE)
        log.info("  NUM_RESERVED      : %d  [v4-I]", NUM_RESERVED)
        log.info("  RAYON threads     : %d", NUM_THREADS)
        log.info("  CPU count         : %s", os.cpu_count())

        pattern, pname = get_pretokenizer_pattern()
        log.info("  Active pattern    : %s  ([0-9] single-digit rule)", pname)

        try:
            from tokenizers.pre_tokenizers import ByteLevel as BL
            log.info("  ByteLevel.alphabet() size: %d symbols", len(BL.alphabet()))
        except Exception as e:
            log.error("  ByteLevel.alphabet() error: %s", e)

        try:
            from tokenizers import Regex
            Regex(BRAHMAI_PATTERN)
            log.info("  BRAHMAI_v4 pattern: OK  ([0-9] single-digit active)")
        except Exception as e:
            log.error("  BRAHMAI_v4 pattern: INVALID — %s", e)

        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATE SPECIAL TOKENS"
        log.info("  Duplicate check   : OK")

        test_enc = encode_for_phase2("by the way")
        assert "Ġ" in test_enc and " " not in test_enc, f"Byte encoder broken: {test_enc}"
        log.info("  Byte encoder      : OK  → %r", test_enc[:20])

        test_indent = encode_for_phase2("        items:")
        g_count = test_indent.count("Ġ")
        assert g_count <= 2, f"Ġ-run collapse broken: {repr(test_indent)} Ġ×{g_count}"
        log.info("  Ġ-run collapse    : OK  → %r  (Ġ×%d)", test_indent, g_count)

        log_vocab_summary(DEFAULT_PHASE1_VOCAB)

        # Verify [0-9] single-digit is in the compiled pattern
        import regex as _rx
        pat = _rx.compile(BRAHMAI_PATTERN)
        test_num = "1947"
        pieces = [m.group(0) for m in pat.finditer(test_num)]
        assert all(len(p.strip()) == 1 and p.strip().isdigit() for p in pieces), \
            f"[0-9] rule not single-digit: {pieces}"
        log.info("  [0-9] digit rule  : OK  '1947' → %s (each digit isolated)", pieces)

        log.info("=" * 65)
        return

    run(
        shards_dir          = args.shards_dir,
        output_dir          = args.output,
        num_shards          = args.num_shards,
        seed                = args.seed,
        validate            = args.validate,
        monitor_interval    = args.monitor_interval,
        phase1_vocab_size   = args.phase1_vocab,
        phase1_only         = args.phase1_only,
        phase2_only         = args.phase2_only,
        phase1_checkpoint   = args.phase1_checkpoint,
        phase2_num_shards   = args.phase2_shards,
        phase2_max_chars    = phase2_max_chars,
        oversample_newlang  = args.oversample_newlang,
    )


if __name__ == "__main__":
    main()