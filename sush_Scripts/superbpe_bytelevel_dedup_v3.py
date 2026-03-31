#!/usr/bin/env python3
"""
superbpe_bytelevel_dedup_v3.py
================================
SuperBPE  |  BYTE-LEVEL BPE  |  BRAHMAI_PATTERN Phase 1  |  WITH line-level dedup

CHANGES OVER v2  (annotated with fix tag)
==========================================

[v3-FIX-1]  DIGIT_TOKENS eliminated
    v2 added 1,100 forced digit tokens (00–99, 000–999) as AddedTokens.
    AddedToken entries create hard merge boundaries — BPE can never merge
    ACROSS a digit token into its neighbors.  On Indic text that interleaves
    digits (e.g. Hindi dates), this silently blocks syllable merges.
    The \\p{N}{1,3} pre-tokeniser rule ALREADY splits numbers into ≤3-digit
    chunks before BPE sees them, making the AddedTokens 100% redundant.
    Single digits 0–9 are in ByteLevel.alphabet() (initial_alphabet) so
    they are native vocabulary from day 1.  Freeing the 1,100 slots gives
    Phase 1 an equivalent budget increase with zero cost.

    DIGIT STRATEGY (DeepSeek-style vs {1,3}-chunks):
    DeepSeek uses individual digit splitting (123 → 1,2,3).  This is optimal
    for pure arithmetic reasoning but slightly hurts compression of text
    containing small numbers (e.g. "128", "2024").  The \\p{N}{1,3} rule is
    a better balance: common 2–3-digit sequences (256, 001, 42) become single
    BPE tokens naturally.  To go full DeepSeek-style, change {1,3} to {1} in
    BRAHMAI_PATTERN — but this is NOT recommended for a general-purpose
    multilingual tokenizer.

[v3-FIX-2]  PHASE1_VOCAB_SIZE restored to 200,000  (was 180,000 in v2)
    Byte-level encoding inflates Indic token length by ~2.5x:
      v1 Devanagari avg token length = 5.6 chars  (Unicode tokens)
      v2 Devanagari avg token length = 14.4 chars (GPT-2 byte surrogates)
    Each extra 2–3 chars costs 2 additional Phase 1 merges just to reassemble
    one Unicode character.  With 20k fewer Phase 1 merges, v2 ran out of
    budget before reaching the syllable-level coverage v1 had.
    Restoring to 200k recovers ~8–10k Indic subword slots.

[v3-FIX-3]  FIXED_SCI_VOCAB trimmed to user-specified compulsory set
    v2 added 270 extra LaTeX tokens including \\frac{, \\begin{equation}, etc.
    Every cmd+brace AddedToken creates a merge BARRIER: BPE cannot continue
    merging \\frac{ with the numerator content, so '\\frac{x+y}' stays as
    three tokens instead of merging into one.  v1 proved that BPE learns
    \\frac, \\nabla, \\alpha (IDs 27–29k) naturally and then merges freely
    into longer forms.  The fix: keep only truly indivisible atoms (Greek
    letters, core operators, structural bigrams) and let BPE learn the
    compound forms from corpus frequency.

[v3-FIX-4]  FIXED_CODE_VOCAB trimmed to atomic operators only
    Long library-call AddedTokens (np.array(, torch.tensor(, etc.) create
    merge barriers just like LaTeX cmd+brace forms.  Only genuinely
    indivisible operators (==, !=, ->, =>, ++, --, **, //, etc.),
    dunder names, and preprocessor directives are kept.  Method chains
    are learned by BPE.  This also reduces boundary friction for
    Indic/foreign text that happens to contain these ASCII sequences.

[v3-FIX-5]  MIN_FREQUENCY_P2 = 3  (was 10 in v2, was 2 in v1)
    min_frequency=10 starved Phase 2: the boilerplate filter already removes
    the genuine low-quality lines.  Setting it to 3 lets uncommon-but-real
    phrases earn superword slots while still blocking corpus-specific noise
    that appears only twice.

[v3-FIX-6]  Ġ-run collapse in encode_for_phase2
    v2 Phase 2 contained tokens like 'list:ĠĠĠĠĠĠĠĠĠ' and 'Ġs;ĠĠĠĠĠĠĠĠĠ'
    — indent noise from code blocks.  Collapsing runs of 3+ Ġ to ĠĠ
    (two-space equivalent) prevents these from burning superword slots.

[v3-FIX-7]  Whitespace token set optimised
    Added document-separator triple-newline and Python indent combos.
    Removed triple-tab (rare, wastes a slot).

WHAT IS UNCHANGED FROM v2  (and why it stays)
==============================================
  - Split-first pre-tokeniser order in Phase 1  (correct: \\p{N}{1,3} sees raw spaces)
  - Inference bridge: ByteLevel(use_regex=False) restored after Phase 2 save
  - Two-bucket rolling dedup (strictly better than cliff-clear)
  - Boilerplate filter (prevents homework-site superwords)
  - Ġ pre-encoding of Phase 2 corpus (space consistency)
  - NUM_RESERVED = 1,024  (was 8,192 in v1 — frees 7,168 real vocab slots)
  - ByteLevel byte_fallback = False, initial_alphabet = ByteLevel.alphabet()
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

SCRIPT_NAME        = "superbpe_bytelevel_dedup_v3"
TOTAL_VOCAB_SIZE   = 262_144
PHASE1_VOCAB_SIZE  = 200_000          # [v3-FIX-2] restored from 180k
MIN_FREQUENCY_P1   = 2
MIN_FREQUENCY_P2   = 3                # [v3-FIX-5] was 10; boilerplate filter handles quality
NUM_THREADS        = 64
MODEL_MAX_LENGTH   = 8_192

DEFAULT_OUTPUT             = Path("./superbpe_v3_out")
DEFAULT_SHARD_DIR          = Path("/path/to/shards")
DEFAULT_NUM_SHARDS: Optional[int] = None
DEFAULT_SEED               = 42
DEFAULT_PHASE1_VOCAB       = PHASE1_VOCAB_SIZE
DEFAULT_PHASE2_MAX_CHARS   = 1_000
DEFAULT_PHASE2_NUM_SHARDS  = None

# Two-bucket dedup (unchanged from v2)
DEDUP_ENABLED      = True
DEDUP_BUCKET_SIZE  = 3_000_000

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
# 3.  FIXED SCIENTIFIC VOCABULARY  [v3-FIX-3]
#
#  RULE: AddedTokens must be ATOMIC — units BPE would otherwise fragment into
#  raw bytes, AND that should NOT block further BPE merges with neighbors.
#
#  KEPT:   Greek letters, bare LaTeX commands, math operators, structural
#          bigrams (_{, ^{), physics units.
#  REMOVED (vs v2): all cmd+brace forms (\\frac{, \\begin{equation}, etc.),
#          all space-prefixed commands, all PDE ASCII forms.
#          These create merge barriers that prevent BPE from learning
#          longer LaTeX expressions naturally.
# ══════════════════════════════════════════════════════════════════════════════

FIXED_SCI_VOCAB = [

    # ── COMPULSORY BASE (Do not remove) ───────────────────────────────────────

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

    # ── NECESSARY LATEX ATOMS (Greek & Operators) ─────────────────────────────
    # Bare commands only.  NO cmd+brace forms — those create merge barriers.
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta",
    "\\eta", "\\theta", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi",
    "\\pi", "\\rho", "\\sigma", "\\tau", "\\phi", "\\chi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Sigma", "\\Phi", "\\Psi", "\\Omega",
    "\\partial", "\\nabla", "\\infty", "\\times", "\\pm", "\\cdot", "\\neq",
    "\\approx", "\\equiv", "\\propto", "\\leq", "\\geq", "\\to", "\\leftarrow",
    "\\rightarrow", "\\Leftrightarrow", "\\forall", "\\exists", "\\in", "\\subset",

    # ── NECESSARY STRUCTURAL BIGRAMS ─────────────────────────────────────────
    # Extremely frequent in LaTeX; keeping atomic saves ≥10% context on heavy
    # math docs.  Safe: BPE CAN still merge these further (e.g. _{x}).
    "_{", "^{", "}{", "^2", "_i", "_n", "_j", "_k", "$$", "^{-1}",

    # ── CORE PHYSICS UNITS & CONSTANTS ───────────────────────────────────────
    "eV", "nm", "μm", "kg", "m/s", "Pa", "Hz", "k_B", "\\hbar", "c_0",
]

_N_SCI = len(FIXED_SCI_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIXED CODE VOCABULARY  [v3-FIX-4]
#
#  RULE: Only operators and short structural tokens.
#  REMOVED (vs v2): all long method-chain bindings (np.array(, torch.tensor(,
#  optimizer.zero_grad(), etc.) — these create merge barriers for function-
#  argument patterns and are learned naturally by BPE.
# ══════════════════════════════════════════════════════════════════════════════

FIXED_CODE_VOCAB = [

    # ── Comparison operators (confirmed missing from v1 vocab) ────────────────
    "==", "!=", ">=", "<=", "===", "!==", "<=>",

    # ── Arithmetic / augmented-assignment operators ───────────────────────────
    "**", "//", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "**=", "//=",
    "&=", "|=", "^=", ">>=", "<<=",
    ">>", "<<",

    # ── Arrow, membership, and other structural operators ─────────────────────
    "->", "=>", "::", "..", "...", "??", "?.", "?..",

    # ── Comment delimiters ────────────────────────────────────────────────────
    "/*", "*/", "/**", "///", "//!",

    # ── Python: dunder / magic method names ──────────────────────────────────
    # Atomic identifiers; BPE must never split __init__ into __ + init.
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

    # ── C++ / Rust: namespace prefix and generic openers ─────────────────────
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
# 5.  DIGIT TOKENS  [v3-FIX-1]
#
#  DECISION: Empty — zero digit AddedTokens.
#
#  Single digits 0–9 are already in ByteLevel.alphabet().
#  \\p{N}{1,3} in BRAHMAI_PATTERN caps number chunk size at 3 digits.
#  BPE learns frequent 2/3-digit combinations naturally from corpus.
#  Adding forced digit tokens would create merge barriers around numbers
#  and waste 1,100 Phase 1 slots.
# ══════════════════════════════════════════════════════════════════════════════

DIGIT_TOKENS: list = []
_N_DIGITS = 0


# ══════════════════════════════════════════════════════════════════════════════
# 6.  WHITESPACE TOKENS  [v3-FIX-7]
# ══════════════════════════════════════════════════════════════════════════════

WHITESPACE_TOKENS = [
    "\n", "\n\n", "\n\n\n", "\r\n",       # newlines
    "\t", "\v", "\f",                      # tab / control chars
    "  ", "    ", "        ",              # 2/4/8-space indents
    "\n  ", "\n    ", "\n        ",        # newline + indent combos
    "\t\t",                                # double-tab
]
_N_WHITESPACE = len(WHITESPACE_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BRAHMAI PRETOKENIZER PATTERN
#     \\p{N}{1,3}: numbers split into ≤3-digit chunks.
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
    r"|[ ]?\p{N}{1,3}"
    r"|[ ]?[^\s\p{L}\p{N}\\{}\[\]^_$&#%~]+"
    r"|\r?\n+"
    r"|\s+"
)

V3_FALLBACK_PATTERN = r"[ \t]*[^ \t\n]+|[ \t]*\n"


def get_pretokenizer_pattern() -> Tuple[str, str]:
    try:
        import regex as _regex
        _regex.compile(r"\p{Tamil}+")
        return BRAHMAI_PATTERN, "BRAHMAI"
    except (ImportError, Exception):
        log.warning("'regex' unavailable — falling back to V3_FALLBACK. "
                    "Install: pip install regex --break-system-packages")
        return V3_FALLBACK_PATTERN, "V3_FALLBACK"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GPT-2 BYTE ENCODER
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

# [v3-FIX-6] Collapse runs of 3+ Ġ → ĠĠ (indent noise suppression)
_GSPACE_RUN = _re.compile(r"Ġ{3,}")


def encode_for_phase2(text: str) -> str:
    """
    GPT-2 byte-level encode + Ġ-run collapse.

    Space → Ġ, then runs of 3+ Ġ become ĠĠ to prevent indent noise from
    occupying superword slots (e.g. 'list:ĠĠĠĠĠĠĠĠĠ' → 'list:ĠĠ').

    Examples:
        "by the way"  → "byĠtheĠway"
        "        x"   → "ĠĠx"         (was "ĠĠĠĠĠĠĠĠx" before collapse)
    """
    encoded = "".join(_BYTE_ENCODER[b] for b in text.encode("utf-8"))
    return _GSPACE_RUN.sub("ĠĠ", encoded)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  BOILERPLATE PATTERNS  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

_BOILERPLATE_PATTERNS = [
    _re.compile(r"this solution is locked", _re.IGNORECASE),
    _re.compile(r"to view this solution", _re.IGNORECASE),
    _re.compile(r"you need to log in", _re.IGNORECASE),
    _re.compile(r"please sign up", _re.IGNORECASE),
    _re.compile(r"sign up or log in", _re.IGNORECASE),
    _re.compile(r"upgrade to premium", _re.IGNORECASE),
    _re.compile(r"community treasure hunt", _re.IGNORECASE),
    _re.compile(r"find the treasures in matlab central", _re.IGNORECASE),
    _re.compile(r"discover what matlab", _re.IGNORECASE),
    _re.compile(r"brainly\.", _re.IGNORECASE),
    _re.compile(r"chegg\.com", _re.IGNORECASE),
    _re.compile(r"coursehero\.com", _re.IGNORECASE),
    _re.compile(r"we use cookies", _re.IGNORECASE),
    _re.compile(r"accept all cookies", _re.IGNORECASE),
    _re.compile(r"cookie policy", _re.IGNORECASE),
    _re.compile(
        r"(home|about|contact|privacy|terms)\s*\|\s*(home|about|contact|privacy|terms)",
        _re.IGNORECASE,
    ),
    _re.compile(r"(\b\w+\b)(?:\s+\1){7,}", _re.IGNORECASE),
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
    shard_dir:  Path,
    num_shards: Optional[int] = None,
    seed:       int = DEFAULT_SEED,
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
# 12. CORPUS ITERATORS
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB = 500


def _two_bucket_dedup_check(
    seen_a: set, seen_b: set, h: int, bucket_size: int
) -> Tuple[bool, set, set]:
    if h in seen_a or h in seen_b:
        return True, seen_a, seen_b
    seen_a.add(h)
    if len(seen_a) >= bucket_size:
        seen_a, seen_b = set(), seen_a
    return False, seen_a, seen_b


def corpus_iterator(
    shard_dir:          Path,
    num_shards:         Optional[int] = None,
    seed:               int = DEFAULT_SEED,
    max_line_chars:     Optional[int] = None,
    dedup:              bool = DEDUP_ENABLED,
    bucket_size:        int  = DEDUP_BUCKET_SIZE,
    phase2_encode:      bool = False,
    filter_boilerplate: bool = False,
) -> Iterator[str]:
    paths         = select_shards(shard_dir, num_shards, seed)
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

    for path in paths:
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

    # DIGIT_TOKENS is intentionally empty — no digit AddedTokens [v3-FIX-1]
    all_tokens = special_objs + sci_objs + code_objs + ws_objs
    seen, deduped = set(), []
    for tok in all_tokens:
        if tok.content not in seen:
            seen.add(tok.content)
            deduped.append(tok)

    log.info(
        "Added tokens: %d special + %d sci + %d code + 0 digit + %d ws"
        " = %d total (after dedup: %d)",
        len(special_objs), _N_SCI, _N_CODE, _N_WHITESPACE,
        len(special_objs) + _N_SCI + _N_CODE + _N_WHITESPACE,
        len(deduped))
    log.info("  Digit AddedTokens: NONE — BPE learns freely [v3-FIX-1]")
    log.info("  No <0xXX> tokens — byte-level handles all bytes natively")
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# 14. POST-SAVE JSON PATCH
# ══════════════════════════════════════════════════════════════════════════════

def _patch_tokenizer_json(tok_path: Path, is_phase2: bool = False) -> None:
    with open(tok_path, encoding="utf-8") as f:
        state = json.load(f)

    # Ensure decoder is ByteLevel
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

    # Inference bridge: restore ByteLevel(use_regex=False) for Phase 2
    if is_phase2:
        state["pre_tokenizer"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        log.info("  [patch] pre_tokenizer → ByteLevel(use_regex=False) for inference")

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

    # Inject any whitespace tokens in vocab but not in added_tokens
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
# 15. PHASE 1  —  BYTE-LEVEL BPE + BRAHMAI_PATTERN
#     vocab_size = 200,000  [v3-FIX-2]
#     Pre-tokenizer order: Split FIRST, ByteLevel SECOND  (from v2, correct)
# ══════════════════════════════════════════════════════════════════════════════

def build_phase1_tokenizer(phase1_vocab_size: int, pattern: str, pattern_name: str):
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())
    tokenizer.decoder = ByteLevelDecoder()

    # CRITICAL ORDER: Split FIRST so \\p{N}{1,3} and [ ]? patterns see raw ASCII.
    # ByteLevel SECOND with use_regex=False: byte-encodes only, no GPT-2 word-split.
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
    log.info("  Pre-tokenizer : Split(%s, first) → ByteLevel(use_regex=False)",
             pattern_name)
    log.info("  Numbers       : \\p{N}{1,3} — ≤3-digit chunks, no AddedTokens")
    log.info("  SCI atoms     : %d (no cmd+brace) [v3-FIX-3]", _N_SCI)
    log.info("  CODE operators: %d [v3-FIX-4]", _N_CODE)
    log.info("  DIGIT tokens  : 0 [v3-FIX-1]")
    log.info("  initial_alphabet: ByteLevel.alphabet() = 256 byte symbols")
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 16. PHASE 2  —  SuperBPE: remove pretokenizer for training
#     min_frequency = 3  [v3-FIX-5]
#     Ġ pre-encoding with run collapse  [v3-FIX-6]
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
    log.info("  Phase 1 vocab=%d  merges=%d", p1_vocab_size, p1_merges)
    log.info("  Superword slots: %d", superword_slots)

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

    log.info("Phase 2: pretokenizer=None (training) | "
             "ByteLevel(use_regex=False) (inference) | "
             "target=%d | min_freq=%d [v3-FIX-5]",
             TOTAL_VOCAB_SIZE, MIN_FREQUENCY_P2)
    log.info("  Ġ pre-encoding + Ġ-run collapse  [v3-FIX-6]")
    log.info("  Boilerplate filter: ON")
    log.info("  Alphabet: 256 bytes → pair table = 65,536 entries (~50 MB)")
    return tokenizer, trainer, p1_vocab_size


# ══════════════════════════════════════════════════════════════════════════════
# 17. TRAINING RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def train_phase1(tokenizer, trainer, shard_dir: Path,
                 num_shards: Optional[int], seed: int):
    log.info("Training Phase 1 ...")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(
            shard_dir, num_shards, seed,
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


def train_phase2(tokenizer, trainer, shard_dir: Path,
                 num_shards: Optional[int], seed: int,
                 max_line_chars: int):
    log.info("Training Phase 2 SuperBPE ...")
    log.info("  max_line_chars  : %d", max_line_chars)
    log.info("  Ġ pre-encoding  : ON (with Ġ-run collapse [v3-FIX-6])")
    log.info("  Boilerplate     : ON")
    log.info("  min_frequency   : %d [v3-FIX-5]", MIN_FREQUENCY_P2)
    log.info("  Alphabet        : 256 bytes → 65,536 pair table (~50 MB)")
    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(
            shard_dir, num_shards, seed,
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
# 18. SAVE + PATCH + HF EXPORT
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
        log.info("Loaded via PreTrainedTokenizerFast")
    except Exception as e:
        log.warning("HF load failed (%s) — using raw tokenizer", e)
        from tokenizers import Tokenizer as _Tok
        tok = _Tok.from_file(str(load_dir / "tokenizer.json"))
        hf  = False

    vocab = tok.get_vocab()
    log.info("Vocab size: %d", len(vocab))

    # Byte-fallback
    byte_tokens = sum(1 for i in range(256) if f"<0x{i:02X}>" in vocab)
    log.info("  <0xXX> tokens: %d  %s",
             byte_tokens, "✓" if byte_tokens == 0 else "⚠ unexpected")

    # Special token IDs
    log.info("Special token IDs:")
    expected = FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING + LANGUAGE_TAGS + FIM
    mismatches = 0
    for eid, tstr in enumerate(expected):
        aid = vocab.get(tstr, -1)
        if aid != eid:
            log.warning("  MISMATCH %r  expected=%d  actual=%d", tstr, eid, aid)
            mismatches += 1
    log.info("  %d IDs correct %s", len(expected) - mismatches,
             "✓" if mismatches == 0 else f"✗ ({mismatches} mismatches)")

    # v3-FIX-1: no forced digit AddedTokens
    log.info("Digit token check [v3-FIX-1]:")
    at_boundary = len(SPECIAL_TOKENS) + _N_SCI + _N_CODE + _N_WHITESPACE
    forced_digits = sum(
        1 for i in range(100)
        if f"{i:02d}" in vocab and vocab[f"{i:02d}"] < at_boundary
    )
    log.info("  Forced 2-digit AddedTokens: %d  %s",
             forced_digits, "✓" if forced_digits == 0 else "⚠")
    for d in ["0", "1", "9", "42", "128", "256", "1024"]:
        log.info("  %-8r  id=%d", d, vocab.get(d, -1))

    # v3-FIX-4: operator coverage
    ops = ["==", "!=", ">=", "<=", "//", "/*", "->", "=>", "++", "--", "**",
           "===", "!==", "::", "..", "??"]
    log.info("Operator coverage [v3-FIX-4]:")
    missing_ops = [op for op in ops if op not in vocab]
    log.info("  %d / %d present%s",
             len(ops) - len(missing_ops), len(ops),
             f"  MISSING: {missing_ops}" if missing_ops else " ✓")

    # Round-trip tests
    rt_tests = [
        ("LaTeX frac",     r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"),
        ("LaTeX env",      r"\begin{equation} E = mc^2 \end{equation}"),
        ("LaTeX subscript","$x_{i+1} = x_i^2 + c$"),
        ("ASCII math",     "d/dt(T) = alpha * laplacian(T)"),
        ("Unicode math",   "∂u/∂t = α∇²u"),
        ("Hindi",          "नमस्ते दुनिया, यह परीक्षण है।"),
        ("Tamil",          "போகிறார்கள்"),
        ("Telugu",         "నమస్కారం"),
        ("Kannada",        "ನಮಸ್ಕಾರ"),
        ("Bengali",        "বাংলাদেশ"),
        ("Arabic",         "مرحبا بالعالم"),
        ("Chinese",        "你好世界"),
        ("Japanese",       "こんにちは世界"),
        ("Korean",         "안녕하세요 세계"),
        ("Code Python",    "    def fib(n: int) -> int:\n        return n"),
        ("Code operators", "if x == y and a != b and c >= 0:"),
        ("Code C++",       "std::vector<int> v; v.push_back(42);"),
        ("Code arrow",     "const fn = (x) => x * 2;"),
        ("Code comment",   "// single\n/* block */"),
        ("Rust macro",     'fn main() { println!("{}", Ok(42).unwrap()); }'),
        ("Preprocessor",   "#include <stdio.h>\n#define MAX 100"),
        ("Numbers 1-3d",   "lr=0.001 batch=128 epoch=2048"),
        ("Numbers 5d",     "12345 and 99999"),
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
        log.info("Fertility report (lower = better compression):")
        samples = {
            "English text": "The quick brown fox jumps over the lazy dog.",
            "Python code":  "    def fib(n):\n        if n<=1: return n\n"
                            "        return fib(n-1)+fib(n-2)",
            "LaTeX math":   r"\frac{\partial u}{\partial t}=\alpha\nabla^2 u"
                            r"+\beta u\cdot\nabla u",
            "Hindi":        "नमस्ते दुनिया, यह एक परीक्षण है।",
            "Arabic":       "مرحبا بالعالم هذا اختبار",
            "Numbers":      "batch=128 lr=0.001 epochs=10000 seed=42",
        }
        for name, sample in samples.items():
            ids = tok.encode(sample) if hf else tok.encode(sample).ids
            fertility = len(ids) / len(sample)
            log.info("  %-18s  %3d chars → %3d toks  (%.3f tok/char)",
                     name, len(sample), len(ids), fertility)

        log.info("Superword spot check:")
        for sw in ["ĠbyĠtheĠway", "ĠofĠthe", "ĠinĠthe", "ĠtoĠthe"]:
            log.info("  %-25r  %s", sw,
                     "IN VOCAB ✓" if sw in vocab else "not yet")

    log.info("VALIDATION %s", "ALL PASSED ✓" if all_ok else "SOME FAILED ✗")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 20. VOCABULARY BUDGET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def log_vocab_summary(phase1_vocab_size: int):
    fixed = _N_SPECIAL + _N_SCI + _N_CODE + _N_WHITESPACE
    bpe_p1_slots = phase1_vocab_size - fixed - 256
    log.info("━" * 65)
    log.info("VOCABULARY BUDGET  (v3)")
    log.info("━" * 65)
    log.info("  Special tokens    : %6d  (reserved=%d)", _N_SPECIAL, _N_RESERVED)
    log.info("  FIXED_SCI_VOCAB   : %6d  (atoms only) [v3-FIX-3]", _N_SCI)
    log.info("  FIXED_CODE_VOCAB  : %6d  (operators only) [v3-FIX-4]", _N_CODE)
    log.info("  DIGIT_TOKENS      :      0  (removed) [v3-FIX-1]")
    log.info("  WHITESPACE_TOKENS : %6d  [v3-FIX-7]", _N_WHITESPACE)
    log.info("  ─────────────────────────────")
    log.info("  Total fixed       : %6d", fixed)
    log.info("  ByteLevel.alphabet:    256  (initial_alphabet)")
    log.info("  Phase 1 BPE merges: ~%5d  (vocab=%d − fixed − 256)",
             bpe_p1_slots, phase1_vocab_size)
    log.info("  Phase 2 superwords: %6d", TOTAL_VOCAB_SIZE - phase1_vocab_size)
    log.info("  Total vocab target: %6d", TOTAL_VOCAB_SIZE)
    log.info("━" * 65)
    log.info("Phase 1 budget comparison across versions:")
    log.info("  v1: 200k vocab, ~8,400 fixed  → ~191,300 BPE merges")
    log.info("  v2: 180k vocab, ~2,900 fixed  → ~176,800 BPE merges  (-7.6%%)")
    log.info("  v3: %dk vocab, ~%d fixed  → ~%d BPE merges  [RESTORED + IMPROVED]",
             phase1_vocab_size // 1000, fixed, bpe_p1_slots)
    log.info("━" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 21. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(
    shards_dir       : Path,
    output_dir       : Path,
    num_shards       : Optional[int],
    seed             : int,
    validate         : bool,
    monitor_interval : float,
    phase1_vocab_size: int,
    phase1_only      : bool,
    phase2_only      : bool,
    phase1_checkpoint: Optional[Path],
    phase2_num_shards: Optional[int],
    phase2_max_chars : int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase1_checkpoint is None:
        phase1_checkpoint = output_dir / "phase1_checkpoint"

    p2_shards = phase2_num_shards if phase2_num_shards is not None else num_shards
    pattern, pattern_name = get_pretokenizer_pattern()

    log.info("=" * 65)
    log.info("  %s", SCRIPT_NAME.upper())
    log.info("=" * 65)
    log.info("  Phase 1 vocab    : %d  [v3-FIX-2]", phase1_vocab_size)
    log.info("  Phase 1 preток   : Split(%s) → ByteLevel(use_regex=False)",
             pattern_name)
    log.info("  Phase 1 min_freq : %d", MIN_FREQUENCY_P1)
    log.info("  Digit tokens     : NONE  [v3-FIX-1]")
    log.info("  SCI vocab        : %d atoms  [v3-FIX-3]", _N_SCI)
    log.info("  CODE vocab       : %d operators  [v3-FIX-4]", _N_CODE)
    log.info("  Phase 2 min_freq : %d  [v3-FIX-5]", MIN_FREQUENCY_P2)
    log.info("  Phase 2 Ġ-encode : ON + run-collapse  [v3-FIX-6]")
    log.info("  Phase 2 boilerplt: ON")
    log.info("  Phase 2 max_chars: %d", phase2_max_chars)
    log.info("  Dedup            : two-bucket rolling")
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
        "dedup_strategy"      : "two-bucket-rolling",
        "dedup_bucket_size"   : DEDUP_BUCKET_SIZE,
        "digit_tokens"        : "none",
        "phase2_g_encode"     : True,
        "phase2_g_run_collapse": True,
        "phase2_boilerplate"  : True,
        "n_sci_vocab"         : _N_SCI,
        "n_code_vocab"        : _N_CODE,
        "n_digit_tokens"      : 0,
        "num_reserved"        : NUM_RESERVED,
        "seed"                : seed,
        "p2_max_chars"        : phase2_max_chars,
        "p1_num_shards"       : num_shards,
        "p2_num_shards"       : p2_shards,
        "byte_fallback"       : False,
        "v3_fixes"            : [
            "FIX-1:no-digit-tokens",
            "FIX-2:phase1-200k",
            "FIX-3:sci-atoms-only-no-cmd-brace",
            "FIX-4:code-ops-only",
            "FIX-5:min-freq-p2-3",
            "FIX-6:g-run-collapse",
            "FIX-7:ws-optimised",
        ],
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not phase2_only:
        with stage("Phase 1: Build tokenizer"):
            tok1, tr1 = build_phase1_tokenizer(
                phase1_vocab_size, pattern, pattern_name)

        with stage("Phase 1: Train"):
            tok1 = train_phase1(tok1, tr1, shards_dir, num_shards, seed)

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
        description="SuperBPE v3 — 200k Phase 1, atom-only fixed vocab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python superbpe_bytelevel_dedup_v3.py \\
      --shards-dir ~/data/shards --output ~/superbpe_v3_out --validate

  # Phase 1 only
  python superbpe_bytelevel_dedup_v3.py \\
      --shards-dir ~/data/shards --output ~/superbpe_v3_out --phase1-only --validate

  # Phase 2 only
  python superbpe_bytelevel_dedup_v3.py \\
      --shards-dir ~/data/shards --output ~/superbpe_v3_out \\
      --phase2-only --phase1-checkpoint ~/superbpe_v3_out/phase1_checkpoint --validate

  # Check config
  python superbpe_bytelevel_dedup_v3.py --check-only
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
                   help=f"Max chars per Phase 2 line (default {DEFAULT_PHASE2_MAX_CHARS}; 0=no limit).")
    p.add_argument("--check-only",        action="store_true")

    args = p.parse_args()

    if args.phase1_only and args.phase2_only:
        p.error("--phase1-only and --phase2-only are mutually exclusive.")

    phase2_max_chars = args.phase2_max_chars if args.phase2_max_chars > 0 else None

    if args.check_only:
        log.info("=" * 65)
        log.info("%s — Check Mode", SCRIPT_NAME)
        log.info("=" * 65)
        log.info("  Phase 1 vocab     : %d  [v3-FIX-2]", DEFAULT_PHASE1_VOCAB)
        log.info("  Phase 2 min_freq  : %d  [v3-FIX-5]", MIN_FREQUENCY_P2)
        log.info("  Digit tokens      : NONE  [v3-FIX-1]")
        log.info("  FIXED_SCI_VOCAB   : %d  (atoms only)  [v3-FIX-3]", _N_SCI)
        log.info("  FIXED_CODE_VOCAB  : %d  (operators)   [v3-FIX-4]", _N_CODE)
        log.info("  WHITESPACE_TOKENS : %d  [v3-FIX-7]", _N_WHITESPACE)
        log.info("  NUM_RESERVED      : %d", NUM_RESERVED)
        log.info("  RAYON threads     : %d", NUM_THREADS)
        log.info("  CPU count         : %s", os.cpu_count())

        pattern, pname = get_pretokenizer_pattern()
        log.info("  Active pattern    : %s  (\\p{N}{1,3} rule active)", pname)

        try:
            from tokenizers.pre_tokenizers import ByteLevel as BL
            log.info("  ByteLevel.alphabet() size: %d symbols", len(BL.alphabet()))
        except Exception as e:
            log.error("  ByteLevel.alphabet() error: %s", e)

        try:
            from tokenizers import Regex
            Regex(BRAHMAI_PATTERN)
            log.info("  BRAHMAI_PATTERN   : OK")
        except Exception as e:
            log.error("  BRAHMAI_PATTERN   : INVALID — %s", e)

        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATE SPECIAL TOKENS"
        log.info("  Duplicate check   : OK")

        test_enc = encode_for_phase2("def foo() -> int:")
        assert "Ġ" in test_enc and " " not in test_enc, "Byte encoder broken"
        log.info("  Byte encoder      : OK  (%r)", test_enc[:30])

        test_indent = encode_for_phase2("        items:")
        g_count = test_indent.count("Ġ")
        assert g_count <= 2, f"Ġ-run collapse broken: {repr(test_indent)} (Ġ×{g_count})"
        log.info("  Ġ-run collapse    : OK  ('        items:' → %r)", test_indent)

        log_vocab_summary(DEFAULT_PHASE1_VOCAB)
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