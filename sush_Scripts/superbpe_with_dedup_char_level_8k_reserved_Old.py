#!/usr/bin/env python3
"""
superbpe_with_dedup.py
======================
SuperBPE  |  No sci_preprocessor  |  BRAHMAI_PATTERN  |  WITH line-level dedup

Bug fixes applied in both ablations:
  Fix 2 — FIXED_SCI_VOCAB special=False enforced post-save.
  Fix 3 — single_word=True for alphabetic FIXED_SCI_VOCAB tokens.

Ablation axis (this script):
  Fix 1 (dedup) — ENABLED.
  Rolling hash-set cache per corpus scan eliminates exact-duplicate lines.
  Prevents boilerplate paragraphs (e.g. Hindi copyright disclaimers repeating
  1000s of times) from dominating Phase 2 superword merges.

Special token layout:
  IDs  0 –   3   Foundational  : <pad> <eos> <bos> <unk>
  IDs  4 –   7   Utility       : <mask> [multimodal] [@BOS@] <|image_soft_token|>
  IDs  8 –  12   Chat          : <|system|> <|user|> <|assistant|> turn delimiters
  IDs 13 –  24   Tool use      : <tools> <tool_call> arg/response tokens
  IDs 25 –  27   Reasoning     : <think> </think> <|nothink|>
  IDs 28 –  52   Language tags : 25 tags (hin tam tel kan mal mar guj ben pan
                                  ory urd npi pus sin mya fas bod dzo eng deu
                                  fra rus cmn jpn kor)
  IDs 53 –  55   FIM           : fim_prefix fim_middle fim_suffix
  IDs 56 – 8247  Reserved      : 8192 slots

Usage
-----
  python superbpe_with_dedup.py --check-only

  # Phase 1 only
  python superbpe_with_dedup.py \\
      --shards-dir /path/to/shards \\
      --output ./out --phase1-only --validate

  # Phase 2 only (memory-optimized, dedup on)
  python superbpe_with_dedup.py \\
      --shards-dir /path/to/filtered_shards \\
      --seed 42 --output ./out \\
      --phase2-only \\
      --phase1-checkpoint ./out/phase1_checkpoint \\
      --phase2-max-chars 10000 \\
      --validate

  # Full two-phase
  python superbpe_with_dedup.py \\
      --shards-dir /path/to/shards \\
      --num-shards 5 --seed 42 \\
      --output ./out \\
      --phase2-max-chars 10000 --validate

  # Patch existing checkpoint (no retraining)
  python superbpe_with_dedup.py \\
      --patch-only \\
      --phase1-checkpoint /path/to/phase1_checkpoint --validate
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

ABLATION_NAME       = "superbpe_with_dedup"
TOTAL_VOCAB_SIZE    = 262_144
PHASE1_VOCAB_SIZE   = 200_000
MIN_FREQUENCY       = 2
NUM_THREADS         = 64
MODEL_MAX_LENGTH    = 8_192

DEFAULT_OUTPUT              = Path("./superbpe_with_dedup_out")
DEFAULT_SHARD_DIR           = Path("/path/to/shards")
DEFAULT_NUM_SHARDS: Optional[int] = None
DEFAULT_SEED                = 42
DEFAULT_PHASE1_VOCAB        = PHASE1_VOCAB_SIZE
DEFAULT_PHASE2_MAX_CHARS    = 10_000   # raised from 500; monitor RAM
DEFAULT_PHASE2_NUM_SHARDS   = None     # None = all available

# ── Fix 1: dedup config (ENABLED in this script) ─────────────────────────────
DEDUP_ENABLED    = True
DEDUP_MAX_CACHE  = 5_000_000  # rolling hash-set cap (~400 MB for 5M hashes)

os.environ["RAYON_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPECIAL TOKEN DEFINITIONS
#     IDs assigned in list order — do NOT reorder within groups.
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

# ── Language tags (IDs 28–52) ─────────────────────────────────────────────────
# 25 languages using ISO 639-3 codes.
# Indic first (by approximate training data volume), then other scripts.
LANGUAGE_TAGS = [
    "<lang_hin>",   # 28  Hindi        (Devanagari)
    "<lang_tam>",   # 29  Tamil
    "<lang_tel>",   # 30  Telugu
    "<lang_kan>",   # 31  Kannada
    "<lang_mal>",   # 32  Malayalam
    "<lang_mar>",   # 33  Marathi      (Devanagari)
    "<lang_guj>",   # 34  Gujarati
    "<lang_ben>",   # 35  Bengali
    "<lang_pan>",   # 36  Punjabi      (Gurmukhi)
    "<lang_ory>",   # 37  Odia
    "<lang_urd>",   # 38  Urdu
    "<lang_npi>",   # 39  Nepali       (Devanagari)
    "<lang_pus>",   # 40  Pashto
    "<lang_sin>",   # 41  Sinhala
    "<lang_mya>",   # 42  Burmese
    "<lang_fas>",   # 43  Dari/Persian
    "<lang_bod>",   # 44  Tibetan
    "<lang_dzo>",   # 45  Dzongkha
    "<lang_eng>",   # 46  English
    "<lang_deu>",   # 47  German
    "<lang_fra>",   # 48  French
    "<lang_rus>",   # 49  Russian      (Cyrillic)
    "<lang_cmn>",   # 50  Mandarin Chinese
    "<lang_jpn>",   # 51  Japanese
    "<lang_kor>",   # 52  Korean
]

assert len(LANGUAGE_TAGS) == 25, (
    f"Expected 25 LANGUAGE_TAGS (IDs 28–52), got {len(LANGUAGE_TAGS)}."
)

# ── FIM (IDs 53–55) — shifted up because LANGUAGE_TAGS grew from 9 → 25 ──────
FIM = [
    "<|fim_prefix|>",   # 53
    "<|fim_middle|>",   # 54
    "<|fim_suffix|>",   # 55
]

# ── Reserved (IDs 56–8247) ────────────────────────────────────────────────────
NUM_RESERVED = 8_192
RESERVED     = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]

# ── Assembled list ─────────────────────────────────────────────────────────────
SPECIAL_TOKENS = (
    FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING
    + LANGUAGE_TAGS + FIM + RESERVED
)

assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), \
    "DUPLICATE SPECIAL TOKENS FOUND"

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
# 3.  BYTE FALLBACK TOKENS
# ══════════════════════════════════════════════════════════════════════════════

BYTE_FALLBACK_TOKENS = [f"<0x{i:02X}>" for i in range(256)]
_N_BYTE = len(BYTE_FALLBACK_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  WHITESPACE TOKENS  (Fix 4 — common to both ablations)
# ══════════════════════════════════════════════════════════════════════════════

WHITESPACE_TOKENS = [
    "\n",     # line break
    "\n\n",   # paragraph separator
    "\r\n",   # Windows CRLF
    "\t",     # horizontal tab
    "\v",     # vertical tab
    "\f",     # form feed
]
_N_WHITESPACE = len(WHITESPACE_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIXED SCIENTIFIC VOCABULARY  (Fix 2 + Fix 3 — common to both ablations)
# ══════════════════════════════════════════════════════════════════════════════

FIXED_SCI_VOCAB = [
    "=", "+", "-", "*", "/", "^", "\u221a", "\u2211", "\u220f", "!",
    "<", ">", "\u2264", "\u2265", "\u2260",
    "\u2202\u209c", "\u2202\u2093", "\u2202\u1d67", "\u2202_z",
    "\u2202\u209c\u209c", "\u2202\u2093\u2093", "\u2202\u1d67\u1d67", "\u2202_zz",
    "\u2202_xy", "\u2202_xz", "\u2202_yz",
    "\u2207", "\u2207\u00b2", "\u2207\u00b7", "\u2207\u00d7",
    "\u222b", "\u222e", "\u2202",
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
    "dim_M", "dim_L", "dim_T", "dim_I", "dim_\u03b8", "dim_N", "dim_J",
    "atom_C", "atom_N", "atom_O", "atom_H", "atom_S", "atom_P",
    "atom_F", "atom_Cl", "atom_Br", "atom_I",
    "bond_single", "bond_double", "bond_triple", "bond_aromatic",
    "ring_open", "ring_close",
]
_N_SCI = len(FIXED_SCI_VOCAB)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BRAHMAI_PATTERN
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


def get_pretokenizer_pattern(use_brahmai: bool = True):
    if not use_brahmai:
        return V3_FALLBACK_PATTERN, "V3_FALLBACK"
    try:
        import regex as _regex
        _regex.compile(r"\p{Tamil}+")
        return BRAHMAI_PATTERN, "BRAHMAI"
    except (ImportError, Exception):
        log.warning("  'regex' unavailable — falling back to V3. "
                    "Install: pip install regex --break-system-packages")
        return V3_FALLBACK_PATTERN, "V3_FALLBACK"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RESOURCE MONITOR
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
        mem   = self._read_meminfo()
        tot   = mem.get("MemTotal", 1)
        avail = mem.get("MemAvailable", mem.get("MemFree", 0))
        used  = tot - avail
        swap  = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
        cache = mem.get("Buffers", 0) + mem.get("Cached", 0)
        rss   = self._read_rss_kb()
        l1, l5, l15 = self._read_load()
        ncpu  = os.cpu_count() or 1
        G     = 1 / (1024 * 1024)
        pct   = (used / tot) * 100 if tot else 0
        rss_gb = rss * G
        self._peak_rss = max(self._peak_rss, rss_gb)
        self._peak_ram = max(self._peak_ram, pct)
        warn = " ⚠⚠ CRITICAL" if pct >= self.ram_crit else (
               " ⚠ WARN"     if pct >= self.ram_warn  else "")
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
# 8.  SHARD DISCOVERY + SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def discover_shards(shard_dir: Path) -> list[Path]:
    shards = sorted(shard_dir.glob("shard_*.txt.gz"))
    if not shards:
        shards = sorted(shard_dir.glob("shard_*.txt"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.txt.gz (or shard_*.txt) files found in {shard_dir}"
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
# 9.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_line(raw: str) -> str:
    return raw.rstrip('\n\r')


# ══════════════════════════════════════════════════════════════════════════════
# 10. CORPUS ITERATOR  — Fix 1 (dedup) ENABLED
#
#  How dedup works here:
#    - A single hash-set spans all shards (catches cross-shard duplicates).
#    - hash(line) is fast and good enough for exact-match boilerplate removal.
#    - When the set grows beyond DEDUP_MAX_CACHE, it is cleared. This trades
#      perfect recall for bounded RAM: any line repeated > DEDUP_MAX_CACHE
#      apart is seen twice, but lines repeated thousands of times (the actual
#      problem) are almost certainly within a single cache window.
#    - For near-duplicate removal (slight whitespace variation) you'd need
#      datasketch MinHashLSH — pre-filter shards offline in that case.
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB  = 500
MIN_DOC_CHARS = 20


def corpus_iterator(
    shard_dir      : Path,
    num_shards     : Optional[int] = None,
    seed           : int            = DEFAULT_SEED,
    dry_run        : bool           = False,
    max_line_chars : Optional[int]  = None,
    dedup          : bool           = DEDUP_ENABLED,
    dedup_max_cache: int            = DEDUP_MAX_CACHE,
) -> Iterator[str]:
    paths          = select_shards(shard_dir, num_shards, seed)
    total_bytes    = 0
    total_yielded  = 0
    total_skipped  = 0
    long_skipped   = 0
    dedup_skipped  = 0
    last_log_bytes = 0
    t_start        = time.perf_counter()

    seen_hashes: set[int] = set() if dedup else set()
    if dedup:
        log.info("  Fix 1 (dedup): ENABLED  |  cache_size=%d  (~%.0f MB RAM)",
                 dedup_max_cache, dedup_max_cache * 28 / 1_048_576)
    else:
        log.info("  Fix 1 (dedup): DISABLED")

    if max_line_chars is not None:
        log.info("  max_line_chars=%d", max_line_chars)

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

                # ── Fix 1: dedup ───────────────────────────────────────────
                if dedup:
                    h = hash(line)
                    if h in seen_hashes:
                        dedup_skipped += 1
                        continue
                    seen_hashes.add(h)
                    if len(seen_hashes) > dedup_max_cache:
                        seen_hashes.clear()  # rolling window
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
# 11. ADDED TOKEN BUILDER  — Fix 2 + Fix 3 + Fix 4
# ══════════════════════════════════════════════════════════════════════════════

def _is_alpha_token(content: str) -> bool:
    return content.replace("_", "").isalpha()


def _build_added_tokens():
    from tokenizers import AddedToken

    special_objs = [
        AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS
    ]
    byte_objs = [
        AddedToken(t, special=True, normalized=False) for t in BYTE_FALLBACK_TOKENS
    ]
    sci_objs = []
    for t in FIXED_SCI_VOCAB:
        is_alpha = _is_alpha_token(t)
        sci_objs.append(AddedToken(
            t, special=False, normalized=False,
            single_word=is_alpha, lstrip=False, rstrip=False,
        ))
    ws_objs = [
        AddedToken(t, special=False, normalized=False,
                   single_word=False, lstrip=False, rstrip=False)
        for t in WHITESPACE_TOKENS
    ]

    all_tokens = special_objs + byte_objs + sci_objs + ws_objs
    seen, deduped = set(), []
    for tok in all_tokens:
        if tok.content not in seen:
            seen.add(tok.content)
            deduped.append(tok)

    n_alpha = sum(1 for t in FIXED_SCI_VOCAB if _is_alpha_token(t))
    log.info(
        "  Added tokens: %d special + %d byte + %d sci "
        "(%d alpha→single_word) + %d whitespace  =  %d total",
        _N_SPECIAL, _N_BYTE, _N_SCI, n_alpha, _N_WHITESPACE, len(deduped),
    )
    log.info(
        "  Special breakdown: %d foundational + %d utility + %d chat + "
        "%d tool + %d reasoning + %d lang + %d fim + %d reserved",
        len(FOUNDATIONAL), len(UTILITY), len(CHAT), len(TOOL_USE),
        len(REASONING), len(LANGUAGE_TAGS), len(FIM), NUM_RESERVED,
    )
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# 12. POST-SAVE JSON PATCH  — Fix 2 + Fix 3 + Fix 4  (shared with no_dedup)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_tokenizer_json(tok_path: Path) -> None:
    with open(tok_path, encoding="utf-8") as f:
        state = json.load(f)

    old_decoder = state.get("decoder")
    state["decoder"] = {"type": "Sequence", "decoders": []}
    log.info("  [patch] decoder: %s → Sequence([])", old_decoder)

    sci_set = set(FIXED_SCI_VOCAB)
    ws_set  = set(WHITESPACE_TOKENS)
    n_special = n_sword = n_ws = 0

    for entry in state.get("added_tokens", []):
        content = entry.get("content", "")
        if content in sci_set:
            if entry.get("special", True):
                entry["special"] = False
                n_special += 1
            if _is_alpha_token(content) and not entry.get("single_word", False):
                entry["single_word"] = True
                n_sword += 1
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
                "id": vocab[ws], "content": ws,
                "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": False,
            })
            injected += 1

    log.info("  [patch] sci special=False=%d  single_word=%d  "
             "ws normalized=False=%d  injected=%d",
             n_special, n_sword, n_ws, injected)

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("  [patch] Saved → %s (%d bytes)", tok_path, tok_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# 13. PHASE 1 — BRAHMAI_PATTERN char-level BPE
# ══════════════════════════════════════════════════════════════════════════════

def build_phase1_tokenizer(phase1_vocab_size: int, pattern_name: str, pattern: str):
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Split
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = Split(
        pattern=Regex(pattern), behavior="isolated", invert=True)

    deduped = _build_added_tokens()
    trainer = BpeTrainer(
        vocab_size=phase1_vocab_size, min_frequency=MIN_FREQUENCY,
        special_tokens=deduped, show_progress=True,
    )
    log.info("Phase 1: %s | char-level | vocab=%d", pattern_name, phase1_vocab_size)
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 14. PHASE 2 — SuperBPE: no pretokenizer
# ══════════════════════════════════════════════════════════════════════════════

def build_phase2_tokenizer(phase1_checkpoint_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.trainers import BpeTrainer

    tok_path = phase1_checkpoint_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {tok_path}\n"
            "Run Phase 1 first (--phase1-only) or full training."
        )

    log.info("Loading Phase 1 checkpoint: %s", tok_path)
    with open(tok_path, encoding="utf-8") as f:
        p1_state = json.load(f)

    p1_vocab  = len(p1_state.get("model", {}).get("vocab", {}))
    p1_merges = len(p1_state.get("model", {}).get("merges", []))
    log.info("  Phase 1 vocab=%d  merges=%d", p1_vocab, p1_merges)

    p2_state                  = p1_state.copy()
    p2_state["pre_tokenizer"] = None

    tokenizer = Tokenizer.from_str(json.dumps(p2_state))
    deduped   = _build_added_tokens()
    trainer   = BpeTrainer(
        vocab_size=TOTAL_VOCAB_SIZE, min_frequency=MIN_FREQUENCY,
        special_tokens=deduped, show_progress=True,
    )

    remaining = TOTAL_VOCAB_SIZE - p1_vocab
    log.info("Phase 2: no pretokenizer (SuperBPE) | +%d superword merges", remaining)
    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# 15. TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def train_phase(phase_num, tokenizer, trainer, shard_dir, num_shards, seed,
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
    log.info("TRAINING PHASE %d  |  target=%d  |  threads=%d  |  max_chars=%s",
             phase_num, trainer.vocab_size, NUM_THREADS,
             str(max_line_chars) if max_line_chars else "∞")
    log.info("=" * 65)

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator=corpus_iterator(shard_dir, num_shards, seed,
                                  max_line_chars=max_line_chars),
        trainer=trainer, length=None,
    )
    elapsed = time.perf_counter() - t0
    actual  = tokenizer.get_vocab_size()
    log.info("Phase %d complete in %.1f min. Vocab: %d", phase_num, elapsed / 60, actual)
    if phase_num == 1 and actual < trainer.vocab_size * 0.95:
        log.warning("Phase 1 vocab (%d) below target (%d) — increase shards?",
                    actual, trainer.vocab_size)
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 16. SAVE + PATCH + HF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_tokenizer(tokenizer, output_dir: Path, meta: dict, phase: str = "final"):
    save_dir = (output_dir / "phase1_checkpoint") if phase == "phase1" else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    tok_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    log.info("Saved raw tokenizer.json (%d bytes)", tok_path.stat().st_size)

    log.info("Applying patches (Sequence decoder + Fix 2+3+4)...")
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
        "unk_token"                   : UNK_TOKEN,
        "pad_token"                   : PAD_TOKEN,
        "mask_token"                  : "<mask>",
        "model_max_length"            : MODEL_MAX_LENGTH,
        "clean_up_tokenization_spaces": False,
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
    for t in BYTE_FALLBACK_TOKENS:
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
        "unk_token" : UNK_TOKEN,
        "pad_token" : PAD_TOKEN,
        "mask_token": {"content": "<mask>", "lstrip": False, "normalized": False,
                       "rstrip": False, "single_word": False},
        "additional_special_tokens": UTILITY + CHAT + TOOL_USE + REASONING
                                     + LANGUAGE_TAGS + FIM,
    }
    with open(save_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(stm, f, indent=2, ensure_ascii=False)

    with open(save_dir / "ablation_metadata.json", "w") as f:
        json.dump({**meta, "vocab_size_actual": len(sv),
                   "merges": len(merges), "phase": phase,
                   "special_tokens_count": _N_SPECIAL,
                   "reserved_slots": NUM_RESERVED,
                   "dedup_enabled": DEDUP_ENABLED,
                   "dedup_max_cache": DEDUP_MAX_CACHE}, f, indent=2)

    log.info("All files saved → %s", save_dir)
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# 17. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_tokenizer(output_dir: Path, phase: str = "final"):
    load_dir = (output_dir / "phase1_checkpoint") if phase == "phase1" else output_dir
    log.info("=" * 65)
    log.info("VALIDATION — %s  [%s]", load_dir, phase)

    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(str(load_dir))
        hf  = True
        log.info("Loaded via PreTrainedTokenizerFast")
    except Exception:
        from tokenizers import Tokenizer as _Tok
        tok = _Tok.from_file(str(load_dir / "tokenizer.json"))
        hf  = False
        log.info("Loaded via raw tokenizers.Tokenizer")

    vocab = tok.get_vocab()
    log.info("Vocab size: %d", len(vocab))

    log.info("")
    log.info("Special token ID checks:")
    for expected_id, tok_str in enumerate(
        FOUNDATIONAL + UTILITY + CHAT + TOOL_USE + REASONING + LANGUAGE_TAGS + FIM
    ):
        actual_id = vocab.get(tok_str, -1)
        if actual_id != expected_id:
            log.warning("  ID MISMATCH: %r  expected=%d  actual=%d",
                        tok_str, expected_id, actual_id)
    log.info("  First %d special tokens ID-verified",
             len(FOUNDATIONAL + UTILITY + CHAT + TOOL_USE
                 + REASONING + LANGUAGE_TAGS + FIM))

    log.info("")
    log.info("Language tag presence (all 25 expected):")
    for lt in LANGUAGE_TAGS:
        in_v = lt in vocab
        log.info("  %-16r → %s (id=%s)", lt,
                 "IN VOCAB ✓" if in_v else "NOT IN VOCAB ✗",
                 vocab.get(lt, "—"))

    log.info("")
    log.info("Fix 4 — whitespace token presence:")
    for ws in WHITESPACE_TOKENS:
        in_v = ws in vocab
        log.info("  %-6r → %s (id=%s)", ws,
                 "IN VOCAB ✓" if in_v else "NOT IN VOCAB ✗",
                 vocab.get(ws, "—"))

    tests = [
        ("LaTeX",        r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"),
        ("ASCII math",   "d/dt(T) = alpha * laplacian(T)"),
        ("Unicode math", "∂u/∂t = α∇²u"),
        ("Numbers",      "k = 3.14159 and T = 1e-5"),
        ("Hindi",        "नमस्ते दुनिया"),
        ("Tamil",        "போகிறார்கள்"),
        ("Telugu",       "నమస్కారం"),
        ("Kannada",      "ನಮಸ್ಕಾರ"),
        ("Code indent",  "    def foo():"),
        ("Code 8sp",     "        return x + 1"),
        ("Newline",      "def foo():\n    return x"),
        ("Multiline",    "x = 1\ny = 2\nz = x + y"),
        ("Tab indent",   "\tdef bar():\n\t\treturn 42"),
        ("tr-safety",    "trailing spaces   "),
        ("sin-safety",   "single sinusoidal"),
        ("exp-safety",   "experiment export expert"),
        ("operators",    "a = b + c - d * e / f"),
        ("sci-ops",      "sin(x) + cos(y) = tan(z)"),
        ("tr-word",      "tr(A) is the trace operator"),
        ("No indent",    "no indent here"),
        ("Multi space",  "a  =  b  +  c"),
        ("Mixed",        "The heat equation ∂u/∂t = α∇²u holds for நமஸ்தே"),
        ("<think> tag",  "<think>reasoning here</think>"),
        ("FIM",          "<|fim_prefix|>def f():<|fim_suffix|>    pass<|fim_middle|>"),
        ("Lang tag hin", "<lang_hin>नमस्ते दुनिया</lang_hin>"),
        ("Lang tag tam", "<lang_tam>தமிழ் உரை</lang_tam>"),
        ("Lang tag deu", "<lang_deu>Guten Morgen</lang_deu>"),
        ("Lang tag cmn", "<lang_cmn>你好世界</lang_cmn>"),
        ("Tool call",    "<tool_call><arg_key>fn</arg_key></tool_call>"),
    ]

    log.info("")
    log.info("Round-trip encode/decode tests (%s):", phase)
    all_pass = True
    for label, t in tests:
        ids = tok.encode(t) if hf else tok.encode(t).ids
        dec = (tok.decode(ids, skip_special_tokens=False)
               if hf else tok.decode(ids))
        match = (t == dec)
        if not match:
            all_pass = False
        log.info("  [%-18s] %3d toks | %s | %r",
                 label, len(ids), "OK " if match else "ERR", t[:50])
        if not match:
            log.warning("    MISMATCH: out=%r", dec[:50])

    if phase == "final":
        log.info("")
        log.info("Superword token checks (Phase 2 expected merges):")
        for sw in [r"\frac{\partial", r"\nabla^2", "d/dt",
                   "\n    ", "    return", " the", " equation"]:
            log.info("  %-28r  %s", sw, "IN VOCAB ✓" if sw in vocab else "not yet")

    log.info("")
    log.info("ALL PASSED ✓" if all_pass else "SOME FAILED ✗ — see mismatches above")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 18. PATCH-ONLY MODE
# ══════════════════════════════════════════════════════════════════════════════

def patch_existing_checkpoint(checkpoint_dir: Path) -> None:
    tok_path = checkpoint_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {checkpoint_dir}")
    log.info("Patching: %s", checkpoint_dir)
    _patch_tokenizer_json(tok_path)

    cfg_path = checkpoint_dir / "tokenizer_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(tok_path, encoding="utf-8") as f:
            vocab = json.load(f).get("model", {}).get("vocab", {})
        sci_set  = set(FIXED_SCI_VOCAB)
        ws_set   = set(WHITESPACE_TOKENS)
        existing = {e.get("content") for e in
                    cfg.get("added_tokens_decoder", {}).values()}
        n_sci = n_ws = 0
        for entry in cfg.get("added_tokens_decoder", {}).values():
            c = entry.get("content", "")
            if c in sci_set:
                entry["special"]     = False
                entry["single_word"] = _is_alpha_token(c)
                n_sci += 1
            if c in ws_set:
                entry["special"]    = False
                entry["normalized"] = False
                n_ws += 1
        for ws in WHITESPACE_TOKENS:
            if ws not in existing and ws in vocab:
                cfg["added_tokens_decoder"][str(vocab[ws])] = {
                    "content": ws, "lstrip": False, "normalized": False,
                    "rstrip": False, "single_word": False, "special": False,
                }
                n_ws += 1
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        log.info("  tokenizer_config.json: %d sci + %d ws patched", n_sci, n_ws)
    log.info("Patch complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 19. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(shards_dir, output_dir, num_shards, seed, dry_run, validate,
        monitor_interval, phase1_vocab_size, phase1_only, phase2_only,
        phase1_checkpoint, phase2_num_shards, phase2_max_chars):

    if phase1_checkpoint is None:
        phase1_checkpoint = output_dir

    p2_shards = phase2_num_shards if phase2_num_shards is not None else num_shards

    log.info("=" * 65)
    log.info("  %s", ABLATION_NAME.upper())
    log.info("=" * 65)
    log.info("  Fix 1 (dedup)    : ENABLED  (cache=%d)", DEDUP_MAX_CACHE)
    log.info("  Fix 2+3 (sci)    : ENABLED")
    log.info("  Fix 4 (ws)       : ENABLED")
    log.info("  Special tokens   : %d  (%d reserved)", _N_SPECIAL, NUM_RESERVED)
    log.info("  Language tags    : %d  (IDs 28–52)", len(LANGUAGE_TAGS))
    log.info("  FIM IDs          : 53–55")
    log.info("  Phase 1 shards   : %s", num_shards or "all")
    log.info("  Phase 1 vocab    : %d", phase1_vocab_size)
    log.info("  Phase 2 shards   : %s", p2_shards or "all")
    log.info("  Phase 2 max_chars: %s", str(phase2_max_chars) or "∞")
    log.info("  Phase 2 vocab    : %d  (%d superword slots)",
             TOTAL_VOCAB_SIZE, TOTAL_VOCAB_SIZE - phase1_vocab_size)
    log.info("  Shards dir       : %s", shards_dir)
    log.info("  Seed             : %d", seed)
    log.info("  Output           : %s", output_dir)
    log.info("=" * 65)

    pattern, pattern_name = get_pretokenizer_pattern(use_brahmai=True)
    log.info("  Active pattern   : %s", pattern_name)
    log.info("=" * 65)

    monitor = ResourceMonitor(interval_sec=monitor_interval)
    monitor.start()
    monitor.log_now("BASELINE")
    t_total = time.perf_counter()

    meta_base = {
        "ablation"           : ABLATION_NAME,
        "preprocessor"       : "none",
        "akshara_seg"        : False,
        "phase1_pattern"     : pattern_name,
        "phase2_pattern"     : "none (SuperBPE)",
        "phase1_vocab"       : phase1_vocab_size,
        "total_vocab"        : TOTAL_VOCAB_SIZE,
        "superword_slots"    : TOTAL_VOCAB_SIZE - phase1_vocab_size,
        "bpe_level"          : "character",
        "min_frequency"      : MIN_FREQUENCY,
        "phase1_num_shards"  : num_shards,
        "phase2_num_shards"  : p2_shards,
        "phase2_max_chars"   : phase2_max_chars,
        "seed"               : seed,
        "threads"            : NUM_THREADS,
        "special_tokens"     : _N_SPECIAL,
        "reserved_slots"     : NUM_RESERVED,
        "dedup_enabled"      : DEDUP_ENABLED,
        "dedup_max_cache"    : DEDUP_MAX_CACHE,
        "bug_fixes"          : ["Fix1:dedup=ON", "Fix2:special=F",
                                "Fix3:single_word", "Fix4:newline"],
    }

    if not phase2_only:
        with stage("Phase 1: Build"):
            tok1, tr1 = build_phase1_tokenizer(
                phase1_vocab_size, pattern_name, pattern)

        with stage("Phase 1: Train"):
            tok1 = train_phase(1, tok1, tr1, shards_dir, num_shards, seed,
                               dry_run, max_line_chars=None)

        if dry_run:
            monitor.stop()
            return

        with stage("Phase 1: Save + patch"):
            save_tokenizer(tok1, output_dir,
                           {**meta_base, "phase_description": "Phase 1 checkpoint"},
                           phase="phase1")

        if validate:
            with stage("Phase 1: Validate"):
                validate_tokenizer(output_dir, phase="phase1")

        if phase1_only:
            res = monitor.stop()
            log.info("Phase 1 done. Time: %.2f h  Peak RSS: %.1f GB",
                     (time.perf_counter() - t_total) / 3600,
                     res["peak_process_rss_gb"])
            return

    with stage("Phase 2: Build (SuperBPE — no pretokenizer)"):
        tok2, tr2 = build_phase2_tokenizer(
            phase1_checkpoint if phase2_only else output_dir)

    with stage("Phase 2: Train (superword merges, dedup=ON)"):
        tok2 = train_phase(2, tok2, tr2, shards_dir, p2_shards, seed,
                           dry_run, max_line_chars=phase2_max_chars)

    res     = monitor.stop()
    elapsed = time.perf_counter() - t_total

    with stage("Phase 2: Save + patch final"):
        save_tokenizer(tok2, output_dir,
                       {**meta_base,
                        "phase_description": "Final SuperBPE (Phase 1 + Phase 2)",
                        "time_total_sec"   : round(elapsed, 1),
                        "resources"        : res},
                       phase="final")

    if validate:
        with stage("Phase 2: Validate final"):
            validate_tokenizer(output_dir, phase="final")

    log.info("=" * 65)
    log.info("  DONE  |  %.2f h  |  Peak RSS %.1f GB  |  %s",
             elapsed / 3600, res["peak_process_rss_gb"], output_dir)
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 20. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=(
            f"{ABLATION_NAME}\n"
            "SuperBPE + BRAHMAI_PATTERN.\n"
            "Fix 1 (dedup): ENABLED. Fix 2+3+4: ENABLED."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 2 command:
  python superbpe_with_dedup.py \\
      --shards-dir /home/.../filtered_shards \\
      --seed 42 \\
      --output /home/.../superbpe_with_dedup_out \\
      --phase2-only \\
      --phase1-checkpoint /home/.../superbpe_with_dedup_out/phase1_checkpoint \\
      --phase2-max-chars 10000 \\
      --validate
        """,
    )
    p.add_argument("--shards-dir",         type=Path,  default=DEFAULT_SHARD_DIR)
    p.add_argument("--output",             type=Path,  default=DEFAULT_OUTPUT)
    p.add_argument("--num-shards",         type=int,   default=DEFAULT_NUM_SHARDS)
    p.add_argument("--seed",               type=int,   default=DEFAULT_SEED)
    p.add_argument("--monitor-interval",   type=float, default=30.0)
    p.add_argument("--dry-run",            action="store_true")
    p.add_argument("--validate",           action="store_true")
    p.add_argument("--phase1-vocab",       type=int,   default=DEFAULT_PHASE1_VOCAB)
    p.add_argument("--phase1-only",        action="store_true")
    p.add_argument("--phase2-only",        action="store_true")
    p.add_argument("--phase1-checkpoint",  type=Path,  default=None)
    p.add_argument("--phase2-shards",      type=int,   default=DEFAULT_PHASE2_NUM_SHARDS)
    p.add_argument("--phase2-max-chars",   type=int,   default=DEFAULT_PHASE2_MAX_CHARS,
                   help="Skip lines >N chars in Phase 2 (default 10000). 0=disable.")
    p.add_argument("--patch-only",         action="store_true")
    p.add_argument("--check-only",         action="store_true")
    args = p.parse_args()

    if args.phase1_only and args.phase2_only:
        p.error("--phase1-only and --phase2-only are mutually exclusive.")

    phase2_max_chars = args.phase2_max_chars if args.phase2_max_chars > 0 else None

    if args.patch_only:
        ckpt = args.phase1_checkpoint
        if ckpt is None:
            p.error("--patch-only requires --phase1-checkpoint <dir>")
        patch_existing_checkpoint(ckpt)
        if args.validate:
            validate_tokenizer(ckpt, phase="phase1")
        return

    if args.check_only:
        log.info("=" * 65)
        log.info("%s — Check Mode", ABLATION_NAME)
        log.info("=" * 65)
        log.info("  Fix 1 (dedup)   : ENABLED (cache=%d)", DEDUP_MAX_CACHE)
        log.info("  Special tokens  : %d", _N_SPECIAL)
        log.info("    Foundational  : %d  %s", len(FOUNDATIONAL), FOUNDATIONAL)
        log.info("    Utility       : %d  %s", len(UTILITY), UTILITY)
        log.info("    Chat          : %d  %s", len(CHAT), CHAT)
        log.info("    Tool use      : %d", len(TOOL_USE))
        log.info("    Reasoning     : %d  %s", len(REASONING), REASONING)
        log.info("    Language tags : %d", len(LANGUAGE_TAGS))
        for lt in LANGUAGE_TAGS:
            log.info("      %s", lt)
        log.info("    FIM           : %d  %s  (IDs 53–55)", len(FIM), FIM)
        log.info("    Reserved      : %d  (IDs 56–%d)", NUM_RESERVED, _N_SPECIAL - 1)
        log.info("  Whitespace toks : %s", WHITESPACE_TOKENS)
        try:
            import regex as _r; _r.compile(r"\p{Tamil}+")
            log.info("  regex library   : OK")
        except ImportError:
            log.warning("  regex library   : NOT FOUND — pip install regex")
        try:
            from tokenizers import Regex; Regex(BRAHMAI_PATTERN)
            log.info("  BRAHMAI_PATTERN : OK")
        except Exception as e:
            log.error("  BRAHMAI_PATTERN : INVALID — %s", e)
        pattern, pname = get_pretokenizer_pattern()
        log.info("  Active pattern  : %s", pname)
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATES!"
        log.info("  Duplicate check : OK")
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
        phase1_vocab_size = args.phase1_vocab,
        phase1_only       = args.phase1_only,
        phase2_only       = args.phase2_only,
        phase1_checkpoint = args.phase1_checkpoint,
        phase2_num_shards = args.phase2_shards,
        phase2_max_chars  = phase2_max_chars,
    )


if __name__ == "__main__":
    main()