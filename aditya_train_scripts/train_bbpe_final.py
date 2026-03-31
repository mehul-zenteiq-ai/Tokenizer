#!/usr/bin/env python3
"""
train_tokenizer_bbpe.py
=======================
Byte-level BPE, 262,144 vocab, random shard selection.

Derived from train_tokenizer_bpe.py (character-level BPE v3).

Key difference from the char-level script:
  - Pre-tokenizer: Sequence([ Split(regex), ByteLevel(add_prefix_space=False) ])
    Regex runs first to produce pretokens, then ByteLevel encodes each piece
    into the GPT-2 byte→unicode mapping (256 base tokens).
  - Decoder: ByteLevel() — reverses the byte mapping on decode.
  - BPE model: byte_fallback=False — all 256 bytes are in vocab natively,
    no fallback needed.
  - Initial alphabet: ByteLevel.alphabet() (256 tokens) passed to BpeTrainer.

Everything else is identical to the char-level script:
  - Same PATTERN_DENSE regex options
  - Same SPECIAL_TOKENS / ADDED_TOKENS
  - Same shard discovery, streaming, random selection
  - Same save & verify pipeline

Vocabulary budget:
    IDs     0 –   N_special-1  :  special tokens  (meaningful + reserved)
    IDs  N_special – N_special+N_added-1  :  added tokens (LaTeX + indent)
    IDs  next 256 slots        :  byte-level base alphabet (ByteLevel.alphabet())
    IDs  remaining – 262,143   :  BPE merges (learned from corpus)
    Total = exactly 262,144

    NOTE: BpeTrainer fills merges automatically to hit vocab_size exactly.

Loading:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(
        tokenizer_file="/path/to/tokenizer_bbpe/tokenizer.json"
    )

Usage:
    python train_tokenizer_bbpe.py --check-only
    python train_tokenizer_bbpe.py --dry-run
    python train_tokenizer_bbpe.py --num-shards 15 --seed 42
    python train_tokenizer_bbpe.py --num-shards 22 --seed 42  # all shards
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import argparse
import contextlib
import gzip
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

# ── HuggingFace tokenizers ─────────────────────────────────────────────────────
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

# ── HuggingFace transformers ───────────────────────────────────────────────────
from transformers import PreTrainedTokenizerFast

# ── project token lists ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from special_tokens import (  # noqa: E402
    SPECIAL_TOKENS,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
)
from added_tokens import ADDED_TOKENS  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE       = 262144   # 2^18, divisible by 128
MIN_FREQUENCY    = 2
NUM_THREADS      = 64
os.environ["RAYON_NUM_THREADS"] = str(NUM_THREADS)

MODEL_MAX_LENGTH = 8_192     # metadata only

DEFAULT_OUTPUT = Path(
    "/home/sushmetha_zenteiq_com/normal_bbpe_code_v2"
)
# python train_bbpe_final.py --num-shards 5 --seed 42
SHARD_DIR = Path("/home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/Curated_Corpus_180GB/filtered_shards_all_langs")
# /home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/Curated_Corpus_180GB/shards

DEFAULT_NUM_SHARDS = None   # None = use all available
DEFAULT_SEED       = 42

# ── Token counts (for logging only) ──────────────────────────────────────────
_N_SPECIAL = len(SPECIAL_TOKENS)
_N_ADDED   = len(ADDED_TOKENS)

# ── Logging ────────────────────────────────────────────────────────────────────

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
# A.  PRETOKENIZER REGEX
# ══════════════════════════════════════════════════════════════════════════════

PATTERN_VERBOSE = r"""(?x)

    # ── Priority 1a: LaTeX commands ──────────────────────────────────────────
    [ ]?\\[a-zA-Z]+\*?

    # ── Priority 1b: LaTeX non-letter backslash escapes ───────────────────────
    |\\[^a-zA-Z\s]

    # ── Priority 2: LaTeX structural characters ───────────────────────────────
    |[ ]?[{}\[\]^_$&#%~]

    # ── Priority 3: English contractions ──────────────────────────────────────
    |(?i:'s|'t|'re|'ve|'m|'ll|'d)

    # ── Priority 4: Indic scripts (one branch per script = no cross-script BPE)
    |[ ]?\p{Devanagari}+
    |[ ]?\p{Tamil}+
    |[ ]?\p{Telugu}+
    |[ ]?\p{Kannada}+
    |[ ]?\p{Malayalam}+
    |[ ]?\p{Gujarati}+
    |[ ]?\p{Bengali}+

    # ── Priority 5: Greek letters ──────────────────────────────────────────────
    |[ ]?\p{Greek}+

    # ── Priority 6: Mathematical Unicode symbols ───────────────────────────────
    |[\u2200-\u22FF\u2100-\u214F\u27C0-\u27EF\u2980-\u29FF\u2A00-\u2AFF]+

    # ── Priority 7: Latin words ────────────────────────────────────────────────
    |[ ]?\p{Latin}+

    # ── Priority 7b: Remaining Unicode letter scripts ─────────────────────────
    |[ ]?\p{L}+

    # ── Priority 8: Individual digits (one at a time) ─────────────────────────
    |[ ]?[0-9]

    # ── Priority 9: Operators and remaining punctuation ───────────────────────
    |[ ]?[^\s\p{L}\p{N}\\{}\[\]^_$&#%~]+

    # ── Priority 10: Newlines ─────────────────────────────────────────────────
    |\r?\n

    # ── Priority 11: Remaining whitespace ─────────────────────────────────────
    |\s+
"""

# ── Dense pattern (full Unicode — uncomment to use) ───────────────────────────
# PATTERN_DENSE = (
#     r"[ ]?\\[a-zA-Z]+\*?"
#     r"|\\[^a-zA-Z\s]"
#     r"|[ ]?[{}\[\]^_$&#%~]"
#     r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)"
#     r"|[ ]?\p{Devanagari}+"
#     r"|[ ]?\p{Tamil}+"
#     r"|[ ]?\p{Telugu}+"
#     r"|[ ]?\p{Kannada}+"
#     r"|[ ]?\p{Malayalam}+"
#     r"|[ ]?\p{Gujarati}+"
#     r"|[ ]?\p{Bengali}+"
#     r"|[ ]?\p{Greek}+"
#     r"|[\u2200-\u22FF\u2100-\u214F\u27C0-\u27EF\u2980-\u29FF\u2A00-\u2AFF]+"
#     r"|[ ]?\p{Latin}+"
#     r"|[ ]?\p{L}+"
#     r"|[ ]?[0-9]"
#     r"|[ ]?[^\s\p{L}\p{N}\\{}\[\]^_$&#%~]+"
#     r"|\r?\n+"
#     r"|\s+"
# )

# ── Dense pattern ────────────────────────────────
# PATTERN_DENSE = r" ?[0-9]| ?[^(\s|[.,!?…。，、।۔،])]+"

PATTERN_DENSE = (
    # === Indic scripts ===
    r" ?\p{Devanagari}+"        # Hindi, Marathi, Nepali
    r"| ?\p{Tamil}+"
    r"| ?\p{Telugu}+"
    r"| ?\p{Kannada}+"
    r"| ?\p{Malayalam}+"
    r"| ?\p{Gujarati}+"
    r"| ?\p{Bengali}+"
    r"| ?\p{Oriya}+"            # Odia
    r"| ?\p{Gurmukhi}+"         # Punjabi
    r"| ?\p{Sinhala}+"

    # === Arabic-script languages (Urdu, Farsi, Pashto, Arabic) ===
    r"| ?[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"

    # === CJK (Mandarin, Japanese Kanji) ===
    r"| ?[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF\uF900-\uFAFF]+"

    # === Japanese (Hiragana + Katakana) ===
    r"| ?[\u3040-\u309F]+"
    r"| ?[\u30A0-\u30FF\u31F0-\u31FF]+"

    # === Korean (Hangul) ===
    r"| ?[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]+"

    # === Cyrillic (Russian) ===
    r"| ?\p{Cyrillic}+"

    # === Myanmar / Burmese ===
    r"| ?[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]+"

    # === Greek ===
    r"| ?\p{Greek}+"

    # === Latin (English, French, German, etc.) ===
    r"| ?\p{Latin}+"

    # === Fallback (Bloom-style): any non-whitespace, non-punctuation run ===
    r"| ?[^(\s|[.,!?…。，、।۔،])]+"
)

PATTERN_ASCII_FALLBACK = (
    r"[ ]?\\[a-zA-Z]+\*?"
    r"|\\[^a-zA-Z\s]"
    r"|[ ]?[{}\[\]^_$&#%~]"
    r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[ ]?\w+"
    r"|[ ]?[0-9]"
    r"|[ ]?[^\s\w\\{}\[\]^_$&#%~]+"
    r"|\r?\n"
    r"|\s+"
)

TEST_STRINGS = {
    "latex_math"        : r"\frac{\partial^2 u}{\partial x^2}",
    "env_wrap"          : r"\begin{equation} E = mc^2 \end{equation}",
    "mixed_latin_math"  : r"The Schrödinger equation is \hat{H}\psi = E\psi",
    "devanagari"        : "यह एक परीक्षण है",
    "tamil"             : "தமிழ் மொழி",
    "digits_subscripts" : r"x_1 + x_2 = 3.14",
    "code_indent"       : "def solve(x):\n    return x**2\n",
}

_EXPECTED_MIN_TOKENS = {
    "latex_math"        : 6,
    "env_wrap"          : 5,
    "devanagari"        : 2,
    "digits_subscripts" : 8,
    "code_indent"       : 5,
}


def _hf_pretokenize(pattern_str: str, text: str) -> list[str]:
    """Pretokenize using Split only (no ByteLevel) — for regex check."""
    pre_tok = Split(
        pattern=Regex(pattern_str),
        behavior="isolated",
        invert=True,
    )
    return [piece for piece, _span in pre_tok.pre_tokenize_str(text)]


def run_compatibility_check() -> str:
    import regex as _regex

    log.info("=" * 65)
    log.info("SECTION A — PRETOKENIZER COMPATIBILITY CHECK")
    log.info("=" * 65)

    pattern_to_use = PATTERN_DENSE
    try:
        Split(pattern=Regex(PATTERN_DENSE), behavior="isolated", invert=True)
        log.info("PASS  Dense pattern compiles in Rust backend.")
    except Exception as exc:
        log.warning("FAIL  Dense pattern rejected: %s", exc)
        try:
            Split(pattern=Regex(PATTERN_ASCII_FALLBACK),
                  behavior="isolated", invert=True)
            log.warning(
                "WARN  Using ASCII fallback. \\p{Script=X} unsupported. "
                "Indic cross-script isolation NOT guaranteed. "
                "Run: pip install -U tokenizers"
            )
            pattern_to_use = PATTERN_ASCII_FALLBACK
        except Exception as exc2:
            raise RuntimeError(f"ASCII fallback also failed: {exc2}") from exc2

    py_regex = _regex.compile(PATTERN_VERBOSE, _regex.UNICODE)

    all_pass = True
    for name, text in TEST_STRINGS.items():
        py_tokens = py_regex.findall(text)
        hf_pieces = _hf_pretokenize(pattern_to_use, text)
        match = (py_tokens == hf_pieces)
        if not match:
            all_pass = False
        log.info("%s  [%s]  py=%d  hf=%d",
                 "PASS" if match else "DIFF", name,
                 len(py_tokens), len(hf_pieces))
        if not match:
            log.warning("     Python : %s", py_tokens[:8])
            log.warning("     HF     : %s", hf_pieces[:8])

    log.info("-" * 65)
    _tok = Tokenizer(BPE())
    _tok.pre_tokenizer = Split(
        pattern=Regex(pattern_to_use), behavior="isolated", invert=True
    )
    _tok.add_special_tokens([
        AddedToken(r"\begin{",          special=False, normalized=False),
        AddedToken(r"\begin{equation}", special=False, normalized=False),
    ])
    enc = _tok.encode(r"\begin{equation} E = mc^2 \end{equation}")
    if enc.tokens and enc.tokens[0] == r"\begin{equation}":
        log.info("PASS  \\begin{equation} wins over \\begin{ (longest-match).")
    else:
        log.warning(
            "FAIL  First token is %r — check tokenizers version (need ≥0.13).",
            enc.tokens[0] if enc.tokens else "<empty>")
        all_pass = False

    log.info("-" * 65)
    for name, min_count in _EXPECTED_MIN_TOKENS.items():
        py_tokens = py_regex.findall(TEST_STRINGS[name])
        ok = len(py_tokens) >= min_count
        if not ok:
            all_pass = False
        log.info("%s  [%s] %d pretokens (expected ≥%d)",
                 "PASS" if ok else "FAIL", name,
                 len(py_tokens), min_count)

    log.info("=" * 65)
    log.info("%s  Using: %s",
             "ALL CHECKS PASSED." if all_pass else "SOME CHECKS FAILED.",
             "PATTERN_DENSE" if pattern_to_use == PATTERN_DENSE
             else "PATTERN_ASCII_FALLBACK")
    log.info("=" * 65)
    return pattern_to_use


# ══════════════════════════════════════════════════════════════════════════════
# C.  SHARD DISCOVERY + RANDOM SELECTION
#
#     Why sort before sampling:
#       os.listdir / glob order is filesystem-dependent and non-deterministic.
#       Sorting first ensures seed=42 always picks the same shards regardless
#       of OS, filesystem, or directory state.
# ══════════════════════════════════════════════════════════════════════════════

def discover_shards(shard_dir: Path) -> list[Path]:
    """Return all shard_*.txt.gz files in shard_dir, sorted by filename."""
    shards = sorted(shard_dir.glob("shard_*.txt.gz"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.txt.gz files found in {shard_dir}"
        )
    return shards


def select_shards(
    shard_dir  : Path,
    num_shards : int | None = None,
    seed       : int        = DEFAULT_SEED,
) -> list[Path]:
    """
    Discover all shards and randomly select num_shards of them.

    Args:
        shard_dir  : directory containing shard_*.txt.gz files
        num_shards : how many to select. None = use all.
        seed       : RNG seed — same seed always picks the same shards.

    Returns:
        Sorted list of selected paths (sorted for deterministic iteration).
    """
    available = discover_shards(shard_dir)

    if num_shards is None or num_shards >= len(available):
        if num_shards is not None and num_shards > len(available):
            log.warning(
                "Requested %d shards but only %d available — using all.",
                num_shards, len(available)
            )
        selected = available
    else:
        rng      = random.Random(seed)
        selected = sorted(rng.sample(available, num_shards))

    log.info("Shard selection (seed=%d):", seed)
    log.info("  Available : %d", len(available))
    log.info("  Selected  : %d", len(selected))
    for p in selected:
        log.info("    %s", p.name)

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# C.  DATA ITERATOR
# ══════════════════════════════════════════════════════════════════════════════

LOG_EVERY_MB  = 500
MIN_DOC_CHARS = 20


def corpus_iterator(
    shard_dir  : Path       = SHARD_DIR,
    num_shards : int | None = None,
    seed       : int        = DEFAULT_SEED,
    dry_run    : bool       = False,
):
    """
    Yields one document (str) at a time from the selected shards.

    Args:
        shard_dir  : directory containing shard_*.txt.gz files
        num_shards : how many shards to use (None = all available)
        seed       : RNG seed for shard selection
        dry_run    : iterate without yielding (data access check only)
    """
    paths = select_shards(shard_dir, num_shards, seed)

    total_bytes   = 0
    total_docs    = 0
    total_skipped = 0
    last_log      = 0
    t_start       = time.perf_counter()

    pbar = tqdm(desc="Docs", unit=" docs", smoothing=0.1,
                disable=dry_run, mininterval=5.0)

    for shard_idx, path in enumerate(paths):
        shard_bytes = 0
        shard_docs  = 0
        t_shard     = time.perf_counter()
        pbar.set_postfix(shard=f"{path.name} [{shard_idx+1}/{len(paths)}]")

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if len(line) < MIN_DOC_CHARS:
                    total_skipped += 1
                    continue

                byte_len    = len(line.encode("utf-8"))
                total_bytes += byte_len
                shard_bytes += byte_len
                total_docs  += 1
                shard_docs  += 1

                if not dry_run:
                    yield line

                pbar.update(1)

                if total_bytes - last_log >= LOG_EVERY_MB * 1_048_576:
                    elapsed = time.perf_counter() - t_start
                    log.info(
                        "  progress  [%s]  total=%.1f GB  docs=%d  "
                        "speed=%.1f MB/s",
                        path.name,
                        total_bytes / 1_073_741_824,
                        total_docs,
                        (total_bytes / 1_048_576) / elapsed if elapsed else 0,
                    )
                    last_log = total_bytes

        log.info("  %-22s  %.2f GB  %d docs  %.1f s",
                 path.name,
                 shard_bytes / 1_073_741_824,
                 shard_docs,
                 time.perf_counter() - t_shard)

    pbar.close()
    log.info(
        "Corpus done.  %.2f GB | %d docs | %d skipped | %.1f s total",
        total_bytes / 1_073_741_824,
        total_docs,
        total_skipped,
        time.perf_counter() - t_start,
    )


# ══════════════════════════════════════════════════════════════════════════════
# B.  TOKENIZER + TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def build_tokenizer_and_trainer(pattern: str):
    """
    Byte-level BPE.

    Pre-tokenizer = Sequence([ Split(regex), ByteLevel(add_prefix_space=False) ]):
      1. Regex splits text into pretokens (LaTeX-aware, script-aware, etc.)
      2. ByteLevel encodes each pretoken into GPT-2 byte→unicode mapping.
         e.g. space → 'Ġ', newline → 'Ċ', multi-byte UTF-8 → multiple
         single-byte unicode chars from the 256-entry mapping.

    Decoder = ByteLevel():
      Reverses the byte→unicode mapping on decode.
      Each byte-mapped unicode char is converted back to its original byte,
      then the resulting bytes are decoded as UTF-8.

    BPE model: byte_fallback=False because all 256 bytes are natively in
    the vocabulary via ByteLevel.alphabet() — no fallback needed.

    Initial alphabet = ByteLevel.alphabet() (256 tokens) so BpeTrainer
    starts from the full byte vocabulary and learns merges on top.
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN, byte_fallback=False, fuse_unk=True))

    # Regex split first, then byte-level encoding
    tokenizer.pre_tokenizer = Sequence([
        Split(
            pattern=Regex(pattern),
            behavior="isolated",
            invert=True,
        ),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevel decoder reverses the byte→unicode mapping
    tokenizer.decoder = ByteLevelDecoder()

    special_token_objects = [
        AddedToken(tok, special=True, normalized=False)
        for tok in SPECIAL_TOKENS
    ]
    added_token_objects = [
        AddedToken(tok, special=False, normalized=False,
                   lstrip=False, rstrip=False)
        for tok in ADDED_TOKENS
    ]
    all_trainer_specials = special_token_objects + added_token_objects

    trainer = BpeTrainer(
        vocab_size        = VOCAB_SIZE,
        min_frequency     = MIN_FREQUENCY,
        special_tokens    = all_trainer_specials,
        initial_alphabet  = ByteLevel.alphabet(),
        show_progress     = True,
    )

    log.info("Tokenizer and trainer configured.")
    log.info("  mode              = BYTE-LEVEL BPE")
    log.info("  vocab_size        = %d", VOCAB_SIZE)
    log.info("  min_frequency     = %d", MIN_FREQUENCY)
    log.info("  num_threads       = %d", NUM_THREADS)
    log.info("  special_tokens    = %d  (%d special + %d added)",
             len(all_trainer_specials),
             len(special_token_objects),
             len(added_token_objects))
    log.info("  decoder           = ByteLevel()")
    log.info("  pre_tokenizer     = Sequence([ Split(regex), ByteLevel() ])")
    log.info("  initial_alphabet  = ByteLevel.alphabet() (256 bytes)")
    log.info("  byte_fallback     = False")
    log.info("  add_prefix_space  = False")

    return tokenizer, trainer


# ══════════════════════════════════════════════════════════════════════════════
# D.  TRAINING CALL
# ══════════════════════════════════════════════════════════════════════════════

def train(
    tokenizer,
    trainer,
    shard_dir  : Path,
    num_shards : int | None,
    seed       : int,
    dry_run    : bool = False,
):
    if dry_run:
        log.info("DRY RUN — iterating corpus only, no training.")
        for _ in corpus_iterator(shard_dir, num_shards, seed, dry_run=True):
            pass
        return tokenizer

    log.info("=" * 65)
    log.info("TRAINING — byte-level BPE")
    log.info("  Threads : %d", NUM_THREADS)
    log.info("  RAM peak: ~40-80 GB expected")
    log.info("=" * 65)

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator = corpus_iterator(shard_dir, num_shards, seed),
        trainer  = trainer,
        length   = None,
    )
    elapsed = time.perf_counter() - t0

    actual_vocab = tokenizer.get_vocab_size()
    log.info("Training complete in %.1f minutes.", elapsed / 60)
    log.info("Vocabulary size: %d (target: %d)", actual_vocab, VOCAB_SIZE)

    if actual_vocab != VOCAB_SIZE:
        log.error(
            "VOCAB SIZE MISMATCH: got %d, expected %d.",
            actual_vocab, VOCAB_SIZE
        )

    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# E.  SAVE AND VERIFY
# ══════════════════════════════════════════════════════════════════════════════

def save_and_verify(tokenizer: Tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("SECTION E — SAVE & VERIFY")
    log.info("  Output: %s", output_dir)
    log.info("=" * 65)

    tok_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_json_path))
    log.info("Saved tokenizer.json  (%d bytes)", tok_json_path.stat().st_size)

    tok_config_path = output_dir / "tokenizer_config.json"
    tok_config_path.write_text(json.dumps({
        "bos_token"       : BOS_TOKEN,
        "eos_token"       : EOS_TOKEN,
        "pad_token"       : PAD_TOKEN,
        "unk_token"       : UNK_TOKEN,
        "model_max_length": MODEL_MAX_LENGTH,
        "padding_side"    : "right",
        "truncation_side" : "right",
    }, indent=2))
    log.info("Saved tokenizer_config.json")

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file   = str(tok_json_path),
        bos_token        = BOS_TOKEN,
        eos_token        = EOS_TOKEN,
        pad_token        = PAD_TOKEN,
        unk_token        = UNK_TOKEN,
        model_max_length = MODEL_MAX_LENGTH,
    )

    vocab = fast_tok.get_vocab()

    # ── Vocab size ─────────────────────────────────────────────────────────────
    if len(vocab) == VOCAB_SIZE:
        log.info("PASS  vocab size = %d", len(vocab))
    else:
        log.error("FAIL  vocab size = %d (expected %d)", len(vocab), VOCAB_SIZE)

    # ── Special token IDs ──────────────────────────────────────────────────────
    id_errors = []
    for expected_id, token_str in enumerate(SPECIAL_TOKENS):
        actual_id = vocab.get(token_str, -1)
        if actual_id != expected_id:
            id_errors.append(
                f"  '{token_str}'  expected={expected_id}  actual={actual_id}"
            )
    if not id_errors:
        log.info("PASS  All %d special token IDs match.", len(SPECIAL_TOKENS))
    else:
        log.error("FAIL  %d special token ID mismatches:", len(id_errors))
        for err in id_errors[:20]:
            log.error(err)
        if len(id_errors) > 20:
            log.error("  ... and %d more", len(id_errors) - 20)

    # ── Added token IDs (spot-check first 10 + last 10) ───────────────────────
    added_errors = []
    indices = set(
        list(range(min(10, len(ADDED_TOKENS)))) +
        list(range(max(0, len(ADDED_TOKENS) - 10), len(ADDED_TOKENS)))
    )
    for i in sorted(indices):
        expected_id = _N_SPECIAL + i
        actual_id   = vocab.get(ADDED_TOKENS[i], -1)
        if actual_id != expected_id:
            added_errors.append(
                f"  ADDED[{i}] '{ADDED_TOKENS[i]}'  "
                f"expected={expected_id}  actual={actual_id}"
            )
    if not added_errors:
        log.info("PASS  Added token ID spot-check passed.")
    else:
        log.error("FAIL  Added token ID mismatches:")
        for err in added_errors:
            log.error(err)

    # ── Foundational tokens ────────────────────────────────────────────────────
    for actual, expected, label in [
        (fast_tok.pad_token, PAD_TOKEN, "pad_token"),
        (fast_tok.eos_token, EOS_TOKEN, "eos_token"),
        (fast_tok.bos_token, BOS_TOKEN, "bos_token"),
        (fast_tok.unk_token, UNK_TOKEN, "unk_token"),
    ]:
        if actual == expected:
            log.info("PASS  %-15s = %r", label, actual)
        else:
            log.error("FAIL  %-15s = %r  (expected %r)", label, actual, expected)

    # ── Vocab breakdown ────────────────────────────────────────────────────────
    n_fixed     = _N_SPECIAL + _N_ADDED
    n_remaining = len(vocab) - n_fixed
    log.info(
        "Vocab breakdown: %d special | %d added | 256 byte-base | ~%d BPE merges",
        _N_SPECIAL, _N_ADDED, max(0, n_remaining - 256)
    )

    # ── Round-trip tests ───────────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("Round-trip encode/decode tests:")
    rt_strings = [
        r"\frac{\partial^2 u}{\partial x^2}",
        r"\begin{equation} E = mc^2 \end{equation}",
        "यह एक परीक्षण है",
        "தமிழ் மொழி",
        "ఉష్ణ సమీకరణం",
        "ಶಾಖ ಸಮೀಕರಣ",
        r"x_1 + x_2 = 3.14",
        "def solve(x):\n    return x**2\n",
    ]
    all_rt_pass = True
    for s in rt_strings:
        ids     = fast_tok.encode(s, add_special_tokens=False)
        decoded = fast_tok.decode(ids, skip_special_tokens=False)
        ok      = (decoded == s)
        if not ok:
            all_rt_pass = False
        log.info("  %s  %r", "OK " if ok else "ERR", s[:60])
        if not ok:
            log.warning("       expected : %r", s)
            log.warning("       got      : %r", decoded)

    if all_rt_pass:
        log.info("PASS  All round-trip tests passed.")
    else:
        log.error("FAIL  Some round-trip tests failed.")

    log.info("=" * 65)
    log.info("Save complete.")
    log.info("")
    log.info("Load with:")
    log.info("  from transformers import PreTrainedTokenizerFast")
    log.info("  tok = PreTrainedTokenizerFast(")
    log.info("      tokenizer_file='%s'", tok_json_path)
    log.info("  )")

    return fast_tok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 262,144-vocab byte-level BPE tokenizer."
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="Regex compatibility check only; do not train.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Iterate corpus without training; verifies data access.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Save directory (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--shard-dir", type=Path, default=SHARD_DIR,
        help="Directory containing shard_*.txt.gz files.",
    )
    p.add_argument(
        "--num-shards", type=int, default=DEFAULT_NUM_SHARDS,
        help=(
            "How many shards to randomly select. "
            "Default: all available shards. "
            "Must be ≤ total shards in --shard-dir."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=(
            f"RNG seed for shard selection (default: {DEFAULT_SEED}). "
            "Same seed = same subset, guaranteed reproducible."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    log.info("Scientific Tokenizer — BYTE-LEVEL BPE")
    log.info("  vocab_size    = %d", VOCAB_SIZE)
    log.info("  min_frequency = %d", MIN_FREQUENCY)
    log.info("  num_threads   = %d", NUM_THREADS)
    log.info("  shard_dir     = %s", args.shard_dir)
    log.info("  num_shards    = %s", args.num_shards or "all available")
    log.info("  seed          = %d", args.seed)
    log.info("  output_dir    = %s", args.output_dir)

    with stage("Regex compatibility check"):
        active_pattern = run_compatibility_check()

    if args.check_only:
        log.info("--check-only set; done.")
        return

    # Preview shard selection before committing to training
    log.info("Previewing shard selection...")
    select_shards(args.shard_dir, args.num_shards, args.seed)

    with stage("Build tokenizer and trainer"):
        tokenizer, trainer = build_tokenizer_and_trainer(active_pattern)

    with stage("Training"):
        tokenizer = train(
            tokenizer,
            trainer,
            shard_dir  = args.shard_dir,
            num_shards = args.num_shards,
            seed       = args.seed,
            dry_run    = args.dry_run,
        )

    if args.dry_run:
        log.info("--dry-run complete. No tokenizer saved.")
        return

    with stage("Save and verify"):
        save_and_verify(tokenizer, args.output_dir)


if __name__ == "__main__":
    main()


"""
# ── Run commands ─────────────────────────────────────────────────────────────

# 1. Regex check (always first)
python aditya/train_tokenizer_bbpe.py --check-only

# 2. Dry run — verifies shard selection + data access
python aditya/train_tokenizer_bbpe.py --dry-run --num-shards 15 --seed 42

# 3. Full training
tmux new -s tok_bbpe
systemd-run --uid=$(id -u) --gid=$(id -g) --scope \\
  -p MemoryMax=200G -p OOMScoreAdjust=-900 \\
  python aditya/train_tokenizer_bbpe.py --num-shards 15 --seed 42

# Use all 22 shards
python aditya/train_tokenizer_bbpe.py --num-shards 22 --seed 42

# 4. Load
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(
    tokenizer_file="/home/sushmetha_zenteiq_com/aditya/tokenizer_bbpe_v1/tokenizer.json"
)
"""