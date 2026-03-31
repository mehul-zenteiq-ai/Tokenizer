#!/usr/bin/env python3
"""
filter_shard.py — Two-layer Unicode filter for tokenizer training shard.

Layer 1 (document-level):  Drop lines where >30% of letter characters
                           fall outside the allowed Unicode ranges.
Layer 2 (character-level): In surviving lines, delete every character
                           whose codepoint is outside the allowed set.
Safety rail:               If a line shrinks below min-chars after Layer 2,
                           discard it.

Allowed Unicode ranges — covers ALL 26 target languages:

  ── Core (English, LaTeX, Code, Math) ────────────────────────────────────
  ASCII                         U+0000–007F
  Latin Extended + IPA + Mods   U+00A0–02FF   English, French, German, IPA
  Combining Diacritical Marks   U+0300–036F
  Greek and Coptic              U+0370–03FF   α β γ in math/physics
  General Punct → Math Ops      U+2000–2AFF
  Math Alphanumeric Symbols     U+1D400–1D7FF

  ── Cyrillic (Russian) ──────────────────────────────────────────────────
  Cyrillic + Extended           U+0400–052F

  ── Arabic (Urdu, Pashto, Dari) ─────────────────────────────────────────
  Arabic + Supplement + Ext-A   U+0600–06FF, U+0750–077F, U+08A0–08FF
  Arabic Presentation Forms     U+FB50–FDFF, U+FE70–FEFF

  ── South Asian (10 Indic + Sinhala) ─────────────────────────────────────
  Devanagari → Sinhala          U+0900–0DFF  (contiguous block)

  ── Myanmar ───────────────────────────────────────────────────────────
  Myanmar                       U+1000–109F

  ── CJK + Japanese + Korean ──────────────────────────────────────────────
  Hangul Jamo                   U+1100–11FF
  CJK Symbols + Hiragana + Katakana  U+3000–30FF
  Hangul Compat Jamo            U+3130–318F
  CJK Extension A + Unified    U+3400–9FFF
  Hangul Extended + Syllables   U+A960–A97F, U+AC00–D7AF

What gets REMOVED:
  - Emojis (all blocks)
  - Thai, Lao, Ethiopic, Hebrew, Armenian, Georgian, Khmer, Tibetan
  - Variation selectors, ZWJ, tags
  - All other scripts not in the target list

Usage:
  python filter_shard.py --input shard.txt.gz --output-dir out/ --output-name filtered.txt.gz
  python filter_shard.py --dry-run --input shard.txt.gz
"""

import argparse
import gzip
import logging
import os
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PATH = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount/Tokenizer_Data/Curated_Corpus_all_langs/shard_newlang_01.txt.gz"
)
OUTPUT_DIR = Path(
    "/home/sushmetha_zenteiq_com/raw_storage_mount"
    "/Tokenizer_Data/Curated_Corpus_180GB/filtered_shards_all_langs"
)
OUTPUT_FILENAME = "shard_newlang_01_filtered.txt.gz"

DEFAULT_WORKERS   = 64
DEFAULT_THRESHOLD = 0.30
MIN_CHARS_AFTER   = 0
BATCH_SIZE        = 50_000

# ══════════════════════════════════════════════════════════════════════════════
# ALLOWED UNICODE RANGES — ALL 26 TARGET LANGUAGES
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_RANGES = [
    # ── Core: ASCII, Latin, Math ──────────────────────────────────────────
    (0x0000, 0x007F),    # ASCII (English, LaTeX, code, digits)
    (0x00A0, 0x02FF),    # Latin Extended A/B + IPA + Spacing Modifiers (ˆ ˜ ˙)
    (0x0300, 0x036F),    # Combining Diacritical Marks
    (0x0370, 0x03FF),    # Greek and Coptic (α β γ Σ Δ)

    # ── Cyrillic (Russian) ────────────────────────────────────────────────
    (0x0400, 0x052F),    # Cyrillic + Cyrillic Supplement

    # ── Arabic (Urdu, Pashto, Dari/Farsi) ─────────────────────────────────
    (0x0600, 0x06FF),    # Arabic
    (0x0750, 0x077F),    # Arabic Supplement
    (0x08A0, 0x08FF),    # Arabic Extended-A

    # ── South Asian: Devanagari through Sinhala (contiguous) ──────────────
    # Devanagari 0900  Hindi, Marathi, Nepali
    # Bengali    0980  Bengali, Bangla (BD)
    # Gurmukhi   0A00  Punjabi
    # Gujarati   0A80
    # Oriya      0B00  Odia
    # Tamil      0B80
    # Telugu     0C00
    # Kannada    0C80
    # Malayalam  0D00
    # Sinhala    0D80
    (0x0900, 0x0DFF),

    # ── Myanmar (Burmese) ─────────────────────────────────────────────────
    (0x1000, 0x109F),

    # ── Hangul Jamo (Korean) ──────────────────────────────────────────────
    (0x1100, 0x11FF),

    # ── General Punctuation → Supplemental Math Operators ─────────────────
    (0x2000, 0x2AFF),

    # ── CJK Symbols + Hiragana + Katakana ─────────────────────────────────
    (0x3000, 0x30FF),    # CJK Symbols (3000) + Hiragana (3040) + Katakana (30A0)

    # ── Hangul Compatibility Jamo ─────────────────────────────────────────
    (0x3130, 0x318F),

    # ── CJK Ideographs (Mandarin, Japanese kanji) ─────────────────────────
    (0x3400, 0x9FFF),    # CJK Extension A (3400–4DBF) + CJK Unified (4E00–9FFF)

    # ── Hangul Extended + Syllables (Korean) ──────────────────────────────
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),    # Hangul Syllables

    # ── Arabic Presentation Forms (Urdu/Pashto ligatures) ─────────────────
    (0xFB50, 0xFDFF),    # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),    # Arabic Presentation Forms-B

    # ── Mathematical Alphanumeric Symbols ─────────────────────────────────
    (0x1D400, 0x1D7FF),  # 𝐴 𝑥 𝐵 𝑦 etc.
]

ALLOWED_RANGES.sort()


def is_allowed(cp: int) -> bool:
    """Check if a codepoint falls within any allowed range."""
    for lo, hi in ALLOWED_RANGES:
        if cp < lo:
            return False
        if cp <= hi:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT IDENTIFICATION  (for stats reporting only)
# ══════════════════════════════════════════════════════════════════════════════

def script_label(cp: int) -> str:
    """Best-effort label for stripped codepoints — used only in the report."""
    # Emoji (the main target of this filter)
    if 0x1F600 <= cp <= 0x1F64F: return "Emoji_Face"
    if 0x1F300 <= cp <= 0x1F5FF: return "Emoji_Symbol"
    if 0x1F680 <= cp <= 0x1F6FF: return "Emoji_Transport"
    if 0x1F900 <= cp <= 0x1F9FF: return "Emoji_Supplement"
    if 0x1FA00 <= cp <= 0x1FAFF: return "Emoji_Extended"
    if 0xFE00 <= cp <= 0xFE0F:  return "Variation_Selector"
    if cp == 0x200D:             return "ZWJ"
    if 0xE0000 <= cp <= 0xE007F: return "Tag"
    # Scripts we exclude
    if 0x0E00 <= cp <= 0x0E7F:  return "Thai"
    if 0x0E80 <= cp <= 0x0EFF:  return "Lao"
    if 0x1200 <= cp <= 0x137F:  return "Ethiopic"
    if 0x0590 <= cp <= 0x05FF:  return "Hebrew"
    if 0x0530 <= cp <= 0x058F:  return "Armenian"
    if 0x10A0 <= cp <= 0x10FF:  return "Georgian"
    if 0x1780 <= cp <= 0x17FF:  return "Khmer"
    if 0xFF00 <= cp <= 0xFFEF:  return "Halfwidth_Fullwidth"
    if cp == 0xFFFD:            return "REPLACEMENT"
    try:
        name = unicodedata.name(chr(cp), "UNKNOWN")
        return name.split()[0]
    except Exception:
        return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def filter_batch(args):
    lines, threshold, min_chars_after = args

    kept = []
    stats = {
        "docs_in": len(lines), "docs_dropped_l1": 0, "docs_dropped_l2": 0,
        "docs_kept": 0, "bytes_in": 0, "bytes_out": 0, "chars_stripped": 0,
        "l1_dominant_script": Counter(), "l2_scripts_stripped": Counter(),
    }

    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        stats["bytes_in"] += line_bytes

        # ── Layer 1: Document-level ──────────────────────────────────────
        total_letters = 0
        unwanted_letters = 0
        unwanted_by_script = Counter()

        for ch in line:
            cp = ord(ch)
            if ch.isalpha():
                total_letters += 1
                if not is_allowed(cp):
                    unwanted_letters += 1
                    unwanted_by_script[script_label(cp)] += 1

        if total_letters > 0 and (unwanted_letters / total_letters) > threshold:
            stats["docs_dropped_l1"] += 1
            if unwanted_by_script:
                stats["l1_dominant_script"][unwanted_by_script.most_common(1)[0][0]] += 1
            continue

        # ── Layer 2: Character-level ─────────────────────────────────────
        cleaned_chars = []
        for ch in line:
            cp = ord(ch)
            if is_allowed(cp):
                cleaned_chars.append(ch)
            else:
                stats["chars_stripped"] += 1
                stats["l2_scripts_stripped"][script_label(cp)] += 1

        cleaned_line = "".join(cleaned_chars)

        if len(cleaned_line) < min_chars_after:
            stats["docs_dropped_l2"] += 1
            continue

        stats["docs_kept"] += 1
        stats["bytes_out"] += len(cleaned_line.encode("utf-8"))
        kept.append(cleaned_line)

    return kept, stats


def merge_stats(accumulated, batch_stats):
    for key in ("docs_in", "docs_dropped_l1", "docs_dropped_l2",
                "docs_kept", "bytes_in", "bytes_out", "chars_stripped"):
        accumulated[key] += batch_stats[key]
    accumulated["l1_dominant_script"] += batch_stats["l1_dominant_script"]
    accumulated["l2_scripts_stripped"] += batch_stats["l2_scripts_stripped"]


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
GB = 1024 ** 3
MB = 1024 ** 2


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-layer Unicode filter for tokenizer training shard."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=OUTPUT_FILENAME)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS_AFTER)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    input_size = args.input.stat().st_size
    output_path = args.output_dir / args.output_name

    log.info("=" * 70)
    log.info("SHARD FILTER — Two-Layer Unicode Cleaning (ALL LANGUAGES)")
    log.info("=" * 70)
    log.info("  Input          : %s  (%.2f GB compressed)", args.input, input_size / GB)
    log.info("  Output         : %s", output_path if not args.dry_run else "(dry run)")
    log.info("  Workers        : %d", args.workers)
    log.info("  L1 threshold   : %.0f%% unwanted letters -> drop doc", args.threshold * 100)
    log.info("  L2 min chars   : %d (after strip)", args.min_chars)
    log.info("  Allowed ranges : %d blocks", len(ALLOWED_RANGES))
    for lo, hi in ALLOWED_RANGES:
        log.info("    U+%04X – U+%04X", lo, hi)
    log.info("=" * 70)

    # ── Read ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("Reading input shard...")
    t_read = time.perf_counter()

    batches = []
    current_batch = []
    total_lines_read = 0

    with gzip.open(args.input, "rt", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            current_batch.append(line)
            total_lines_read += 1
            if len(current_batch) >= args.batch_size:
                batches.append(current_batch)
                current_batch = []
            if total_lines_read % 500_000 == 0:
                log.info("  ... read %d lines (%d batches)", total_lines_read, len(batches))

    if current_batch:
        batches.append(current_batch)

    read_time = time.perf_counter() - t_read
    log.info("Read: %d lines in %d batches (%.1f s)", total_lines_read, len(batches), read_time)

    # ── Filter ────────────────────────────────────────────────────────────
    log.info("")
    log.info("Filtering with %d workers...", args.workers)
    t_filter = time.perf_counter()

    batch_args = [(batch, args.threshold, args.min_chars) for batch in batches]

    total_stats = {
        "docs_in": 0, "docs_dropped_l1": 0, "docs_dropped_l2": 0,
        "docs_kept": 0, "bytes_in": 0, "bytes_out": 0, "chars_stripped": 0,
        "l1_dominant_script": Counter(), "l2_scripts_stripped": Counter(),
    }

    all_kept_lines = []
    batches_done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for kept_lines, batch_stats in pool.map(filter_batch, batch_args):
            all_kept_lines.extend(kept_lines)
            merge_stats(total_stats, batch_stats)
            batches_done += 1
            if batches_done % 10 == 0 or batches_done == len(batches):
                log.info("  batch %d/%d  kept=%d  dropped_l1=%d  stripped=%d chars",
                         batches_done, len(batches),
                         total_stats["docs_kept"],
                         total_stats["docs_dropped_l1"],
                         total_stats["chars_stripped"])

    filter_time = time.perf_counter() - t_filter
    log.info("Filtering complete (%.1f s)", filter_time)
    del batch_args, batches

    # ── Write ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        log.info("")
        log.info("Writing output...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        t_write = time.perf_counter()

        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for i, line in enumerate(all_kept_lines):
                fh.write(line + "\n")
                if (i + 1) % 500_000 == 0:
                    log.info("  ... wrote %d / %d lines", i + 1, len(all_kept_lines))

        write_time = time.perf_counter() - t_write
        output_size = output_path.stat().st_size
        log.info("Write complete: %s (%.2f GB compressed, %.1f s)",
                 output_path, output_size / GB, write_time)
    else:
        log.info("")
        log.info("DRY RUN — no output written.")

    # ── Report ────────────────────────────────────────────────────────────
    s = total_stats
    total_time = time.perf_counter() - t_read

    log.info("")
    log.info("=" * 70)
    log.info("FILTER REPORT")
    log.info("=" * 70)
    log.info("")
    log.info("Documents:")
    log.info("  Input              : %12d", s["docs_in"])
    log.info("  Dropped (Layer 1)  : %12d  (%.2f%%)",
             s["docs_dropped_l1"], 100 * s["docs_dropped_l1"] / max(1, s["docs_in"]))
    log.info("  Dropped (Layer 2)  : %12d  (%.2f%%)",
             s["docs_dropped_l2"], 100 * s["docs_dropped_l2"] / max(1, s["docs_in"]))
    log.info("  Kept               : %12d  (%.2f%%)",
             s["docs_kept"], 100 * s["docs_kept"] / max(1, s["docs_in"]))
    log.info("")
    log.info("Bytes (uncompressed):")
    log.info("  Input              : %12d  (%.3f GB)", s["bytes_in"], s["bytes_in"] / GB)
    log.info("  Output             : %12d  (%.3f GB)", s["bytes_out"], s["bytes_out"] / GB)
    log.info("  Reduction          : %.2f%%", 100 * (1 - s["bytes_out"] / max(1, s["bytes_in"])))
    log.info("")
    log.info("Layer 2 — Characters stripped: %d", s["chars_stripped"])

    if s["l1_dominant_script"]:
        log.info("")
        log.info("Layer 1 — Docs dropped by dominant unwanted script:")
        for script, count in s["l1_dominant_script"].most_common(20):
            log.info("  %-25s  %6d docs", script, count)

    if s["l2_scripts_stripped"]:
        log.info("")
        log.info("Layer 2 — Characters stripped by script (top 20):")
        for script, count in s["l2_scripts_stripped"].most_common(20):
            log.info("  %-25s  %6d chars", script, count)

    log.info("")
    log.info("Total time: %.1f s (read=%.1f  filter=%.1f  write=%.1f)",
             total_time, read_time, filter_time,
             total_time - read_time - filter_time)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
