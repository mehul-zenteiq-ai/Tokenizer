#!/usr/bin/env python3
"""
inspect_mathpile_stackexchange.py — Examine key structure of MathPile stackexchange files.

Prints:
  1. Top-level keys and their types
  2. If 'question' is a dict, its sub-keys and types
  3. If 'answers' is a list, the structure of the first answer
  4. A fully reconstructed text sample showing what the final output should look like
  5. Checks ALL stackexchange files (one record each) to see if structure is consistent

Usage:
    python inspect_mathpile_stackexchange.py
"""

import gzip
import json
import glob
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

BASE = "/home/sushmetha_zenteiq_com/raw_storage_mount"
PATTERN = f"{BASE}/math/MathPile/train/stackexchange/*.jsonl.gz"

def inspect_record(obj, label=""):
    """Print detailed structure of one JSON record."""
    print(f"\n  {'─'*80}")
    if label:
        print(f"  {label}")
        print(f"  {'─'*80}")

    # Top-level keys
    print(f"\n  Top-level keys ({len(obj)}):")
    for key in sorted(obj.keys()):
        val = obj[key]
        if isinstance(val, str):
            print(f"    {key:20s} → str  (len={len(val)}, preview: {val[:100]!r})")
        elif isinstance(val, dict):
            print(f"    {key:20s} → dict (keys: {sorted(val.keys())})")
        elif isinstance(val, list):
            print(f"    {key:20s} → list (len={len(val)}, item type: {type(val[0]).__name__ if val else '?'})")
        elif isinstance(val, (int, float)):
            print(f"    {key:20s} → {type(val).__name__}  (value: {val})")
        elif val is None:
            print(f"    {key:20s} → None")
        else:
            print(f"    {key:20s} → {type(val).__name__}")

    # Dig into 'question' if it's a dict
    if "question" in obj:
        q = obj["question"]
        if isinstance(q, dict):
            print(f"\n  'question' is a DICT — sub-keys:")
            for k in sorted(q.keys()):
                v = q[k]
                if isinstance(v, str):
                    print(f"    question.{k:20s} → str  (len={len(v)}, preview: {v[:100]!r})")
                elif isinstance(v, list):
                    print(f"    question.{k:20s} → list (len={len(v)})")
                elif isinstance(v, dict):
                    print(f"    question.{k:20s} → dict (keys: {sorted(v.keys())})")
                else:
                    print(f"    question.{k:20s} → {type(v).__name__}  (value: {v!r})")
        elif isinstance(q, str):
            print(f"\n  'question' is a STRING (len={len(q)}, preview: {q[:200]!r})")

    # Dig into 'answers' if it's a list
    if "answers" in obj:
        answers = obj["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            print(f"\n  'answers' is a list with {len(answers)} items")
            ans0 = answers[0]
            if isinstance(ans0, dict):
                print(f"  First answer keys:")
                for k in sorted(ans0.keys()):
                    v = ans0[k]
                    if isinstance(v, str):
                        print(f"    answers[0].{k:20s} → str  (len={len(v)}, preview: {v[:100]!r})")
                    else:
                        print(f"    answers[0].{k:20s} → {type(v).__name__}  (value: {str(v)[:100]!r})")
            else:
                print(f"  First answer type: {type(ans0).__name__}")
                print(f"  Value: {str(ans0)[:200]!r}")
        elif isinstance(answers, list):
            print(f"\n  'answers' is an EMPTY list")

    # Try to reconstruct full text
    print(f"\n  RECONSTRUCTED TEXT (best guess):")
    print(f"  {'='*60}")
    text_parts = []

    # Try question.Body, question.Title, or question as string
    if "question" in obj:
        q = obj["question"]
        if isinstance(q, dict):
            if "Body" in q:
                text_parts.append(q["Body"])
            elif "body" in q:
                text_parts.append(q["body"])
            elif "Title" in q:
                text_parts.append(q["Title"])
        elif isinstance(q, str):
            text_parts.append(q)

    if "answers" in obj and isinstance(obj["answers"], list):
        for i, ans in enumerate(obj["answers"][:2]):  # first 2 answers
            if isinstance(ans, dict):
                body = ans.get("Body", ans.get("body", ans.get("text", "")))
                if body:
                    text_parts.append(body)
            elif isinstance(ans, str):
                text_parts.append(ans)

    full_text = "\n\n".join(str(p) for p in text_parts)
    print(f"  {full_text[:500]}")
    print(f"  {'='*60}")
    print(f"  Total reconstructed length: {len(full_text)} chars")


def main():
    files = sorted(glob.glob(PATTERN))
    print(f"Found {len(files)} stackexchange files:\n")
    for f in files:
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  {os.path.basename(f):50s}  {size_mb:>8.1f} MB")

    if not files:
        print("No files found!")
        return

    # ── Deep inspection of first record from first file ──
    print(f"\n{'='*80}")
    print("  DEEP INSPECTION — first record from first file")
    print(f"{'='*80}")

    with gzip.open(files[0], "rt", encoding="utf-8") as fh:
        line = fh.readline().strip()
        obj = json.loads(line)
        inspect_record(obj, label=f"File: {os.path.basename(files[0])}")

    # ── Quick check: is structure consistent across all files? ──
    print(f"\n{'='*80}")
    print("  CONSISTENCY CHECK — first record from each file")
    print(f"{'='*80}\n")

    print(f"  {'File':<50s}  {'Top keys':>10s}  {'question type':>15s}  "
          f"{'answers type':>15s}  {'n_answers':>10s}")
    print(f"  {'─'*110}")

    for filepath in files:
        basename = os.path.basename(filepath)
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as fh:
                line = fh.readline().strip()
                obj = json.loads(line)

            n_keys = len(obj.keys())
            q = obj.get("question")
            q_type = type(q).__name__
            if isinstance(q, dict):
                q_type += f"({','.join(sorted(q.keys())[:5])})"

            ans = obj.get("answers", [])
            a_type = type(ans).__name__
            n_ans = len(ans) if isinstance(ans, list) else "?"

            if isinstance(ans, list) and ans:
                a0 = ans[0]
                if isinstance(a0, dict):
                    a_type += f"({','.join(sorted(a0.keys())[:5])})"

            print(f"  {basename:<50s}  {n_keys:>10d}  {q_type:>15s}  "
                  f"{a_type:>15s}  {str(n_ans):>10s}")

        except Exception as e:
            print(f"  {basename:<50s}  ERROR: {e}")

    # ── Also check 5th record from a larger file to catch variations ──
    big_files = [f for f in files if os.path.getsize(f) > 10*1024*1024]
    if big_files:
        print(f"\n{'='*80}")
        print(f"  SPOT CHECK — 5th record from largest file")
        print(f"{'='*80}")

        biggest = max(big_files, key=os.path.getsize)
        with gzip.open(biggest, "rt", encoding="utf-8") as fh:
            for i in range(5):
                line = fh.readline().strip()
            obj = json.loads(line)
            inspect_record(obj, label=f"File: {os.path.basename(biggest)}, record #5")

    print("\nDone.")


if __name__ == "__main__":
    main()