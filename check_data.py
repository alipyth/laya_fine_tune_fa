"""Check the training data before every training run. Standard library only.

    python check_data.py

Fails when:
  * a label is not one of the four benchmark labels, or a text is empty or duplicated;
  * a dev text also appears in train (the dev score would be inflated);
  * any train or dev text is a near-duplicate of a held-out benchmark message
    (character 3-gram Jaccard similarity >= FAIL_AT after light normalisation).
Warns for similarity >= WARN_AT so you can read those pairs yourself.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _stream in (sys.stdout, sys.stderr):   # Windows consoles/pipes may not be UTF-8
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
LABELS = ("billing", "technical", "cancel", "other")
FAIL_AT, WARN_AT = 0.50, 0.35

_MAP = str.maketrans({"ي": "ی", "ك": "ک", "‌": " "})
_DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def load(name):
    path = HERE / "data" / name
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def grams(text, n=3):
    t = re.sub(r"[^\w ]", " ", text.lower().translate(_MAP).translate(_DIG))
    t = " " + re.sub(r"\s+", " ", t).strip() + " "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    train, dev = load("train.jsonl"), load("dev.jsonl")
    heldout = load("heldout_benchmark_DO_NOT_TRAIN.jsonl")
    problems, warnings = [], []

    for split, rows in (("train", train), ("dev", dev)):
        for r in rows:
            if r.get("label") not in LABELS:
                problems.append(f"{split} {r.get('id')}: bad label {r.get('label')!r}")
            if not str(r.get("text", "")).strip():
                problems.append(f"{split} {r.get('id')}: empty text")
        for text, n in Counter(r["text"] for r in rows).items():
            if n > 1:
                problems.append(f"{split}: duplicate text {text[:50]}")

    train_texts = {r["text"] for r in train}
    for r in dev:
        if r["text"] in train_texts:
            problems.append(f"dev {r['id']} also in train")

    held = [(h["id"], h["text"], grams(h["text"])) for h in heldout]
    worst = 0.0
    for split, rows in (("train", train), ("dev", dev)):
        for r in rows:
            g = grams(r["text"])
            for hid, htext, hg in held:
                s = jaccard(g, hg)
                worst = max(worst, s)
                msg = f"{split} {r['id']} ~ benchmark {hid} (similarity {s:.2f})\n    {r['text']}\n    {htext}"
                if s >= FAIL_AT:
                    problems.append("near-duplicate of held-out test: " + msg)
                elif s >= WARN_AT:
                    warnings.append(msg)

    print(f"train: {len(train)}  {dict(Counter(r['label'] for r in train))}")
    print(f"dev:   {len(dev)}  {dict(Counter(r['label'] for r in dev))}")
    print(f"highest similarity to any held-out benchmark message: {worst:.2f} (fail at {FAIL_AT})")
    for w in warnings:
        print("CHECK:", w)
    for p in problems:
        print("FAIL:", p)
    if problems:
        print(f"\ndata check FAILED ({len(problems)} problems)")
        return 1
    print("data check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
