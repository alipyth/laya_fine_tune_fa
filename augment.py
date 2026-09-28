"""Light Persian surface-form augmentation for training rows.

Real users type Arabic ي/ك, drop the zero-width non-joiner, and mix digit systems. Randomly
applying those variants during training teaches the model that they mean the same thing.
It never changes words, so it cannot change the label.
"""
import random

ZWNJ = "‌"
_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك"})
_FA_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_LATIN_TO_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def augment(text: str, rng: random.Random, p: float = 0.5) -> str:
    if rng.random() >= p or not any("؀" <= c <= "ۿ" for c in text):
        return text
    choice = rng.randrange(4)
    if choice == 0:
        return text.translate(_TO_ARABIC)
    if choice == 1:
        return text.replace(ZWNJ, rng.choice(["", " "]))
    if choice == 2:
        return text.translate(_FA_TO_LATIN if rng.random() < 0.5 else _LATIN_TO_FA)
    return text.replace(ZWNJ, "").translate(_TO_ARABIC)
