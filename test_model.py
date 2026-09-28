"""Try the fine-tuned model on a few messages, and check the Hub copy matches the local one.

    python test_model.py                                   # Hub model only
    python test_model.py --local out\\laya-fa-support       # also compare with the local folder

It asks with the full prompt from the model card and with a deliberately terse prompt
(one-word option descriptions), to show how much the option wording matters.
"""
import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import laya

FULL = {"department": {
    "type": "choice",
    "instructions": "این پیام پشتیبانی باید به کدام بخش ارجاع داده شود؟",
    "criteria": {
        "billing": "پرداخت، کسر وجه، استرداد پول، فاکتور، قیمت و هزینه‌ی اشتراک",
        "technical": "خطا، باگ، کرش، مشکل ورود یا رمز، کندی و قطعی برنامه",
        "cancel": "کاربر می‌خواهد اشتراک را لغو کند، تمدید را متوقف کند یا حساب را ببندد یا حذف کند",
        "other": "هر چیز دیگر: تشکر، پیشنهاد، همکاری، استخدام، سؤال عمومی",
    }}}
TERSE = {"department": {
    "type": "choice", "instructions": "کدام بخش؟",
    "criteria": {"billing": "مالی", "technical": "فنی", "cancel": "لغو", "other": "سایر"}}}

# New messages written for this test: not in the training data, dev set or benchmark.
MESSAGES = [
    ("اپ باز نمیشه", "technical"),
    ("برنامه موقع ورود ارور میده", "technical"),
    ("پول دو بار از کارتم کم شده", "billing"),
    ("فاکتور ماه قبل رو می‌خوام", "billing"),
    ("لطفاً اشتراکمو لغو کنید", "cancel"),
    ("حسابمو پاک کنید دیگه نمی‌خوام", "cancel"),
    ("مرسی از پشتیبانی خوبتون", "other"),
    ("دفترتون کجاست؟", "other"),
]


def run(agent, question):
    out = []
    for text, _ in MESSAGES:
        a = agent.predict({"message": text}, question)["answers"]["department"]
        out.append((a["choice"], a["answer_confidence"]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="tarfandoon/laya-fa-support")
    p.add_argument("--local", help="local checkpoint folder to compare with")
    args = p.parse_args()

    hub = laya.load(args.repo)
    for name, q in (("FULL prompt (model card)", FULL), ("TERSE prompt (one-word options)", TERSE)):
        res = run(hub, q)
        ok = sum(c == gold for (c, _), (_, gold) in zip(res, MESSAGES))
        print(f"\n== {name}: {ok}/{len(MESSAGES)} correct")
        for (text, gold), (choice, conf) in zip(MESSAGES, res):
            print(f"  {'OK ' if choice == gold else 'XX '} {choice:<9} ({conf:.2f})  gold={gold:<9}  {text}")

    if args.local:
        local = laya.load(args.local)
        same = run(local, FULL) == run(hub, FULL)
        print(f"\nlocal folder and Hub model give identical answers: {same}")


if __name__ == "__main__":
    main()
