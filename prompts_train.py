"""Training prompts: paraphrases of the task, deliberately NOT the benchmark's exact wording.

The benchmark (persian_fa/prompts.py) asks with its own frozen instructions. Training on
different wordings, with the options in a random order each time, means a benchmark gain
has to come from understanding the task, not from memorising one prompt or one option
position. The option keys are the same four labels the benchmark uses.
"""

LABELS = ("billing", "technical", "cancel", "other")

TRAIN_PROMPTS = [
    {   # Persian, variant A
        "instructions": "پیام کاربر مربوط به کدام واحد است؟",
        "criteria": {
            "billing": "مسائل مالی: پرداخت، برداشت پول، بازگشت وجه، صورتحساب، قیمت پلن",
            "technical": "مشکل فنی: خطا، کرش، کندی، مشکل ورود، باگ یا از کار افتادن قابلیت",
            "cancel": "کاربر می‌خواهد اشتراکش تمام شود، تمدید نشود یا حسابش بسته یا پاک شود",
            "other": "موارد دیگر مثل تشکر، پیشنهاد، سؤال عمومی، همکاری یا استخدام",
        },
    },
    {   # Persian, variant B
        "instructions": "این درخواست پشتیبانی را به کدام تیم بفرستیم؟",
        "criteria": {
            "billing": "تیم مالی (پول، پرداخت، فاکتور، تخفیف، استرداد)",
            "technical": "تیم فنی (باگ، خطا، اپ باز نمی‌شود، ورود یا همگام‌سازی کار نمی‌کند)",
            "cancel": "تیم لغو اشتراک و حذف حساب",
            "other": "سایر درخواست‌ها (قدردانی، ایده، سؤال درباره‌ی شرکت، رسانه، شغل)",
        },
    },
    {   # English, variant A
        "instructions": "Route this customer message to the right team.",
        "criteria": {
            "billing": "money matters: charges, payments, refunds, invoices, plan prices, discounts",
            "technical": "something is broken: errors, crashes, slowness, login or sync failures",
            "cancel": "the customer wants to end the subscription, stop renewal, or close or delete the account",
            "other": "anything else: thanks, ideas, company questions, press, jobs, partnerships",
        },
    },
    {   # English, variant B
        "instructions": "What is the main request in this support ticket?",
        "criteria": {
            "billing": "a payment, charge, refund, invoice or pricing question",
            "technical": "a bug, error or feature that does not work",
            "cancel": "ending the subscription or removing the account",
            "other": "none of the above, such as feedback, suggestions or general questions",
        },
    },
]


def question(variant: dict, order) -> dict:
    """A Laya choice question with the options in the given key order."""
    return {"type": "choice", "instructions": variant["instructions"],
            "criteria": {k: variant["criteria"][k] for k in order}}
