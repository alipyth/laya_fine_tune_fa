"""Write a model card and upload a fine-tuned checkpoint to the Hugging Face Hub.

    python push_to_hub.py --repo alipyth/laya-fa-support ^
        --before C:\\...\\persian_fa\\results\\v1\\multilingual\\summary.json ^
        --after  C:\\...\\persian_fa\\runs\\finetuned\\summary.json ^
        --dry-run

--before / --after are the benchmark summaries written by `audit.py --write`.
--dry-run only writes README.md into the model folder, so you can read it first.
Log in once before uploading:  python -c "from huggingface_hub import login; login()"
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ("formal", "colloquial", "finglish", "orthography", "code_mixed", "negation", "sarcasm_taarof", "digits")
LABELS = ("billing", "technical", "cancel", "other")
MODES = (("choice_fa", "Persian instructions"), ("choice_en", "English instructions"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(c, n):
    return f"{c}/{n} ({100 * c / n:.1f}%)" if n else "-"


def _delta(b, a):
    d = a - b
    return f"**{'+' if d > 0 else ''}{d}**" if d else "0"


def headline(before, after):
    if not (before and after):
        return ""
    rows = ["| Prompt | Base `laya-multilingual` | **This model** | Change |", "|---|---|---|---|"]
    for mode, title in MODES:
        b, a = before["modes"].get(mode), after["modes"].get(mode)
        if b and a:
            rows.append(f"| {title} | {_pct(b['correct'], b['n'])} | **{_pct(a['correct'], a['n'])}** | "
                        f"{'+' if a['correct'] >= b['correct'] else ''}{a['correct'] - b['correct']} cases |")
    return "\n".join(rows) + "\n"


def detail_tables(before, after):
    if not (before and after):
        return "_Benchmark results were not attached to this upload._\n"
    out = []
    for mode, title in MODES:
        b, a = before["modes"].get(mode), after["modes"].get(mode)
        if not (b and a):
            continue
        out += [f"### {title} (`{mode}`)", "",
                "| Metric | Base | This model | Change |", "|---|---|---|---|",
                f"| **Accuracy** | {b['correct']}/{b['n']} | **{a['correct']}/{a['n']}** | {_delta(b['correct'], a['correct'])} |",
                f"| Macro-F1 | {_f(b['macro_f1'])} | {_f(a['macro_f1'])} | |",
                f"| False `cancel` (would wrongly close an account) | {b['false_cancel']} | {a['false_cancel']} | |",
                f"| Missed `cancel` | {b['missed_cancel']} | {a['missed_cancel']} | |",
                f"| Mean answer confidence | {_f(b['mean_answer_confidence'], 2)} | {_f(a['mean_answer_confidence'], 2)} | |",
                f"| ECE (calibration error, lower is better) | {_f(b['ece'])} | {_f(a['ece'])} | |",
                "", "| Family | Base | This model | Change |", "|---|---|---|---|"]
        for fam in FAMILIES:
            fb, fa = b["by_family"].get(fam), a["by_family"].get(fam)
            if fb and fa:
                out.append(f"| {fam} | {fb['correct']}/{fb['n']} | {fa['correct']}/{fa['n']} | "
                           f"{_delta(fb['correct'], fa['correct'])} |")
        out += ["", "| Label (recall) | Base | This model |", "|---|---|---|"]
        for lab in LABELS:
            lb, la = b["by_label"][lab], a["by_label"][lab]
            out.append(f"| `{lab}` | {lb['correct']}/{lb['n']} | {la['correct']}/{la['n']} |")
        out += ["", "<details><summary>Confusion matrices (rows = gold, columns = predicted)</summary>", ""]
        for name, s in (("Base", b), ("This model", a)):
            out += [f"**{name}**", "", "| | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
            for g in LABELS:
                out.append(f"| {g} | " + " | ".join(str(s["confusion"][g].get(p, 0)) for p in LABELS) + " |")
            out.append("")
        out += ["</details>", ""]
    return "\n".join(out) + "\n"


def weak_spots(before, after):
    """Families that did not improve, stated from the data rather than hard-coded."""
    if not (before and after):
        return ""
    lines = []
    for mode, title in MODES:
        b, a = before["modes"].get(mode), after["modes"].get(mode)
        if not (b and a):
            continue
        flat = [f"{f} ({a['by_family'][f]['correct']}/{a['by_family'][f]['n']})" for f in FAMILIES
                if f in a["by_family"] and a["by_family"][f]["correct"] <= b["by_family"][f]["correct"]]
        if flat:
            lines.append(f"- With {title}, these families did **not** improve: {', '.join(flat)}.")
        gain = a["correct"] - b["correct"]
        if gain <= 3:
            lines.append(f"- With {title} the overall change is small ({'+' if gain >= 0 else ''}{gain} of "
                         f"{a['n']}), which is within noise for 64 cases.")
    return "\n".join(lines)


def training_table(log):
    hist = (log or {}).get("history") or []
    if not hist:
        return ""
    pre = log.get("before") or {}
    rows = ["| Epoch | Train loss | Dev accuracy | Dev NLL |", "|---|---|---|---|",
            f"| 0 (before) | - | {_f(pre.get('acc'))} | {_f(pre.get('nll'))} |"]
    best_epoch = (log.get("best") or {}).get("epoch")
    for h in hist:
        mark = " ✅ selected" if h["epoch"] == best_epoch else ""
        rows.append(f"| {h['epoch']}{mark} | {_f(h.get('train_loss'))} | {_f(h['acc'])} | {_f(h['nll'])} |")
    return "\n".join(rows) + "\n"


def model_card(repo, cfg, log, before, after, github, benchmark, author, author_url):
    ft = cfg.get("finetune", {})
    best = (log or {}).get("best") or {}
    pre = (log or {}).get("before") or {}
    rev = ft.get("base_revision") or "unpinned"
    name = repo.split("/")[-1]
    unfrozen = f" + last {ft['unfreeze']} encoder layers" if ft.get("unfreeze") else ""
    return f"""---
license: apache-2.0
language:
- fa
base_model: convaiinnovations/laya
tags:
- laya
- persian
- farsi
- text-classification
- customer-support
- decision-model
- fine-tuned
---

# {name}

**Fine-tuned by [{author}]({author_url})** · Telegram: [{author_url.replace('https://', '')}]({author_url})

A **Persian customer-support triage** model, fine-tuned from
[Laya](https://github.com/NandhaKishorM/laya)'s `laya-multilingual` checkpoint. It reads a
Persian support message and decides which department should handle it (`billing`,
`technical`, `cancel` or `other`) in a single forward pass. It returns a probability for every
option and generates no text, so there is nothing to parse and nothing to hallucinate.

## Results at a glance

These scores come from the held-out [`persian_fa` benchmark]({benchmark}): 64 Persian
support messages the model never saw during training, asked with prompts it never saw
during training.

{headline(before, after)}
## Quick start

```bash
pip install laya
```

```python
import laya

agent = laya.load("{repo}")            # add device="cuda" for a GPU

question = {{
    "department": {{
        "type": "choice",
        "instructions": "این پیام پشتیبانی باید به کدام بخش ارجاع داده شود؟",
        "criteria": {{
            "billing": "پرداخت، کسر وجه، استرداد پول، فاکتور، قیمت و هزینه‌ی اشتراک",
            "technical": "خطا، باگ، کرش، مشکل ورود یا رمز، کندی و قطعی برنامه",
            "cancel": "کاربر می‌خواهد اشتراک را لغو کند، تمدید را متوقف کند یا حساب را ببندد یا حذف کند",
            "other": "هر چیز دیگر: تشکر، پیشنهاد، همکاری، استخدام، سؤال عمومی",
        }},
    }}
}}

result = agent.predict({{"message": "سلام، دو بار پول از حسابم کم شده ولی اشتراکم فعال نشده"}}, question)
answer = result["answers"]["department"]
print(answer["choice"])              # one of: billing, technical, cancel, other
print(answer["probabilities"])       # probability for every department
print(answer["answer_confidence"])   # calibrated probability of the chosen answer
```

### Many messages at once

```python
messages = ["اپ باز نمیشه", "mikham eshterakam ro laghv konam", "مرسی از پشتیبانی‌تون"]
results = agent.predict_batch([{{"message": m}} for m in messages], question)
for m, r in zip(messages, results):
    print(r["answers"]["department"]["choice"], "<-", m)
```

### Send uncertain cases to a human

```python
THRESHOLD = 0.6   # choose this on your own labelled data
if answer["answer_confidence"] >= THRESHOLD:
    route_to(answer["choice"])
else:
    send_to_human(answer)
```

### Tips

- **Keep the four option keys** (`billing`, `technical`, `cancel`, `other`) and **give every
  option a full description**, like the example above. The wording can change, because the
  model was trained on several paraphrases in Persian and English. **Persian instructions
  work best** for this model (see the results).
- ⚠️ **Do not use one-word descriptions** such as `"billing": "مالی"`. In a spot check on 8 new
  messages, the full prompt above got 7/8 right, while one-word descriptions got 5/8. The
  one-word version was also confidently wrong: it labelled «اپ باز نمیشه» as `cancel` with
  0.92 confidence.
- The input can be formal or colloquial Persian, Finglish, or Persian mixed with English words.
  Arabic ي/ك and missing half-spaces are handled.
- On a CPU a decision takes roughly 150-200 ms. On a GPU it is much faster.

## Full benchmark results: before vs after

Base model: `convaiinnovations/laya` / `multilingual` @ `{rev}`. Scores are from the first
of three repeats, and every repeat gave identical predictions. Neither model saw these 64
messages or these two prompts during training.

{detail_tables(before, after)}
## Training

| | |
|---|---|
| Base checkpoint | `laya-multilingual` (mmBERT-base, 322M parameters) @ `{rev}` |
| Method | `{ft.get('mode')}` mode: decision head{unfrozen}. Token embeddings frozen. |
| Objective | cross-entropy on the gold option, AdamW, fp16 autocast |
| Data | 200 training and 40 dev Persian support messages (50 and 10 per label), all synthetic |
| Prompts | 4 paraphrased questions (2 Persian, 2 English), **none of them the benchmark prompts** |
| Robustness | options shuffled with probability {ft.get('shuffle_prob')}; random ي/ك, half-space and digit variants |
| Model selection | best epoch by dev NLL (epoch {ft.get('epoch')}) |
| Calibration | one choice temperature fitted on dev ({ft.get('choice_temperature')}) |
| Hardware | a single NVIDIA GeForce GTX 1650 (4 GB), about 10 minutes |
| Software | laya {ft.get('laya_version')}, torch {ft.get('torch_version')}, seed {ft.get('seed')} |

The dev set has 40 messages x 4 training prompts. Dev accuracy went from {_f(pre.get('acc'))}
before training to {_f(best.get('acc'))} at the selected epoch.

{training_table(log)}
A near-duplicate check (character 3-gram Jaccard < 0.5) guarantees that no training or dev
message copies a benchmark message.

## Limitations

- **The data is small and synthetic.** The training set and the benchmark were drafted the
  same way, with AI assistance. Real customer traffic will be harder, so expect a smaller
  gain there than the benchmark shows. Evaluate on your own messages before relying on the
  model.
{weak_spots(before, after)}
- **It is sensitive to how the options are described.** Use descriptive option texts (see
  Tips). Terse one-word options made it much worse, with high confidence.
- **It is specialised to one 4-way decision.** Other Laya tasks and languages were not
  re-evaluated and may have degraded. Use the original Laya checkpoints for those.
- Confidence was calibrated on 160 dev rows only. Refit the threshold or temperature on your
  own data before gating decisions on it.

## Links

- 📊 Benchmark (data, code, all raw results): {benchmark}
- 🛠️ Training code and data: {github}
- 🧠 Laya: https://github.com/NandhaKishorM/laya (Apache-2.0)
- 📣 {author}: {author_url}

## License

Apache-2.0, the same as the base Laya weights.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=str(HERE / "out" / "laya-fa-support"))
    p.add_argument("--repo", required=True, help="e.g. alipyth/laya-fa-support")
    p.add_argument("--before", help="benchmark summary.json of the base model")
    p.add_argument("--after", help="benchmark summary.json of this model")
    p.add_argument("--author", default="Ali Jahani")
    p.add_argument("--author-url", default="https://t.me/tarfandoonchannel")
    p.add_argument("--github", default="https://github.com/alipyth/laya-persian-finetune")
    p.add_argument("--benchmark", default="https://github.com/alipyth/laya-persian-benchmark")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write README.md only, do not upload")
    p.add_argument("--card-only", action="store_true", help="upload only README.md (update the model card)")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    for needed in ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder"):
        if not (model_dir / needed).exists():
            raise SystemExit(f"{model_dir} is missing {needed}; is this a trained checkpoint folder?")
    cfg = load(model_dir / "rl_agent_config.json")
    log = load(model_dir / "training_log.json") if (model_dir / "training_log.json").exists() else None
    card = model_card(args.repo, cfg, log, load(args.before), load(args.after), args.github,
                      args.benchmark, args.author, args.author_url)
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {model_dir / 'README.md'}")
    if args.dry_run:
        print("dry run: nothing uploaded. Read the README, then run again without --dry-run.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    if args.card_only:
        api.upload_file(path_or_fileobj=str(model_dir / "README.md"), path_in_repo="README.md",
                        repo_id=args.repo, commit_message="Update model card")
        print(f"model card updated: https://huggingface.co/{args.repo}")
        return
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(model_dir),
                      commit_message="Upload Persian support-triage fine-tune of laya-multilingual")
    print(f"uploaded: https://huggingface.co/{args.repo}")
    print(f'test it:  python -c "import laya; print(laya.load(\'{args.repo}\'))"')


if __name__ == "__main__":
    main()
