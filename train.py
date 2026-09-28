"""Fine-tune a Laya checkpoint on Persian support triage.

Designed for a small GPU (GTX 1650, 4 GB) but runs anywhere, including CPU:

    python train.py                                  # defaults: laya-multilingual, --mode top
    python train.py --mode head                      # lightest: only the decision head trains
    python train.py --mode head --device cpu         # no GPU at all (slower, still works)
    python train.py --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851   # pin the base

Modes (the 197M-parameter token embedding table is always frozen):
    head  decision head only, encoder frozen                  ~1.8 GB VRAM
    top   head + the last --unfreeze encoder layers (default)  ~2.5 GB VRAM
    full  head + every encoder layer                           >4 GB, use Kaggle/Colab

The result is a normal Laya checkpoint directory (rl_agent_config.json, model.safetensors,
tokenizer/, encoder/) that laya.load() and the persian_fa benchmark can read directly.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import laya
from laya.agent import Agent
from laya.common import collate_items

from augment import augment
from prompts_train import LABELS, TRAIN_PROMPTS, question

HERE = Path(__file__).resolve().parent
for _stream in (sys.stdout, sys.stderr):   # Windows consoles/pipes may not be UTF-8
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
HUB_REPO = "convaiinnovations/laya"
NAMED = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
QID = "department"


# --------------------------------------------------------------------------- data
def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def encode(agent, text, variant, order, max_len):
    """Tokenise one (message, prompt variant, option order) exactly the way Laya does at inference."""
    q = question(variant, order)
    Agent._check_question(QID, q)
    internal = {QID: Agent._to_internal(q)}
    item = agent._encode_state({"message": text}, [QID], internal, max_len=max_len)[0]
    return item


def train_rows(agent, rows, rng, max_len, use_aug, shuffle_prob=0.5):
    """One epoch of training items: random prompt variant, option order shuffled with
    probability `shuffle_prob` (else the canonical order), light augmentation."""
    out = []
    for r in rows:
        variant = rng.choice(TRAIN_PROMPTS)
        order = list(LABELS)
        if rng.random() < shuffle_prob:
            rng.shuffle(order)
        text = augment(r["text"], rng) if use_aug else r["text"]
        item = encode(agent, text, variant, order, max_len)
        item["label"] = order.index(r["label"])
        out.append(item)
    rng.shuffle(out)
    return out


def dev_rows(agent, rows, max_len):
    """Fixed dev items: every dev message under every training prompt, canonical option order."""
    out = []
    for r in rows:
        for variant in TRAIN_PROMPTS:
            item = encode(agent, r["text"], variant, list(LABELS), max_len)
            item["label"] = LABELS.index(r["label"])
            out.append(item)
    return out


# --------------------------------------------------------------------------- model setup
def resolve_base(base, revision):
    """Return (local checkpoint dir, resolved hub commit or None)."""
    if os.path.isdir(base):
        return os.path.abspath(base), None
    from huggingface_hub import snapshot_download
    repo, sub = (HUB_REPO, NAMED[base]) if base in NAMED else (base, None)
    prefix = f"{sub}/" if sub else ""
    kw = {"allow_patterns": [prefix + n for n in ("rl_agent_config.json", "model.safetensors",
                                                  "tokenizer/*", "encoder/*")]}
    if revision:
        kw["revision"] = revision
    root = snapshot_download(repo, **kw)
    m = re.search(r"snapshots[\\/]([0-9a-f]{40})", root)
    return (os.path.join(root, sub) if sub else root), (m.group(1) if m else revision)


def configure_trainable(model, mode, unfreeze):
    for p in model.parameters():
        p.requires_grad = False
    enc = model.encoder
    groups = {"head": [], "encoder": []}
    for mod in (model.head, model.type_emb, model.scorer):
        if mod is not None:
            groups["head"] += list(mod.parameters())
    if mode in ("top", "full"):
        layers = list(enc.layers)
        chosen = layers if mode == "full" else layers[-unfreeze:]
        for layer in chosen:
            groups["encoder"] += list(layer.parameters())
        if getattr(enc, "final_norm", None) is not None:
            groups["encoder"] += list(enc.final_norm.parameters())
    for ps in groups.values():
        for p in ps:
            p.requires_grad = True
    return groups


def amp_setup(device):
    if device.type != "cuda":
        return torch.float32, False
    major, _ = torch.cuda.get_device_capability(device)
    return (torch.bfloat16, True) if major >= 8 else (torch.float16, True)


def forward(model, batch, device, dtype, use_amp, detach):
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else torch.autocast("cpu", enabled=False)
    with ctx:
        logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                          batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                          batch["qtype"].to(device), detach_encoder=detach)
    return logits.float()


@torch.no_grad()
def evaluate(model, items, pad_id, device, dtype, use_amp, batch_size):
    model.eval()
    all_logits, labels = [], []
    for i in range(0, len(items), batch_size):
        b = collate_items([[it] for it in items[i:i + batch_size]], pad_id)
        all_logits.append(forward(model, b, device, dtype, use_amp, detach=True)[:, :len(LABELS)].cpu())
        labels.append(b["label"])
    logits, labels = torch.cat(all_logits), torch.cat(labels)
    nll = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(-1) == labels).float().mean().item()
    return {"acc": acc, "nll": nll}, logits, labels


def fit_temperature(logits, labels, lo=0.5, hi=5.0):
    """Grid-search the one temperature that minimises dev NLL; Laya clamps to [0.5, 5]."""
    best_t, best = 1.0, float("inf")
    for i in range(0, 451):
        t = lo + (hi - lo) * i / 450
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best:
            best_t, best = t, nll
    return round(best_t, 4), best


# --------------------------------------------------------------------------- saving
def save_checkpoint(model, cfg, base_dir, out_dir, temperature, meta):
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for sub in ("tokenizer", "encoder"):
        src = Path(base_dir) / sub
        if src.is_dir():
            shutil.copytree(src, tmp / sub)
    cfg = json.loads(json.dumps(cfg))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[0] = temperature                      # index 0 = choice
    cfg["temperature"] = temps
    # Inherited bucket temperatures would override the new fit at inference; drop choice buckets.
    cfg["temperature_by_options"] = {k: v for k, v in cfg.get("temperature_by_options", {}).items()
                                     if not k.startswith("choice:")}
    cfg["finetune"] = meta
    (tmp / "rl_agent_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {k: v.detach().to("cpu", torch.float32 if v.is_floating_point() else v.dtype).clone().contiguous()
             for k, v in model.state_dict().items()}
    if "temperature" in state:
        state["temperature"][0] = temperature
    save_file(state, str(tmp / "model.safetensors"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="multilingual", help="multilingual | english | typed-decisions | local dir | hub id")
    p.add_argument("--revision", default=None, help="hub commit of the base checkpoint (recommended)")
    p.add_argument("--out", default="out/laya-fa-support")
    p.add_argument("--mode", choices=("head", "top", "full"), default="top")
    p.add_argument("--unfreeze", type=int, default=4, help="encoder layers to train in --mode top")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--passes", type=int, default=2, help="times each training message is seen per epoch")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
    p.add_argument("--patience", type=int, default=3, help="stop after N epochs without dev improvement")
    p.add_argument("--max-len", type=int, default=320, help="token budget per row during training")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--shuffle-prob", type=float, default=0.5,
                   help="chance a training row gets its options in random order (0 = always canonical)")
    p.add_argument("--grad-checkpointing", action="store_true", help="less VRAM, slower")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", default=str(HERE / "data" / "train.jsonl"))
    p.add_argument("--dev", default=str(HERE / "data" / "dev.jsonl"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available: install a CUDA build of PyTorch (see README) or use --device cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    base_dir, base_rev = resolve_base(args.base, args.revision)
    print(f"base: {args.base} -> {base_dir} (revision {base_rev})")
    agent = laya.load(base_dir, device="cpu")          # load on CPU, move after freezing
    model, tok, cfg = agent.model, agent.tok, agent.cfg
    pad_id = tok.pad_token_id

    groups = configure_trainable(model, args.mode, args.unfreeze)
    if args.grad_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        model.head_checkpointing = True
    model.to(device)
    dtype, use_amp = amp_setup(device)
    n_train = sum(p.numel() for ps in groups.values() for p in ps)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"mode {args.mode}: training {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters"
          f"{f', autocast {dtype}' if use_amp else ''}")

    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)
    dev_items = dev_rows(agent, dev, args.max_len)

    param_groups = [{"params": groups["head"], "lr": args.lr_head}]
    if groups["encoder"]:
        param_groups.append({"params": groups["encoder"], "lr": args.lr_encoder})
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train) * args.passes / args.batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    detach = args.mode == "head"

    before, _, _ = evaluate(model, dev_items, pad_id, device, dtype, use_amp, args.batch * 2)
    print(f"dev before training: acc {before['acc']:.3f}  nll {before['nll']:.3f}")

    out_dir = Path(args.out)
    history, best, bad_epochs, step = [], None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "head":
            model.encoder.eval()      # frozen encoder: no dropout noise
        items = [it for _ in range(args.passes) for it in train_rows(agent, train, rng, args.max_len, not args.no_augment, args.shuffle_prob)]
        rng.shuffle(items)
        run_loss = 0.0
        for i in range(0, len(items), args.batch):
            b = collate_items([[it] for it in items[i:i + args.batch]], pad_id)
            logits = forward(model, b, device, dtype, use_amp, detach)[:, :len(LABELS)]
            loss = F.cross_entropy(logits, b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            n_done = i // args.batch + 1
            print(f"\repoch {epoch}/{args.epochs}  step {n_done}/{steps_per_epoch}  loss {run_loss / n_done:.3f}",
                  end="", flush=True)
        m, dev_logits, dev_labels = evaluate(model, dev_items, pad_id, device, dtype, use_amp, args.batch * 2)
        m.update(epoch=epoch, train_loss=run_loss / max(1, steps_per_epoch), minutes=round((time.time() - t0) / 60, 1))
        history.append(m)
        improved = best is None or m["nll"] < best["nll"] - 1e-4
        print(f"\n  dev acc {m['acc']:.3f}  nll {m['nll']:.3f}" + ("  <- best, saving" if improved else ""))
        if improved:
            best, bad_epochs = m, 0
            temp, temp_nll = fit_temperature(dev_logits, dev_labels)
            meta = {"base": args.base, "base_revision": base_rev, "mode": args.mode,
                    "unfreeze": args.unfreeze if args.mode == "top" else None, "epoch": epoch,
                    "dev_acc": round(m["acc"], 4), "dev_nll": round(m["nll"], 4),
                    "dev_nll_after_temperature": round(temp_nll, 4), "choice_temperature": temp,
                    "train_examples": len(train), "dev_examples": len(dev),
                    "prompts": "paraphrased training prompts (not the benchmark prompts)",
                    "shuffle_prob": args.shuffle_prob,
                    "seed": args.seed, "laya_version": getattr(laya, "__version__", None),
                    "torch_version": torch.__version__}
            save_checkpoint(model, cfg, base_dir, out_dir, temp, meta)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping")
                break

    summary = {"before": before, "best": best, "history": history, "args": vars(args), "base_revision": base_rev}
    (out_dir / "training_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest dev acc {best['acc']:.3f} (before {before['acc']:.3f}) at epoch {best['epoch']}")
    print(f"saved: {out_dir.resolve()}")

    # Sanity check: the saved directory loads with plain laya.load and answers.
    del model, agent
    if device.type == "cuda":
        torch.cuda.empty_cache()
    check = laya.load(str(out_dir), device=str(device))
    ans = check.predict({"message": dev[0]["text"]}, {QID: question(TRAIN_PROMPTS[0], LABELS)})
    print(f"reload check: {dev[0]['text']!r} -> {ans['answers'][QID]['choice']} (gold {dev[0]['label']})")
    print("\nnext: evaluate on the benchmark, e.g.\n"
          f"  cd ..\\persian_fa && python run.py --checkpoint {out_dir.resolve()} --output runs\\finetuned")


if __name__ == "__main__":
    main()
