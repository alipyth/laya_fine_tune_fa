"""Check your setup in about a minute, without downloading the real model.

    python tests/smoke_test.py            # uses the GPU if PyTorch can see one
    python tests/smoke_test.py --cpu

It builds a tiny random checkpoint, trains it for two epochs with train.py (in every mode),
reloads the result with laya.load(), and writes a model card with push_to_hub.py --dry-run.
A random tiny model cannot learn the real task, so ignore its accuracy: the test only
checks that nothing crashes. For that reason it trains on an easy synthetic variant.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def run(*args):
    print("$", " ".join(str(a) for a in args), flush=True)
    res = subprocess.run([sys.executable, *map(str, args)], cwd=HERE, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise SystemExit(f"FAILED: {args[0]}")
    return res.stdout


def main():
    cpu = "--cpu" in sys.argv
    import torch
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run("check_data.py")
        run("tools/make_tiny_checkpoint.py", tmp / "tiny")
        for split in ("train", "dev"):   # easy variant: the label word is in the message
            rows = [json.loads(l) for l in open(HERE / "data" / f"{split}.jsonl", encoding="utf-8")]
            with open(tmp / f"{split}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    r["text"] = f"{r['label']} {r['text']}"
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for mode in ("head", "top", "full"):
            out = run("train.py", "--base", tmp / "tiny", "--mode", mode, "--unfreeze", "2", "--epochs", "2",
                      "--device", device, "--out", tmp / f"out_{mode}", "--train", tmp / "train.jsonl",
                      "--dev", tmp / "dev.jsonl", "--lr-head", "1e-3", "--lr-encoder", "1e-3")
            last = [l for l in out.replace("\r", "\n").splitlines() if l.startswith(("best dev", "reload check"))]
            print("   " + "\n   ".join(last))
        run("push_to_hub.py", "--model-dir", tmp / "out_top", "--repo", "you/test", "--dry-run")
    print("\nsmoke test OK: training, saving, reloading and the model card all work on this machine")


if __name__ == "__main__":
    main()
