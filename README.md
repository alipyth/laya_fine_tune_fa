# laya-persian-finetune

[فارسی](README.fa.md)

This project fine-tunes [Laya](https://github.com/NandhaKishorM/laya)'s `laya-multilingual`
checkpoint for **Persian customer-support triage**. Each message is routed to `billing`,
`technical`, `cancel` or `other`. The result is evaluated on the held-out
[`persian_fa` benchmark](https://github.com/alipyth/laya-persian-benchmark) and published on
the Hugging Face Hub.

It is built to run on a 4 GB laptop GPU (GTX 1650), and on CPU in `--mode head`.

## Quick start

```bash
pip install -r requirements.txt        # install a CUDA build of PyTorch first, see README.fa.md
python tests/smoke_test.py             # about 1 minute, no download: checks the whole pipeline
python check_data.py                   # labels, duplicates, similarity to the benchmark
python train.py --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
```

Evaluate with the benchmark, which reads the output folder directly:

```bash
cd ../persian_fa
python run.py --checkpoint ../laya-persian-finetune/out/laya-fa-support --output runs/finetuned
python audit.py --run-dir runs/finetuned --write
python compare.py results/v1/multilingual runs/finetuned --mode choice_fa
```

Publish:

```bash
hf auth login
python push_to_hub.py --repo alipyth/laya-fa-support \
    --before ../persian_fa/results/v1/multilingual/summary.json \
    --after  ../persian_fa/runs/finetuned/summary.json
```

## How it works

- **Data:** 200 training messages (50 per label) and 40 dev messages, in `data/`. They are
  synthetic, drafted with AI assistance, and cover formal and colloquial Persian, Finglish,
  Arabic ي/ك, missing ZWNJ, code-mixing, negation, sarcasm/taarof and both digit systems.
  `check_data.py` fails if any message is a near-duplicate (character 3-gram Jaccard ≥ 0.5)
  of a benchmark case. The benchmark cases are included only for that check.
- **Prompts:** four paraphrased questions (two Persian, two English) in `prompts_train.py`.
  The benchmark's own prompts are never used in training. Option order is shuffled for half of
  the rows, so the model cannot rely on option position alone.
- **Augmentation:** random ي/ك, ZWNJ and digit-system variants (`augment.py`). These never
  change the label.
- **Model:** Laya's own tokenisation and `DecisionModel` forward (`laya.agent.Agent._encode_state`
  and `laya.common.collate_items`), with cross-entropy on the gold option.
  - `--mode head` trains only the decision head.
  - `--mode top` (default) also trains the last `--unfreeze` encoder layers (default 4).
  - `--mode full` trains every encoder layer.
  - The 197M-parameter token embedding table is always frozen. fp16 autocast is used on
    pre-Ampere GPUs, bf16 on Ampere and newer.
- **Selection and calibration:** the checkpoint with the lowest dev NLL is kept, and one choice
  temperature is fitted on dev. Inherited `choice:*` bucket temperatures are removed so they
  cannot override the fit.
- **Output:** a standard Laya checkpoint folder (`rl_agent_config.json`, `model.safetensors`,
  `tokenizer/`, `encoder/`) that `laya.load()` reads. It carries a `finetune` block in its
  config that records the base revision, mode, epoch and dev metrics.

## Results

_Fill in after training: paste the `compare.py` tables for `choice_fa` and `choice_en`,
the base revision, the mode and the hardware._

## Limitations

- The training and benchmark data are both small, synthetic, and written the same way.
  Gains on the benchmark will likely overstate gains on real customer messages.
- The model is specialised to one 4-way decision. Other Laya tasks were not re-evaluated.
- Do not tune on the benchmark. Choose settings on `dev`, and evaluate on the benchmark once or
  twice, reporting every attempt.

## License

Code and data: MIT. The fine-tuned weights are a derivative of Laya (Apache-2.0).
