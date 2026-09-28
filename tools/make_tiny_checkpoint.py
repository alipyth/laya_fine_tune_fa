"""Build a tiny RANDOM checkpoint in Laya's on-disk format, for pipeline tests only.

It downloads nothing and trains nothing useful: it exists so tests/smoke_test.py can check
that training, saving and reloading work on your machine before you download ~700 MB.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from laya.common import DecisionModel  # noqa: E402
from prompts_train import TRAIN_PROMPTS  # noqa: E402

out = Path(sys.argv[1])
(out / "tokenizer").mkdir(parents=True, exist_ok=True)
(out / "encoder").mkdir(parents=True, exist_ok=True)

texts = []
for f in ("train.jsonl", "dev.jsonl"):
    texts += [json.loads(l)["text"] for l in open(HERE / "data" / f, encoding="utf-8")]
for v in TRAIN_PROMPTS:
    texts += [v["instructions"], *v["criteria"].values()]
texts += ["billing", "technical", "cancel", "other", "choice question:"]

specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
tk = Tokenizer(models.BPE(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(texts * 3, trainers.BpeTrainer(vocab_size=3000, special_tokens=specials))
fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                               cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
fast.save_pretrained(out / "tokenizer")

ecfg = ModernBertConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                        num_attention_heads=2, max_position_embeddings=1024,
                        pad_token_id=fast.pad_token_id, cls_token_id=fast.cls_token_id,
                        sep_token_id=fast.sep_token_id, bos_token_id=fast.cls_token_id,
                        eos_token_id=fast.sep_token_id)
ecfg.save_pretrained(out / "encoder")
torch.manual_seed(0)
enc = ModernBertModel(ecfg)
model = DecisionModel(enc, head_layers=2, n_act=2)
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
cfg = {"encoder": "tiny-test-encoder", "head_layers": 2, "act_costs": {"act": 1.0}, "max_len": 1024,
       "head_max_len": 256, "temperature": [1.0, 1.0, 1.0],
       "temperature_by_options": {"choice:3-5": 0.9, "noul:2": 1.1}, "amp_dtype": "bf16"}
(out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
print("tiny checkpoint:", out, "vocab", len(fast))
