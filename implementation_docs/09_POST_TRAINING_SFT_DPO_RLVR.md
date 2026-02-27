# 09 — Post-Training Alignment: SFT, DPO, and RLVR

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete post-training alignment pipeline — from base model to instruction-following, preference-aligned, reasoning-capable NanoSeek
**Prerequisite**: Read `00_MASTER_PLAN.md` (context), `06_MULTI_PHASE_TRAINING_ORCHESTRATION.md` (pre-training)

---

## 1. Problem Statement

Pre-trained NanoSeek is a base model: it completes text but cannot follow instructions,
refuses nothing, and hallucinates freely. Post-training bridges this gap through three stages.

| Stage | Input Model | Output Model | What It Does |
|-------|------------|-------------|--------------|
| SFT | Base (text completion) | Instruction-following | Teaches format + task compliance |
| DPO | SFT model | Preference-aligned | Learns "good vs bad" from human preferences |
| RLVR | DPO model | Reasoning-enhanced | Reinforces verifiable math/code correctness |

### Current State

- `model/model.py`: `NanoSeekModel` accepts `input_ids` and `labels`, computes loss via
  `_compute_loss()` with `cross_entropy(ignore_index=-100)` + MTP auxiliary loss.
- `scripts/tokenizer.py`: `render_conversation()` tokenizes chat turns with proper masking
  (mask=1 for assistant tokens, mask=0 for user/system). Special tokens: `<|user_start|>`,
  `<|assistant_start|>`, `<|assistant_end|>`, etc.
- **No SFT, DPO, or RLVR training scripts exist.**

### Compute Budget

Post-training consumes ~10% of the total ~$300 budget. In 2026, frontier labs allocate
55% of compute to post-training (up from ~5% in 2022).

| Stage | Data | Wall Time (8×H100) | Cost |
|-------|------|-----------------:|-----:|
| SFT | 1M conversations | ~40 min | ~$12 |
| DPO | 200K preference pairs | ~15 min | ~$5 |
| RLVR | 5K update steps | ~8 min | ~$3 |
| **Total** | | **~63 min** | **~$20** |

---

## 2. First Principles

### Why SFT Before Preference Learning

SFT establishes the instruction-following format. Without it, DPO receives a model that
outputs random web text — the chosen/rejected signal has no grounding. Empirically, skipping
SFT degrades DPO performance by 15-30% on MT-Bench (Tunstall et al., 2024).

```
Base Model ──SFT──▶ Follows instructions ──DPO──▶ Prefers good outputs ──RLVR──▶ Reasons correctly
```

### DPO vs RLHF

| | RLHF (PPO) | DPO |
|--|-----------|-----|
| Reward model | Required (separate training) | Eliminated (implicit) |
| Compute | RM + PPO = 2.5× SFT cost | 1.5× SFT cost (**40% savings**) |
| Quality | Gold standard | Equivalent on most benchmarks |

DPO loss directly optimizes the preference objective without a reward model:

```
L_DPO = -log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
```

### Numerical Example: DPO Loss Computation

```
β = 0.1

Policy log-probs:      log π_θ(y_w|x) = -2.3,   log π_θ(y_l|x) = -3.1
Reference log-probs:   log π_ref(y_w|x) = -2.5,  log π_ref(y_l|x) = -2.9

Reward margin = β · ((−2.3 − (−2.5)) − (−3.1 − (−2.9)))
              = 0.1 · ((0.2) − (−0.2))
              = 0.1 · 0.4 = 0.04

Loss = −log σ(0.04) = −log(0.510) = 0.673

Gradient pushes policy to increase π_θ(y_w) and decrease π_θ(y_l).
```

### RLVR: Verifiable Rewards

For math and code, correctness is machine-verifiable — no human labels needed, infinite
scale. REINFORCE with baseline: `∇J = E[(R(y) - b) · ∇log π_θ(y|x)]`
where R(y) ∈ {0, 1} for answer correctness, and b is a running baseline.

### Catastrophic Forgetting

Post-training on narrow instruction data erases general knowledge. Mitigations:
1. **KL constraint** (DPO's β): penalizes deviation from reference model
2. **Replay buffer**: mix 10% pre-training data into SFT batches
3. **LoRA**: only train low-rank adapters, freezing base weights
4. **Monitor**: track perplexity on FineWeb-Edu eval set across all stages

---

## 3. Production Code

### 9a. Chat Data Format & Loading (`scripts/chat_data.py`)

```python
"""
Chat data loading and tokenization for NanoSeek post-training.
Supports ShareGPT, Alpaca, UltraChat formats. Loss ONLY on assistant tokens.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset

def normalize_sharegpt(raw: Dict) -> Dict:
    messages = []
    for turn in raw["conversations"]:
        role = "user" if turn["from"] in ("human", "user") else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}

def normalize_alpaca(raw: Dict) -> Dict:
    user_content = raw["instruction"]
    if raw.get("input"):
        user_content += "\n\n" + raw["input"]
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": raw["output"]},
    ]}

def normalize_ultrachat(raw: Dict) -> Dict:
    return {"messages": raw["messages"]}

FORMAT_NORMALIZERS = {"sharegpt": normalize_sharegpt, "alpaca": normalize_alpaca,
                      "ultrachat": normalize_ultrachat}

def detect_format(record: Dict) -> str:
    if "conversations" in record: return "sharegpt"
    if "instruction" in record:   return "alpaca"
    if "messages" in record:      return "ultrachat"
    raise ValueError(f"Unknown format. Keys: {list(record.keys())}")

@dataclass
class TokenizedConversation:
    input_ids: List[int]
    labels: List[int]       # -100 for non-assistant tokens
    attention_mask: List[int]
    def __len__(self) -> int: return len(self.input_ids)

def tokenize_conversation(tokenizer, conversation: Dict, max_length: int = 4096):
    """Uses tokenizer.render_conversation(); converts mask to labels with -100."""
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_length)
    labels = [tok if m == 1 else -100 for tok, m in zip(ids, mask)]
    return TokenizedConversation(input_ids=ids, labels=labels, attention_mask=[1]*len(ids))

def pack_conversations(convs: List[TokenizedConversation], max_length: int, pad_id: int):
    """Greedy bin-pack multiple short conversations into single sequences."""
    batches, cur_ids, cur_lab, cur_mask = [], [], [], []
    for c in convs:
        if len(cur_ids) + len(c) > max_length:
            if cur_ids:
                pad = max_length - len(cur_ids)
                batches.append({"input_ids": torch.tensor(cur_ids + [pad_id]*pad),
                                "labels": torch.tensor(cur_lab + [-100]*pad),
                                "attention_mask": torch.tensor(cur_mask + [0]*pad)})
            cur_ids, cur_lab, cur_mask = [], [], []
        cur_ids.extend(c.input_ids); cur_lab.extend(c.labels); cur_mask.extend(c.attention_mask)
    if cur_ids:
        pad = max_length - len(cur_ids)
        batches.append({"input_ids": torch.tensor(cur_ids + [pad_id]*pad),
                        "labels": torch.tensor(cur_lab + [-100]*pad),
                        "attention_mask": torch.tensor(cur_mask + [0]*pad)})
    return batches

class ChatDataset(Dataset):
    """In-memory chat dataset for SFT. Supports packing for GPU efficiency."""
    def __init__(self, data_path: str, tokenizer, max_length=4096,
                 data_format="auto", pack_sequences=True):
        raw = self._load(data_path)
        fmt = data_format if data_format != "auto" else detect_format(raw[0])
        norm = FORMAT_NORMALIZERS[fmt]
        tokenized = [tokenize_conversation(tokenizer, norm(r), max_length) for r in raw]
        if pack_sequences:
            self.samples = pack_conversations(tokenized, max_length, tokenizer.get_pad_token_id())
        else:
            self.samples = [{"input_ids": torch.tensor(t.input_ids),
                             "labels": torch.tensor(t.labels),
                             "attention_mask": torch.tensor(t.attention_mask)} for t in tokenized]
    def _load(self, path):
        p = Path(path)
        with open(p) as f:
            return [json.loads(l) for l in f] if p.suffix == ".jsonl" else json.load(f)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class PreferenceDataset(Dataset):
    """Preference pairs for DPO: {prompt, chosen, rejected}."""
    def __init__(self, data_path: str, tokenizer, max_length=4096):
        self.tokenizer, self.max_length = tokenizer, max_length
        with open(data_path) as f:
            raw = [json.loads(l) for l in f] if data_path.endswith(".jsonl") else json.load(f)
        self.pairs = [self._process(r) for r in raw]

    def _process(self, raw):
        def _tok(content):
            c = {"messages": [{"role":"user","content":raw["prompt"]},
                              {"role":"assistant","content":content}]}
            return tokenize_conversation(self.tokenizer, c, self.max_length)
        chosen, rejected = _tok(raw["chosen"]), _tok(raw["rejected"])
        pad_id = self.tokenizer.get_pad_token_id()
        def _pad(ids, val):
            t = ids[:self.max_length]
            return torch.tensor(t + [val]*(self.max_length - len(t)), dtype=torch.long)
        return {"chosen_ids": _pad(chosen.input_ids, pad_id),
                "chosen_labels": _pad(chosen.labels, -100),
                "chosen_mask": _pad(chosen.attention_mask, 0),
                "rejected_ids": _pad(rejected.input_ids, pad_id),
                "rejected_labels": _pad(rejected.labels, -100),
                "rejected_mask": _pad(rejected.attention_mask, 0)}
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]
```

---

### 9b. SFT Training Script (`scripts/sft.py`)

```python
"""
Supervised Fine-Tuning for NanoSeek. Conversation masking: loss ONLY on assistant tokens.
Optional LoRA, cosine LR, WandB logging, checkpoint management.
"""
from __future__ import annotations
import argparse, logging, math, os, sys
from pathlib import Path
import torch, torch.distributed as dist, torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import get_nanoseek_config
from model.model import NanoSeekModel
from scripts.chat_data import ChatDataset
from scripts.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup: return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

def train_sft(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [SFT] %(message)s")
        try: import wandb; wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except ImportError: logger.warning("wandb not installed")

    config = get_nanoseek_config()
    model = NanoSeekModel(config)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=False)
    if args.use_lora:
        from model.lora import apply_lora
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        if is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"LoRA: {trainable:,} trainable params")
    model = model.to(device)
    if is_distributed: model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    tokenizer = get_tokenizer(args.tokenizer_type)
    dataset = ChatDataset(args.data_path, tokenizer, args.max_length, args.data_format, args.pack_sequences)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs
    global_step = 0; model.train()

    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        for batch in loader:
            ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)

            lr = cosine_lr(global_step, args.warmup_steps, total_steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups: pg["lr"] = lr
            loss = model(input_ids=ids, labels=labels, attention_mask=mask)["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(); global_step += 1

            if is_main and global_step % args.log_every == 0:
                logger.info(f"Step {global_step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")
                try: import wandb; wandb.log({"sft/loss": loss.item(), "sft/lr": lr})
                except: pass
            if is_main and global_step % args.save_every == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), f"{args.output_dir}/sft_step_{global_step}.pt")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        raw = model.module if hasattr(model, "module") else model
        torch.save(raw.state_dict(), f"{args.output_dir}/sft_final.pt")
        logger.info("SFT complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default="checkpoints/sft")
    p.add_argument("--data_format", default="auto"); p.add_argument("--tokenizer_type", default="auto")
    p.add_argument("--max_length", type=int, default=4096); p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--use_lora", action="store_true"); p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--pack_sequences", action="store_true", default=True)
    p.add_argument("--log_every", type=int, default=10); p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--wandb_project", default="nanoseek-sft"); p.add_argument("--run_name", default=None)
    train_sft(p.parse_args())
```

---

### 9c. DPO Training Script (`scripts/dpo.py`)

```python
"""
Direct Preference Optimization for NanoSeek.
Loss: -log σ(β · (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
Reference model: frozen SFT copy. β annealing: 0.1 → 0.5.
"""
from __future__ import annotations
import argparse, copy, logging, math, os, sys
from pathlib import Path
import torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import get_nanoseek_config
from model.model import NanoSeekModel
from scripts.chat_data import PreferenceDataset
from scripts.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

def compute_log_probs(model, input_ids, labels, attention_mask):
    """Per-sequence sum of log probs on label positions. Returns [batch]."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(dim=-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    """DPO loss with diagnostics dict."""
    chosen_rewards = beta * (pi_w - ref_w)
    rejected_rewards = beta * (pi_l - ref_l)
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    return loss, {"dpo/loss": loss.item(), "dpo/reward_margin": margin.mean().item(),
                  "dpo/accuracy": (margin > 0).float().mean().item(),
                  "dpo/chosen_reward": chosen_rewards.mean().item(),
                  "dpo/rejected_reward": rejected_rewards.mean().item()}

def beta_schedule(step, total, start=0.1, end=0.5):
    return start + min(step / max(total, 1), 1.0) * (end - start)

def train_dpo(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [DPO] %(message)s")
        try: import wandb; wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except ImportError: pass

    config = get_nanoseek_config()
    policy = NanoSeekModel(config)
    if args.sft_checkpoint:
        policy.load_state_dict(torch.load(args.sft_checkpoint, map_location="cpu", weights_only=True), strict=False)

    ref_model = copy.deepcopy(policy); ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False

    if args.use_lora:
        from model.lora import apply_lora
        policy = apply_lora(policy, rank=args.lora_rank, alpha=args.lora_alpha)

    policy, ref_model = policy.to(device), ref_model.to(device)
    if is_distributed: policy = nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank])

    tokenizer = get_tokenizer(args.tokenizer_type)
    dataset = PreferenceDataset(args.data_path, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy.parameters()),
                                   lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs
    global_step = 0; policy.train()

    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        for batch in loader:
            beta = beta_schedule(global_step, total_steps, args.beta_start, args.beta_end)
            c_ids, c_lab, c_mask = [batch[k].to(device) for k in ("chosen_ids","chosen_labels","chosen_mask")]
            r_ids, r_lab, r_mask = [batch[k].to(device) for k in ("rejected_ids","rejected_labels","rejected_mask")]

            pi_w = compute_log_probs(policy, c_ids, c_lab, c_mask)
            pi_l = compute_log_probs(policy, r_ids, r_lab, r_mask)
            with torch.no_grad():
                ref_w = compute_log_probs(ref_model, c_ids, c_lab, c_mask)
                ref_l = compute_log_probs(ref_model, r_ids, r_lab, r_mask)

            loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
            kl = (pi_w - ref_w).mean().item()
            metrics["dpo/kl"] = kl; metrics["dpo/beta"] = beta

            loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(); global_step += 1

            if is_main and global_step % args.log_every == 0:
                logger.info(f"Step {global_step}/{total_steps} loss={metrics['dpo/loss']:.4f} "
                            f"acc={metrics['dpo/accuracy']:.3f} β={beta:.3f} KL={kl:.3f}")
                try: import wandb; wandb.log(metrics)
                except: pass
            if is_main and global_step % args.save_every == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                raw = policy.module if hasattr(policy, "module") else policy
                torch.save(raw.state_dict(), f"{args.output_dir}/dpo_step_{global_step}.pt")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        raw = policy.module if hasattr(policy, "module") else policy
        torch.save(raw.state_dict(), f"{args.output_dir}/dpo_final.pt")
        logger.info("DPO complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--sft_checkpoint", required=True)
    p.add_argument("--output_dir", default="checkpoints/dpo"); p.add_argument("--tokenizer_type", default="auto")
    p.add_argument("--max_length", type=int, default=4096); p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1); p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta_start", type=float, default=0.1); p.add_argument("--beta_end", type=float, default=0.5)
    p.add_argument("--use_lora", action="store_true"); p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--log_every", type=int, default=10); p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--wandb_project", default="nanoseek-dpo"); p.add_argument("--run_name", default=None)
    train_dpo(p.parse_args())
```

---

### 9d. RLVR Training Script (`scripts/rlvr.py`)

```python
"""
Reinforcement Learning with Verifiable Rewards for NanoSeek.
REINFORCE with baseline. Rewards: math answer correctness, code test execution.
"""
from __future__ import annotations
import argparse, json, logging, math, os, re, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, List, Optional
import torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import get_nanoseek_config
from model.model import NanoSeekModel
from scripts.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

def extract_boxed_answer(text: str) -> Optional[str]:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match: return match.group(1).strip()
    match = re.search(r"(?:answer is|answer:)\s*([^\n.]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def math_reward(generation: str, ground_truth: str) -> float:
    predicted = extract_boxed_answer(generation)
    if predicted is None: return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0

def code_reward(generation: str, test_cases: List[Dict]) -> float:
    code_match = re.search(r"```python\n(.*?)```", generation, re.DOTALL)
    if not code_match: return 0.0
    code = code_match.group(1); passed = 0
    for test in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code + "\n" + test["test_code"]); f.flush()
                if subprocess.run([sys.executable, f.name], capture_output=True, timeout=5).returncode == 0:
                    passed += 1
        except: pass
        finally:
            try: os.unlink(f.name)
            except: pass
    return passed / max(len(test_cases), 1)

class RewardNormalizer:
    def __init__(self, gamma=0.99): self.mean, self.var, self.gamma = 0.0, 1.0, gamma
    def normalize(self, rewards):
        self.mean = self.gamma * self.mean + (1-self.gamma) * rewards.mean().item()
        self.var = self.gamma * self.var + (1-self.gamma) * (rewards.var().item() if rewards.numel()>1 else 1.0)
        return (rewards - self.mean) / max(math.sqrt(self.var), 1e-8)

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, max_new_tokens=512, temperature=0.7):
    completions = []
    for prompt in prompts:
        conv = {"messages": [{"role":"user","content":prompt}, {"role":"assistant","content":""}]}
        prompt_ids = tokenizer.render_for_completion(conv)
        gen = []
        for tok in model.generate_simple(prompt_ids, max_tokens=max_new_tokens, temperature=temperature):
            gen.append(tok)
            if tok == tokenizer.encode_special("<|assistant_end|>"): break
        completions.append(tokenizer.decode(gen))
    return completions

def train_rlvr(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0
    if is_main: logging.basicConfig(level=logging.INFO, format="%(asctime)s [RLVR] %(message)s")

    config = get_nanoseek_config(); model = NanoSeekModel(config)
    if args.dpo_checkpoint:
        model.load_state_dict(torch.load(args.dpo_checkpoint, map_location="cpu", weights_only=True), strict=False)
    model = model.to(device)
    tokenizer = get_tokenizer(args.tokenizer_type)
    with open(args.data_path) as f: problems = [json.loads(l) for l in f if l.strip()]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    normalizer = RewardNormalizer(); baseline = 0.0

    for step in range(args.total_steps):
        batch = [problems[i % len(problems)] for i in range(step*args.batch_size, (step+1)*args.batch_size)]
        model.eval()
        completions = generate_completions(model, tokenizer, [p["prompt"] for p in batch],
                                            args.max_gen_tokens, args.temperature)
        model.train()

        rewards = []
        for prob, comp in zip(batch, completions):
            r = code_reward(comp, prob.get("test_cases",[])) if prob.get("type")=="code" else math_reward(comp, prob["answer"])
            rewards.append(r)
        reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        baseline = 0.95 * baseline + 0.05 * reward_t.mean().item()
        advantages = normalizer.normalize(reward_t - baseline)

        all_ids, all_labels = [], []
        for prompt, comp in zip([p["prompt"] for p in batch], completions):
            ids, mask = tokenizer.render_conversation(
                {"messages":[{"role":"user","content":prompt},{"role":"assistant","content":comp}]},
                max_tokens=args.max_length)
            pad = args.max_length - len(ids); pad_id = tokenizer.get_pad_token_id()
            all_ids.append(ids + [pad_id]*pad)
            all_labels.append([t if m==1 else -100 for t,m in zip(ids,mask)] + [-100]*pad)

        input_ids = torch.tensor(all_ids, device=device)
        labels = torch.tensor(all_labels, device=device)

        # REINFORCE: -advantage * log π(y|x)
        logits = model(input_ids=input_ids)["logits"]
        shift_logits = logits[:, :-1, :]; shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != -100).float()
        per_seq = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        loss = -(advantages * per_seq).mean()

        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()

        if is_main and (step+1) % args.log_every == 0:
            logger.info(f"Step {step+1}/{args.total_steps} loss={loss.item():.4f} "
                        f"reward={reward_t.mean().item():.3f} baseline={baseline:.3f}")
        if is_main and (step+1) % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/rlvr_step_{step+1}.pt")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{args.output_dir}/rlvr_final.pt"); logger.info("RLVR complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--dpo_checkpoint", default=None)
    p.add_argument("--output_dir", default="checkpoints/rlvr"); p.add_argument("--tokenizer_type", default="auto")
    p.add_argument("--max_length", type=int, default=4096); p.add_argument("--max_gen_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8); p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-6); p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10); p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--wandb_project", default="nanoseek-rlvr"); p.add_argument("--run_name", default=None)
    train_rlvr(p.parse_args())
```

---

### 9e. LoRA Adapter Module (`model/lora.py`)

```python
"""
Low-Rank Adaptation (LoRA) for NanoSeek.
MoE strategy: apply to shared experts + MLA projections, NOT all 64 routed experts.
"""
from __future__ import annotations
import math
from typing import Optional, Set
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class LoRALinear(nn.Module):
    """y = W·x + (α/r)·B·A·x. Merge for zero inference overhead."""
    def __init__(self, base_layer: nn.Linear, rank=16, alpha=32.0, dropout=0.0):
        super().__init__()
        self.base_layer, self.rank, self.scaling = base_layer, rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        base_layer.weight.requires_grad = False
        if base_layer.bias is not None: base_layer.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

    def merge_weights(self) -> nn.Linear:
        merged = nn.Linear(self.base_layer.in_features, self.base_layer.out_features,
                           bias=self.base_layer.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A))
            if self.base_layer.bias is not None: merged.bias.copy_(self.base_layer.bias)
        return merged

MLA_TARGETS = {"wq_a", "wq_b", "wkv_a", "wkv_b", "wo"}
SHARED_FFN_TARGETS = {"gate_proj", "up_proj", "down_proj"}
DEFAULT_TARGETS = MLA_TARGETS | SHARED_FFN_TARGETS

def apply_lora(model: nn.Module, rank=16, alpha=32.0, dropout=0.0,
               target_modules: Optional[Set[str]] = None) -> nn.Module:
    """Apply LoRA to MLA + shared experts. Skip 64 routed experts (too many adapters)."""
    if target_modules is None: target_modules = DEFAULT_TARGETS
    for name, module in model.named_modules():
        if ".experts." in name and "shared_experts" not in name: continue
        for attr in list(dir(module)):
            if attr not in target_modules: continue
            target = getattr(module, attr, None)
            if isinstance(target, nn.Linear):
                setattr(module, attr, LoRALinear(target, rank, alpha, dropout))
    for n, p in model.named_parameters():
        if "lora_" not in n: p.requires_grad = False
    return model

def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base layers for deployment."""
    for _, module in model.named_modules():
        for attr in list(dir(module)):
            target = getattr(module, attr, None)
            if isinstance(target, LoRALinear):
                setattr(module, attr, target.merge_weights())
    return model

def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
```

---

## 4. File Placement

```
scripts/
├── sft.py                 # SFT training (§9b)
├── dpo.py                 # DPO training (§9c)
├── rlvr.py                # RLVR training (§9d)
├── chat_data.py           # Chat data loading & packing (§9a)
model/
└── lora.py                # LoRA adapter module (§9e)
```

---

## 5. Performance Targets

| Stage | Data Scale | Time (8×H100) | Key Metric |
|-------|-----------|:-------------:|------------|
| SFT | 1M conversations, 3 epochs | ~40 min | Loss < 1.5 on held-out |
| DPO | 200K preference pairs, 1 epoch | ~15 min | Accuracy > 65% (chosen > rejected) |
| RLVR | 5K update steps | ~8 min | Mean reward > 0.3 (from 0.05 baseline) |

### SFT Loss Trajectory (Expected)

```
Step 0:     loss ≈ 8.5  (random on assistant tokens)
Step 500:   loss ≈ 2.1  (learning format)
Step 2000:  loss ≈ 1.4  (instruction following)
Step 5000:  loss ≈ 1.2  (converged)
```

### DPO Diagnostics

```
reward_margin > 0      → model prefers chosen over rejected (target: >0.5)
accuracy > 0.65        → policy beats reference on >65% of pairs
KL(π‖π_ref) < 5.0     → not diverging too far from SFT model
```

---

## 6. Gotchas

1. **Conversation masking is critical.** If you train loss on user/system tokens, the model
   learns to *parrot prompts* instead of *answering them*. The `labels[i] = -100` mask in
   `tokenize_conversation()` ensures `cross_entropy(ignore_index=-100)` skips those positions.
   Always verify with `tokenizer.visualize_tokenization()` before a full run.

2. **DPO β too low → mode collapse.** At β=0.01, the KL penalty is negligible and the model
   collapses to always outputting the chosen response verbatim. At β=1.0, the penalty
   dominates and the model barely moves from reference. The 0.1→0.5 anneal balances
   exploration and exploitation.

3. **LoRA on MoE: apply to shared experts + MLA, NOT all 64 experts.** Each expert has
   3 projections. At rank=16 with 64 experts × 3 × 2 matrices = 384 adapter pairs — this
   defeats the purpose. Shared experts (2×3=6) + MLA (5×16 layers=80) = 86 adapters total.

4. **Catastrophic forgetting: monitor pre-training eval perplexity.** Evaluate on 1000
   held-out FineWeb-Edu documents every 500 SFT steps. If perplexity rises >20% from
   baseline, reduce learning rate or increase replay buffer fraction.

5. **RLVR code execution is a security risk.** `code_reward()` runs model-generated code
   in a subprocess. In production, sandbox with `firejail` or container isolation.

6. **DPO reference model doubles memory.** The frozen `ref_model` holds a full copy of SFT
   weights: 4.75B × 2 bytes = 9.5 GB extra. With LoRA on the policy, you can alternatively
   compute reference log-probs from the base weights inside `LoRALinear` (skip the delta).

---

*"Post-training is where models become products. SFT teaches the format, DPO teaches
taste, and RLVR teaches truth. Skip any stage and users will notice within 5 messages."*

— Principal Engineer's Note, Foundation Models Division, 2026
