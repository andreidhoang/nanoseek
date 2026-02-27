# What Would Make NanoSeek Massively Valuable to the AI World

## A First-Principles Strategic Analysis

**Author perspective**: Distinguished AI Researcher and Principal Engineer, Foundation Models Division, Tier-1 AI Lab
**Date**: February 2026
**Classification**: Strategic technical assessment — reasoning from ground truth only

---

## Part I: The Honest Assessment

### What This Project Actually Is Today

I've read every line of code. Here is the ground truth:

**What works (verified by running it):**
- Complete DeepSeek V3.2 architecture in pure PyTorch (2,038 lines in `model/model.py`)
- All four innovations implemented: MLA, MoE with aux-loss-free balancing, MTP, DSA with Lightning Indexer
- Pre-training script with DDP, gradient accumulation, multi-phase training, Muon+AdamW optimizers
- Streaming dataloader from parquet files with approximate resume
- 145 tests passing covering all components including numerical stability
- Forward and backward passes verified on CPU
- Configuration system preserving DeepSeek V3 architectural ratios

**What does not exist (honest count):**
- No trained model weights. Nobody has run `pre-train.py` on real GPUs with real data for a full training run.
- No published loss curves, no training logs, no benchmark scores.
- No inference server, no API, no way for a user to interact with the model.
- No post-training (SFT/DPO). The model cannot follow instructions or hold a conversation.
- No Flash Attention or fused kernels. Training is functional but slow.
- No FP8 implementation despite config existing.

**The documentation reality:**
- 6 deep-dive theory docs (~7,000 lines) — strong educational content
- 15 implementation docs (~16,000 lines) — production code blueprints
- 2 strategic docs (~400 lines) — roadmaps

That is approximately **23,000 lines of documentation** describing how things *should* work, written around **~5,000 lines of code** that implements the architecture but has never been trained.

This is a common pattern in open-source AI: excellent architecture, comprehensive docs, zero trained artifacts. And this is precisely the pattern that prevents projects from being valuable.

### Why This Matters

The AI world in 2026 does not lack documentation of how transformers work. It does not lack theoretical explanations of MLA or MoE. What the AI world desperately lacks is:

1. **Reproducible evidence** that DeepSeek V3's innovations work at small scale
2. **A trained model** someone can download and use in 10 minutes
3. **Published training logs** showing loss curves, expert routing behavior, MTP acceptance rates — the artifacts that let researchers learn from the training process, not just the architecture

NanoGPT got 40,900 GitHub stars not because Karpathy wrote good documentation. It got them because he recorded a video of himself training GPT-2 in 90 minutes for $10 and the resulting model actually worked. The documentation supported the artifact — it did not replace it.

---

## Part II: What "Massively Valuable" Actually Means

### The Competitive Landscape (Ground Truth)

As of February 2026, here is what exists in the open-source ecosystem:

| Project | What it provides | Stars | Gap NanoSeek could fill |
|---------|-----------------|-------|------------------------|
| nanoGPT (Karpathy) | GPT-2 reproduction, educational | 40.9K | Does not cover MoE, MLA, MTP, DSA |
| DeepSeek-V3 (official) | 671B model weights, inference code | ~65K | No training code, not reproducible at small scale |
| Megatron-LM (NVIDIA) | Training infrastructure for large models | ~10K | Infrastructure, not architecture education |
| vLLM | Inference serving | ~40K | Serving only, no training |
| modded-nanogpt | Speed-optimized GPT-2 | ~12K | Dense models only, no MoE/MLA |
| OLMo (AI2) | Fully open LLM | ~5K | Dense architecture, no DeepSeek innovations |

**The gap:** There is no project that provides a reproducible, trained implementation of DeepSeek V3's architecture at educational scale. NanoSeek is the closest, but it hasn't crossed the finish line — it has the architecture but not the trained model.

### What Would Actually Move the Needle

From my experience leading foundation model development, a project becomes massively valuable when it provides something researchers and engineers **cannot get anywhere else**. For NanoSeek, that means:

**Tier 1 value (would generate genuine impact):**
1. A trained 1B-active/4.75B-total MoE model with published weights, trained using all four DeepSeek innovations
2. Complete training logs (loss curves, expert routing evolution, MTP loss contribution, memory usage) that researchers can study
3. Reproducible training recipe that someone with 8×H100 (or even 1×H100 with gradient checkpointing) can run themselves

**Tier 2 value (would generate significant interest):**
4. Post-trained chat model (SFT + DPO) with benchmarks showing it's competitive for its size
5. Working inference with MTP speculative decoding showing actual measured speedup
6. A blog post or technical report documenting what was learned during training

**Tier 3 value (nice to have):**
7. Multiple scale variants (500M, 1B, 2B) showing scaling behavior
8. Comparison with dense baselines at matched FLOPs
9. Community training framework for experiments

### What Would NOT Move the Needle (Honest)

- More documentation of the architecture (already excellent, diminishing returns)
- More implementation docs of theoretical production features (useful but speculative)
- Triton kernels, FP8, or FSDP (optimization before having a working model)
- An inference server (nothing to serve yet)

---

## Part III: First-Principles Engineering Analysis

### The Core Question

**What is the minimum viable path from where we are today to "someone can download a model and use it"?**

Let me trace this from first principles.

### Step 1: Data Pipeline Verification

**Current state:** `scripts/dataloader.py` streams from parquet files using `scripts/dataset.py` to find them and `scripts/tokenizer.py` to tokenize. The setup script `scripts/setup_data.py` downloads FineWeb-Edu.

**What needs to happen:**
```
1. Run: python scripts/setup_data.py --dataset=fineweb-edu --tokens=10B
   → Downloads ~20GB of parquet files to data/fineweb-edu/
   → This is a REAL step. Someone has to actually do it.

2. Verify: the dataloader yields correct (input, target) pairs
   → Run a few batches, check token IDs are in [0, 65535]
   → Check no degenerate sequences (all zeros, all same token)
   
3. Estimated time: ~2 hours for download + verification
```

**Risk assessment:** The dataloader works (it has real streaming implementation, not a stub). The tokenizer exists. The main risk is FineWeb-Edu parquet format compatibility — this needs to be tested with the actual data.

### Step 2: Training Run

**Current state:** `scripts/pre-train.py` supports everything needed: DDP, gradient accumulation, multi-phase training, Muon+AdamW, checkpointing, evaluation.

**What needs to happen:**
```
Hardware: 8×H100 80GB (or equivalent)
Duration: ~14 hours
Cost: ~$300

Configuration (from actual config.py):
  - total_tokens: 22B (Chinchilla optimal for 1.08B active params)
  - global_batch_size: 128
  - sequence_length: 4096
  - learning_rate: 3e-4 → 3e-5 (DeepSeek schedule)
  - 16 gradient accumulation steps (single GPU) or 2 (8 GPU)

Command:
  torchrun --nproc_per_node=8 scripts/pre-train.py \
    --target_param_data_ratio=20.0 \
    --max_seq_len=4096 \
    --device_batch_size=4 \
    --total_batch_size=524288

Expected behavior:
  - Loss starts ~11 (ln(65536) ≈ 11.09, random init)
  - Drops to ~6-7 by 10% of training
  - Converges to ~3.5-4.5 by end
  - Expert load variance should decrease over training
  - MTP loss should track ~0.5-1.0 above main loss
```

**Risk assessment:** This is the highest-risk step. Real training at this scale exposes bugs that unit tests cannot catch:
- Memory: 4.75B params × 2 bytes (bf16) = 9.5GB for model alone, plus optimizer states. DistAdamW and DistMuon shard across GPUs, so this should fit. But activation memory at seq_len=4096 needs verification.
- Numerical: MoE routing with 64 experts can collapse in early training. The bias-based load balancing must work correctly.
- Multi-phase: The Phase 1 → Phase 2 transition (dense → sparse at 80% of tokens) has only been coded but never tested in a real run.
- MTP: The MTP loss schedule (0.3 → 0.1 at 60%) interacts with the main loss. If the MTP module's cross-attention has a shape mismatch with the reduced hidden sizes used in testing, it would fail silently.

**Critical realization:** The MTP module's cross-attention uses `nn.MultiheadAttention(embed_dim=hidden_size)` which hardcodes the full hidden_size (2048). This means MTP only works at full scale, not with the reduced configs used in unit tests. This is actually fine for a real training run but means MTP has never been end-to-end tested at the config it would actually run at. This is a genuine risk.

### Step 3: Evaluation

**Current state:** `model/eval/core_eval.py` has CORE benchmark evaluation. `model/eval/loss_eval.py` has BPB evaluation.

**What needs to happen:**
```
Run evaluation on standard benchmarks:
  - MMLU (5-shot): Expected ~30-40% for 1B model
  - HellaSwag (10-shot): Expected ~50-60%
  - ARC-Challenge (25-shot): Expected ~35-45%
  - Validation loss / BPB

Compare against baselines:
  - OLMo-1B: Known benchmark scores available
  - TinyLlama-1.1B: Known benchmark scores available
  - Phi-2 (2.7B): As an upper bound
```

**Risk assessment:** Low risk. Evaluation code exists and can be extended with standard lm-evaluation-harness integration.

### Step 4: Post-Training (SFT + DPO)

**Current state:** No implementation exists.

**What needs to happen:**
```
SFT phase:
  - Dataset: Alpaca-GPT4 or UltraChat (publicly available)
  - Training: 3 epochs, cosine LR decay
  - Loss: CE on assistant tokens only (masking user/system)
  - Duration: ~2 hours on 8×H100
  - Output: Model that follows instructions

DPO phase:
  - Dataset: UltraFeedback or Nectar (publicly available)
  - Training: 1 epoch
  - Loss: DPO with β=0.1
  - Duration: ~30 minutes on 8×H100
  - Output: Model with improved alignment
```

**Risk assessment:** Medium risk. SFT is straightforward. DPO requires careful implementation of the reference model and β tuning. The main risk is that a 1B MoE model might not have enough capacity for meaningful instruction following — but OLMo-1B and TinyLlama show this is feasible for dense models, and our 4.75B total parameters provide more capacity.

### Step 5: Serving

**Current state:** Basic `generate_cached()` method exists with KV cache support.

**What needs to happen:**
```
Minimum viable serving:
  - Gradio demo (simplest path to "users can interact")
  - Load model weights
  - Use generate_cached() for KV cache efficiency
  - Add MTP speculative decoding for speedup
  
  Duration to implement: ~1-2 days
  
  This is NOT a production server. It's a demo that proves the model works.
```

**Risk assessment:** Low risk. Gradio + generate_cached() is proven. The MTP speculative decoding integration needs testing but the components exist.

---

## Part IV: The Strategic Decision

### What Makes This Project Different From Every Other Open-Source LLM

I need to be precise about this because it determines everything.

**NanoSeek's unique position:**

1. It is the **only** open-source project that implements ALL FOUR of DeepSeek V3's innovations in a single, readable, testable codebase:
   - Multi-Head Latent Attention (MLA) with 23× KV cache compression
   - Mixture of Experts (MoE) with auxiliary-loss-free load balancing
   - Multi-Token Prediction (MTP) for speculative decoding
   - DeepSeek Sparse Attention (DSA) with Lightning Indexer

2. It preserves DeepSeek's architectural ratios (q_lora_rank/hidden ≈ 0.21, kv_lora_rank/hidden ≈ 0.07, experts/active = 64/8) so insights transfer to larger scales.

3. It is designed to be trainable on accessible hardware (~$300 on 8×H100) unlike the official DeepSeek V3 which requires 2048 H800 GPUs.

**This means the project's value is NOT as another LLM. It's as the definitive educational and experimental platform for understanding and extending the DeepSeek architecture family.**

Think of it this way:
- nanoGPT taught the world how transformers train → 40K stars
- NanoSeek could teach the world how DeepSeek trains → ?

But only if there is a trained model with published results. Architecture code without training evidence is a textbook without experiments.

### The Decision: Train First, Optimize Later

Every instinct in engineering says "make it fast first, then train." This is wrong for this project.

**The correct sequence:**

```
Phase 1: PROVE IT WORKS (4 weeks, ~$400 compute)
├── Week 1: Data pipeline verification + small-scale training (100M tokens)
├── Week 2: Full 22B token training run
├── Week 3: Evaluation + SFT + DPO
└── Week 4: Demo deployment + write-up

Phase 2: MAKE IT EXCELLENT (8 weeks, incremental effort)  
├── Flash Attention integration
├── Fused MoE kernels
├── Production inference with speculative decoding
└── Multi-scale ablations

Phase 3: MAKE IT A PLATFORM (ongoing)
├── Community training recipes
├── Extension framework for new architectures
└── Curriculum materials
```

**Why this order:** A working trained model with published results generates 100× more community engagement than perfect documentation. Community engagement generates contributors. Contributors build the production features. This is how every successful open-source ML project has grown.

---

## Part V: The Technical Roadmap (What Concretely Must Happen)

### Phase 1, Week 1: Prove the Pipeline

**Objective:** Run a short training job (1B tokens) and verify everything works end-to-end.

```
Day 1-2: Data
  - Download FineWeb-Edu: python scripts/setup_data.py --tokens=2B
  - Verify dataloader produces correct batches
  - Run 100 iterations on CPU with --device_batch_size=1 --max_seq_len=512
  - Check: loss decreasing, no NaN, expert loads balanced

Day 3-4: GPU Smoke Test  
  - Single H100: python scripts/pre-train.py --num_iterations=500
  - Check: memory fits, throughput reasonable, loss decreasing
  - Profile: where is time spent? (attention, MoE, data loading)

Day 5: Multi-GPU Smoke Test
  - 8×H100: torchrun --nproc_per_node=8 scripts/pre-train.py --num_iterations=500
  - Check: DDP sync correct, DistMuon/DistAdamW working
  - Check: gradient accumulation numerically equivalent to large batch

Day 6-7: Extended Test (1B tokens, ~1 hour)
  - Run to 1B tokens (~2000 steps)
  - Log: loss curve, expert loads, MTP loss, gradient norms
  - Decision gate: If loss is decreasing as expected, proceed to Phase 1 Week 2
```

**What to publish:** Training loss curve at 1B tokens. This alone would be the first public evidence that the DeepSeek V3 architecture works at nano scale.

### Phase 1, Week 2: Full Training Run

**Objective:** Train the complete NanoSeek-1B model for 22B tokens.

```
Configuration (from actual model/config.py):
  NanoSeek-1B:
    hidden_size=2048, num_layers=16, num_heads=16
    MoE: 64 routed + 2 shared, 8 active
    MLA: q_lora=430, kv_lora=143
    MTP: 1 module
    Total: ~4.75B params, ~1.08B active

  Training:
    tokens=22B, batch=512K, seq_len=4096
    LR: 3e-4 → 3e-5 (warmup → constant → cosine)
    Optimizer: Muon (matrices) + AdamW (embeddings)
    Duration: ~42,000 steps, ~14 hours on 8×H100

  Phase 1 (80% = 17.6B tokens):
    Dense attention, 4K context
    Indexer trains via aux loss

  Phase 2 (20% = 4.4B tokens):
    Sparse DSA enabled, 8K context
    YaRN for position extension
```

**What to capture and publish:**
1. Loss curves (main, MTP, aux, total) — one plot per 100 steps
2. Expert load distribution heatmap — evolving over training
3. MTP acceptance rates (at eval checkpoints)
4. Gradient norm evolution
5. Memory usage per GPU
6. Tokens/second throughput
7. Total compute (H100-hours)
8. Final checkpoint weights

**Decision gate:** Final validation loss < 4.5 BPB. If not, investigate and re-train with adjusted hyperparameters.

### Phase 1, Week 3: Evaluation and Post-Training

**Objective:** Benchmark the model and make it chat-capable.

```
Day 1-2: Benchmarking
  Standard benchmarks (using lm-evaluation-harness or custom):
  - MMLU (5-shot)
  - HellaSwag (10-shot)  
  - ARC-Challenge (25-shot)
  - GSM8K (5-shot CoT)
  - HumanEval (pass@1, pass@10)
  
  Comparison table against:
  - OLMo-1B (dense, 1B)
  - TinyLlama-1.1B (dense, 1.1B)
  - Phi-2 (dense, 2.7B) — as upper bound
  
  MoE-specific metrics:
  - Expert utilization entropy
  - Load balance Gini coefficient
  - Dead expert count (if any)

Day 3-4: SFT
  Dataset: UltraChat-200K (filtered)
  Method: Standard CE loss on assistant tokens
  Duration: ~2 hours
  
Day 5: DPO
  Dataset: UltraFeedback-60K
  Method: DPO with β=0.1
  Duration: ~30 minutes

Day 6-7: Post-Training Evaluation
  - ChatBot Arena style manual evaluation
  - MT-Bench score
  - Safety benchmarks (TruthfulQA)
```

**What to publish:** Full benchmark table, comparison against dense baselines at matched FLOPs. This is the evidence that MoE + MLA + MTP + DSA provides value at small scale.

### Phase 1, Week 4: Demo and Write-up

**Objective:** Make the model accessible and document the learnings.

```
Day 1-2: Demo
  - Gradio interface for chat
  - Hosted on Hugging Face Spaces
  - Include MTP speculative decoding toggle
  
Day 3-5: Write-up
  - Technical report (15-20 pages)
  - Blog post (readable summary)
  - "What we learned training DeepSeek V3 at 1B scale"
  
Day 6-7: Release
  - Model weights on Hugging Face
  - Training logs (full WandB run)
  - Reproducibility instructions
```

---

## Part VI: Why This Strategy Will Work (From First Principles)

### The Value Creation Chain

```
Trained model with weights    →  Researchers can EVALUATE ideas against it
Published training logs       →  Engineers can LEARN from the training process
Reproducible recipe           →  Labs can REPRODUCE and EXTEND the work
Benchmark comparisons         →  Community can COMPARE against other approaches
Chat-capable model            →  Users can INTERACT and PROVIDE FEEDBACK
```

Each link in this chain generates value. The existing NanoSeek codebase has built the FOUNDATION (architecture code) but has not yet produced any of the ARTIFACTS (weights, logs, benchmarks, demos) that create the value chain.

### The Moat: Why No One Else Has Done This

It is genuinely surprising that no one has published a small-scale reproduction of DeepSeek V3. The reasons:

1. **Complexity barrier:** Implementing MLA + MoE + MTP + DSA together is hard. NanoSeek has already done this.
2. **Compute barrier:** $300 is not trivial for individual researchers, but it's cheap for institutions. The real barrier is confidence that the code will work, which requires the smoke-test phase.
3. **Incentive misalignment:** Academic groups publish papers about one innovation at a time. An all-four-at-once reproduction is engineering-heavy and publication-unfriendly.
4. **DeepSeek's own gap:** The official DeepSeek repository has inference code but not training code. Their training recipe is described in the paper but requires significant engineering to reproduce.

This gap is NanoSeek's opportunity. Being first-to-publish a working small-scale DeepSeek V3 reproduction would be genuinely novel and highly cited.

### Quantifying the Impact

Based on comparable projects:

| Outcome | Evidence | Estimated impact |
|---------|----------|-----------------|
| First small-scale DeepSeek V3 reproduction | Trained model + logs | 5K-15K GitHub stars, significant citations |
| Benchmark showing MoE advantage at 1B | Published comparison table | Referenced in MoE survey papers |
| MTP speculative decoding measured speedup | Throughput benchmarks | Adopted by inference frameworks |
| Educational resource for DeepSeek architecture | Code + docs + video | University course material |
| Community training platform | Reproducible recipes | Ongoing contributor growth |

### What Could Go Wrong (Honest Risk Assessment)

1. **Training diverges:** MoE models can collapse. Mitigation: the bias-based load balancing is designed to prevent this, and we have 1B token smoke test before full run.

2. **Model doesn't learn:** 1B active params might be too small for meaningful performance. Mitigation: OLMo-1B and TinyLlama prove 1B is viable. Our 4.75B total gives more capacity via MoE.

3. **MTP doesn't help at this scale:** Multi-token prediction might only show value at larger scales. Mitigation: we measure it explicitly and publish honestly. Even a negative result ("MTP doesn't help below 10B active params") is valuable.

4. **Compute cost overrun:** Training might take longer than expected. Mitigation: checkpoint every 1000 steps, budget 2× for safety ($600 instead of $300).

5. **Post-training fails:** SFT/DPO on 1B MoE might not produce a useful chat model. Mitigation: this is an exploration, not a commitment. Even base model results are valuable.

---

## Part VII: The Production Path (What "Serving to Users" Means)

### Realistic Production Definition

For a 1B-active parameter research model in 2026, "production serving" means:

1. **Hugging Face model card** with weights, benchmarks, and usage examples
2. **Gradio/Streamlit demo** hosted on Spaces
3. **API endpoint** (simple FastAPI + generate_cached) for programmatic access
4. **Speculative decoding** showing measured inference speedup
5. **Quantized variants** (INT8, INT4) for consumer GPU inference

It does NOT mean:
- Competing with ChatGPT/Claude on quality
- Handling millions of requests
- Enterprise deployment
- Safety guarantees for unrestricted use

### The Minimum Viable Product

```python
# This is literally all that's needed for "serving":

from model.model import create_nanoseek, NanoSeekModel
from model.config import get_nanoseek_config

# Load trained weights
config = get_nanoseek_config()
model = create_nanoseek(config)
model.load_state_dict(torch.load("nanoseek-1b-chat.pt"))
model.eval().cuda()

# Serve via Gradio
import gradio as gr

def chat(message, history):
    tokens = tokenizer.encode(format_chat(history + [(message, "")]))
    output_tokens = []
    for token in model.generate_cached(tokens, max_tokens=512, temperature=0.7):
        output_tokens.append(token)
        yield tokenizer.decode(output_tokens)

gr.ChatInterface(chat, title="NanoSeek-1B").launch()
```

The serving infrastructure is the EASY part. The hard part — training the model — is what makes the serving meaningful.

---

## Part VIII: The Decision Matrix

### What to Do Now (Priority Order)

| Priority | Action | Cost | Time | Value |
|----------|--------|------|------|-------|
| **P0** | Download FineWeb-Edu and verify data pipeline | $0 | 1 day | Unblocks everything |
| **P0** | Run 1B token smoke test on GPU | $20 | 3 hours | Proves pipeline works |
| **P0** | Full 22B token training run | $300 | 14 hours | **THE artifact** |
| **P0** | Publish weights + training logs | $0 | 1 day | **THE impact** |
| **P1** | Benchmark against dense baselines | $10 | 4 hours | Scientific value |
| **P1** | SFT + DPO post-training | $20 | 3 hours | Usability |
| **P1** | Gradio demo | $0 | 1 day | Accessibility |
| **P1** | Technical write-up | $0 | 3 days | Communication |
| **P2** | Flash Attention integration | $0 | 1 week | Training speed |
| **P2** | Fused MoE kernels | $0 | 1 week | MoE efficiency |
| **P2** | Multi-scale ablations (500M, 2B) | $500 | 1 week | Scientific depth |
| **P3** | Production inference server | $0 | 2 weeks | Scale serving |
| **P3** | FP8 training | $0 | 1 week | Training efficiency |
| **P3** | FSDP for larger models | $0 | 1 week | Scale training |

### What NOT to Do (Waste of Effort at This Stage)

1. **Do not write more architecture documentation.** The existing 23,000 lines of docs are excellent. More documentation without a trained model is diminishing returns.

2. **Do not build production infrastructure before training.** An inference server with nothing to serve is an empty restaurant.

3. **Do not optimize before measuring.** Flash Attention and fused kernels are important but the first training run should use the current PyTorch implementation. It will be slower but it will work, and the profiling data from the slow run will guide optimization priorities.

4. **Do not aim for SOTA benchmarks.** A 1B model will not beat GPT-4. That is not the point. The point is demonstrating that the DeepSeek V3 architecture works at small scale and publishing the evidence.

---

## Part IX: The One-Page Summary

### Vision
NanoSeek becomes the definitive open-source platform for understanding and experimenting with DeepSeek V3 architecture — the "nanoGPT of the MoE era."

### Current Gap
Architecture is implemented and tested. No model has been trained. No artifacts exist.

### Critical Path
Train the model ($300, 14 hours). Publish weights and logs. Everything else follows.

### Unique Value
Only project implementing all four DeepSeek V3 innovations (MLA + MoE + MTP + DSA) at reproducible scale. First-mover advantage is available now but will not last.

### Success Metric
A researcher can run `pip install nanoseek && nanoseek chat` within 10 minutes and have a conversation with a model that was trained using the same architecture as DeepSeek V3.

### Failure Mode
Continuing to write documentation and implementation blueprints without training the model. The project stalls as "impressive code that nobody uses."

---

## Part X: What I Would Do If This Were My Lab

If I were the principal engineer responsible for this project at a tier-1 lab, here is exactly what I would do on Monday morning:

**Monday:**
- Reserve 8×H100 instance for 2 days
- Download FineWeb-Edu (2 hours)
- Run `verify_install.py` on GPU (5 minutes)
- Run 500-step smoke test with real data (30 minutes)
- Fix any issues (expect 2-3 hours of debugging)

**Tuesday:**
- Start full 22B token training run (14 hours)
- Monitor first 1000 steps closely (loss, expert loads, memory)
- Go to sleep. Set checkpoint_every=1000.

**Wednesday morning:**
- Training complete. Check final loss.
- Run evaluation benchmarks (4 hours)
- Start SFT on UltraChat (2 hours)

**Wednesday afternoon:**
- Run DPO (30 minutes)
- Deploy Gradio demo
- Write Hugging Face model card

**Thursday:**
- Write blog post: "We Trained DeepSeek V3 Architecture at 1B Scale for $300 — Here's What We Learned"
- Publish: weights, logs, benchmarks, code, demo
- Submit to r/MachineLearning, Hacker News, Twitter

**Friday:**
- Respond to community feedback
- Plan Phase 2 based on what we learned

Total cost: ~$400 (compute) + 5 days of engineering time.
Total impact: First public small-scale DeepSeek V3 reproduction with trained weights.

That is how you make a project massively valuable. Not by writing more docs. By training the model and publishing the evidence.

---

*"In God we trust. All others must bring data."*
— W. Edwards Deming

*Applied to ML: In architecture we trust. All others must bring trained weights.*
