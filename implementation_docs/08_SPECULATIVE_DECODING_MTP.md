# 08 — Production Speculative Decoding with MTP Draft Models

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete draft-verify-accept pipeline using MTP modules as the draft model
**Prerequisite**: Read `07_INFERENCE_ENGINE.md` (continuous batching), `00_MASTER_PLAN.md` (context)

---

## 1. Problem Statement

Autoregressive decoding is **memory-bandwidth bound**: every token requires a full forward
pass through all 16 transformer layers, but arithmetic intensity is ~1 FLOP/byte — 99% of
GPU compute sits idle waiting on HBM.

| Metric | Standard Decode | With Speculation |
|--------|----------------:|----------------:|
| Forward passes per token | 1 | 1/(1+α) |
| GPU compute utilization | ~1% | ~5-15% |
| Effective tokens/second | 1× | 1.4-1.8× |

**Speculative decoding** breaks this: draft K tokens cheaply, verify all K in a single
parallel forward pass, accept a prefix, reject the rest. Verification is parallel because
the main model processes [prompt + drafts] as a sequence, producing logits at every
position simultaneously.

**NanoSeek's structural advantage**: MTP modules trained alongside the main model *are*
the draft model. No separate model needed. Shared embeddings and lm_head yield higher
acceptance rates than an external draft model.

### Current State (`model/model.py`)

`MultiTokenPrediction.speculative_decode()` generates draft tokens via MTP modules but
has **no verification, no acceptance/rejection, no adjusted sampling**. We need the
complete pipeline: draft → verify → accept/reject → residual sample → repeat.

### Targets

- **Acceptance rate**: >60% for MTP-1 drafts
- **Wall-clock speedup**: 1.4-1.8× over vanilla autoregressive
- **Correctness**: EXACT distribution preservation (lossless)
- **Memory overhead**: <5% for draft state

---

## 2. First Principles

### Why Speculative Decoding Works

Standard decode: N tokens = N sequential forward passes. Speculation: draft K tokens
cheaply, verify all K+1 positions in one forward pass, accept prefix of length n ≤ K.
Amortized cost per token drops to ~1/(1 + E[n]) forward passes.

### The Acceptance Criterion

Given p(x) (main model) and q(x) (draft model) for token x at position t:

```
Accept x with probability:  α(x) = min(1, p(x) / q(x))
If rejected, sample from:   p'(x) = max(0, p(x) - q(x)) / Z
```

This preserves the **exact** target distribution p(x). Proof in Section 4.

### Expected Acceptance Rate

```
P(accept) = Σ_x min(p(x), q(x)) = 1 - TV(p, q)
```

The closer draft to target, the higher acceptance. MTP modules are co-trained with the
main model on identical data, sharing embeddings and lm_head — distribution alignment
by construction.

### Why MTP is Ideal for Drafting

| Property | External Draft Model | MTP Draft Module |
|----------|---------------------|------------------|
| Parameter sharing | None | Embeddings + lm_head |
| Distribution alignment | Low-moderate | High (co-trained) |
| Expected acceptance rate | 40-60% | 60-80% |
| Additional memory | Full small model | Single lightweight block |
| Cost per draft token | Small model forward | One MTPBlock forward |

### Speculation Economics

Cost ratio c = MTP forward / main forward ≈ 1/16 (one MTPBlock vs 16 layers).

```
Speedup = (1 + E[accepted]) / (1 + K·c)

K=1, α=0.65: Speedup = 1.65 / 1.0625 ≈ 1.55×
K=3, α=0.60: Speedup ≈ 2.36 / 1.19 ≈ 1.99×
```

---

## 3. Production Code

### File: `model/serving/speculative.py`

```python
"""
Production Speculative Decoding Engine for NanoSeek.

Implements the complete draft-verify-accept pipeline using MTP modules
as the draft model.  Preserves the exact target distribution (lossless).

Reference: Leviathan et al. (2023), Chen et al. (2023)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 8a. Speculative Decoding Engine
# ============================================================================

@dataclass
class SpeculativeConfig:
    """Configuration for the speculative decoding engine."""
    max_draft_tokens: int = 1
    min_draft_tokens: int = 1
    initial_draft_tokens: int = 1
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # Adaptive draft length (Section 8b)
    adaptive_draft_length: bool = True
    acceptance_rate_window: int = 64
    acceptance_rate_high: float = 0.8
    acceptance_rate_low: float = 0.3
    # Batching and caching
    max_batch_size: int = 64
    draft_cache_enabled: bool = False


@dataclass
class SpeculativeMetrics:
    """Runtime metrics for speculative decoding."""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rounds: int = 0
    total_main_forward_ms: float = 0.0
    total_draft_forward_ms: float = 0.0
    acceptance_history: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_accepted_per_round(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_rounds

    @property
    def tokens_per_main_forward(self) -> float:
        if self.total_rounds == 0:
            return 1.0
        return (self.total_accepted_tokens + self.total_rounds) / self.total_rounds

    @property
    def speedup_estimate(self) -> float:
        if self.total_main_forward_ms == 0:
            return 1.0
        draft_overhead = self.total_draft_forward_ms / max(self.total_main_forward_ms, 1e-6)
        return self.tokens_per_main_forward / (1.0 + draft_overhead)

    def report(self) -> Dict[str, float]:
        return {
            "acceptance_rate": self.acceptance_rate,
            "avg_accepted_per_round": self.avg_accepted_per_round,
            "tokens_per_main_forward": self.tokens_per_main_forward,
            "estimated_speedup": self.speedup_estimate,
            "total_rounds": self.total_rounds,
        }


def apply_sampling_params(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tensor:
    """Apply temperature, top-k, top-p to logits, return probabilities.

    Temperature 0.0 returns one-hot on argmax (greedy).
    """
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k_clamped = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k_clamped, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = sorted_probs.scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return probs


def sample_from_probs(probs: Tensor) -> Tuple[Tensor, Tensor]:
    """Sample tokens from a probability distribution.

    Returns (tokens, token_probs) both of shape [batch].
    """
    is_greedy = (probs.max(dim=-1).values > 0.999).all()
    if is_greedy:
        tokens = probs.argmax(dim=-1)
    else:
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_probs = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, token_probs


class SpeculativeDecodingEngine:
    """Complete draft-verify-accept pipeline for speculative decoding.

    Each step():
      1. DRAFT  — MTP module(s) generate K candidate tokens
      2. VERIFY — Main model forward on [context + drafts]
      3. ACCEPT/REJECT — Walk drafts comparing p vs q
      4. SAMPLE — On rejection, sample correction from max(0, p-q)/Z
      5. REPEAT — Advance by accepted + 1 tokens
    """

    def __init__(self, model: nn.Module, config: Optional[SpeculativeConfig] = None):
        self.model = model
        self.config = config or SpeculativeConfig()
        self.metrics = SpeculativeMetrics()
        if not hasattr(model, "mtp") or model.mtp is None:
            raise ValueError("Model must have MTP modules for speculative decoding")
        self._current_draft_len = self.config.initial_draft_tokens
        self._device = next(model.parameters()).device

    # ── Core: single speculation round ────────────────────────────────

    @torch.inference_mode()
    def speculative_step(
        self,
        input_ids: Tensor,
        past_key_values: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[list]]:
        """Execute one draft-verify-accept round.

        Returns (new_tokens [batch, n_accepted+1], updated_past_key_values).
        """
        batch_size = input_ids.shape[0]
        K = self._current_draft_len

        # 1. DRAFT
        t0 = time.perf_counter()
        draft_tokens, draft_probs = self._generate_drafts(input_ids, past_key_values, K)
        draft_ms = (time.perf_counter() - t0) * 1000

        # 2. VERIFY — forward [last_token, d1, ..., dK] through main model
        verify_ids = torch.cat([input_ids[:, -1:], draft_tokens], dim=1)
        t1 = time.perf_counter()
        outputs = self.model(
            input_ids=verify_ids, past_key_values=past_key_values, use_cache=True,
        )
        main_ms = (time.perf_counter() - t1) * 1000

        main_logits = outputs["logits"]  # [batch, 1+K, vocab]
        new_past = outputs.get("past_key_values", None)

        main_probs_all = []
        for i in range(K + 1):
            p = apply_sampling_params(
                main_logits[:, i], self.config.temperature,
                self.config.top_k, self.config.top_p,
            )
            main_probs_all.append(p)
        main_probs = torch.stack(main_probs_all, dim=1)  # [batch, K+1, V]

        # 3. ACCEPT / REJECT
        accepted_tokens, n_accepted = self._accept_reject(
            draft_tokens, draft_probs, main_probs, batch_size, K,
        )

        # 4. UPDATE METRICS
        total_accepted = n_accepted.sum().item()
        self.metrics.total_draft_tokens += K * batch_size
        self.metrics.total_accepted_tokens += int(total_accepted)
        self.metrics.total_rounds += batch_size
        self.metrics.total_main_forward_ms += main_ms
        self.metrics.total_draft_forward_ms += draft_ms
        self.metrics.acceptance_history.append(total_accepted / max(K * batch_size, 1))

        # 5. TRIM KV CACHE — remove rejected draft positions
        if new_past is not None:
            positions_to_remove = K - int(n_accepted.max().item())
            if positions_to_remove > 0:
                new_past = [
                    (kv[:, :-positions_to_remove], kpe[:, :-positions_to_remove])
                    if kv is not None else None
                    for kv, kpe in (layer for layer in new_past)
                ]

        # 6. ADAPTIVE DRAFT LENGTH
        if self.config.adaptive_draft_length:
            self._adapt_draft_length()

        return accepted_tokens, new_past

    # ── Draft generation ──────────────────────────────────────────────

    def _generate_drafts(
        self, input_ids: Tensor, past_key_values: Optional[list], K: int,
    ) -> Tuple[Tensor, Tensor]:
        """Generate K draft tokens using MTP modules.

        Returns draft_tokens [batch, K], draft_probs [batch, K, vocab].
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                past_key_values=past_key_values, use_cache=False,
            )
            main_hidden = outputs["hidden_states"]

        draft_tokens_list, draft_probs_list = [], []
        prev_hidden = main_hidden
        prev_token = input_ids[:, -1:]
        mtp_modules = self.model.mtp.mtp_modules

        for i in range(K):
            module = mtp_modules[min(i, len(mtp_modules) - 1)]
            logits, hidden = module(
                prev_hidden=prev_hidden[:, -1:],
                target_tokens=prev_token,
                main_hidden=main_hidden[:, -1:],
            )
            probs = apply_sampling_params(
                logits[:, -1], self.config.temperature,
                self.config.top_k, self.config.top_p,
            )
            token, _ = sample_from_probs(probs)
            draft_tokens_list.append(token)
            draft_probs_list.append(probs)
            prev_hidden = hidden
            prev_token = token.unsqueeze(-1)

        return torch.stack(draft_tokens_list, dim=1), torch.stack(draft_probs_list, dim=1)

    # ── Accept / Reject ───────────────────────────────────────────────

    def _accept_reject(
        self,
        draft_tokens: Tensor,   # [batch, K]
        draft_probs: Tensor,    # [batch, K, vocab]
        main_probs: Tensor,     # [batch, K+1, vocab]
        batch_size: int,
        K: int,
    ) -> Tuple[Tensor, Tensor]:
        """Token-by-token acceptance with adjusted residual sampling.

        For each draft position i:
          q_x = draft_probs[:, i, x_i]
          p_x = main_probs[:, i, x_i]
          Accept with prob min(1, p_x / q_x).
          On rejection: sample from max(0, p - q) / Z.

        After all K drafts (or first rejection), sample one bonus token
        from main_probs at the next position.

        Returns accepted_tokens [batch, max_n+1], n_accepted [batch].
        """
        device = draft_tokens.device
        n_accepted = torch.zeros(batch_size, dtype=torch.long, device=device)
        all_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        result_tokens = []

        for i in range(K):
            x_i = draft_tokens[:, i]
            q_x = draft_probs[:, i].gather(-1, x_i.unsqueeze(-1)).squeeze(-1)
            p_x = main_probs[:, i].gather(-1, x_i.unsqueeze(-1)).squeeze(-1)

            if self.config.temperature == 0.0:
                accept_mask = (main_probs[:, i].argmax(dim=-1) == x_i) & ~all_done
            else:
                r = torch.rand(batch_size, device=device)
                ratio = (p_x / q_x.clamp(min=1e-10)).clamp(max=1.0)
                accept_mask = (r < ratio) & ~all_done

            correction = self._sample_correction(main_probs[:, i], draft_probs[:, i])
            token = torch.where(accept_mask, x_i, correction)
            result_tokens.append(token)
            n_accepted += accept_mask.long()
            all_done = all_done | (~accept_mask & ~all_done)

            if all_done.all():
                break

        # Bonus token for sequences that accepted all K drafts
        all_accepted_mask = (n_accepted == K) & ~all_done
        if not all_done.all() or all_accepted_mask.any():
            bonus_probs = main_probs[:, min(K, main_probs.shape[1] - 1)]
            bonus_token, _ = sample_from_probs(bonus_probs)
            if all_accepted_mask.any():
                result_tokens.append(bonus_token)

        if not result_tokens:
            token, _ = sample_from_probs(main_probs[:, 0])
            return token.unsqueeze(1), n_accepted

        result = torch.stack(result_tokens, dim=1)
        max_len = n_accepted.max().item() + 1
        return result[:, :max_len], n_accepted

    def _sample_correction(self, p: Tensor, q: Tensor) -> Tensor:
        """Sample from adjusted residual: max(0, p - q) / Z.

        Guarantees exact target distribution preservation on rejection.
        """
        if self.config.temperature == 0.0:
            return p.argmax(dim=-1)

        residual = (p - q).clamp(min=0.0)
        residual_sum = residual.sum(dim=-1, keepdim=True)

        # Fallback to p if residual is zero everywhere (numerical edge case)
        fallback = residual_sum.squeeze(-1) < 1e-10
        if fallback.any():
            residual[fallback] = p[fallback]
            residual_sum[fallback] = p[fallback].sum(dim=-1, keepdim=True)

        residual = residual / residual_sum.clamp(min=1e-10)
        return torch.multinomial(residual, num_samples=1).squeeze(-1)

    # ── 8b. Adaptive Draft Length ─────────────────────────────────────

    def _adapt_draft_length(self) -> None:
        """Dynamically adjust draft length based on recent acceptance.

        High acceptance (>0.8) → increase K (model is predictable)
        Low acceptance  (<0.3) → decrease K (wasting draft compute)
        """
        history = self.metrics.acceptance_history
        if len(history) < 4:
            return
        window = self.config.acceptance_rate_window
        recent = history[-window:]
        rate = sum(recent) / len(recent)

        if rate > self.config.acceptance_rate_high:
            self._current_draft_len = min(
                self._current_draft_len + 1, self.config.max_draft_tokens,
            )
        elif rate < self.config.acceptance_rate_low:
            self._current_draft_len = max(
                self._current_draft_len - 1, self.config.min_draft_tokens,
            )

    @property
    def current_draft_length(self) -> int:
        return self._current_draft_len

    # ── Complete generation loop ──────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self, input_ids: Tensor, max_new_tokens: int = 128,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[Tensor, SpeculativeMetrics]:
        """Full generation loop with speculative decoding.

        Returns (generated_ids [batch, prompt_len + n], metrics).
        """
        self.model.eval()
        self.metrics = SpeculativeMetrics()
        generated = input_ids.clone()
        tokens_generated = 0

        # Prefill
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.get("past_key_values", None)

        # First token from prefill logits
        first_probs = apply_sampling_params(
            outputs["logits"][:, -1], self.config.temperature,
            self.config.top_k, self.config.top_p,
        )
        first_token, _ = sample_from_probs(first_probs)
        generated = torch.cat([generated, first_token.unsqueeze(1)], dim=1)
        tokens_generated += 1

        while tokens_generated < max_new_tokens:
            new_tokens, past_key_values = self.speculative_step(generated, past_key_values)
            generated = torch.cat([generated, new_tokens], dim=1)
            tokens_generated += new_tokens.shape[1]

            if eos_token_id is not None and (new_tokens == eos_token_id).any(dim=-1).all():
                break

        return generated[:, : input_ids.shape[1] + max_new_tokens], self.metrics


# ============================================================================
# 8c. MTP-Specific Optimizations
# ============================================================================

class MTPDraftOptimizer:
    """Optimizations specific to MTP-as-draft-model.

    1. KV cache sharing: Reuse main model's hidden states for MTP cross-attn
    2. Parallel MTP: Run independent MTP modules concurrently
    3. Draft caching: Reuse distributions for shared prefixes (best-of-N)
    """

    def __init__(self, model: nn.Module, config: SpeculativeConfig):
        self.model = model
        self.config = config
        self._draft_cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    def get_shared_hidden_state(
        self, input_ids: Tensor, past_key_values: Optional[list],
    ) -> Tensor:
        """Extract main model hidden states for MTP cross-attention.

        MTPModule uses main_hidden as cross-attn K/V. Extracting once
        and reusing across modules avoids redundant computation.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                past_key_values=past_key_values, use_cache=False,
            )
        return outputs["hidden_states"]

    def parallel_mtp_draft(
        self, main_hidden: Tensor, first_token: Tensor,
        mtp_modules: nn.ModuleList,
    ) -> Tuple[Tensor, Tensor]:
        """Draft from multiple MTP modules (sequential for 1 module,
        infrastructure ready for multi-module tree speculation)."""
        results = []
        prev_hidden, prev_token = main_hidden, first_token

        for module in mtp_modules:
            logits, hidden = module(
                prev_hidden=prev_hidden, target_tokens=prev_token,
                main_hidden=main_hidden,
            )
            probs = F.softmax(logits[:, -1], dim=-1)
            token = probs.argmax(dim=-1)
            results.append((token, probs, hidden))
            prev_hidden = hidden
            prev_token = token.unsqueeze(-1)

        tokens = torch.stack([r[0] for r in results], dim=1)
        probs = torch.stack([r[1] for r in results], dim=1)
        return tokens, probs

    def cache_draft(self, prefix_hash: int, probs: Tensor, tokens: Tensor):
        """Cache a draft distribution for prefix reuse."""
        if not self.config.draft_cache_enabled:
            return
        if len(self._draft_cache) >= 1024:
            self._draft_cache.pop(next(iter(self._draft_cache)))
        self._draft_cache[prefix_hash] = (probs.detach().clone(), tokens.detach().clone())

    def lookup_cached_draft(self, prefix_hash: int) -> Optional[Tuple[Tensor, Tensor]]:
        if not self.config.draft_cache_enabled:
            return None
        return self._draft_cache.get(prefix_hash)


# ============================================================================
# 8d. Integration with Continuous Batching
# ============================================================================

@dataclass
class BatchedSequenceState:
    """Per-sequence state within a speculative decoding batch."""
    sequence_id: int
    input_ids: Tensor
    n_generated: int = 0
    max_tokens: int = 128
    finished: bool = False


class BatchedSpeculativeEngine:
    """Speculative decoding integrated with continuous batching (Doc 07).

    Key challenges:
    - Variable acceptance lengths across sequences in the same batch
    - Draft tokens temporarily consume KV cache slots
    - Scheduler must account for draft overhead in memory budget

    Memory overhead per speculation round:
      B × K × (kv_lora_rank + qk_rope_head_dim) × 2 bytes × num_layers
      NanoSeek-1B, K=1, B=32: 32 × 1 × 175 × 2 × 16 = ~175 KB (negligible)
    """

    def __init__(self, model: nn.Module, config: Optional[SpeculativeConfig] = None):
        self.config = config or SpeculativeConfig()
        self.engine = SpeculativeDecodingEngine(model, self.config)
        self.active_sequences: Dict[int, BatchedSequenceState] = {}

    def add_sequence(self, seq_id: int, input_ids: Tensor, max_tokens: int = 128):
        self.active_sequences[seq_id] = BatchedSequenceState(
            sequence_id=seq_id, input_ids=input_ids, max_tokens=max_tokens,
        )

    def step(self) -> Dict[int, Tensor]:
        """One speculation round for all active sequences."""
        results = {}
        for seq_id, state in list(self.active_sequences.items()):
            if state.finished:
                continue
            new_tokens, _ = self.engine.speculative_step(state.input_ids.unsqueeze(0), None)
            new_tokens = new_tokens.squeeze(0)
            state.input_ids = torch.cat([state.input_ids, new_tokens])
            state.n_generated += new_tokens.shape[0]
            results[seq_id] = new_tokens
            if state.n_generated >= state.max_tokens:
                state.finished = True
        return results

    def remove_sequence(self, seq_id: int) -> Optional[Tensor]:
        state = self.active_sequences.pop(seq_id, None)
        return state.input_ids if state else None

    def memory_budget_check(self, max_kv_slots: int) -> bool:
        """Verify draft tokens won't exceed KV cache budget."""
        n_active = sum(1 for s in self.active_sequences.values() if not s.finished)
        return n_active * (1 + self.config.max_draft_tokens) <= max_kv_slots


# ============================================================================
# 8e. Benchmarking & Metrics
# ============================================================================

class SpeculativeBenchmark:
    """Benchmarking harness: acceptance rate, speedup, distribution fidelity."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.inference_mode()
    def measure_acceptance_rate(
        self, prompts: List[Tensor], max_tokens: int = 64,
        config: Optional[SpeculativeConfig] = None,
    ) -> Dict[str, float]:
        engine = SpeculativeDecodingEngine(self.model, config or SpeculativeConfig())
        for prompt in prompts:
            engine.generate(prompt.unsqueeze(0), max_new_tokens=max_tokens)
        return engine.metrics.report()

    @torch.inference_mode()
    def measure_speedup(
        self, prompt: Tensor, max_tokens: int = 128,
        config: Optional[SpeculativeConfig] = None, n_runs: int = 3,
    ) -> Dict[str, float]:
        """Wall-clock speedup vs vanilla autoregressive (median of n_runs)."""
        prompt = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt

        ar_times = []
        for _ in range(n_runs):
            if prompt.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            if prompt.is_cuda:
                torch.cuda.synchronize()
            ar_times.append(time.perf_counter() - t0)

        engine = SpeculativeDecodingEngine(self.model, config or SpeculativeConfig())
        spec_times = []
        for _ in range(n_runs):
            if prompt.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompt, max_new_tokens=max_tokens)
            if prompt.is_cuda:
                torch.cuda.synchronize()
            spec_times.append(time.perf_counter() - t0)

        ar_med = sorted(ar_times)[len(ar_times) // 2]
        spec_med = sorted(spec_times)[len(spec_times) // 2]
        return {
            "autoregressive_ms": ar_med * 1000,
            "speculative_ms": spec_med * 1000,
            "speedup": ar_med / max(spec_med, 1e-6),
            "acceptance_rate": engine.metrics.acceptance_rate,
        }

    @torch.inference_mode()
    def verify_distribution_preservation(
        self, prompt: Tensor, n_samples: int = 1000,
        position: int = 0, temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Empirically verify speculative decoding preserves target distribution.

        Draws n_samples first-tokens from both vanilla and speculative decoding,
        computes KL divergence and total variation distance.
        Expected: KL < 0.01, TVD < 0.02 for correct implementation.
        """
        prompt = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
        V = self.model.vocab_size

        ar_counts = torch.zeros(V, device=prompt.device)
        for _ in range(n_samples):
            out = self.model.generate(prompt, max_new_tokens=position + 1, temperature=temperature)
            ar_counts[out[0, prompt.shape[1] + position]] += 1

        cfg = SpeculativeConfig(temperature=temperature)
        engine = SpeculativeDecodingEngine(self.model, cfg)
        spec_counts = torch.zeros(V, device=prompt.device)
        for _ in range(n_samples):
            out, _ = engine.generate(prompt, max_new_tokens=position + 1)
            spec_counts[out[0, prompt.shape[1] + position]] += 1

        ar_dist = ar_counts / ar_counts.sum()
        spec_dist = spec_counts / spec_counts.sum()

        mask = (ar_dist > 0) & (spec_dist > 0)
        kl = (spec_dist[mask] * (spec_dist[mask] / ar_dist[mask]).log()).sum()
        tvd = 0.5 * (ar_dist - spec_dist).abs().sum()

        return {"kl_divergence": kl.item(), "total_variation_distance": tvd.item()}
```

---

## 4. Verification: Proof of Correctness

### Theorem

Speculative decoding with α(x) = min(1, p(x)/q(x)) and correction sampling from
max(0, p(x) − q(x))/Z produces tokens distributed **exactly** according to p(x).

### Proof

Consider a single position. Draft proposes x ~ q(x), accepted with probability α(x).

**Case 1 — Accepted** (probability q(x) · α(x)):

```
P(output=x, accepted) = q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))
```

**Case 2 — Rejected, correction y sampled** (probability P(reject) · p'(y)):

```
P(reject) = 1 − Σ_x min(q(x), p(x)) = Σ_x max(0, p(x) − q(x))

p'(y) = max(0, p(y) − q(y)) / Σ_x max(0, p(x) − q(x))

P(output=y, rejected) = max(0, p(y) − q(y))
```

**Combined:**

```
P(output=y) = min(q(y), p(y)) + max(0, p(y) − q(y))

If p(y) ≤ q(y):  p(y) + 0 = p(y)  ✓
If p(y) > q(y):  q(y) + (p(y) − q(y)) = p(y)  ✓
```

**∴ P(output=y) = p(y) for all y.** ∎

For K draft tokens, sequential application preserves the joint distribution by the
chain rule: each accepted position is correct conditioned on the prefix, and the
first rejected position is corrected to match p exactly.

### Empirical Check

`verify_distribution_preservation()` samples from both methods and compares.
For a correct implementation: KL divergence < 0.01, TVD < 0.02 (with n ≥ 1000 samples).

---

## 5. Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Acceptance rate (MTP-1) | >60% | `SpeculativeMetrics.acceptance_rate` |
| Wall-clock speedup | 1.4-1.8× | `SpeculativeBenchmark.measure_speedup` |
| Distribution KL | <0.01 | `verify_distribution_preservation` |
| Memory overhead | <5% | Peak memory delta vs vanilla |

### Speedup Table (K=1, c=1/16)

| Acceptance Rate | Speedup |
|:-:|:-:|
| 50% | 1.41× |
| 60% | 1.51× |
| 65% | 1.55× |
| 70% | 1.60× |
| 80% | 1.69× |

### When Speculation Hurts

- **Low acceptance (<30%)**: Draft compute wasted. Adaptive K mitigates this.
- **Very short sequences (<10 tokens)**: Prefill dominates, no decode to optimize.
- **High batch + memory pressure**: Draft KV slots compete with batch size.

---

## 6. Integration Guide

```python
from model.model import create_nanoseek
from model.serving.speculative import SpeculativeDecodingEngine, SpeculativeConfig

model = create_nanoseek()
model.eval()

config = SpeculativeConfig(max_draft_tokens=1, temperature=0.0)
engine = SpeculativeDecodingEngine(model, config)
output, metrics = engine.generate(prompt_ids, max_new_tokens=128)

print(f"Acceptance rate: {metrics.acceptance_rate:.2%}")
print(f"Speedup: {metrics.speedup_estimate:.2f}×")
```

### Continuous Batching Integration (Doc 07)

1. Scheduler assigns sequences to speculation batches based on draft length and memory.
2. Draft phase runs MTP modules for all sequences.
3. Verify phase runs main model on the batch (padded for variable draft lengths).
4. Accept/reject per-sequence; KV cache trimmed to remove rejected positions.
5. Scheduler reclaims slots for completed sequences / new arrivals.

Temporary KV overhead per round: B × K × (kv_lora_rank + qk_rope_head_dim) × 2 × L.
NanoSeek-1B, K=1, B=32: 32 × 175 × 2 × 16 ≈ **175 KB** (negligible on H100).

---

## 7. Gotchas & Edge Cases

1. **KV cache rollback**: Must precisely trim rejected positions from every layer's
   cache. Off-by-one → silent correctness bugs in subsequent attention patterns.
2. **Greedy mode**: Accept iff argmax(p) == argmax(q). Binary decision, no stochastic
   acceptance. Greedy acceptance rates may be lower than stochastic for the same model.
3. **FP16 residual underflow**: `max(0, p−q)` can underflow to all-zeros. Always
   compute in FP32 or check for the zero-sum fallback case.
4. **Variable-length batches**: Different sequences accept different counts. Padding
   and masking must be correct or sequences leak tokens into each other's context.
5. **EOS handling**: If an accepted draft is EOS, stop that sequence immediately —
   don't process remaining drafts or sample a bonus token.
6. **First step warm-up**: MTP needs main model hidden states. The first decode
   step must run prefill before speculation can begin (handled by `generate()`).

---

*"Speculative decoding is the rare optimization that gives you a free lunch — faster
generation with mathematically identical output. The catch: the implementation must be
exactly right, or you silently break the distribution guarantee."*

— Principal Engineer's Note, Foundation Models Division, 2026
