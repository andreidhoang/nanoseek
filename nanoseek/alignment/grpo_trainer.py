"""
GRPO (Group Relative Policy Optimization) trainer for NanoSeek.

Implements correct GRPO with:
- Group-relative advantage estimation
- Importance sampling ratios (log-space for stability)
- PPO clipped objective (eps=0.2)
- Unbiased KL estimator (always >= 0, lower variance)
- 4 DeepSeek V3.2 MoE RL stabilization techniques

Reference: DeepSeek-R1 (Guo et al., 2025) — GRPO for reasoning RL
Reference: DeepSeek V3.2 — MoE stabilization during RL
"""

import copy
import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rewards import composite_reward, math_reward, code_reward, format_reward

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Generation
    group_size: int = 4           # G completions per prompt
    max_gen_len: int = 512        # max tokens per completion
    temperature: float = 0.7      # sampling temperature

    # PPO / GRPO
    clip_eps: float = 0.2         # PPO clipping epsilon
    kl_beta: float = 0.01         # KL penalty coefficient
    off_policy_delta: float = 0.3 # off-policy masking threshold

    # Optimization
    lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 20
    total_steps: int = 1000
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # MoE stabilization
    freeze_router: bool = True    # Keep Routing (technique 3)

    # Reward
    reward_type: str = 'math'     # 'math' or 'code'
    task_weight: float = 0.6
    format_weight: float = 0.3
    length_weight: float = 0.1

    # Eval
    eval_every: int = 100
    save_dir: str = "checkpoints/nanoseek_grpo"

    # Batch
    prompts_per_step: int = 4     # B prompts per training step


def _freeze_router_params(model: nn.Module):
    """Technique 3 — Keep Routing: freeze all Gate parameters during RL."""
    frozen = 0
    for name, param in model.named_parameters():
        if 'gate' in name.lower() or 'router' in name.lower():
            param.requires_grad = False
            frozen += 1
    logger.info(f"Froze {frozen} router parameters (Keep Routing)")


def _unbiased_kl(log_pi: torch.Tensor, log_pi_ref: torch.Tensor) -> torch.Tensor:
    """
    Technique 1 — Unbiased KL estimator.

    Standard KL: E[log(pi/pi_ref)] is biased with finite samples.
    Unbiased form: E[pi/pi_ref - 1 - log(pi/pi_ref)] >= 0 always, lower variance.

    Args:
        log_pi: [B, T] log-probs under current policy
        log_pi_ref: [B, T] log-probs under reference policy

    Returns:
        Scalar KL estimate.
    """
    ratio = (log_pi - log_pi_ref).exp()  # pi / pi_ref
    kl = ratio - 1.0 - (log_pi - log_pi_ref)  # always >= 0
    return kl.mean()


@torch.no_grad()
def _generate_completions(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_gen_len: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate G completions per prompt.

    Args:
        model: policy model
        prompt_ids: [B, S] prompt token IDs
        group_size: number of completions per prompt
        max_gen_len: maximum generation length
        temperature: sampling temperature
        device: compute device

    Returns:
        completions: [B * G, S + max_gen_len] padded sequences
    """
    model.eval()
    B, S = prompt_ids.shape

    # Expand prompts: each prompt repeated G times
    # [B, S] -> [B*G, S]
    expanded = prompt_ids.unsqueeze(1).expand(B, group_size, S).reshape(B * group_size, S)

    all_tokens = expanded.clone()
    for _ in range(max_gen_len):
        outputs = model(all_tokens)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B*G, 1]
        all_tokens = torch.cat([all_tokens, next_token], dim=1)

    model.train()
    return all_tokens  # [B*G, S + max_gen_len]


def _compute_log_probs(
    model: nn.Module,
    sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute per-token log-probs for completion tokens.

    Args:
        sequences: [N, T] full sequences (prompt + completion)
        prompt_len: length of prompt prefix

    Returns:
        log_probs: [N, T - prompt_len] log-probs of completion tokens
    """
    outputs = model(sequences[:, :-1])
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs

    # Only take completion positions
    completion_logits = logits[:, prompt_len - 1:, :]  # [N, gen_len, V]
    completion_targets = sequences[:, prompt_len:]      # [N, gen_len]

    log_probs = F.log_softmax(completion_logits, dim=-1)
    # Gather log-probs of actual tokens
    token_log_probs = log_probs.gather(
        dim=-1, index=completion_targets.unsqueeze(-1)
    ).squeeze(-1)  # [N, gen_len]

    return token_log_probs


def _compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Compute group-relative advantages.

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    Args:
        rewards: [B * G] reward for each completion
        group_size: G

    Returns:
        advantages: [B * G] normalized advantages
    """
    B = rewards.shape[0] // group_size
    grouped = rewards.view(B, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + 1e-8)
    return advantages.view(-1)


class GRPOTrainer:
    """
    Full GRPO trainer with MoE RL stabilization.

    Implements per-step:
    1. Sample B prompts
    2. Generate G completions per prompt
    3. Score with reward functions
    4. Compute group-relative advantages
    5. Compute importance sampling ratios (log-space)
    6. PPO clipped objective
    7. Unbiased KL penalty
    8. Off-policy masking
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        config: GRPOConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or GRPOConfig()

        # Reference model (frozen copy for KL)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        logger.info("Created frozen reference model for KL computation")

        # Technique 3 — Keep Routing
        if self.config.freeze_router:
            _freeze_router_params(self.model)

        # Optimizer (only trainable params)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine LR schedule
        total = self.config.total_steps
        warmup = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.step = 0

    def _decode_sequence(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to string, handling edge cases."""
        ids = token_ids.tolist()
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(ids)
        # Fallback: join individual token strings
        tokens = [self.tokenizer.decode([t]) if hasattr(self.tokenizer, 'decode')
                  else str(t) for t in ids]
        return ''.join(tokens)

    def _score_completions(
        self,
        sequences: torch.Tensor,
        prompt_len: int,
        ground_truths: list[str],
        test_cases_list: list[list[dict]] | None = None,
    ) -> torch.Tensor:
        """
        Score completions using reward functions.

        Args:
            sequences: [B*G, T] full sequences
            prompt_len: length of prompt
            ground_truths: [B] ground truth answers (repeated per group)
            test_cases_list: [B] test cases for code tasks

        Returns:
            rewards: [B*G] scalar rewards
        """
        cfg = self.config
        B_times_G = sequences.shape[0]
        G = cfg.group_size
        B = B_times_G // G
        rewards = []

        for i in range(B_times_G):
            completion_ids = sequences[i, prompt_len:]
            response = self._decode_sequence(completion_ids)
            gt_idx = i // G

            if cfg.reward_type == 'math':
                r = composite_reward(
                    response, ground_truth=ground_truths[gt_idx],
                    reward_type='math',
                    task_weight=cfg.task_weight,
                    format_weight=cfg.format_weight,
                    length_weight=cfg.length_weight,
                )
            elif cfg.reward_type == 'code' and test_cases_list is not None:
                r = composite_reward(
                    response, test_cases=test_cases_list[gt_idx],
                    reward_type='code',
                    task_weight=cfg.task_weight,
                    format_weight=cfg.format_weight,
                    length_weight=cfg.length_weight,
                )
            else:
                r = format_reward(response)

            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        ground_truths: list[str],
        test_cases_list: list[list[dict]] | None = None,
    ) -> dict[str, float]:
        """
        Single GRPO training step.

        Args:
            prompt_ids: [B, S] tokenized prompts
            ground_truths: [B] ground truth answers
            test_cases_list: [B] test cases (for code reward)

        Returns:
            Dict with training metrics.
        """
        cfg = self.config
        B, S = prompt_ids.shape
        G = cfg.group_size

        # Step 1-2: Generate G completions per prompt
        sequences = _generate_completions(
            self.model, prompt_ids, G, cfg.max_gen_len,
            cfg.temperature, self.device,
        )  # [B*G, S + max_gen_len]

        # Step 3: Score all completions
        rewards = self._score_completions(
            sequences, S, ground_truths, test_cases_list,
        )

        # Step 4: Group-relative advantages
        advantages = _compute_group_advantages(rewards, G)

        # Step 5: Compute log-probs under old policy (before gradient update)
        with torch.no_grad():
            old_log_probs = _compute_log_probs(self.model, sequences, S)
            ref_log_probs = _compute_log_probs(self.ref_model, sequences, S)

        # Step 6: Forward pass for current policy log-probs (with gradients)
        self.model.train()
        cur_log_probs = _compute_log_probs(self.model, sequences, S)

        # Per-sequence log-prob (sum over tokens)
        cur_seq_lp = cur_log_probs.sum(dim=-1)   # [B*G]
        old_seq_lp = old_log_probs.sum(dim=-1)    # [B*G]
        ref_seq_lp = ref_log_probs.sum(dim=-1)    # [B*G]

        # Importance sampling ratio (log-space for stability)
        log_ratio = cur_seq_lp - old_seq_lp
        ratio = log_ratio.exp()  # rho = pi_theta / pi_old

        # Technique 2 — Off-policy masking: skip sequences where |rho - 1| > delta
        on_policy_mask = (ratio - 1.0).abs() <= cfg.off_policy_delta
        n_on_policy = on_policy_mask.sum().item()

        if n_on_policy == 0:
            # All sequences are off-policy, skip this step
            logger.warning("All sequences off-policy, skipping gradient update")
            return {
                'reward_mean': rewards.mean().item(),
                'reward_std': rewards.std().item(),
                'on_policy_frac': 0.0,
                'loss': 0.0,
                'kl': 0.0,
                'step': self.step,
            }

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # Apply off-policy mask
        policy_loss = (policy_loss * on_policy_mask.float()).sum() / n_on_policy

        # Technique 1 — Unbiased KL penalty
        kl_loss = _unbiased_kl(cur_seq_lp, ref_seq_lp)

        # Total loss
        total_loss = policy_loss + cfg.kl_beta * kl_loss

        # Backward + optimize
        total_loss.backward()

        if (self.step + 1) % cfg.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.step += 1

        metrics = {
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'on_policy_frac': n_on_policy / (B * G),
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.step,
        }
        return metrics

    def train(
        self,
        prompt_dataset,
        ground_truths: list[str],
        test_cases_list: list[list[dict]] | None = None,
    ):
        """
        Full GRPO training loop.

        Args:
            prompt_dataset: iterable yielding [B, S] prompt batches
            ground_truths: all ground truth answers
            test_cases_list: all test cases (for code reward)
        """
        import os
        cfg = self.config
        logger.info(
            f"Starting GRPO: {cfg.total_steps} steps, "
            f"G={cfg.group_size}, clip_eps={cfg.clip_eps}, "
            f"kl_beta={cfg.kl_beta}, freeze_router={cfg.freeze_router}"
        )

        for step_idx, (prompt_batch, gt_batch) in enumerate(prompt_dataset):
            if step_idx >= cfg.total_steps:
                break

            prompt_batch = prompt_batch.to(self.device)
            tc_batch = None
            if test_cases_list is not None:
                # Get corresponding test cases for this batch
                start = step_idx * cfg.prompts_per_step
                end = start + cfg.prompts_per_step
                tc_batch = test_cases_list[start:end]

            metrics = self.train_step(prompt_batch, gt_batch, tc_batch)

            if step_idx % 10 == 0:
                logger.info(
                    f"GRPO step {metrics['step']}/{cfg.total_steps} | "
                    f"reward: {metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f} | "
                    f"loss: {metrics['total_loss']:.4f} | "
                    f"KL: {metrics['kl_loss']:.4f} | "
                    f"on-policy: {metrics['on_policy_frac']:.1%}"
                )

            # Eval + save
            if step_idx > 0 and step_idx % cfg.eval_every == 0:
                logger.info(f"Checkpoint at step {step_idx}")
                os.makedirs(cfg.save_dir, exist_ok=True)
                save_path = os.path.join(cfg.save_dir, f"grpo_step_{step_idx}.pt")
                torch.save(self.model.state_dict(), save_path)

        # Final save
        os.makedirs(cfg.save_dir, exist_ok=True)
        final_path = os.path.join(cfg.save_dir, "grpo_final.pt")
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"GRPO training complete. Final checkpoint: {final_path}")
