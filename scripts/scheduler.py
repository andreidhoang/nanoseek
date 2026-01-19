# Learning Rate Scheduler for NanoSeek Pre-training
#
# This module implements the DeepSeek-style learning rate schedule:
# 1. Linear warmup from 0 to lr_max
# 2. Constant phase at lr_max (70% of training)
# 3. Cosine decay to lr_min (70% to 95%)
# 4. Constant at lr_min (final 5%)
#
# This schedule provides stable training with gradual cooldown.

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional
import math


class DeepSeekLRScheduler(_LRScheduler):
    """
    DeepSeek-style learning rate scheduler.

    Schedule phases:
    1. Warmup (0 to warmup_steps): Linear 0 -> lr_max
    2. Constant (warmup_steps to constant_end): lr_max
    3. Decay (constant_end to decay_end): Cosine lr_max -> lr_min
    4. Final (decay_end to end): lr_min

    This schedule differs from standard cosine by having:
    - Extended constant phase (better gradient signal)
    - Shorter decay (focused fine-tuning phase)
    - Minimum LR floor (prevents underflow)

    Args:
        optimizer: PyTorch optimizer
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        total_steps: Total training steps
        warmup_steps: Steps for linear warmup
        constant_phase_ratio: Fraction of training at constant LR
        cosine_decay_end_ratio: Fraction where decay ends
        last_step: Last step (for resuming)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        total_steps: int,
        warmup_steps: int = 1000,
        constant_phase_ratio: float = 0.70,
        cosine_decay_end_ratio: float = 0.95,
        last_step: int = -1,
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        # Calculate phase boundaries
        self.constant_end = int(total_steps * constant_phase_ratio)
        self.decay_end = int(total_steps * cosine_decay_end_ratio)

        # Store initial LRs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        """Calculate learning rate for current step."""
        step = self.last_epoch + 1  # 0-indexed to 1-indexed

        if step < self.warmup_steps:
            # Phase 1: Linear warmup
            lr = self.lr_max * step / self.warmup_steps
        elif step < self.constant_end:
            # Phase 2: Constant at max
            lr = self.lr_max
        elif step < self.decay_end:
            # Phase 3: Cosine decay
            progress = (step - self.constant_end) / (self.decay_end - self.constant_end)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        else:
            # Phase 4: Constant at min
            lr = self.lr_min

        return [lr for _ in self.base_lrs]

    def get_lr_info(self) -> dict:
        """Get information about current LR schedule state."""
        step = self.last_epoch + 1
        current_lr = self.get_last_lr()[0]

        if step < self.warmup_steps:
            phase = "warmup"
            phase_progress = step / self.warmup_steps
        elif step < self.constant_end:
            phase = "constant"
            phase_progress = (step - self.warmup_steps) / (self.constant_end - self.warmup_steps)
        elif step < self.decay_end:
            phase = "decay"
            phase_progress = (step - self.constant_end) / (self.decay_end - self.constant_end)
        else:
            phase = "final"
            phase_progress = (step - self.decay_end) / (self.total_steps - self.decay_end) if self.total_steps > self.decay_end else 1.0

        return {
            "step": step,
            "lr": current_lr,
            "phase": phase,
            "phase_progress": phase_progress,
            "overall_progress": step / self.total_steps,
        }


class WarmupCosineScheduler(_LRScheduler):
    """
    Simple warmup + cosine decay scheduler.

    Alternative to DeepSeek's multi-phase schedule.
    Uses linear warmup followed by cosine decay to min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        total_steps: int,
        warmup_steps: int = 1000,
        last_step: int = -1,
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            lr = self.lr_max * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]


class MuonScheduler(_LRScheduler):
    """
    Scheduler for Muon optimizer (as used in nanochat).

    Muon uses:
    - No warmup
    - Constant phase followed by decay
    - 20% warmdown period

    This is included for comparison with nanochat training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        total_steps: int,
        warmdown_ratio: float = 0.2,
        last_step: int = -1,
    ):
        self.lr_max = lr_max
        self.total_steps = total_steps
        self.warmdown_start = int(total_steps * (1 - warmdown_ratio))

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmdown_start:
            # Constant phase
            lr = self.lr_max
        else:
            # Cosine warmdown
            progress = (step - self.warmdown_start) / (self.total_steps - self.warmdown_start)
            lr = self.lr_max * 0.5 * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]


class LoadBalanceBiasScheduler:
    """
    Scheduler for MoE load balancing bias update rate (gamma).

    The bias update rate starts at gamma_initial and is frozen
    (set to 0) at gamma_freeze_at fraction of training.

    This prevents the bias from continuing to change late in
    training, which could destabilize convergence.
    """

    def __init__(
        self,
        gamma_initial: float = 0.001,
        gamma_freeze_at: float = 0.95,
        total_steps: int = 21400,
    ):
        self.gamma_initial = gamma_initial
        self.gamma_freeze_at = gamma_freeze_at
        self.freeze_step = int(total_steps * gamma_freeze_at)

    def get_gamma(self, step: int) -> float:
        """Get gamma value for current step."""
        if step >= self.freeze_step:
            return 0.0  # Frozen
        return self.gamma_initial


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with proper parameter groups.

    Separates parameters into:
    - decay_params: Linear layer weights (apply weight decay)
    - no_decay_params: Embeddings, norms, biases (no weight decay)
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if weight decay should apply
        if any(nd in name.lower() for nd in ['bias', 'norm', 'embed']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    lr_max: float,
    lr_min: float,
    total_steps: int,
    warmup_steps: int = 1000,
    **kwargs,
) -> _LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: "deepseek", "cosine", or "muon"
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        total_steps: Total training steps
        warmup_steps: Warmup steps
        **kwargs: Additional scheduler-specific arguments
    """
    if scheduler_type == "deepseek":
        return DeepSeekLRScheduler(
            optimizer,
            lr_max=lr_max,
            lr_min=lr_min,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            constant_phase_ratio=kwargs.get('constant_phase_ratio', 0.70),
            cosine_decay_end_ratio=kwargs.get('cosine_decay_end_ratio', 0.95),
        )
    elif scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            lr_max=lr_max,
            lr_min=lr_min,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
    elif scheduler_type == "muon":
        return MuonScheduler(
            optimizer,
            lr_max=lr_max,
            total_steps=total_steps,
            warmdown_ratio=kwargs.get('warmdown_ratio', 0.2),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test schedulers
    print("Learning Rate Scheduler Test")
    print("=" * 60)

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Test DeepSeek scheduler
    total_steps = 21400
    scheduler = DeepSeekLRScheduler(
        optimizer,
        lr_max=3e-4,
        lr_min=3e-5,
        total_steps=total_steps,
        warmup_steps=1000,
    )

    print("\nDeepSeek Schedule (NanoSeek-1B):")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup: 0-1000")
    print(f"  Constant: 1000-{scheduler.constant_end}")
    print(f"  Decay: {scheduler.constant_end}-{scheduler.decay_end}")
    print(f"  Final: {scheduler.decay_end}-{total_steps}")

    # Sample LR values
    checkpoints = [0, 500, 1000, 5000, 10000, 15000, 18000, 20000, 21400]
    print("\n  LR at checkpoints:")
    for step in checkpoints:
        scheduler.last_epoch = step - 1
        lr = scheduler.get_lr()[0]
        info = scheduler.get_lr_info()
        print(f"    Step {step:5d}: {lr:.2e} ({info['phase']})")

    # Test gamma scheduler
    print("\nLoad Balance Bias Schedule:")
    gamma_scheduler = LoadBalanceBiasScheduler(
        gamma_initial=0.001,
        gamma_freeze_at=0.95,
        total_steps=total_steps,
    )

    for step in [0, 10000, 20000, 20400]:
        gamma = gamma_scheduler.get_gamma(step)
        print(f"  Step {step}: gamma = {gamma}")
