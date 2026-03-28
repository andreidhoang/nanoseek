"""Shared utilities for NanoSeek benchmark scripts.

Provides: SHAPE_CONFIGS (production tensor shapes), Timer, model creation,
MFU calculation, and CLI helpers.
"""

import argparse
import time
from contextlib import nullcontext
from statistics import mean, stdev

import torch

from nanoseek.nanoseek.config import (
    get_nanoseek_ablation_config, get_nanoseek_config,
    NanoSeekConfig, MLAConfig, MoEConfig, MTPConfig,
)
from nanoseek.nanoseek.model import NanoSeekModel
from nanoseek.nanoseek.common import get_peak_flops


# ─── Production Tensor Shapes ──────────────────────────────────────────────

SHAPE_CONFIGS = {
    "ablation": {
        "D": 1280, "n_layers": 16, "n_heads": 10, "vocab": 32768,
        "E": 64, "K": 8, "moe_inter": 480, "shared_inter": 960,
        "n_group": 8, "topk_group": 4,
        "q_lora_rank": 275, "kv_lora_rank": 90,
        "qk_nope": 128, "qk_rope": 64, "v_head_dim": 128,
        "B": 4, "S": 512,
    },
    "1b": {
        "D": 2048, "n_layers": 16, "n_heads": 16, "vocab": 32768,
        "E": 64, "K": 8, "moe_inter": 768, "shared_inter": 1536,
        "n_group": 8, "topk_group": 4,
        "q_lora_rank": 440, "kv_lora_rank": 143,
        "qk_nope": 128, "qk_rope": 64, "v_head_dim": 128,
        "B": 4, "S": 512,
    },
}

for _s in SHAPE_CONFIGS.values():
    _s["N"] = _s["B"] * _s["S"]
    _s["NK"] = _s["N"] * _s["K"]
    _s["tokens_per_expert"] = _s["NK"] // _s["E"]
    _s["qk_head_dim"] = _s["qk_nope"] + _s["qk_rope"]


# ─── Tiny Model Config (smoke tests only) ──────────────────────────────────

def get_bench_config() -> NanoSeekConfig:
    """~1M param model preserving full architecture. For harness smoke tests."""
    return NanoSeekConfig(
        vocab_size=512, hidden_size=128, num_layers=4, num_heads=2,
        intermediate_size=256, sequence_length=256, max_position_embeddings=512,
        gradient_checkpointing=False,
        mla=MLAConfig(q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=32,
                      qk_rope_head_dim=16, v_head_dim=32, rope_theta=10000.0,
                      original_max_position_embeddings=512),
        moe=MoEConfig(n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
                      moe_intermediate_size=48, shared_inter_dim=96,
                      n_group=2, topk_group=1, first_k_dense_replace=1),
        mtp=MTPConfig(num_mtp_modules=1, mtp_num_heads=2),
    )


# ─── Model Creation ────────────────────────────────────────────────────────

def create_model(scale: str, device: torch.device, use_compile: bool = False):
    """Create NanoSeekModel. scale: 'bench' | 'ablation' | '1b'."""
    if scale == "bench":
        config = get_bench_config()
    elif scale == "ablation":
        config = get_nanoseek_ablation_config()
    else:
        config = get_nanoseek_config()

    with torch.device("meta"):
        model = NanoSeekModel(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    config.gradient_checkpointing = False

    param_counts = model.num_parameters()
    if use_compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
    return model, config, param_counts


def create_input(config, device: torch.device, batch_size: int, seq_len: int = None):
    if seq_len is None:
        seq_len = min(config.sequence_length, 512)
    return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ─── Timer ──────────────────────────────────────────────────────────────────

class Timer:
    """CUDA event timer (GPU) or perf_counter timer (CPU)."""
    def __init__(self, device: torch.device):
        self.is_cuda = device.type == "cuda"
        if self.is_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = self._t1 = 0.0

    def start(self):
        if self.is_cuda: self._start.record()
        else: self._t0 = time.perf_counter()

    def stop(self):
        if self.is_cuda: self._end.record()
        else: self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.is_cuda:
            torch.cuda.synchronize()
            return self._start.elapsed_time(self._end)
        return (self._t1 - self._t0) * 1000


# ─── Warmup / Autocast / MFU ───────────────────────────────────────────────

def run_warmup(model, x, device: torch.device, num_warmup: int = 5):
    is_cuda = device.type == "cuda"
    for i in range(num_warmup):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
            outputs = model(x, labels=x, mtp_lambda=0.3)
        outputs["loss"].backward()
        model.zero_grad(set_to_none=True)
        if i == 0 and is_cuda:
            torch.cuda.synchronize()


def get_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def compute_mfu(n_active: int, batch_tokens: int, step_ms: float, device_name: str) -> dict:
    flops_per_token = 6 * n_active
    flops_per_step = flops_per_token * batch_tokens
    step_sec = step_ms / 1000.0
    flops_per_sec = flops_per_step / step_sec
    gpu_peak = get_peak_flops(device_name)
    return {
        "mfu": flops_per_sec / gpu_peak if gpu_peak < float('inf') else 0.0,
        "achieved_tflops": flops_per_sec / 1e12,
        "tokens_per_sec": batch_tokens / step_sec,
    }


# ─── CLI / Reporting ────────────────────────────────────────────────────────

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--scale", default="bench", choices=["bench", "ablation", "1b"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser


def stats(times: list) -> dict:
    return {
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }
