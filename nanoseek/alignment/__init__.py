"""
NanoSeek post-training infrastructure.

Modules:
- rewards: Rule-based reward functions (math, code, format)
- sft_warmup: Cold-start SFT on GSM8K before RL
- grpo_trainer: Full GRPO with importance sampling + PPO clipping + MoE stabilization
- run_grpo: CLI entry point for GRPO training
"""
