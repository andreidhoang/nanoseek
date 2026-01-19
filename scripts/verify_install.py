#!/usr/bin/env python3
"""
NanoSeek Installation Verification Script

Runs quick checks to verify the installation is correct.

Usage:
    python scripts/verify_install.py
    python scripts/verify_install.py --full  # Run full tests
"""

import argparse
import sys
import time
from typing import Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_status(message: str, status: str):
    """Print a status message with color."""
    if status == "ok":
        symbol = f"{GREEN}[OK]{RESET}"
    elif status == "fail":
        symbol = f"{RED}[FAIL]{RESET}"
    elif status == "warn":
        symbol = f"{YELLOW}[WARN]{RESET}"
    elif status == "info":
        symbol = f"{BLUE}[INFO]{RESET}"
    else:
        symbol = "[??]"

    print(f"  {symbol} {message}")


def print_header(title: str):
    """Print a section header."""
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 50)


def check_python_version() -> bool:
    """Check Python version >= 3.11."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 11:
        print_status(f"Python version: {version_str}", "ok")
        return True
    else:
        print_status(f"Python version: {version_str} (need >= 3.11)", "fail")
        return False


def check_pytorch() -> Tuple[bool, bool]:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch

        version = torch.__version__
        print_status(f"PyTorch version: {version}", "ok")

        # Check CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print_status(f"CUDA available: {cuda_version} ({device_name})", "ok")
            return True, True
        else:
            print_status("CUDA not available (CPU mode)", "warn")
            return True, False

    except ImportError:
        print_status("PyTorch not installed", "fail")
        return False, False


def check_model_import() -> bool:
    """Check if model modules can be imported."""
    try:
        from model.config import get_nanoseek_config
        print_status("model.config imported", "ok")

        from model.model import create_nanoseek
        print_status("model.model imported", "ok")

        return True

    except ImportError as e:
        print_status(f"Import error: {e}", "fail")
        return False


def check_model_creation() -> bool:
    """Check if model can be created."""
    try:
        import torch
        from model.config import get_nanoseek_config
        from model.model import create_nanoseek

        # Use minimal config for quick test
        config = get_nanoseek_config()

        # Override for quick test
        config.num_hidden_layers = 2
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.mla.q_lora_rank = 54
        config.mla.kv_lora_rank = 18
        config.mla.v_head_dim = 32
        config.mla.qk_nope_head_dim = 32
        config.mla.qk_rope_head_dim = 16

        model = create_nanoseek(config)
        num_params = sum(p.numel() for p in model.parameters())

        print_status(f"Model created: {num_params:,} params", "ok")
        return True

    except Exception as e:
        print_status(f"Model creation failed: {e}", "fail")
        return False


def check_forward_pass(use_cuda: bool) -> bool:
    """Check if forward pass works."""
    try:
        import torch
        from model.config import get_nanoseek_config
        from model.model import create_nanoseek

        # Minimal config
        config = get_nanoseek_config()
        config.num_hidden_layers = 2
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.mla.q_lora_rank = 54
        config.mla.kv_lora_rank = 18
        config.mla.v_head_dim = 32
        config.mla.qk_nope_head_dim = 32
        config.mla.qk_rope_head_dim = 16

        device = "cuda" if use_cuda else "cpu"
        model = create_nanoseek(config).to(device)
        model.eval()

        # Test input
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Time forward pass
        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
        elapsed = time.time() - start

        logits = outputs['logits']
        expected_shape = (batch_size, seq_len, config.vocab_size)

        if logits.shape == expected_shape:
            print_status(f"Forward pass: {elapsed*1000:.1f}ms ({device})", "ok")
            return True
        else:
            print_status(f"Output shape mismatch: {logits.shape} vs {expected_shape}", "fail")
            return False

    except Exception as e:
        print_status(f"Forward pass failed: {e}", "fail")
        return False


def check_backward_pass(use_cuda: bool) -> bool:
    """Check if backward pass works."""
    try:
        import torch
        from model.config import get_nanoseek_config
        from model.model import create_nanoseek

        # Minimal config
        config = get_nanoseek_config()
        config.num_hidden_layers = 2
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.mla.q_lora_rank = 54
        config.mla.kv_lora_rank = 18
        config.mla.v_head_dim = 32
        config.mla.qk_nope_head_dim = 32
        config.mla.qk_rope_head_dim = 16

        device = "cuda" if use_cuda else "cpu"
        model = create_nanoseek(config).to(device)
        model.train()

        # Test input
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Forward + backward
        start = time.time()
        outputs = model(input_ids)
        loss = outputs['logits'].mean()
        loss.backward()
        elapsed = time.time() - start

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)

        if has_grads:
            print_status(f"Backward pass: {elapsed*1000:.1f}ms ({device})", "ok")
            return True
        else:
            print_status("No gradients computed", "fail")
            return False

    except Exception as e:
        print_status(f"Backward pass failed: {e}", "fail")
        return False


def run_pytest_tests() -> bool:
    """Run pytest tests."""
    try:
        import subprocess

        print_status("Running pytest tests...", "info")
        result = subprocess.run(
            ["pytest", "tests/test_mla.py", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print_status("pytest tests passed", "ok")
            return True
        else:
            print_status("pytest tests failed", "fail")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print_status("pytest timed out", "fail")
        return False
    except Exception as e:
        print_status(f"pytest failed: {e}", "fail")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify NanoSeek installation"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (slower)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip CUDA tests"
    )

    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 50}{RESET}")
    print(f"{BOLD}NanoSeek Installation Verification{RESET}")
    print(f"{BOLD}{'=' * 50}{RESET}")

    results = {}

    # Check Python
    print_header("Python Environment")
    results["python"] = check_python_version()

    # Check PyTorch
    pytorch_ok, cuda_ok = check_pytorch()
    results["pytorch"] = pytorch_ok

    if args.cpu_only:
        cuda_ok = False

    # Check imports
    print_header("Model Imports")
    results["imports"] = check_model_import()

    # Check model creation
    print_header("Model Creation")
    results["model"] = check_model_creation()

    # Check forward pass
    print_header("Forward Pass")
    results["forward"] = check_forward_pass(cuda_ok)

    # Check backward pass
    print_header("Backward Pass")
    results["backward"] = check_backward_pass(cuda_ok)

    # Optional: Full tests
    if args.full:
        print_header("Full Test Suite")
        results["tests"] = run_pytest_tests()

    # Summary
    print_header("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    if passed == total:
        print(f"\n{GREEN}{BOLD}All checks passed! ({passed}/{total}){RESET}")
        print("\nNanoSeek is ready to use.")
        print("\nNext steps:")
        print("  1. Setup data: python scripts/setup_data.py")
        print("  2. Start training: python scripts/pre-train.py")
        return 0
    else:
        print(f"\n{RED}{BOLD}Some checks failed ({passed}/{total}){RESET}")
        print("\nPlease fix the issues above and try again.")
        print("See README.md for installation instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
