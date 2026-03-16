"""
Rule-based reward functions for GRPO post-training.

No reward model needed — uses verifiable rewards at 1B scale:
- math_reward: Extract answer from \\boxed{} or final number, compare to ground truth
- code_reward: Execute code in sandbox, check test cases
- format_reward: Check <think>...</think> format compliance

Reference: DeepSeek-R1 (Guo et al., 2025) — rule-based rewards for reasoning RL
"""

import re
import logging
import sys
import os

logger = logging.getLogger(__name__)


def math_reward(response: str, ground_truth: str) -> float:
    """
    Score a math response by comparing extracted answer to ground truth.

    Extraction priority:
    1. \\boxed{answer}
    2. Last number in response
    3. "The answer is X" pattern

    Returns: 1.0 (exact match), 0.5 (close within 1%), 0.0 (wrong)
    """
    extracted = _extract_math_answer(response)
    if extracted is None:
        return 0.0

    gt = _normalize_number(ground_truth)
    pred = _normalize_number(extracted)

    if gt is None or pred is None:
        # String comparison fallback
        return 1.0 if extracted.strip().lower() == ground_truth.strip().lower() else 0.0

    if abs(gt - pred) < 1e-6:
        return 1.0
    if abs(gt) > 1e-6 and abs(gt - pred) / abs(gt) < 0.01:
        return 0.5
    return 0.0


def _extract_math_answer(text: str) -> str | None:
    """Extract math answer from response text."""
    # Priority 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1]

    # Priority 2: "The answer is X" pattern
    answer_match = re.search(r'(?:the answer is|answer:)\s*([^\n.]+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Priority 3: Last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def _normalize_number(s: str) -> float | None:
    """Try to parse a string as a number."""
    s = s.strip().replace(',', '').replace('$', '').replace('%', '')
    # Handle fractions like 3/4
    frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        return num / den if den != 0 else None
    try:
        return float(s)
    except ValueError:
        return None


def code_reward(response: str, test_cases: list[dict]) -> float:
    """
    Score a code response by executing in sandbox and checking test cases.

    Reuses nanochat/nanochat/execution.py sandbox for safe code execution.

    Args:
        response: Model response containing code.
        test_cases: List of dicts with 'input' and 'expected_output' keys.

    Returns: 1.0 (all pass), fraction (partial), 0.0 (all fail/error/timeout)
    """
    # Extract code from response
    code = _extract_code(response)
    if not code:
        return 0.0

    try:
        # Import sandbox execution from nanochat
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from nanochat.execution import execute_code
    except ImportError:
        logger.warning("nanochat.execution not available, using basic exec")
        return _basic_code_eval(code, test_cases)

    passed = 0
    for tc in test_cases:
        test_code = code + "\n" + tc.get('test_code', '')
        result = execute_code(test_code, timeout=5)
        if result.get('success', False):
            passed += 1

    return passed / max(len(test_cases), 1)


def _extract_code(text: str) -> str | None:
    """Extract code block from model response."""
    # Try fenced code block
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Try indented code (lines starting with spaces/tabs after "def" or "class")
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r'^(def |class |import |from )', line):
            in_code = True
        if in_code:
            code_lines.append(line)
            if line.strip() == '' and code_lines:
                # End of code block (blank line after code)
                break

    if code_lines:
        return '\n'.join(code_lines).strip()
    return None


def _basic_code_eval(code: str, test_cases: list[dict]) -> float:
    """Fallback code eval without sandbox (less safe, for development only)."""
    passed = 0
    for tc in test_cases:
        try:
            local_ns = {}
            exec(code + "\n" + tc.get('test_code', ''), {}, local_ns)
            passed += 1
        except Exception:
            pass
    return passed / max(len(test_cases), 1)


def format_reward(response: str) -> float:
    """
    Score format compliance of a reasoning response.

    Checks for proper <think>...</think> structure.

    Returns: 0.0-0.5 based on format quality.
    """
    score = 0.0

    # Has think tags
    if '<think>' in response and '</think>' in response:
        score += 0.2

        # Properly closed (think section before answer)
        think_end = response.rfind('</think>')
        if think_end < len(response) - 1:
            # There's content after </think> (the actual answer)
            score += 0.1

        # Reasonable think length (not too short, not too long)
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1)
            think_words = len(think_content.split())
            if 20 <= think_words <= 500:
                score += 0.1

            # Step-by-step indicators
            step_patterns = ['step', 'first', 'then', 'next', 'therefore', 'so', 'because']
            step_count = sum(1 for p in step_patterns if p in think_content.lower())
            if step_count >= 2:
                score += 0.1

    return score


def composite_reward(
    response: str,
    ground_truth: str | None = None,
    test_cases: list[dict] | None = None,
    reward_type: str = 'math',
    task_weight: float = 0.6,
    format_weight: float = 0.3,
    length_weight: float = 0.1,
    max_length: int = 1024,
) -> float:
    """
    Compute composite reward combining task + format + length.

    R = task_weight * task_reward + format_weight * format_reward + length_weight * length_penalty
    """
    # Task reward
    if reward_type == 'math' and ground_truth is not None:
        task_score = math_reward(response, ground_truth)
    elif reward_type == 'code' and test_cases is not None:
        task_score = code_reward(response, test_cases)
    else:
        task_score = 0.0

    # Format reward
    fmt_score = format_reward(response)

    # Length penalty (penalize very long responses)
    response_tokens = len(response.split())
    if response_tokens > max_length:
        length_penalty = -0.5 * (response_tokens - max_length) / max_length
    else:
        length_penalty = 0.0

    total = task_weight * task_score + format_weight * fmt_score + length_weight * length_penalty
    return max(0.0, total)
