"""
FineWeb-style heuristic text quality filters.

Each filter checks a statistical property of the document text.
Documents failing any filter are rejected. Filter thresholds are based on
HuggingFace FineWeb (Penedo et al., 2024) with adjustments for ClimbMix
(already partially curated, so thresholds are slightly relaxed).
"""

import re
from dataclasses import dataclass, field


@dataclass
class FilterThresholds:
    """Configurable thresholds for heuristic filters."""
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_ellipsis_line_fraction: float = 0.30
    max_duplicate_line_fraction: float = 0.30
    max_short_line_fraction: float = 0.67
    min_short_line_chars: int = 30
    max_bullet_ratio: float = 0.90
    max_url_ratio: float = 0.20
    max_digit_ratio: float = 0.40
    min_doc_length: int = 100  # minimum characters
    min_word_count: int = 20   # minimum words


# Precompiled patterns
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
_BULLET_PATTERN = re.compile(r'^\s*[\u2022\u2023\u25E6\u2043\u2219•\-\*]\s', re.MULTILINE)


def _compute_metrics(text: str) -> dict[str, float]:
    """Compute all heuristic metrics for a document."""
    metrics = {}

    # Basic stats
    words = text.split()
    lines = text.split('\n')
    num_chars = len(text)
    num_words = len(words)
    num_lines = len(lines)

    metrics['num_chars'] = num_chars
    metrics['num_words'] = num_words
    metrics['num_lines'] = num_lines

    # Mean word length
    if num_words > 0:
        metrics['mean_word_length'] = sum(len(w) for w in words) / num_words
    else:
        metrics['mean_word_length'] = 0.0

    # Lines ending with ellipsis
    if num_lines > 0:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith('...') or line.rstrip().endswith('\u2026'))
        metrics['ellipsis_line_fraction'] = ellipsis_lines / num_lines
    else:
        metrics['ellipsis_line_fraction'] = 0.0

    # Duplicate line fraction
    if num_lines > 0:
        stripped_lines = [line.strip() for line in lines if line.strip()]
        if stripped_lines:
            unique_lines = set(stripped_lines)
            metrics['duplicate_line_fraction'] = 1.0 - len(unique_lines) / len(stripped_lines)
        else:
            metrics['duplicate_line_fraction'] = 0.0
    else:
        metrics['duplicate_line_fraction'] = 0.0

    # Short line fraction
    if num_lines > 0:
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            short_lines = sum(1 for line in non_empty_lines if len(line.strip()) < 30)
            metrics['short_line_fraction'] = short_lines / len(non_empty_lines)
        else:
            metrics['short_line_fraction'] = 0.0
    else:
        metrics['short_line_fraction'] = 0.0

    # Bullet-to-word ratio
    bullet_matches = _BULLET_PATTERN.findall(text)
    metrics['bullet_ratio'] = len(bullet_matches) / max(num_words, 1)

    # URL character ratio
    url_chars = sum(len(m) for m in _URL_PATTERN.findall(text))
    metrics['url_ratio'] = url_chars / max(num_chars, 1)

    # Digit ratio
    digit_chars = sum(1 for c in text if c.isdigit())
    metrics['digit_ratio'] = digit_chars / max(num_chars, 1)

    return metrics


def filter_document(text: str, thresholds: FilterThresholds | None = None) -> tuple[bool, dict[str, float]]:
    """
    Apply heuristic quality filters to a document.

    Returns:
        (keep, metrics): keep=True if document passes all filters,
                         metrics dict with computed values for logging.
    """
    if thresholds is None:
        thresholds = FilterThresholds()

    metrics = _compute_metrics(text)

    # Track which filters triggered rejection
    reject_reasons = []

    # Minimum length checks
    if metrics['num_chars'] < thresholds.min_doc_length:
        reject_reasons.append('too_short_chars')
    if metrics['num_words'] < thresholds.min_word_count:
        reject_reasons.append('too_few_words')

    # Mean word length (removes garbled text)
    mwl = metrics['mean_word_length']
    if mwl < thresholds.min_mean_word_length or mwl > thresholds.max_mean_word_length:
        reject_reasons.append('mean_word_length')

    # Ellipsis lines (removes truncated content)
    if metrics['ellipsis_line_fraction'] > thresholds.max_ellipsis_line_fraction:
        reject_reasons.append('ellipsis_lines')

    # Duplicate lines (removes repetitive content)
    if metrics['duplicate_line_fraction'] > thresholds.max_duplicate_line_fraction:
        reject_reasons.append('duplicate_lines')

    # Short lines (removes lists/menus/navigation)
    if metrics['short_line_fraction'] > thresholds.max_short_line_fraction:
        reject_reasons.append('short_lines')

    # Bullet ratio (removes bullet-only pages)
    if metrics['bullet_ratio'] > thresholds.max_bullet_ratio:
        reject_reasons.append('bullet_ratio')

    # URL ratio (removes link farms)
    if metrics['url_ratio'] > thresholds.max_url_ratio:
        reject_reasons.append('url_ratio')

    # Digit ratio (removes data dumps)
    if metrics['digit_ratio'] > thresholds.max_digit_ratio:
        reject_reasons.append('digit_ratio')

    metrics['reject_reasons'] = reject_reasons
    keep = len(reject_reasons) == 0
    return keep, metrics
