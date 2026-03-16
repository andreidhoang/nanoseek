"""
Domain classification and mixture control.

Classifies documents into 5 domain buckets using keyword heuristics,
then applies reservoir sampling to hit target domain ratios.

Domains: code, math, science, web, other
"""

import re
import random
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Domain detection patterns
_CODE_PATTERNS = re.compile(
    r'(?:'
    r'def\s+\w+\s*\(|'           # Python function
    r'class\s+\w+[\s:(]|'        # Class definition
    r'import\s+\w+|'             # Import statement
    r'from\s+\w+\s+import|'      # From import
    r'function\s+\w+\s*\(|'      # JS function
    r'const\s+\w+\s*=|'          # JS const
    r'public\s+(?:static\s+)?(?:void|int|String)|'  # Java
    r'#include\s*<|'             # C/C++
    r'fn\s+\w+\s*\(|'           # Rust
    r'```(?:python|java|cpp|rust|javascript|typescript|go|c\b)'  # Code blocks
    r')',
    re.MULTILINE
)

_MATH_PATTERNS = re.compile(
    r'(?:'
    r'\\(?:frac|sum|int|prod|lim|sqrt|begin\{(?:equation|align|matrix))|'  # LaTeX math
    r'(?:theorem|lemma|proof|corollary|proposition)\s*[\d.:]+|'
    r'(?:equation|formula|derivative|integral|polynomial)|'
    r'\b(?:QED|q\.e\.d\.)\b|'
    r'(?:\d+\s*[+\-*/^]\s*\d+\s*=)|'  # Arithmetic
    r'(?:solve|calculate|compute|evaluate)\s+(?:the|this|for)'
    r')',
    re.IGNORECASE | re.MULTILINE
)

_SCIENCE_PATTERNS = re.compile(
    r'(?:'
    r'\b(?:hypothesis|experiment|methodology|abstract|conclusion|results)\b|'
    r'\b(?:Fig\.|Figure|Table)\s*\d+|'
    r'\b(?:et\s+al\.|doi:|arXiv:|PMID:)|'
    r'\b(?:molecule|protein|genome|neuron|quantum|photon|electron)|'
    r'\b(?:spectroscopy|chromatography|regression|p-value|confidence interval)'
    r')',
    re.IGNORECASE | re.MULTILINE
)


@dataclass
class DomainMixture:
    """Target domain mixture weights."""
    code: float = 0.15
    math: float = 0.10
    science: float = 0.15
    web: float = 0.50
    other: float = 0.10

    def __post_init__(self):
        total = self.code + self.math + self.science + self.web + self.other
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Domain weights must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            'code': self.code, 'math': self.math, 'science': self.science,
            'web': self.web, 'other': self.other
        }


def classify_domain(text: str) -> str:
    """
    Classify a document into one of 5 domain buckets.

    Uses keyword/pattern matching. Priority: code > math > science > web.
    A document is classified by whichever domain has the most pattern matches,
    with a minimum threshold to avoid false positives.
    """
    scores = {}

    code_matches = len(_CODE_PATTERNS.findall(text[:5000]))  # Check first 5K chars
    math_matches = len(_MATH_PATTERNS.findall(text[:5000]))
    science_matches = len(_SCIENCE_PATTERNS.findall(text[:5000]))

    scores['code'] = code_matches
    scores['math'] = math_matches
    scores['science'] = science_matches

    # Minimum threshold: need at least 3 pattern matches to classify as specialized
    min_threshold = 3
    best_domain = max(scores, key=scores.get)

    if scores[best_domain] >= min_threshold:
        return best_domain
    return 'web'  # Default bucket for general text


class MixtureController:
    """
    Controls domain mixture via reservoir sampling.

    Tracks how many documents from each domain have been accepted,
    and probabilistically accepts/rejects to converge on target ratios.
    """

    def __init__(self, target: DomainMixture | None = None, seed: int = 42):
        self.target = target or DomainMixture()
        self.rng = random.Random(seed)
        self.counts: dict[str, int] = {d: 0 for d in self.target.as_dict()}
        self.total_seen: dict[str, int] = {d: 0 for d in self.target.as_dict()}

    def should_keep(self, domain: str) -> bool:
        """
        Decide whether to keep a document based on domain mixture targets.

        Uses adaptive acceptance probability: if a domain is over-represented,
        reduce its acceptance rate; if under-represented, increase it.
        """
        if domain not in self.counts:
            domain = 'other'

        self.total_seen[domain] += 1
        total_accepted = sum(self.counts.values())

        if total_accepted < 100:
            # Warmup: accept everything to build initial statistics
            self.counts[domain] += 1
            return True

        # Current fraction vs target fraction
        current_fraction = self.counts[domain] / max(total_accepted, 1)
        target_fraction = self.target.as_dict()[domain]

        if target_fraction <= 0:
            return False

        # Acceptance probability: higher when domain is under-represented
        ratio = target_fraction / max(current_fraction, 1e-6)
        # Clamp to [0.1, 1.0] — never fully reject a domain, never accept >100%
        accept_prob = min(1.0, max(0.1, ratio))

        if self.rng.random() < accept_prob:
            self.counts[domain] += 1
            return True
        return False

    def get_stats(self) -> dict:
        """Return current mixture statistics."""
        total = sum(self.counts.values())
        fractions = {d: c / max(total, 1) for d, c in self.counts.items()}
        return {
            'counts': dict(self.counts),
            'total_seen': dict(self.total_seen),
            'total_accepted': total,
            'fractions': fractions,
            'target': self.target.as_dict(),
        }
