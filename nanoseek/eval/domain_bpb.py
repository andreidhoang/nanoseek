"""
Per-domain bits-per-byte (BPB) evaluation.

Measures model perplexity on held-out documents from 5 domains:
code, math, science, web, books. This reveals whether data curation
or training dynamics are causing domain-specific degradation.

Reference: Llama 3 (Dubey et al., 2024) — per-domain loss tracking
"""

import math
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Domain evaluation prompts — curated excerpts representative of each domain.
# These are used when no external eval data is available.
DOMAIN_PROMPTS = {
    "code": [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder_traversal(root):\n    if root is None:\n        return []\n    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)",
        "import torch\nimport torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.n_heads = n_heads\n        self.head_dim = d_model // n_heads\n        self.qkv = nn.Linear(d_model, 3 * d_model)\n        self.out = nn.Linear(d_model, d_model)",
    ],
    "math": [
        "Theorem: For any prime p and integer a not divisible by p, a^(p-1) is congruent to 1 modulo p. Proof: Consider the set S = {a, 2a, 3a, ..., (p-1)a}. Each element of S is distinct modulo p, and none is divisible by p. Therefore S is a permutation of {1, 2, ..., p-1} modulo p.",
        "Let f(x) = x^3 - 3x + 1. To find the critical points, we compute f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 3(x-1)(x+1). Setting f'(x) = 0 gives x = 1 and x = -1. Since f''(x) = 6x, we have f''(1) = 6 > 0 (local minimum) and f''(-1) = -6 < 0 (local maximum).",
        "The probability generating function of a Poisson random variable X with parameter lambda is G(s) = E[s^X] = sum_{k=0}^{inf} s^k * e^{-lambda} * lambda^k / k! = e^{-lambda} * sum_{k=0}^{inf} (s*lambda)^k / k! = e^{lambda(s-1)}.",
    ],
    "science": [
        "The CRISPR-Cas9 system provides a mechanism for targeted genome editing by introducing double-strand breaks at specific genomic loci. The Cas9 endonuclease is guided by a single guide RNA (sgRNA) that is complementary to the target DNA sequence. Upon binding, Cas9 cleaves both strands of the DNA, and the cell's repair machinery can be harnessed to introduce desired modifications.",
        "Recent observations from the James Webb Space Telescope have revealed the chemical composition of exoplanet atmospheres with unprecedented precision. Spectroscopic analysis of WASP-39b detected sulfur dioxide (SO2), marking the first detection of photochemistry in an exoplanet atmosphere. The presence of SO2 suggests active atmospheric chemistry driven by stellar radiation.",
        "Transformer-based language models learn representations that encode syntactic structure in their attention patterns. Recent work using probing classifiers has shown that intermediate layers develop representations of dependency parse trees, while earlier layers encode part-of-speech information and later layers capture semantic relationships.",
    ],
    "web": [
        "Getting started with sourdough baking requires patience and practice. First, create your starter by mixing equal parts flour and water, then feeding it daily for about a week. The wild yeast and bacteria will establish a stable culture. When your starter consistently doubles in size within 4-6 hours of feeding, it's ready to use.",
        "The history of the Silk Road spans over 2,000 years, connecting civilizations from China to the Mediterranean. This vast network of trade routes facilitated not only the exchange of goods like silk, spices, and precious metals, but also ideas, religions, and technologies. Buddhism spread eastward along these routes, while papermaking and gunpowder traveled westward.",
        "Tips for maintaining a healthy garden in dry climates include mulching heavily around plants to retain moisture, choosing drought-resistant native species, and watering deeply but infrequently to encourage deep root growth. Drip irrigation systems are more efficient than sprinklers and reduce water waste.",
    ],
    "books": [
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity. The period was so far like the present period that some of its noisiest authorities insisted on its being received, for good or for evil.",
        "Call me Ishmael. Some years ago, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
    ],
}


@torch.no_grad()
def compute_domain_bpb(
    model,
    tokenizer,
    device: torch.device,
    domain_texts: Optional[dict[str, list[str]]] = None,
    max_length: int = 2048,
) -> dict[str, float]:
    """
    Compute bits-per-byte for each domain.

    Args:
        model: NanoSeekModel in eval mode.
        tokenizer: Tokenizer with encode() method.
        device: Compute device.
        domain_texts: Optional dict mapping domain -> list of text samples.
                      Falls back to built-in DOMAIN_PROMPTS.
        max_length: Maximum sequence length for evaluation.

    Returns:
        Dict mapping domain name -> BPB value.
    """
    if domain_texts is None:
        domain_texts = DOMAIN_PROMPTS

    model.eval()
    results = {}

    for domain, texts in domain_texts.items():
        total_loss = 0.0
        total_bytes = 0

        for text in texts:
            # Encode text
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Count raw bytes for BPB denominator
            text_bytes = len(text.encode('utf-8'))

            # Forward pass
            input_ids = torch.tensor([tokens], device=device)
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            outputs = model(inputs)
            # Handle dict-returning model (NanoSeek returns {'logits': ..., ...})
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Cross-entropy loss (sum, not mean — we'll normalize by bytes)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_bytes += text_bytes

        # Convert nats to bits, normalize by bytes
        if total_bytes > 0:
            bpb = total_loss / (total_bytes * math.log(2))
        else:
            bpb = float('inf')

        results[domain] = round(bpb, 4)
        logger.info(f"Domain '{domain}' BPB: {bpb:.4f}")

    # Also compute overall weighted BPB
    total_loss_all = sum(results[d] for d in results)
    results['overall'] = round(total_loss_all / max(len(results), 1), 4)

    return results
