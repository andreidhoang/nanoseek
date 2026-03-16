"""
SimHash-based near-duplicate detection.

Uses 64-bit SimHash fingerprints with Hamming distance thresholding.
Designed for second-pass dedup on already-curated ClimbMix data,
so uses the simpler SimHash approach (vs MinHash/LSH for raw crawl data).

Memory: ~256MB for 32M documents (8 bytes per hash).
"""

import hashlib
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization for SimHash."""
    return re.findall(r'\w+', text.lower())


def compute_simhash(text: str, hashbits: int = 64) -> int:
    """
    Compute 64-bit SimHash fingerprint for a document.

    Uses token-level hashing with weighted accumulation.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0

    v = [0] * hashbits
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hashbits):
            if token_hash & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hashbits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Count differing bits between two hashes."""
    return bin(hash1 ^ hash2).count('1')


class SimHashIndex:
    """
    In-memory SimHash index for near-duplicate detection.

    Uses bit-sampling buckets for fast approximate nearest neighbor lookup,
    avoiding O(n) scan for each query.
    """

    def __init__(self, hamming_threshold: int = 3, hashbits: int = 64):
        self.hamming_threshold = hamming_threshold
        self.hashbits = hashbits
        self.hashes: set[int] = set()
        # For fast lookup: partition hash into chunks, index by chunk value
        # With 64 bits and threshold=3, we need ceil(64/(3+1))=16 bit chunks
        self.num_chunks = max(1, hashbits // (hamming_threshold + 1))
        self.chunk_bits = hashbits // self.num_chunks
        self.buckets: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(self.num_chunks)]

    def _get_chunks(self, fingerprint: int) -> list[int]:
        """Split fingerprint into chunks for bucket indexing."""
        chunks = []
        mask = (1 << self.chunk_bits) - 1
        for i in range(self.num_chunks):
            chunk = (fingerprint >> (i * self.chunk_bits)) & mask
            chunks.append(chunk)
        return chunks

    def is_duplicate(self, fingerprint: int) -> bool:
        """
        Check if a fingerprint is a near-duplicate of any stored hash.

        Uses bucket-based lookup: if two hashes differ in <= threshold bits,
        at least one chunk must be identical (pigeonhole principle).
        """
        if fingerprint in self.hashes:
            return True

        chunks = self._get_chunks(fingerprint)
        candidates = set()
        for i, chunk_val in enumerate(chunks):
            candidates.update(self.buckets[i].get(chunk_val, set()))

        for candidate in candidates:
            if hamming_distance(fingerprint, candidate) <= self.hamming_threshold:
                return True
        return False

    def add(self, fingerprint: int):
        """Add a fingerprint to the index."""
        self.hashes.add(fingerprint)
        chunks = self._get_chunks(fingerprint)
        for i, chunk_val in enumerate(chunks):
            self.buckets[i][chunk_val].add(fingerprint)

    def __len__(self) -> int:
        return len(self.hashes)

    def __contains__(self, fingerprint: int) -> bool:
        return self.is_duplicate(fingerprint)
