"""
NanoSeek Tokenizer — GPT-4 style BPE with FIM and RL special tokens.

Design decisions (see RESEARCH_ENGINEER.md Section 15):
  - vocab_size = 32768 (32K) — reduces embedding tax from 24.8% to 14.2% of active
    params at 1B scale, and from 114% to 57% at anchor scale (480h)
  - Split pattern uses \p{N}{1,2} (nanochat-validated) instead of GPT-4's \p{N}{1,3}
    — optimal for ≤32K vocab, avoids wasting token space on rare 3-digit combos
  - byte_fallback=True — no OOV tokens, graceful handling of any byte sequence
  - FIM tokens (prefix/suffix/middle) for 10% PSM training from token 1 (RULE 6)
  - Chat tokens for RL Stage 2 agent tool-use environment

Two backends:
  1. HuggingFace Tokenizer — for BPE training on the data mix
  2. RustBPE + tiktoken — for fast inference and batch encoding

Reference: nanochat/tokenizer.py (Karpathy), adapted for NanoSeek's MoE + FIM + RL needs.
"""

from __future__ import annotations

import os
import copy
import pickle
from functools import lru_cache
from typing import List, Optional, Union

# ============================================================================
# Special tokens
# ============================================================================
# Order matters: indices are assigned sequentially after the BPE vocab.

SPECIAL_TOKENS = [
    # --- Document boundary ---
    "<|bos|>",              # Beginning of sequence (document delimiter)
    "<|eos|>",              # End of sequence (generation stop signal)

    # --- FIM (Fill-in-the-Middle) — RULE 6: must be present from token 1 ---
    "<|fim_prefix|>",       # PSM format: marks the prefix section
    "<|fim_suffix|>",       # PSM format: marks the suffix section
    "<|fim_middle|>",       # PSM format: marks the middle (model fills this)

    # --- Chat/RL (Stage 2 agent environment + Stage 3 DPO alignment) ---
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",     # Agent RL: tool invocation
    "<|python_end|>",
    "<|output_start|>",     # Agent RL: tool output (not supervised)
    "<|output_end|>",

    # --- Padding (for batch collation) ---
    "<|pad|>",
]

# GPT-4 split pattern with \p{N}{1,2} optimization for small vocab.
# nanochat validated this: 2-digit grouping is optimal for 32K vocab.
# 1-digit slightly worse, 3-digit worse still (wastes token space on rare combos).
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Default vocab size — must match config.py
VOCAB_SIZE = 32768


# ============================================================================
# HuggingFace Tokenizer (for training)
# ============================================================================
class HuggingFaceTokenizer:
    """BPE tokenizer wrapper using HuggingFace tokenizers library.

    Used for training the tokenizer on the data mix. For fast inference,
    use NanoSeekTokenizer (RustBPE + tiktoken backend).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> HuggingFaceTokenizer:
        from tokenizers import Tokenizer as HFTokenizer
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator,
        vocab_size: int = VOCAB_SIZE,
    ) -> HuggingFaceTokenizer:
        """Train a BPE tokenizer from an iterator of text strings.

        Args:
            text_iterator: Iterator yielding text strings (one document per yield).
            vocab_size: Total vocab including special tokens.
        """
        from tokenizers import Tokenizer as HFTokenizer, pre_tokenizers, decoders, Regex
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))
        tokenizer.normalizer = None

        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode_special(self, text: str) -> int:
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self) -> int:
        bos = self.encode_special("<|bos|>")
        assert bos is not None, "BOS token not found"
        return bos

    def get_eos_token_id(self) -> int:
        eos = self.encode_special("<|eos|>")
        assert eos is not None, "EOS token not found"
        return eos

    def encode(
        self,
        text: Union[str, List[str]],
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._encode_one(text, prepend, append)
        return [self._encode_one(t, prepend, append) for t in text]

    def _encode_one(
        self,
        text: str,
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
    ) -> List[int]:
        ids = []
        if prepend is not None:
            pid = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(pid)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            aid = append if isinstance(append, int) else self.encode_special(append)
            ids.append(aid)
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# ============================================================================
# RustBPE + tiktoken Tokenizer (for fast inference)
# ============================================================================
class RustBPETokenizer:
    """Fast BPE tokenizer using rustbpe for training, tiktoken for inference.

    This is the primary tokenizer backend for NanoSeek training and inference.
    """

    def __init__(self, enc, bos_token: str = "<|bos|>"):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator,
        vocab_size: int = VOCAB_SIZE,
    ) -> RustBPETokenizer:
        """Train tokenizer from text iterator using rustbpe, wrap with tiktoken."""
        import rustbpe
        import tiktoken

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be >= 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )

        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }

        enc = tiktoken.Encoding(
            name="nanoseek",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> RustBPETokenizer:
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def get_eos_token_id(self) -> int:
        return self.encode_special("<|eos|>")

    def get_fim_tokens(self) -> dict:
        """Return FIM special token IDs for dataset.py PSM formatting."""
        return {
            "prefix": self.encode_special("<|fim_prefix|>"),
            "suffix": self.encode_special("<|fim_suffix|>"),
            "middle": self.encode_special("<|fim_middle|>"),
        }

    def get_pad_token_id(self) -> int:
        return self.encode_special("<|pad|>")

    def encode(
        self,
        text: Union[str, List[str]],
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
        num_threads: int = 8,
    ) -> Union[List[int], List[List[int]]]:
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            if append is not None:
                for row in ids:
                    row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer to {pickle_path}")

    def render_conversation(
        self,
        conversation: dict,
        max_tokens: int = 4096,
    ) -> tuple:
        """Tokenize a chat conversation for RL fine-tuning.

        Returns:
            ids:  list[int] — token IDs
            mask: list[int] — 1 for assistant tokens (supervised), 0 otherwise
        """
        ids, mask = [], []

        def add(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge system message into first user message if present
        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            messages = copy.deepcopy(messages)
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        # Fetch special token IDs
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        asst_start = self.encode_special("<|assistant_start|>")
        asst_end = self.encode_special("<|assistant_end|>")
        py_start = self.encode_special("<|python_start|>")
        py_end = self.encode_special("<|python_end|>")
        out_start = self.encode_special("<|output_start|>")
        out_end = self.encode_special("<|output_end|>")

        add(bos, 0)
        for i, msg in enumerate(messages):
            content = msg["content"]
            if msg["role"] == "user":
                add(user_start, 0)
                add(self.encode(content), 0)
                add(user_end, 0)
            elif msg["role"] == "assistant":
                add(asst_start, 0)
                if isinstance(content, str):
                    add(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add(value_ids, 1)
                        elif part["type"] == "python":
                            add(py_start, 1)
                            add(value_ids, 1)
                            add(py_end, 1)
                        elif part["type"] == "python_output":
                            # Tool output is NOT supervised
                            add(out_start, 0)
                            add(value_ids, 0)
                            add(out_end, 0)
                add(asst_end, 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation: dict) -> List[int]:
        """Render conversation priming assistant for completion (RL inference)."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids


# ============================================================================
# Convenience functions
# ============================================================================

def get_tokenizer(tokenizer_dir: Optional[str] = None) -> RustBPETokenizer:
    """Load the NanoSeek tokenizer from a directory.

    Args:
        tokenizer_dir: Path to tokenizer directory. If None, uses default location.
    """
    if tokenizer_dir is None:
        # Default: look relative to this file's parent
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tokenizer_dir = os.path.join(base, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)
