"""
BPE Tokenizer for NanoSeek Training.

Borrowed from nanochat/nanochat/tokenizer.py with modifications.

Two tokenizer backends available:
1) RustBPETokenizer: tiktoken-based for fast inference (recommended)
2) HuggingFaceTokenizer: for training new tokenizers from scratch

For pretraining, we provide a GPT-2 fallback that works out of the box.
"""

import os
import copy
from functools import lru_cache

# Special tokens used in nanochat-style training
SPECIAL_TOKENS = [
    "<|bos|>",           # Beginning of sequence - delimits documents
    "<|user_start|>",    # User message start (finetuning)
    "<|user_end|>",      # User message end
    "<|assistant_start|>",  # Assistant message start
    "<|assistant_end|>",    # Assistant message end
    "<|python_start|>",  # Python tool call start
    "<|python_end|>",    # Python tool call end
    "<|output_start|>",  # Tool output start
    "<|output_end|>",    # Tool output end
]

# GPT-4 style split pattern (modified: \p{N}{1,2} instead of {1,3} to save tokens)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# =============================================================================
# HuggingFace Tokenizer (for training new tokenizers)
# =============================================================================

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import pre_tokenizers, decoders, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for BPE training and inference."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        """Init from a HuggingFace pretrained tokenizer (e.g. 'gpt2')."""
        if not HAS_HF_TOKENIZERS:
            raise ImportError("Install tokenizers: pip install tokenizers")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """Init from a local directory containing tokenizer.json."""
        if not HAS_HF_TOKENIZERS:
            raise ImportError("Install tokenizers: pip install tokenizers")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """Train a new BPE tokenizer from an iterator of text."""
        if not HAS_HF_TOKENIZERS:
            raise ImportError("Install tokenizers: pip install tokenizers")

        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))
        tokenizer.normalizer = None

        # GPT-4 style pre-tokenizer
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
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

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        return [w.content for w in special_tokens_map.values()]

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# =============================================================================
# RustBPE + Tiktoken Tokenizer (fast inference)
# =============================================================================

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import rustbpe
    HAS_RUSTBPE = True
except ImportError:
    HAS_RUSTBPE = False


class RustBPETokenizer:
    """
    Fast tokenizer using tiktoken for inference.
    Can be trained with rustbpe or loaded from pretrained.
    """

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """Train a new BPE tokenizer using rustbpe."""
        if not HAS_RUSTBPE:
            raise ImportError("Install rustbpe: pip install rustbpe")
        if not HAS_TIKTOKEN:
            raise ImportError("Install tiktoken: pip install tiktoken")

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # Build tiktoken encoding
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """Load tokenizer from a pickle file."""
        import pickle
        if not HAS_TIKTOKEN:
            raise ImportError("Install tiktoken: pip install tiktoken")

        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """
        Load a pretrained tiktoken encoding (e.g., 'gpt2', 'cl100k_base').

        Available encodings:
        - 'gpt2': GPT-2 tokenizer (50257 tokens)
        - 'r50k_base': GPT-3 tokenizer
        - 'p50k_base': Codex tokenizer
        - 'cl100k_base': GPT-4/ChatGPT tokenizer (100k tokens)
        - 'o200k_base': GPT-4o tokenizer (200k tokens)
        """
        if not HAS_TIKTOKEN:
            raise ImportError("Install tiktoken: pip install tiktoken")

        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken uses "<|endoftext|>" as document delimiter
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text to token IDs.

        Args:
            text: String or list of strings
            prepend: Token ID or special token string to prepend
            append: Token ID or special token string to append
            num_threads: Number of threads for batch encoding

        Returns:
            List of token IDs or list of lists for batch input
        """
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
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a chat conversation for SFT training.

        Returns:
            ids: List of token IDs for the rendered conversation
            mask: List of mask values (1 for assistant tokens to train on, 0 for context)
        """
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Handle system messages by merging with first user message
        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            import copy
            messages = copy.deepcopy(messages)
            assert messages[1]["role"] == "user", "System message must be followed by user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        assert len(messages) >= 1, f"Conversation has less than 1 message"

        # Get special token IDs
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        # Tokenize the conversation
        add_tokens(bos, 0)

        for i, message in enumerate(messages):
            # Validate alternating roles
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, \
                f"Message {i} is from {message['role']} but should be from {expected_role}"

            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages must be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)

            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)

                if isinstance(content, str):
                    # Simple string content
                    add_tokens(self.encode(content), 1)

                elif isinstance(content, list):
                    # Multi-part content with tool calls
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # Tool outputs are not supervised (come from Python at test time)
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")

                add_tokens(assistant_end, 1)

        # Truncate to max_tokens
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """
        Render conversation for RL completion generation.

        Removes the last assistant message and adds assistant_start token
        to prime the model for generation.

        Returns:
            List of token IDs ready for completion
        """
        import copy
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]

        assert messages[-1]["role"] == "assistant", "Last message must be from assistant"
        messages.pop()  # Remove last assistant message

        # Tokenize without the last message
        ids, _ = self.render_conversation(conversation)

        # Add assistant start token to prime for completion
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """
        Visualize tokenization with color coding.

        Green = supervised (mask=1), Red = context (mask=0)
        """
        RED = '\033[91m'
        GREEN = '\033[92m'
        GRAY = '\033[90m'
        RESET = '\033[0m'

        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def get_pad_token_id(self):
        """Get pad token ID (assistant_end is used as pad token)."""
        return self.encode_special("<|assistant_end|>")

    def save(self, tokenizer_dir):
        import pickle
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")


# =============================================================================
# Convenience functions for NanoSeek training
# =============================================================================

_cached_tokenizer = None

def get_tokenizer(tokenizer_type="auto"):
    """
    Get tokenizer for training/inference.

    Args:
        tokenizer_type: One of:
            - "auto": Try custom tokenizer, fallback to GPT-2
            - "gpt2": Use GPT-2 tokenizer (50257 vocab)
            - "gpt4": Use GPT-4 tokenizer (cl100k_base, 100k vocab)
            - "custom": Load from ~/.cache/nanoseek/tokenizer/

    Returns:
        Tokenizer instance with encode/decode methods
    """
    global _cached_tokenizer
    if _cached_tokenizer is not None:
        return _cached_tokenizer

    # Handle imports for both package and direct execution
    try:
        from .utils import get_base_dir
    except ImportError:
        from utils import get_base_dir

    if tokenizer_type == "auto":
        # Try custom tokenizer first, fallback to GPT-2
        try:
            base_dir = get_base_dir()
            tokenizer_dir = os.path.join(base_dir, "tokenizer")
            if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
                _cached_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
                print(f"Loaded custom RustBPE tokenizer from {tokenizer_dir}")
            elif os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
                _cached_tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
                print(f"Loaded custom HuggingFace tokenizer from {tokenizer_dir}")
            else:
                raise FileNotFoundError("No custom tokenizer found")
        except FileNotFoundError:
            print("No custom tokenizer found, using GPT-2 tokenizer")
            _cached_tokenizer = RustBPETokenizer.from_pretrained("gpt2")

    elif tokenizer_type == "gpt2":
        _cached_tokenizer = RustBPETokenizer.from_pretrained("gpt2")

    elif tokenizer_type == "gpt4":
        _cached_tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")

    elif tokenizer_type == "custom":
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
            _cached_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
        else:
            _cached_tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)

    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    return _cached_tokenizer


def reset_tokenizer_cache():
    """Reset the cached tokenizer (useful for testing)."""
    global _cached_tokenizer
    _cached_tokenizer = None


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing tokenizer...")

    # Test GPT-2 tokenizer
    tok = get_tokenizer("gpt2")
    print(f"Vocab size: {tok.get_vocab_size()}")
    print(f"BOS token ID: {tok.get_bos_token_id()}")

    # Test encoding
    text = "Hello, world! This is a test."
    tokens = tok.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tok.decode(tokens)}")

    # Test batch encoding
    texts = ["Hello world", "How are you?", "Testing batch encoding"]
    batch_tokens = tok.encode(texts, num_threads=2)
    print(f"\nBatch encoding:")
    for t, toks in zip(texts, batch_tokens):
        print(f"  {t} -> {toks}")

    # Test with BOS prepending
    tokens_with_bos = tok.encode(text, prepend=tok.get_bos_token_id())
    print(f"\nWith BOS: {tokens_with_bos}")

    print("\nTokenizer test passed!")
