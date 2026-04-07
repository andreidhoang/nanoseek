"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class MLAKVCache:
    """
    Pre-allocated KV cache for MLA (Multi-head Latent Attention).

    Nanochat equivalent: KVCache with (B, T, H, D) tensors for FA3.
    NanoSeek adaptation: (B, T, kv_lora_rank) + (B, T, 1, rope_dim) for MLA.

    Pre-allocation avoids O(T²) memory allocations from torch.cat at each decode step — the model's MLA.forward() currently grows
    tensors via concat, but the Engine can bypass this by managing the cache externally.
    """
    def __init__(self, batch_size, seq_len, kv_lora_rank, qk_rope_head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # pre-allocate: 1 big tensor per cache type, indexed by layer
        self.c_kv_cache = torch.zeros(
            num_layers, batch_size, seq_len, kv_lora_rank, device=device, dtype=dtype
        )
        self.k_pe_cache = torch.zeros(
            num_layers, batch_size, seq_len, 1, qk_rope_head_dim, device=device, dtype=dtype
        )

        # Current sequence length per batch element
        # Tracks how many tokens have been written to the cache
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """
        Returns (c_kv, k_pe_cache) views for a specific layer.
        Slice to current sequence length for each batch element.
        These are VIEWS into the pre-allocated tensors, not copies.
        """
        pos = self.get_pos()
        return (
            self.c_kv_cache[layer_idx, :, :pos, :], # [B, pos, kv_lora_rank]
            self.k_pe_cache[layer_idx, :, :pos, :, :] # [B, pos, 1, qk_rope_head_dim]
        )

    def update_layer(self, layer_idx, new_c_kv, new_k_pe):
        """
        Write new cache entries at current position.
        Called after model.forward() to store the newly computed KV for this layer.
   
        Args:
            new_c_kv: [B, num_new_tokens, kv_lora_rank]
            new_k_pe: [B, num_new_tokens, 1, qk_rope_head_dim]
        """
        pos = self.get_pos()
        num_new = new_c_kv.shape[1]
        self.c_kv_cache[layer_idx, :, pos:pos+num_new, :] = new_c_kv
        self.k_pe_cache[layer_idx, :, pos:pos+num_new, :, :] = new_k_pe
        
    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from a batch=1 prefill cache into this multi-sample cache.
   
        This is the key trick for efficient multi-sample generation:
        1. Prefill the prompt ONCE with batch=1
        2. Create a batch=num_samples cache
        3. Copy the prefill results to ALL samples via this method
        4. Each sample then diverges independently during decode
   
        Args:
            other: MLAKVCache with batch_size=1 containing prefill results
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty cache"
        assert self.n_layers == other.n_layers
        assert self.kv_lora_rank == other.kv_lora_rank
        assert self.qk_rope_head_dim == other.qk_rope_head_dim
        assert self.max_seq_len >= other.max_seq_len

        other_pos = other.get_pos()
        # Copy and broadcast: other has batch=1, self has batch=num_samples
        # PyTorch broadcasting handles [n_layers, 1, T, ...] → [n_layers, N, T, ...]
        self.c_kv_cache[:, :, :other_pos, :] = other.c_kv_cache[:, :, :other_pos, :]
        self.k_pe_cache[:, :, :other_pos, :, :] = other.k_pe_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)

    def to_past_key_values(self):
        """
        Convert pre-allocated cache to list-of-tuples format expected by
        NanoSeekModel.forward(past_key_values=...).
   
        Returns:
            List of (c_kv, k_pe) tuples, one per layer.
            c_kv: [B, pos, kv_lora_rank]
            k_pe: [B, pos, 1, qk_rope_head_dim]
        """
        pos = self.get_pos()
        if pos == 0:
            return None
        return [
            (
                self.c_kv_cache[i, :, :pos, :].clone(),
                self.k_pe_cache[i, :, :pos, :, :].clone()
            ) for i in range(self.n_layers)
        ]

    def from_past_key_values(self, past_key_values, num_new_tokens):
        """
        Extract ONLY the new entries from model's returned past_key_values
        and write them into the pre-allocated cache.
   
        The model returns the FULL concatenated history. We only need the
        last num_new_tokens entries (the rest are already in our cache).
   
        Args:
            past_key_values: list of (c_kv, k_pe) from model.forward()
            num_new_tokens: how many new tokens were processed this step
        """
        for i, (c_kv, k_pe) in enumerate(past_key_values):
            # c_kv: [B, total_len, 143] — take last num_new_tokens
            new_c_kv = c_kv[:, -num_new_tokens:, :]
            new_k_pe = k_pe[:, -num_new_tokens:, :, :]
            self.update_layer(i, new_c_kv, new_k_pe)
        self.advance(num_new_tokens)
    

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample next token from logits [B, vocab_size]. Returns [B, 1].

    Identical to nanochat — sampling logic is architecture-independent.
    The only thing that changes between nanochat and nanoseek is HOW
    we get the logits (MHA vs MLA), not how we sample from them.

    Args:
        logits: [B, V] unnormalized scores for each vocab token
        rng: torch.Generator for reproducible sampling
        temperature: scaling factor (0 = greedy argmax)
        top_k: if set, only consider top_k highest-scoring tokens
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------
class RowState:
    """Per-row state tracking during generation.
   
    The tool-use state machine is model-agnostic. It operates on token IDs, not on model internals.
    """
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # Full token sequence for this row
        self.forced_tokens = deque()    # Queue of tokens to inject (from calculator)
        self.in_python_block = False    # Inside <|python_start|>...<|python_end|>?
        self.python_expr_tokens = []    # Accumulating expression tokens
        self.completed = False          # Hit <|assistant_end|> or BOS?

class Engine:
    def __init__(self, model, tokenizer=None):
        """
        Args:
            model: NanoSeekModel instance
            tokenizer: Optional — needed only for tool use (calculator).
            Must have .encode(), .decode(), .encode_special(), .get_bos_token_id() methods.
        """
        self.model = model
        self.tokenizer = tokenizer

    def _get_cache_config(self):
        """Extract MLA cache parameters from the model's config."""
        config = self.model.config
        return (
            config.mla.kv_lora_rank,       # 143
            config.mla.qk_rope_head_dim,   # 32
            config.num_layers,             # 16
            config.max_position_embeddings, # 4096
        )

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
    ):
        """
        Streaming generation: yields (token_column, token_masks) per step.
        Adapted from nanochat's Engine.generate(), using MLAKVCache for
        pre-allocated cache management.

        Key differences from nanochat:
            1. model.forward() returns dict, not raw logits tensor
            2. MLAKVCache manages MLA's compressed (c_kv, k_pe) format
            3. Prefill→expand handled by MLAKVCache.prefill()
            4. Tool use only if tokenizer is provided

        Yields:
            (token_column, token_masks) per step
            token_column: list of int, length num_samples — one token per row
            token_masks: list of int, length num_samples — 1=sampled, 0=forced
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        device = next(self.model.parameters()).device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        # Reproducible sampling
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens for tool use state machine
        # Only resolve if tokenizer is available
        if self.tokenizer is not None:
            get_special = lambda s: self.tokenizer.encode_special(s)
            python_start = get_special("<|python_start|>")
            python_end = get_special("<|python_end|>")
            output_start = get_special("<|output_start|>")
            output_end = get_special("<|output_end|>")
            assistant_end = get_special("<|assistant_end|>")
            bos = self.tokenizer.get_bos_token_id()
        else:
            # No tool use — set sentinel values that will never match
            python_start = python_end = output_start = output_end = -1
            assistant_end = bos = -1

        # ─── CACHE CONFIG ───
        kv_lora_rank, qk_rope_head_dim, num_layers, max_pos = self._get_cache_config()
        prompt_len = len(tokens)
        # Total sequence length the cache must hold: prompt + generated tokens
        max_seq = prompt_len + (max_tokens if max_tokens is not None else max_pos - prompt_len)

        # ─── STEP 1: PREFILL (batch=1) ───
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # [1, S]
        outputs = self.model.forward(ids, use_cache=True)
        logits = outputs["logits"][:, -1, :] # [1, V]
        prefill_pkv = outputs["past_key_values"] # list of 16 x (c_kv, k_pe) tuples

        # Store prefill results in a batch=1 MLAKVCache
        prefill_cache = MLAKVCache(
            batch_size=1, seq_len=max_seq,
            kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
            num_layers=num_layers, device=device, dtype=dtype,
        )
        prefill_cache.from_past_key_values(prefill_pkv, num_new_tokens=prompt_len)

        # ─── STEP 2: EXPAND KV CACHE (1 → num_samples) via MLAKVCache.prefill() ───
        cache = MLAKVCache(
            batch_size=num_samples, seq_len=max_seq,
            kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
            num_layers=num_layers, device=device, dtype=dtype,
        )
        cache.prefill(prefill_cache)
        del prefill_cache  # free the batch=1 cache
        logits = logits.expand(num_samples, -1) # [N, V]

        # ─── STEP 3: INITIALIZE ROW STATES ───
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # ─── STEP 4: DECODE LOOP ───
        num_generated = 0
        while True:
            # Stop: max tokens reached
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k) # [N, 1]
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # Check for completion
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Tool use state machine (same as nanochat)
                if self.tokenizer is not None:
                    if next_token == python_start:
                        state.in_python_block = True
                        state.python_expr_tokens = []
                    elif next_token == python_end and state.in_python_block:
                        state.in_python_block = False
                        if state.python_expr_tokens:
                            expr = self.tokenizer.decode(state.python_expr_tokens)
                            result = use_calculator(expr)
                            if result is not None:
                                result_tokens = self.tokenizer.encode(str(result))
                                state.forced_tokens.append(output_start)
                                state.forced_tokens.extend(result_tokens)
                                state.forced_tokens.append(output_end)
                        state.python_expr_tokens = []
                    elif state.in_python_block:
                        state.python_expr_tokens.append(next_token)
            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Forward pass for next step — pass cache as list-of-tuples
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            outputs = self.model.forward(
                ids,
                use_cache=True,
                past_key_values=cache.to_past_key_values(),
            )
            logits = outputs["logits"][:, -1, :] # [N, V]
            # Extract only the 1 new token from returned full cache, write to pre-allocated
            cache.from_past_key_values(outputs["past_key_values"], num_new_tokens=1)
            

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming wrapper — collects all tokens and returns final sequences.
        Returns:
            results: list of num_samples token sequences (list of lists of ints)
            masks: list of num_samples mask sequences (1=sampled, 0=forced)
   
        Terminal tokens (assistant_end, bos) are NOT included in results.
        """
        if self.tokenizer is not None:
            assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
            bos_id = self.tokenizer.get_bos_token_id()
        else:
            assistant_end = bos_id = -1

        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos_id:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks


# =============================================================================
# TEST
# =============================================================================

def test_engine():
    """
    Smoke test for Engine with MLAKVCache integration.

    Tests:
        1. MLAKVCache: position tracking, shapes, prefill broadcast
        2. Engine single-sample greedy vs model.generate() — must match
        3. Engine multi-sample generation — produces valid tokens
        4. generate_batch wrapper — returns correct shapes

    Uses the muP anchor config (~314M params, the smallest real config).
    """
    try:
        from .config import get_config
        from .model import create_nanoseek
    except ImportError:
        from config import get_config
        from model import create_nanoseek

    print("=" * 60)
    print("NanoSeek Engine — Smoke Test (anchor config)")
    print("=" * 60)

    config = get_config("anchor")
    model = create_nanoseek(config)
    model.eval()

    device = torch.device("cpu")
    V = config.vocab_size  # 65536
    prompt_len = 8
    max_new = 4
    prompt_tokens = torch.randint(0, V, (prompt_len,)).tolist()

    # ---- Test 1: MLAKVCache basics ----
    print("\n  Test 1: MLAKVCache basics")
    cache = MLAKVCache(
        batch_size=2, seq_len=32,
        kv_lora_rank=config.mla.kv_lora_rank,
        qk_rope_head_dim=config.mla.qk_rope_head_dim,
        num_layers=config.num_layers,
        device=device, dtype=torch.float32,
    )
    assert cache.get_pos() == 0, "Fresh cache should be at position 0"
    assert cache.c_kv_cache.shape == (config.num_layers, 2, 32, config.mla.kv_lora_rank)
    assert cache.k_pe_cache.shape == (config.num_layers, 2, 32, 1, config.mla.qk_rope_head_dim)
    assert cache.to_past_key_values() is None, "Empty cache should return None"

    # Write some data and check position tracking
    fake_c_kv = torch.randn(2, 4, config.mla.kv_lora_rank)
    fake_k_pe = torch.randn(2, 4, 1, config.mla.qk_rope_head_dim)
    cache.update_layer(0, fake_c_kv, fake_k_pe)
    cache.advance(4)
    assert cache.get_pos() == 4, f"Position should be 4, got {cache.get_pos()}"
    layer_c, layer_k = cache.get_layer_cache(0)
    assert layer_c.shape == (2, 4, config.mla.kv_lora_rank)
    assert layer_k.shape == (2, 4, 1, config.mla.qk_rope_head_dim)

    # Test prefill broadcast (batch=1 → batch=3)
    src_cache = MLAKVCache(
        batch_size=1, seq_len=32,
        kv_lora_rank=config.mla.kv_lora_rank,
        qk_rope_head_dim=config.mla.qk_rope_head_dim,
        num_layers=config.num_layers,
        device=device, dtype=torch.float32,
    )
    src_cache.update_layer(0, fake_c_kv[:1], fake_k_pe[:1])  # batch=1
    src_cache.advance(4)
    dst_cache = MLAKVCache(
        batch_size=3, seq_len=32,
        kv_lora_rank=config.mla.kv_lora_rank,
        qk_rope_head_dim=config.mla.qk_rope_head_dim,
        num_layers=config.num_layers,
        device=device, dtype=torch.float32,
    )
    dst_cache.prefill(src_cache)
    assert dst_cache.get_pos() == 4
    # All 3 batch elements should have the same data as src batch=0
    dst_c, _ = dst_cache.get_layer_cache(0)
    assert dst_c.shape == (3, 4, config.mla.kv_lora_rank)
    assert torch.allclose(dst_c[0], dst_c[1]) and torch.allclose(dst_c[1], dst_c[2])
    print("    MLAKVCache: position tracking, shapes, prefill broadcast — OK")

    # ---- Test 2: Engine greedy vs model.generate() ----
    print("\n  Test 2: Engine greedy vs model.generate()")
    with torch.no_grad():
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        ref_output = model.generate(prompt_tensor, max_new_tokens=max_new, temperature=0.0)
        ref_generated = ref_output[0, prompt_len:].tolist()  # just the new tokens

    engine = Engine(model)
    engine_tokens = []
    for token_column, token_masks in engine.generate(
        prompt_tokens, num_samples=1, max_tokens=max_new, temperature=0.0,
    ):
        engine_tokens.append(token_column[0])
        assert token_masks[0] == 1, "Greedy tokens should all be sampled (mask=1)"

    match = ref_generated == engine_tokens
    if not match:
        print(f"    Reference: {ref_generated}")
        print(f"    Engine:    {engine_tokens}")
        # Find first mismatch
        for i in range(min(len(ref_generated), len(engine_tokens))):
            if ref_generated[i] != engine_tokens[i]:
                print(f"    First mismatch at position {i}: ref={ref_generated[i]} vs eng={engine_tokens[i]}")
                break
    assert match, "Engine greedy output must match model.generate() greedy output"
    print(f"    Generated {len(engine_tokens)} tokens — MATCH")

    # ---- Test 3: Multi-sample generation ----
    print("\n  Test 3: Multi-sample generation")
    num_samples = 3
    columns = []
    for token_column, token_masks in engine.generate(
        prompt_tokens, num_samples=num_samples, max_tokens=max_new,
        temperature=0.8, top_k=50,
    ):
        assert len(token_column) == num_samples
        assert len(token_masks) == num_samples
        columns.append(token_column)
        for tok in token_column:
            assert 0 <= tok < V, f"Token {tok} out of range [0, {V})"

    print(f"    {num_samples} samples × {len(columns)} steps — all tokens valid")

    # ---- Test 4: generate_batch wrapper ----
    print("\n  Test 4: generate_batch()")
    results, masks = engine.generate_batch(
        prompt_tokens, num_samples=2, max_tokens=max_new, temperature=0.0,
    )
    assert len(results) == 2
    assert len(masks) == 2
    for r in results:
        assert r[:prompt_len] == prompt_tokens, "Results should start with prompt"
    # Greedy with same seed → both samples should be identical
    assert results[0] == results[1], "Greedy samples with same seed should match"
    print(f"    2 samples, each {len(results[0])} tokens — OK")

    print()
    print("=" * 60)
    print("ALL 4 ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_engine()