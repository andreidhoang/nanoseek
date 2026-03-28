"""NanoSeek FP8 Training — MoE-aware mixed-precision for DeepSeek V3.2 at nano scale.

Adapted from nanochat/fp8.py with MoE-specific extensions:
  - Selective conversion: skip Gate router (precision-critical for routing)
  - Dimension filter: skip layers where any dim < 128 or not divisible by 16
  - MoE batched dispatch stays BF16 (bmm path can't use _scaled_mm)
  - MLA projections get FP8 (dense path, every token, biggest speedup)
  - Evaluation escape hatch: disable_fp8() context manager for BF16 eval

NanoSeek-specific: what gets FP8 and what doesn't
==================================================
  ✅ FP8:  MLA projections (wq_a, wq_b, wkv_a, wkv_b, wo) — dense, every token
  ✅ FP8:  Shared expert (w_gate, w_up, w_down) — dense, every token
  ✅ FP8:  MTP projection (M_k) — dense, every token
  ✅ FP8:  Routed experts (sequential fallback path only)
  ❌ BF16: Routed experts (batched bmm path — _scaled_mm is 2D only)
  ❌ BF16: Gate router (64 output dim, precision-critical for routing decisions)
  ❌ BF16: Embeddings / LM head (different optimizer group, gradient sensitivity)
  ❌ FP32: RMSNorm, softmax, gate logits, optimizer states (always)

Hardware requirement: H100/H200 (Hopper SM 9.0+) for native FP8 tensor cores.
On Ampere (A100/A6000), FP8 would be software emulation = slower than BF16.
"""

import torch
import torch.nn as nn
from torch import Tensor

# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum absolute value to prevent division by zero in scale computation.
# When all tensor values are zero, amax=0 and we'd divide by zero.
# 1e-12 is safely above FP32 subnormal range (~1.4e-45) and below any
# meaningful activation magnitude.
EPS = 1e-12

# Minimum dimension size for FP8 to be worthwhile.
# Below this, quantize/dequantize overhead dominates the matmul savings.
# Empirically: 128 is the crossover on H100 for tensorwise scaling.
MIN_FP8_DIM = 128

# Hardware alignment requirement for FP8 tensor cores on Hopper.
# The cuBLAS FP8 GEMM kernel processes elements in tiles of 16.
# Dimensions not divisible by 16 would require padding (implicit in cuBLAS
# but wastes tensor core cycles on padding elements).
FP8_ALIGNMENT = 16


# =============================================================================
# QUANTIZATION CORE
# =============================================================================

@torch.no_grad()
def _to_fp8(x: Tensor, fp8_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" = one scalar scale for the entire tensor.
    This maps the tensor's value range [0, amax] into FP8's representable
    range [0, fp8_max], maximizing the use of available precision.

    The math:
        scale = fp8_max / amax(tensor)
        fp8_tensor = clamp(tensor * scale, -fp8_max, fp8_max).to(fp8_dtype)

    Why this works: if your tensor has values in [-5.0, 5.0] and fp8_max=448,
    the scale is 448/5 = 89.6. Now 5.0 maps to 448 (full range) and 0.01
    maps to 0.896 (still representable in E4M3 which has values at 0.875 and 1.0).
    Without scaling, 0.01 would map to the nearest E4M3 value (0.0 or 0.015625)
    — much more quantization error.

    Returns:
        (fp8_data, inverse_scale) — _scaled_mm expects the INVERSE scale
        because it multiplies (not divides) to dequantize during the matmul.
    """
    fp8_max = torch.finfo(fp8_dtype).max  # E4M3: 448.0, E5M2: 57344.0

    # Find the maximum absolute value across the entire tensor.
    # .float() to avoid precision issues in BF16 (7-bit mantissa → amax
    # of nearby values could alias). This is a single reduction → fast.
    amax = x.float().abs().max()

    # Compute scale in float64 for exact division.
    # Without this, torch.compile and eager mode can produce different scales
    # due to FP32 rounding in the division. torchao does the same upcast.
    # clamp(min=EPS) prevents inf scale when tensor is all zeros.
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()

    # Quantize: scale into FP8 range, then saturate (clamp) to prevent
    # overflow. PyTorch's .to(fp8) wraps on overflow by default — clamp
    # ensures we saturate instead (correct behavior for training).
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)

    # _scaled_mm multiplies output by (inv_scale_a * inv_scale_b) to
    # convert FP8 products back to the original value range.
    inv_scale = scale.reciprocal()

    return x_fp8, inv_scale


def _to_col_major(x: Tensor) -> Tensor:
    """Rearrange a 2D tensor to column-major memory layout.

    cuBLAS FP8 kernel requirement:
      - First operand (A): row-major    (strides = [N, 1])
      - Second operand (B): column-major (strides = [1, M])

    The trick: transpose → contiguous (forces physical rewrite in transposed
    order) → transpose back. Result has same logical shape but column-major
    strides.

    Example for [3, 4] tensor:
      Original:   strides (4, 1) — row-major
      After trick: strides (1, 3) — column-major
      Logical shape: still [3, 4], but memory is rearranged.

    This is a copy operation (~microseconds for typical weight sizes).
    Only needed when the tensor isn't already column-major from a transpose.
    """
    return x.t().contiguous().t()


# =============================================================================
# FP8 AUTOGRAD FUNCTION
# =============================================================================

@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd wrapping the three GEMMs of a Linear layer in FP8.

    Why @allow_in_graph:
      torch.compile's dynamo tracer would normally try to decompose this
      Function into individual ops. @allow_in_graph tells it "treat this as
      one opaque node" — the GPU sees one _scaled_mm call, not a graph of
      quantize→clamp→cast→matmul→dequantize ops. This is simpler and avoids
      graph breaks, at the cost of not fusing across the FP8 boundary.

    Why custom autograd instead of just overriding forward():
      A Linear layer does THREE matmuls total (1 forward, 2 backward).
      All three benefit from FP8. We need autograd.Function to intercept
      the backward pass and replace its matmuls with FP8 versions too.

    Memory optimization:
      We save the FP8-quantized input and weight (not the originals) for
      backward. FP8 tensors are 1 byte per element vs 2 bytes for BF16 —
      50% activation memory savings for these saved tensors.
    """

    @staticmethod
    def forward(ctx, input_2d: Tensor, weight: Tensor) -> Tensor:
        # ── Quantize both operands to E4M3 (higher precision format) ──
        # E4M3 for forward because weights and activations have predictable
        # range (post-normalization, post-activation). We want the best
        # precision we can get (3 mantissa bits > 2 mantissa bits of E5M2).
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)

        # Save quantized tensors for backward (saves memory: fp8 = 1 byte/elem)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # ── Forward GEMM: output = input @ weight.T ──
        #
        # Layout analysis:
        #   input_fp8:  [B*S, D_in]  contiguous → row-major ✓ (first arg)
        #   weight_fp8: [D_out, D_in] contiguous → weight.t() is [D_in, D_out]
        #               with strides (1, D_in) → column-major ✓ (second arg)
        #   No copy needed! This is why nn.Linear stores weight as [out, in].
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,  # BF16 output to match rest of computation
            # use_fast_accum=True: accumulates dot products in reduced precision.
            # ~5-10% faster but slightly less accurate. Standard for forward pass
            # because activations are consumed once; gradients need more care.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # ── GEMM 1: grad_input = grad_output @ weight ──
        # Shape: [B*S, D_out] @ [D_out, D_in] → [B*S, D_in]
        #
        # Gradients use E5M2: wider range (±57344 vs ±448) because gradients
        # can be much larger than activations, especially early in training
        # when loss is high. E5M2's 5-bit exponent prevents overflow.
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)

        # go_fp8: [B*S, D_out] row-major ✓ (first arg)
        # w_fp8:  [D_out, D_in] row-major ✗ (second arg needs column-major)
        #         → _to_col_major makes it [D_out, D_in] with col-major strides
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            # use_fast_accum=False: full-precision accumulation for gradients.
            # Gradient precision directly affects weight updates and training
            # stability. The ~5% speed cost is worth the numerical reliability.
            use_fast_accum=False,
        )

        # ── GEMM 2: grad_weight = grad_output.T @ input ──
        # Shape: [D_out, B*S] @ [B*S, D_in] → [D_out, D_in]
        #
        # Subtlety: go_fp8 is [B*S, D_out] row-major.
        # go_fp8.t() gives [D_out, B*S] but with column-major strides.
        # First arg needs row-major → .contiguous() forces a physical copy.
        go_T = go_fp8.t().contiguous()     # [D_out, B*S] row-major
        in_col = _to_col_major(in_fp8)     # [B*S, D_in] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


# =============================================================================
# FP8 LINEAR LAYER
# =============================================================================

class Float8CastLinear(nn.Linear):
    """Drop-in CastLinear replacement that does FP8 compute on H100+.

    Weights remain in their original precision (fp32) for optimizer precision.
    Only the matmul is performed in FP8 via _Float8Matmul autograd function.
    This mirrors CastLinear's philosophy: master weights stay fp32, compute
    happens in reduced precision.

    Compared to nanochat's Float8Linear, this class:
    1. Inherits from nn.Linear (not a custom base) for maximum compatibility
    2. Handles the 3D→2D reshape that _scaled_mm requires (MoE tokens are 2D,
       but MLA outputs may be 3D before reshaping)
    3. Casts input to BF16 before FP8 quantization (matching CastLinear's
       explicit dtype casting instead of relying on autocast)
    """

    def forward(self, x: Tensor) -> Tensor:
        # Cast to BF16 first (same as CastLinear, but we'll quantize to FP8 next)
        x = x.to(torch.bfloat16)

        # _scaled_mm only operates on 2D tensors.
        # If input is [B, S, D], flatten to [B*S, D] for the matmul.
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])

        output = _Float8Matmul.apply(x_2d, self.weight)

        # Restore batch dimensions: [B*S, D_out] → [B, S, D_out]
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_cast_linear(cls, mod: nn.Module) -> "Float8CastLinear":
        """Create Float8CastLinear from CastLinear, sharing weight/bias.

        Uses meta device trick: allocate module shell on meta (no memory),
        then point .weight and .bias to the original module's parameters.
        Zero additional memory — the optimizer sees the same parameter objects.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new_mod.weight = mod.weight  # share, don't copy
        new_mod.bias = mod.bias
        return new_mod


# =============================================================================
# MODEL CONVERSION
# =============================================================================

def _is_fp8_eligible(mod: nn.Module, fqn: str) -> bool:
    """Determine if a CastLinear module should be converted to FP8.

    Filters out modules that are:
    1. Not CastLinear (or nn.Linear)
    2. Too small (any dim < MIN_FP8_DIM) — overhead > benefit
    3. Not aligned (dims not divisible by FP8_ALIGNMENT) — H/W requirement
    4. Gate router — precision-critical for MoE routing decisions
    5. Embedding-related — different optimizer group, gradient sensitivity

    The fully-qualified name (fqn) tells us WHERE in the model this module is:
      "layers.0.mla.wq_a"      → MLA projection     → eligible
      "layers.0.moe.gate.router_weight"  → router    → NEVER eligible
      "layers.0.moe.shared_expert.w_gate" → shared   → eligible
      "tok_emb" / "lm_head"    → embeddings          → skip
    """
    # Only convert CastLinear (our custom dtype-casting Linear)
    # Import here to avoid circular dependency
    from .model import CastLinear
    if not isinstance(mod, (CastLinear, nn.Linear)):
        return False

    # Hardware requirements: dims must be divisible by 16, large enough
    if mod.in_features % FP8_ALIGNMENT != 0 or mod.out_features % FP8_ALIGNMENT != 0:
        return False
    if min(mod.in_features, mod.out_features) < MIN_FP8_DIM:
        return False

    # NEVER FP8-ify the gate router — routing decisions need full precision.
    # Quantization noise in sigmoid gate logits → wrong expert selection →
    # cascading quality degradation across all MoE layers.
    if "gate" in fqn and "router" in fqn:
        return False
    # Also skip the gate's CastLinear if named differently
    if fqn.endswith("gate.router_weight"):
        return False

    # Skip embeddings and LM head (different optimizer group, gradient sensitivity)
    if "tok_emb" in fqn or "lm_head" in fqn or "embed" in fqn:
        return False

    return True


def convert_nanoseek_to_fp8(model: nn.Module) -> tuple[int, int, int]:
    """Replace eligible CastLinear layers with Float8CastLinear throughout model.

    Walks the module tree in post-order (children before parents) and swaps
    each CastLinear that passes the eligibility filter. The new Float8CastLinear
    shares the original weight — no copies, no extra memory.

    Returns:
        (num_converted, num_skipped, num_total) for logging.
    """
    from .model import CastLinear

    num_total = 0
    num_converted = 0
    num_skipped = 0

    def _convert(mod: nn.Module, prefix: str = ""):
        nonlocal num_total, num_converted, num_skipped

        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)

            if isinstance(child, (CastLinear, nn.Linear)):
                num_total += 1
                if _is_fp8_eligible(child, fqn):
                    setattr(mod, name, Float8CastLinear.from_cast_linear(child))
                    num_converted += 1
                else:
                    num_skipped += 1

    _convert(model)
    return num_converted, num_skipped, num_total


# =============================================================================
# EVALUATION ESCAPE HATCH
# =============================================================================

class disable_fp8:
    """Context manager to temporarily revert Float8CastLinear → CastLinear for eval.

    During evaluation, we want deterministic BF16 precision — FP8 quantization
    noise would contaminate val_bpb and make it an unreliable training signal.

    Implementation: swap Float8CastLinear modules with CastLinear, sharing
    the same weight tensors (zero copy). Restore on exit.

    Usage:
        with disable_fp8(model):
            val_bpb = evaluate(model, val_loader)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.fp8_locations: list[tuple[nn.Module, str, nn.Module]] = []

    def __enter__(self):
        from .model import CastLinear

        for name, module in self.model.named_modules():
            if isinstance(module, Float8CastLinear):
                if "." in name:
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                    attr_name = name
                self.fp8_locations.append((parent, attr_name, module))

        # Swap: Float8CastLinear → CastLinear (sharing same weight)
        for parent, attr_name, fp8_mod in self.fp8_locations:
            linear = CastLinear(
                fp8_mod.in_features,
                fp8_mod.out_features,
                bias=fp8_mod.bias is not None,
                device=fp8_mod.weight.device,
                dtype=fp8_mod.weight.dtype,
            )
            linear.weight = fp8_mod.weight  # share, don't copy
            if fp8_mod.bias is not None:
                linear.bias = fp8_mod.bias
            setattr(parent, attr_name, linear)

        return self

    def __exit__(self, *exc):
        # Restore Float8CastLinear modules
        for parent, attr_name, fp8_mod in self.fp8_locations:
            setattr(parent, attr_name, fp8_mod)
        return False


# =============================================================================
# HARDWARE DETECTION
# =============================================================================

def is_fp8_available() -> tuple[bool, str]:
    """Check if FP8 training is possible on current hardware.

    FP8 requires:
    1. CUDA available
    2. Hopper or newer GPU (SM 9.0+) for native FP8 tensor cores
    3. PyTorch with _scaled_mm support (2.1+)

    Returns:
        (available: bool, reason: str)
    """
    if not torch.cuda.is_available():
        return False, "no CUDA available"

    capability = torch.cuda.get_device_capability()
    if capability < (9, 0):
        gpu_name = torch.cuda.get_device_name(0)
        return False, f"GPU {gpu_name} (SM {capability[0]}.{capability[1]}) < SM 9.0 (Hopper required)"

    if not hasattr(torch, "_scaled_mm"):
        return False, f"PyTorch {torch.__version__} missing torch._scaled_mm (need 2.1+)"

    return True, f"H100/H200 detected (SM {capability[0]}.{capability[1]})"
