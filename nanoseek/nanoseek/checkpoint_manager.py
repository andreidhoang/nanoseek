"""
Checkpoint manager for NanoSeek training and evaluation.

Handles saving/loading of:
- Model state dict
- Optimizer state dict
- Training metadata (step, tokens, config, dataloader state)
- EMA state (separate file for tensor compatibility)

Directory structure:
    checkpoints/nanoseek_{scale}/
        model_{step:06d}.pt       # Model state dict
        ema_{step:06d}.pt         # EMA state dict (optional)
        optimizer_{step:06d}.pt   # Optimizer state (optional, large)
        latest.json               # Points to latest checkpoint

Crash Safety:
    All writes use atomic rename (write to .tmp, then os.replace).
    This guarantees that a checkpoint file either exists fully or not at all.
    latest.json is written LAST, so it always points to a complete checkpoint.
    On resume, if latest.json is missing/stale, we fall back to globbing for
    the highest-step model file.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Tuple


# ═══════════════════════════════════════════════════════════════════
# Atomic I/O Primitives
# ═══════════════════════════════════════════════════════════════════
#
# Why atomic writes matter:
#   torch.save writes sequentially to a file. If the process is killed
#   mid-write (SIGKILL, OOM, RunPod preemption), the file is truncated.
#   torch.load on a truncated file raises _pickle.UnpicklingError or
#   RuntimeError — unrecoverable without the previous checkpoint.
#
#   os.replace() is atomic on Linux (ext4, XFS, NVMe) — the old file
#   is replaced in a single filesystem operation. If we crash before
#   os.replace(), the .tmp file is garbage but the original is intact.
# ═══════════════════════════════════════════════════════════════════


def _atomic_torch_save(obj: Any, path: str) -> None:
    """Save a torch object atomically using write-to-tmp + rename.

    Guarantees: `path` either contains the complete new data or the
    previous version. Never a truncated/corrupt intermediate state.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    # fsync to ensure data hits disk before rename (NVMe writeback cache)
    _fsync_file(tmp_path)
    os.replace(tmp_path, path)  # atomic on Linux
    _fsync_dir(os.path.dirname(path))  # durability guarantee on ext4


def _atomic_json_save(obj: Any, path: str) -> None:
    """Save a JSON object atomically using write-to-tmp + rename."""
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(os.path.dirname(path))  # durability guarantee on ext4


def _fsync_file(path: str) -> None:
    """Force file data to disk (bypass OS/NVMe write cache)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(dirpath: str) -> None:
    """Force directory metadata to disk.

    On Linux ext4, file renames (os.replace) are only guaranteed durable
    after a directory fsync. Without this, a power loss right after
    os.replace() could result in both old and new files being missing.
    This matters for bare-metal RunPod with sudden power cuts.
    """
    try:
        fd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # Not all OSes support dir fsync (e.g., macOS)


def _verify_torch_checkpoint(path: str) -> bool:
    """Verify a .pt file is likely valid without loading it into RAM.

    Previous approach: torch.load(path, weights_only=True) — loads the
    entire checkpoint into CPU RAM. For a 1B model + optimizer, that's
    ~8-10 GB of extra memory during verification.

    Current approach: check file size > minimum threshold + verify the
    PyTorch magic number in the file header. This catches truncation
    (the main corruption mode from interrupted saves) without OOM risk.
    """
    try:
        size = os.path.getsize(path)
        if size < 128:  # Any valid checkpoint is at least a few KB
            return False
        # Check for PyTorch/pickle magic bytes at start of file
        with open(path, 'rb') as f:
            header = f.read(2)
            # PyTorch checkpoints start with pickle protocol header (0x80 0x02+)
            # or ZIP archive header (PK = 0x50 0x4B) for newer format
            if header[:2] == b'PK' or (header[0] == 0x80 and header[1] >= 0x02):
                return True
            return False
    except Exception:
        return False


def _cleanup_tmp_files(checkpoint_dir: str) -> int:
    """Remove any leftover .tmp files from interrupted saves.

    Returns number of .tmp files cleaned up.
    """
    count = 0
    if not os.path.exists(checkpoint_dir):
        return 0
    for f in Path(checkpoint_dir).glob("*.tmp"):
        try:
            os.remove(f)
            count += 1
        except OSError:
            pass
    return count


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    rank: int = 0,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    verify: bool = True,
) -> str:
    """
    Save a training checkpoint atomically.

    Write order: model → optimizer → EMA → metadata → latest.json
    Each file is written atomically (tmp + rename).
    latest.json is written LAST so it always points to a complete set.

    Args:
        checkpoint_dir: Directory to save checkpoint
        step: Current training step
        model_state: Model state dict
        optimizer_state: Optimizer state dict (can be None to skip)
        metadata: Training metadata (step, tokens, config, etc.)
        rank: Process rank (only rank 0 writes files)
        ema_state: Optional EMA state dict (saved atomically with checkpoint)
        verify: Whether to verify checkpoint integrity after save

    Returns:
        Path to saved model checkpoint
    """
    if rank != 0:
        return ""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Save model state (atomically)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    _atomic_torch_save(model_state, model_path)

    # 2. Save optimizer state (atomically, optional — large file)
    if optimizer_state is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step:06d}.pt")
        _atomic_torch_save(optimizer_state, optimizer_path)

    # 3. Save EMA state (atomically, alongside checkpoint — not separate)
    if ema_state is not None:
        ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")
        _atomic_torch_save(ema_state, ema_path)

    # 4. Save metadata as JSON (atomically)
    metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")

    # Convert metadata to JSON-serializable format
    json_metadata = _serialize_metadata(metadata)
    _atomic_json_save(json_metadata, metadata_path)

    # 5. Verify checkpoint integrity (lightweight — just check model file)
    if verify and not _verify_torch_checkpoint(model_path):
        raise RuntimeError(
            f"Checkpoint verification FAILED for {model_path}. "
            f"Disk may be corrupt or full. NOT updating latest.json."
        )

    # 6. Update latest.json pointer LAST (atomically)
    #    Uses relative paths so checkpoints are portable across machines/mounts.
    latest_path = os.path.join(checkpoint_dir, "latest.json")
    _atomic_json_save({
        "latest_step": step,
        "model_file": f"model_{step:06d}.pt",       # relative, not absolute
        "metadata_file": f"metadata_{step:06d}.json", # relative, not absolute
    }, latest_path)

    return model_path


def _serialize_metadata(metadata: Any) -> Any:
    """Convert metadata to JSON-serializable format.

    Handles nested dicts, lists, tuples, tensors, and primitive types.
    This is critical for dataloader_state_dict which contains:
    - pq_idx, rg_idx, epoch: ints
    - fim_fraction: float
    - fim_rng_state: list of [int, list-of-625-ints, float]
      (Mersenne Twister state from random.getstate())
    """
    if metadata is None:
        return None
    if isinstance(metadata, (int, float, str, bool)):
        return metadata
    if isinstance(metadata, dict):
        return {k: _serialize_metadata(v) for k, v in metadata.items()}
    if isinstance(metadata, (list, tuple)):
        return [_serialize_metadata(item) for item in metadata]
    if isinstance(metadata, torch.Tensor):
        return metadata.item() if metadata.numel() == 1 else str(metadata.shape)
    # Fallback: stringify unknown types
    return str(metadata)


def save_ema_checkpoint(
    checkpoint_dir: str,
    step: int,
    ema_state: Dict[str, torch.Tensor],
    rank: int = 0,
) -> Optional[str]:
    """
    Save EMA state separately from main checkpoint (atomically).

    NOTE: Prefer passing ema_state to save_checkpoint() directly so that
    EMA is saved as part of the atomic checkpoint sequence. This function
    exists for backward compatibility.

    Args:
        checkpoint_dir: Directory to save checkpoint
        step: Current training step
        ema_state: EMA state dict (shadow parameters)
        rank: Process rank (only rank 0 writes)

    Returns:
        Path to saved EMA checkpoint or None if rank != 0
    """
    if rank != 0:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")
    _atomic_torch_save(ema_state, ema_path)
    return ema_path


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
    """
    Load a training checkpoint.

    Resolution order for step=None (latest):
    1. Read latest.json → use relative paths within checkpoint_dir
    2. If latest.json missing/corrupt: glob for highest-step model file
    3. If no model files found: raise FileNotFoundError

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (None = latest)
        device: Device to load tensors to

    Returns:
        Tuple of (model_state, metadata, loaded_step)

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Determine which step to load
    if step is None:
        # Load latest
        latest_path = os.path.join(checkpoint_dir, "latest.json")
        if os.path.exists(latest_path):
            try:
                with open(latest_path, 'r') as f:
                    latest = json.load(f)
                step = latest["latest_step"]

                # Support both old (absolute path) and new (relative filename) formats
                if "model_file" in latest:
                    model_path = os.path.join(checkpoint_dir, latest["model_file"])
                    metadata_path = os.path.join(checkpoint_dir, latest["metadata_file"])
                else:
                    # Legacy format with absolute paths — try them, fall back to reconstructed
                    model_path = latest.get("model_path", "")
                    metadata_path = latest.get("metadata_path", "")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
                        metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")

            except (json.JSONDecodeError, KeyError) as e:
                # latest.json is corrupt — fall back to glob
                print(f"Warning: latest.json corrupt ({e}), falling back to glob")
                model_path, metadata_path, step = _resolve_latest_by_glob(checkpoint_dir)
        else:
            model_path, metadata_path, step = _resolve_latest_by_glob(checkpoint_dir)
    else:
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load model state
    model_state = torch.load(model_path, map_location=device, weights_only=True)

    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: metadata file corrupt at {metadata_path}, using empty metadata")

    return model_state, metadata, step


def _resolve_latest_by_glob(checkpoint_dir: str) -> Tuple[str, str, int]:
    """Find the latest checkpoint by globbing for model files."""
    model_files = sorted(Path(checkpoint_dir).glob("model_*.pt"))
    # Filter out .tmp files (incomplete writes)
    model_files = [f for f in model_files if not str(f).endswith(".tmp")]
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    model_path = str(model_files[-1])
    step = int(model_files[-1].stem.split('_')[1])
    metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")
    return model_path, metadata_path, step


def load_ema_checkpoint(
    checkpoint_dir: str,
    step: int,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load EMA checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load
        device: Device to load tensors to

    Returns:
        EMA state dict

    Raises:
        FileNotFoundError: If EMA checkpoint not found
    """
    ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")

    if not os.path.exists(ema_path):
        raise FileNotFoundError(f"EMA checkpoint not found: {ema_path}")

    ema_state = torch.load(ema_path, map_location=device, weights_only=True)
    return ema_state


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    List all available checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (step, model_path) tuples, sorted by step
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for model_file in Path(checkpoint_dir).glob("model_*.pt"):
        if str(model_file).endswith(".tmp"):
            continue  # skip incomplete writes
        step = int(model_file.stem.split('_')[1])
        checkpoints.append((step, str(model_file)))

    return sorted(checkpoints)


def load_optimizer_checkpoint(
    checkpoint_dir: str,
    step: int,
    device: torch.device = torch.device("cpu"),
) -> Optional[Dict[str, Any]]:
    """
    Load optimizer checkpoint.

    PyTorch optimizer state_dict() contains nested dicts of tensors and
    Python scalars. weights_only=True supports these types in PyTorch >= 2.1.
    If loading fails with weights_only=True (e.g., custom optimizer with
    non-standard types), falls back to weights_only=False with a warning.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load
        device: Device to load tensors to

    Returns:
        Optimizer state dict or None if not found
    """
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step:06d}.pt")

    if not os.path.exists(optimizer_path):
        return None

    try:
        optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
    except Exception as e:
        # Muon/DistMuonAdamW might store types not supported by safe unpickler.
        # Fall back to weights_only=False (still safe — we wrote this file ourselves).
        print(f"Warning: weights_only=True failed for optimizer ({e}), "
              f"falling back to weights_only=False")
        optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=False)
    return optimizer_state


class CheckpointManager:
    """
    Manages checkpoint lifecycle during training.

    Handles:
    - Periodic checkpoint saving (atomic writes)
    - Keeping only N most recent checkpoints (disk space management)
    - Automatic EMA saving alongside model
    - Emergency checkpoint on signal (SIGTERM/SIGINT)
    - Leftover .tmp cleanup on init
    - Post-save integrity verification
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 1000,
        keep_last_n: int = 3,
        save_optimizer: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
            keep_last_n: Keep only N most recent checkpoints (0 = keep all)
            save_optimizer: Whether to save optimizer state (MUST be True for resume)
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Clean up leftover .tmp files from interrupted saves
        n_cleaned = _cleanup_tmp_files(checkpoint_dir)
        if n_cleaned > 0:
            print(f"CheckpointManager: cleaned up {n_cleaned} leftover .tmp files")

        # Recover saved_steps from existing checkpoints on disk
        # (needed if CheckpointManager is recreated after a crash+resume)
        self.saved_steps = [s for s, _ in list_checkpoints(checkpoint_dir)]

    def should_save(self, step: int, last_step: bool = False) -> bool:
        """Check if we should save at this step."""
        if last_step:
            return True
        if self.save_every <= 0:
            return False
        return step > 0 and step % self.save_every == 0

    def save(
        self,
        step: int,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
        rank: int = 0,
    ) -> Optional[str]:
        """
        Save checkpoint atomically with all components.

        Args:
            step: Current training step
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            metadata: Training metadata
            ema_state: Optional EMA state dict
            rank: Process rank

        Returns:
            Path to saved checkpoint or None if rank != 0
        """
        if rank != 0:
            return None

        # Save all components atomically
        opt_state = optimizer_state if self.save_optimizer else None
        model_path = save_checkpoint(
            self.checkpoint_dir,
            step,
            model_state,
            opt_state,
            metadata,
            rank,
            ema_state=ema_state,
            verify=True,
        )

        self.saved_steps.append(step)

        # Clean up old checkpoints to prevent disk exhaustion
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(rank)

        return model_path

    def maybe_save(
        self,
        step: int,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
        rank: int = 0,
    ) -> Optional[str]:
        """Save checkpoint if it's time to do so (backward compat)."""
        if self.save_every <= 0 or step % self.save_every != 0:
            return None
        return self.save(step, model_state, optimizer_state, metadata, ema_state, rank)

    def _cleanup_old_checkpoints(self, rank: int = 0):
        """Remove old checkpoints, keeping only keep_last_n most recent."""
        if rank != 0:
            return

        if len(self.saved_steps) <= self.keep_last_n:
            return

        steps_to_remove = self.saved_steps[:-self.keep_last_n]

        for step in steps_to_remove:
            for pattern in [
                f"model_{step:06d}.pt",
                f"ema_{step:06d}.pt",
                f"optimizer_{step:06d}.pt",
                f"metadata_{step:06d}.json",
            ]:
                path = os.path.join(self.checkpoint_dir, pattern)
                if os.path.exists(path):
                    os.remove(path)
                # Also clean up any leftover .tmp files
                tmp_path = path + ".tmp"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        self.saved_steps = self.saved_steps[-self.keep_last_n:]


# Convenience functions for evaluation
def load_model_for_eval(
    checkpoint_dir: str,
    config: Any,
    device: torch.device,
    step: Optional[int] = None,
    use_ema: bool = False,
) -> Tuple[torch.nn.Module, int]:
    """
    Load a model for evaluation.

    This is a convenience function that combines model creation,
    checkpoint loading, and optional EMA weight application.

    Args:
        checkpoint_dir: Directory containing checkpoints
        config: Model configuration
        device: Device to load model on
        step: Specific step to load (None = latest)
        use_ema: Whether to use EMA weights

    Returns:
        Tuple of (model, loaded_step)
    """
    from .model import NanoSeekModel, migrate_legacy_moe_state_dict

    # Build model
    with torch.device("meta"):
        model = NanoSeekModel(config)

    model.to_empty(device=device)
    model.init_weights()

    # Load checkpoint (with legacy ModuleList → 3D migration if needed)
    model_state, metadata, loaded_step = load_checkpoint(
        checkpoint_dir, step, device
    )
    model_state = migrate_legacy_moe_state_dict(model_state)
    model.load_state_dict(model_state)

    # Load EMA if requested
    if use_ema:
        try:
            ema_state = load_ema_checkpoint(checkpoint_dir, loaded_step, device)
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name])
            print(f"Loaded EMA weights for step {loaded_step}")
        except FileNotFoundError:
            print(f"Warning: EMA checkpoint not found for step {loaded_step}, using raw weights")

    model.eval()
    return model, loaded_step
