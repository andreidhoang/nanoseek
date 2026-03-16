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
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Tuple


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    rank: int = 0,
) -> str:
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        step: Current training step
        model_state: Model state dict
        optimizer_state: Optimizer state dict (can be None to skip)
        metadata: Training metadata (step, tokens, config, etc.)
        rank: Process rank (only rank 0 writes files)
        
    Returns:
        Path to saved model checkpoint
    """
    if rank != 0:
        return ""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_state, model_path)
    
    # Save optimizer state (optional, large file)
    if optimizer_state is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step:06d}.pt")
        torch.save(optimizer_state, optimizer_path)
    
    # Save metadata as JSON
    metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")
    
    # Convert metadata to JSON-serializable format
    json_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_metadata[key] = value
        elif isinstance(value, dict):
            json_metadata[key] = value
        elif isinstance(value, torch.Tensor):
            json_metadata[key] = value.item() if value.numel() == 1 else str(value.shape)
        else:
            json_metadata[key] = str(value)
    
    with open(metadata_path, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    # Update latest.json pointer
    latest_path = os.path.join(checkpoint_dir, "latest.json")
    with open(latest_path, 'w') as f:
        json.dump({
            "latest_step": step,
            "model_path": model_path,
            "metadata_path": metadata_path,
        }, f, indent=2)
    
    return model_path


def save_ema_checkpoint(
    checkpoint_dir: str,
    step: int,
    ema_state: Dict[str, torch.Tensor],
    rank: int = 0,
) -> Optional[str]:
    """
    Save EMA state separately from main checkpoint.
    
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
    torch.save(ema_state, ema_path)
    return ema_path


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
    """
    Load a training checkpoint.
    
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
            with open(latest_path, 'r') as f:
                latest = json.load(f)
            model_path = latest["model_path"]
            metadata_path = latest["metadata_path"]
            step = latest["latest_step"]
        else:
            # Find latest model file
            model_files = sorted(Path(checkpoint_dir).glob("model_*.pt"))
            if not model_files:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            model_path = str(model_files[-1])
            step = int(model_files[-1].stem.split('_')[1])
            metadata_path = os.path.join(checkpoint_dir, f"metadata_{step:06d}.json")
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
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model_state, metadata, step


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
    
    optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
    return optimizer_state


class CheckpointManager:
    """
    Convenience class for managing checkpoints during training.
    
    Handles:
    - Periodic checkpoint saving
    - Keeping only N most recent checkpoints
    - Automatic EMA saving
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 1000,
        keep_last_n: int = 3,
        save_optimizer: bool = False,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
            keep_last_n: Keep only N most recent checkpoints (0 = keep all)
            save_optimizer: Whether to save optimizer state (large files)
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.saved_steps = []
    
    def maybe_save(
        self,
        step: int,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
        rank: int = 0,
    ) -> Optional[str]:
        """
        Save checkpoint if it's time to do so.
        
        Args:
            step: Current training step
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            metadata: Training metadata
            ema_state: Optional EMA state dict
            rank: Process rank
            
        Returns:
            Path to saved checkpoint or None if not saved
        """
        if self.save_every <= 0 or step % self.save_every != 0:
            return None
        
        if rank != 0:
            return None
        
        # Save main checkpoint
        opt_state = optimizer_state if self.save_optimizer else None
        model_path = save_checkpoint(
            self.checkpoint_dir,
            step,
            model_state,
            opt_state,
            metadata,
            rank,
        )
        
        # Save EMA if provided
        if ema_state is not None:
            save_ema_checkpoint(self.checkpoint_dir, step, ema_state, rank)
        
        self.saved_steps.append(step)
        
        # Clean up old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(rank)
        
        return model_path
    
    def _cleanup_old_checkpoints(self, rank: int = 0):
        """Remove old checkpoints, keeping only keep_last_n most recent."""
        if rank != 0:
            return
        
        if len(self.saved_steps) <= self.keep_last_n:
            return
        
        steps_to_remove = self.saved_steps[:-self.keep_last_n]
        
        for step in steps_to_remove:
            # Remove model checkpoint
            model_path = os.path.join(self.checkpoint_dir, f"model_{step:06d}.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Remove EMA checkpoint
            ema_path = os.path.join(self.checkpoint_dir, f"ema_{step:06d}.pt")
            if os.path.exists(ema_path):
                os.remove(ema_path)
            
            # Remove optimizer checkpoint
            opt_path = os.path.join(self.checkpoint_dir, f"optimizer_{step:06d}.pt")
            if os.path.exists(opt_path):
                os.remove(opt_path)
            
            # Remove metadata
            meta_path = os.path.join(self.checkpoint_dir, f"metadata_{step:06d}.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
        
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
    from .model import NanoSeekModel
    
    # Build model
    with torch.device("meta"):
        model = NanoSeekModel(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    # Load checkpoint
    model_state, metadata, loaded_step = load_checkpoint(
        checkpoint_dir, step, device
    )
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
