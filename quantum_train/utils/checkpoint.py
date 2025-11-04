"""Checkpoint management utilities."""
import torch
from pathlib import Path
import shutil

class CheckpointManager:
    """
    Manage model checkpoints during training.
    """
    
    def __init__(self, checkpoint_dir, keep_best=True, keep_last_n=3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Whether to keep best model checkpoint
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_last_n = keep_last_n
        
        self.best_accuracy = 0.0
        self.checkpoint_history = []
        
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """
        Save a checkpoint.
        
        Args:
            model: Quantum-Train model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'quantum_state_dict': model.quantum_circuit.state_dict(),
            'mapping_state_dict': model.mapping_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)
            print(f"Best model updated: {best_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only last_n."""
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoints = self.checkpoint_history[:-self.keep_last_n]
            for old_checkpoint in old_checkpoints:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            
            self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Quantum-Train model
            optimizer: Optimizer (optional)
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(checkpoint_path)
        
        model.quantum_circuit.load_state_dict(checkpoint['quantum_state_dict'])
        model.mapping_model.load_state_dict(checkpoint['mapping_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        
        return checkpoint
    
    def get_best_checkpoint_path(self):
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
    
    def get_latest_checkpoint_path(self):
        """Get path to latest checkpoint."""
        if self.checkpoint_history:
            return self.checkpoint_history[-1]
        return None
