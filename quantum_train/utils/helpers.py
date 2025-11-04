"""Helper utility functions."""
import torch
import numpy as np
import random
import os
from pathlib import Path

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """
    Get the best available device.
    
    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_directories(paths):
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths or single path string
    """
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def format_time(seconds):
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def save_config(config, path):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        path: Save path
    """
    import json
    
    config_dict = {k: v for k, v in config.__dict__.items() 
                   if not k.startswith('_')}
    
    # Convert non-serializable objects to strings
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(path):
    """
    Load configuration from file.
    
    Args:
        path: Path to config file
        
    Returns:
        Dictionary with configuration
    """
    import json
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict

def print_model_summary(model, input_size=None):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (optional)
    """
    print("=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 70)
