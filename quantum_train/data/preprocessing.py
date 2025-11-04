"""Data preprocessing utilities."""
import torch
import numpy as np

def normalize_data(data, mean, std):
    """
    Normalize data using mean and standard deviation.
    
    Args:
        data: Input data tensor
        mean: Mean value(s)
        std: Standard deviation value(s)
        
    Returns:
        Normalized data
    """
    return (data - mean) / std

def denormalize_data(data, mean, std):
    """
    Denormalize data back to original scale.
    
    Args:
        data: Normalized data tensor
        mean: Mean value(s) used in normalization
        std: Standard deviation value(s) used in normalization
        
    Returns:
        Denormalized data
    """
    return data * std + mean

def augment_images(images, config):
    """
    Apply data augmentation to images.
    
    Args:
        images: Batch of images
        config: Configuration with augmentation parameters
        
    Returns:
        Augmented images
    """
    # Basic augmentation - can be extended
    if hasattr(config, 'use_augmentation') and config.use_augmentation:
        # Add random horizontal flips, rotations, etc.
        pass
    
    return images

def calculate_dataset_statistics(dataset):
    """
    Calculate mean and std of dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        mean, std tuples
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False
    )
    data = next(iter(loader))[0]
    
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3))
    
    return mean, std
