"""Dataset loading and preprocessing."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_dataset(config):
    """Load and prepare dataset."""
    
    if config.dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_train = datasets.MNIST(
            './data/mnist', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            './data/mnist', train=False, transform=transform
        )
        
    elif config.dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        full_train = datasets.CIFAR10(
            './data/cifar10', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            './data/cifar10', train=False, transform=transform
        )
    
    # Split training into train and validation
    train_size = int(config.train_split * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader
