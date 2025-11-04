"""Base configuration for Quantum-Train experiments."""
import torch

class BaseConfig:
    """Base configuration class."""
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Quantum circuit parameters
    n_qubits = None  # To be set by dataset config
    n_blocks = None  # Number of QNN blocks
    
    # Mapping model architecture
    mapping_layers = None  # e.g., [14, 4, 20, 4, 1]
    
    # Training hyperparameters
    batch_size = 128
    learning_rate = 1e-4
    n_epochs = 50
    optimizer = "adam"
    
    # Data parameters
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    # Logging
    log_interval = 10
    save_interval = 5
    
    # Paths
    checkpoint_dir = "experiments/results/checkpoints"
    results_dir = "experiments/results"
