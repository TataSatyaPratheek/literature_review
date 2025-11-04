"""CIFAR-10-specific configuration."""
from .base_config import BaseConfig
import numpy as np

class CIFAR10Config(BaseConfig):
    """Configuration for CIFAR-10 experiments."""
    
    # Dataset
    dataset_name = "cifar10"
    input_shape = (3, 32, 32)
    num_classes = 10
    
    # Classical CNN parameters (from paper)
    classical_params = 285226
    n_qubits = int(np.ceil(np.log2(classical_params)))  # 19 qubits
    
    # Quantum circuit
    n_blocks = 19  # Can vary: 19-323 tested
    
    # Mapping model
    mapping_layers = [20, 40, 200, 40, 1]
    
    # Training
    batch_size = 1000
    n_epochs = 1000
    learning_rate = 1e-4
