"""MNIST-specific configuration."""
from .base_config import BaseConfig
import numpy as np

class MNISTConfig(BaseConfig):
    """Configuration for MNIST experiments."""
    
    # Dataset
    dataset_name = "mnist"
    input_shape = (1, 28, 28)
    num_classes = 10
    
    # Classical CNN parameters (from paper - approximately 6690)
    classical_params = 6676  # Actual count with architecture below
    n_qubits = int(np.ceil(np.log2(classical_params)))  # 13 qubits
    
    # Quantum circuit
    n_blocks = 16  # Can vary: 16-121 tested in paper
    
    # Mapping model [n_qubits+1, hidden, hidden, hidden, 1]
    mapping_layers = [14, 4, 20, 4, 1]
    
    # Training - ADJUSTED
    batch_size = 128
    n_epochs = 50
    learning_rate = 5e-3  # Changed from 1e-3 to 5e-3
    
    # Classical target NN architecture (smaller, matching ~6690 params)
    # Conv1: 1->4 channels, 3x3 kernel (28->26->13 after pool)
    # Conv2: 4->8 channels, 3x3 kernel (13->11->5 after pool)
    # FC1: 200->30, FC2: 30->10
    # Total: 40 + 296 + 6030 + 310 = 6676 params
    classical_architecture = {
        'conv1': {'in_channels': 1, 'out_channels': 4, 'kernel_size': 3},
        'conv2': {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3},
        'fc1': {'in_features': 8*5*5, 'out_features': 30},  # 200 -> 30
        'fc2': {'in_features': 30, 'out_features': 10}
    }
