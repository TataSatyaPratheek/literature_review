from .base_config import BaseConfig
import numpy as np

class MNISTConfig(BaseConfig):
    dataset_name = "mnist"
    input_shape = (1, 28, 28)
    num_classes = 10
    
    classical_params = 6676
    n_qubits = 13
    n_blocks = 8  # REDUCED from 16 to 8 for speed
    
    mapping_layers = []  # Not used with minimal mapping
    
    batch_size = 64  # INCREASED from 32 to 64 for speed
    n_epochs = 50
    learning_rate = 5e-4  # MODERATE: between 1e-5 and 1e-3
    
    classical_architecture = {
        'conv1': {'in_channels': 1, 'out_channels': 4, 'kernel_size': 3},
        'conv2': {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3},
        'fc1': {'in_features': 200, 'out_features': 30},
        'fc2': {'in_features': 30, 'out_features': 10}
    }