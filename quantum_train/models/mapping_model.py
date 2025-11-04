"""Mapping model to transform quantum probabilities to classical parameters."""
import torch
import torch.nn as nn
import math

class MappingModel(nn.Module):
    """
    Mapping model G_gamma that transforms quantum measurement probabilities
    into classical neural network parameters.
    """
    
    def __init__(self, n_qubits, layer_sizes, use_tanh_first=True):
        """
        Initialize mapping model with balanced initialization.
        
        Args:
            n_qubits: Number of qubits (determines input size)
            layer_sizes: List of layer sizes [h1, h2, ..., output]
            use_tanh_first: Use tanh in first layer for unbounded output
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.input_size = n_qubits + 1  # Binary basis + probability
        
        # Build the network
        layers = []
        in_features = self.input_size
        
        for idx, out_features in enumerate(layer_sizes):
            linear = nn.Linear(in_features, out_features)
            
            if idx == 0 and use_tanh_first:
                # Xavier for tanh layer
                nn.init.xavier_uniform_(linear.weight, gain=1.0)
                nn.init.zeros_(linear.bias)
            elif idx == len(layer_sizes) - 1:
                # FINAL LAYER: Small initialization for stability
                nn.init.xavier_uniform_(linear.weight, gain=0.1)  # Increased from 0.01
                nn.init.zeros_(linear.bias)
            else:
                # Hidden layers: Kaiming for ReLU
                nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in')
                nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            
            # Activations
            if idx == 0 and use_tanh_first:
                layers.append(nn.Tanh())
            elif idx < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            # NO activation on final layer!
            
            in_features = out_features
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, basis_vectors, probabilities):
        """
        Map quantum probabilities to classical parameters.
        """
        # Ensure float32 for MPS compatibility
        probabilities = probabilities.float()
        basis_vectors = basis_vectors.float()
        
        # Concatenate basis vectors with probabilities
        x = torch.cat([basis_vectors, probabilities.unsqueeze(-1)], dim=-1)
        
        # Apply mapping network
        theta = self.network(x).squeeze(-1)
        
        # NO SCALING - return raw values (unbounded)
        return theta
    
    def create_basis_vectors(self, device):
        """
        Create binary basis vectors for all computational basis states.
        
        Returns:
            basis_vectors: Tensor (2^n_qubits, n_qubits)
        """
        n_states = 2 ** self.n_qubits
        basis_vectors = torch.zeros(n_states, self.n_qubits, dtype=torch.float32, device=device)
        
        for i in range(n_states):
            # Convert integer to binary representation
            binary = format(i, f'0{self.n_qubits}b')
            basis_vectors[i] = torch.tensor([int(b) for b in binary], 
                                           dtype=torch.float32, device=device)
        
        return basis_vectors
