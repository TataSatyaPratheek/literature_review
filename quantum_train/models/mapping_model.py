"""Mapping model - EXACT paper implementation."""
import torch
import torch.nn as nn

class MappingModel(nn.Module):
    """Mapping model with paper's exact [4, 20, 4] architecture."""
    
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_size = n_qubits + 1
        
        # Paper uses [4, 20, 4] hidden layers
        self.input_layer = nn.Linear(self.input_size, 4)
        self.hidden1 = nn.Linear(4, 20)
        self.hidden2 = nn.Linear(20, 4)
        self.output_layer = nn.Linear(4, 1)
    
    def forward(self, basis_vectors, probabilities):
        """Map quantum probs to classical params."""
        # Concatenate
        x = torch.cat([basis_vectors, probabilities], dim=-1)
        
        # Forward through layers (NO activations!)
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output_layer(x)
        
        # Squeeze and mean center
        theta = x.squeeze(-1)
        theta = theta - torch.mean(theta)
        
        return theta
    
    def create_basis_vectors(self, device):
        """Create basis vectors with -1,+1 encoding (not 0,1)."""
        n_states = 2 ** self.n_qubits
        basis_vectors = torch.zeros(n_states, self.n_qubits, device=device)
        
        for i in range(n_states):
            binary = format(i, f'0{self.n_qubits}b')
            # Use -1, +1 instead of 0, 1 (from paper!)
            basis_vectors[i] = torch.tensor(
                [-1 if b == '0' else 1 for b in binary],
                device=device,
                dtype=torch.float32
            )
        
        return basis_vectors