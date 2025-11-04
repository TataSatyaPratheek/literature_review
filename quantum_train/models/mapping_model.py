"""Mapping model - MINIMAL VERSION."""
import torch
import torch.nn as nn

class MappingModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
        # SIMPLEST possible mapping: just one linear layer
        self.linear = nn.Linear(n_qubits + 1, 1, bias=False)
        
        # Small initialization
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
    
    def forward(self, basis_vectors, probabilities):
        x = torch.cat([basis_vectors, probabilities], dim=-1)
        theta = self.linear(x).squeeze(-1)
        return theta
    
    def create_basis_vectors(self, device):
        n_states = 2 ** self.n_qubits
        basis_vectors = torch.zeros(n_states, self.n_qubits, device=device)
        
        for i in range(n_states):
            binary = format(i, f'0{self.n_qubits}b')
            basis_vectors[i] = torch.tensor([float(b) for b in binary], device=device)
        
        return basis_vectors