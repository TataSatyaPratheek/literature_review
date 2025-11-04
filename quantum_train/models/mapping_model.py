"""Mapping model - IMPROVED VERSION."""
import torch
import torch.nn as nn

class MappingModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Simple 2-layer MLP: (14 -> 20 -> 1)
        self.fc1 = nn.Linear(n_qubits + 1, 20)
        self.fc2 = nn.Linear(20, 1)
        
        # Proper initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, basis_vectors, probabilities):
        x = torch.cat([basis_vectors, probabilities], dim=-1)
        x = torch.relu(self.fc1(x))
        theta = self.fc2(x).squeeze(-1)
        return theta
    
    def create_basis_vectors(self, device):
        n_states = 2 ** self.n_qubits
        basis_vectors = torch.zeros(n_states, self.n_qubits, device=device)
        
        for i in range(n_states):
            binary = format(i, f'0{self.n_qubits}b')
            basis_vectors[i] = torch.tensor([float(b) for b in binary], device=device)
        
        return basis_vectors