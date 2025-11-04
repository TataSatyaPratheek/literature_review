"""Quantum circuit - SIMPLIFIED VERSION."""
import pennylane as qml
import torch
import torch.nn as nn

class QuantumCircuit(nn.Module):
    """Simple quantum circuit with PennyLane."""
    
    def __init__(self, n_qubits, n_blocks, device='default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.n_params = n_qubits * n_blocks
        
        # Quantum device
        self.dev = qml.device(device, wires=n_qubits)
        
        # Simple initialization - LARGER values
        self.phi = nn.Parameter(torch.randn(n_blocks, n_qubits) * 0.1)  # Changed from 0.01 to 2.0
        
        # QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')
        
    def _circuit(self, phi):
        for block in range(self.n_blocks):
            for qubit in range(self.n_qubits):
                qml.RY(phi[block, qubit], wires=qubit)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        return qml.probs(wires=range(self.n_qubits))
    
    def forward(self):
        probs = self.qnode(self.phi)
        if probs.dtype == torch.float64:
            probs = probs.float()
        return probs.unsqueeze(1)  # Shape: (2^n, 1)
    
    def get_n_quantum_params(self):
        return self.n_params
