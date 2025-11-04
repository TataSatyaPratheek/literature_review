"""Quantum circuit implementation using PennyLane."""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumCircuit(nn.Module):
    """Parameterized quantum circuit for parameter generation."""
    
    def __init__(self, n_qubits, n_blocks, device='default.qubit'):
        """
        Initialize quantum circuit.
        
        Args:
            n_qubits: Number of qubits (log2 of classical parameters)
            n_blocks: Number of repeated blocks in the circuit
            device: PennyLane device to use
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.n_params = n_qubits * n_blocks
        
        # Initialize quantum device - ALWAYS use CPU for quantum simulation
        self.dev = qml.device(device, wires=n_qubits)
        
        # Trainable quantum parameters (phi) - keep on CPU initially
        self.phi = nn.Parameter(torch.randn(n_blocks, n_qubits, dtype=torch.float32) * 0.1)
        
        # Create QNode with torch interface
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')
        
    def _circuit(self, phi):
        """
        Define the quantum circuit architecture.
        
        Circuit structure:
        - Repeated blocks of:
          - Ry rotations on all qubits
          - Linear CNOT entanglement
        """
        for block_idx in range(self.n_blocks):
            # Apply Ry gates
            for qubit in range(self.n_qubits):
                qml.RY(phi[block_idx, qubit], wires=qubit)
            
            # Apply CNOT gates (linear connectivity)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        # Return measurement probabilities for all computational basis states
        return qml.probs(wires=range(self.n_qubits))
    
    def forward(self):
        """
        Forward pass: generate measurement probabilities.
        Always returns tensor on CPU (PennyLane requirement).
        """
        # phi is already on CPU
        # Run quantum circuit on CPU
        probs = self.qnode(self.phi)
        
        # Convert to float32 if needed
        if probs.dtype == torch.float64:
            probs = probs.float()
        
        # Return on CPU - let the model handle device transfer
        return probs
    
    def get_n_quantum_params(self):
        """Return number of trainable quantum parameters."""
        return self.n_params
