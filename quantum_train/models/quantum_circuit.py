"""Quantum circuit implementation using TorchQuantum."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumCircuit(nn.Module):
    """Parameterized quantum circuit using TorchQuantum."""
    
    def __init__(self, n_qubits, n_blocks, device='cpu'):
        """
        Initialize quantum circuit with TorchQuantum.
        
        Args:
            n_qubits: Number of qubits
            n_blocks: Number of repeated blocks
            device: Device to run on (cpu or mps)
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.n_params = n_qubits * n_blocks
        self.device_name = device
        
        # Create quantum device (TorchQuantum's quantum state container)
        # This supports both CPU and GPU/MPS
        self.q_device = tq.QuantumDevice(
            n_wires=n_qubits, 
            bsz=1,  # Batch size 1, we'll update for each forward pass
            device=device,
        )
        
        # Create trainable rotation gates
        # Each block has n_qubits RY gates
        self.ry_gates = nn.ModuleList()
        for block in range(n_blocks):
            block_gates = nn.ModuleList()
            for qubit in range(n_qubits):
                # TorchQuantum RY gate with trainable parameter
                ry = tq.RY(has_params=True, trainable=True)
                block_gates.append(ry)
            self.ry_gates.append(block_gates)
        
        # CNOT gates (non-parameterized, applied on-the-fly)
        # Linear connectivity: 0-1, 1-2, 2-3, ..., (n-2)-(n-1)
        
    def forward(self):
        """
        Forward pass: generate measurement probabilities.
        
        Returns:
            probs: Tensor of shape (2^n_qubits,) with measurement probabilities
        """
        # Reset quantum device to |0...0> state
        self.q_device.reset_states(bsz=1)
        
        # Apply circuit: repeated blocks of RY + CNOT
        for block_idx in range(self.n_blocks):
            # Apply RY rotations
            for qubit in range(self.n_qubits):
                self.ry_gates[block_idx][qubit](self.q_device, wires=qubit)
            
            # Apply CNOT entanglement (linear connectivity)
            for qubit in range(self.n_qubits - 1):
                tqf.cnot(self.q_device, wires=[qubit, qubit + 1])
        
        # Get state vector and compute probabilities
        # TorchQuantum stores state as (batch, 2^n_qubits) complex tensor
        state = self.q_device.get_states_1d()  # Shape: (1, 2^n_qubits)
        
        # Compute probabilities: |amplitude|^2
        probs = torch.abs(state) ** 2
        probs = probs.squeeze(0)  # Remove batch dimension: (2^n_qubits,)
        
        return probs
    
    def get_n_quantum_params(self):
        """Return number of trainable quantum parameters."""
        return self.n_params
