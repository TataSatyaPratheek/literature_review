"""Quantum circuit implementation using TorchQuantum - OPTIMIZED."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumCircuit(nn.Module):
    """Parameterized quantum circuit using TorchQuantum - single parameter tensor."""
    
    def __init__(self, n_qubits, n_blocks, device='cpu'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.n_params = n_qubits * n_blocks
        self.device_name = device
        
        # Quantum device
        self.q_device = tq.QuantumDevice(
            n_wires=n_qubits, 
            bsz=1,
            device=device
        )
        
        # SINGLE trainable parameter tensor (NOT 208 separate gates!)
        # Shape: (n_blocks, n_qubits)
        self.phi = nn.Parameter(
            torch.randn(n_blocks, n_qubits, device=device) * 0.5
        )
        
    def forward(self):
        """Forward pass: generate measurement probabilities."""
        # Reset quantum device
        self.q_device.reset_states(bsz=1)
        
        # Apply circuit: RY rotations + CNOT entanglement
        for block_idx in range(self.n_blocks):
            # Apply RY rotations using functional interface
            for qubit in range(self.n_qubits):
                # Use functional RY gate with explicit parameter
                tqf.ry(
                    self.q_device,
                    wires=qubit,
                    params=self.phi[block_idx, qubit]
                )
            
            # Apply CNOT entanglement
            for qubit in range(self.n_qubits - 1):
                tqf.cnot(self.q_device, wires=[qubit, qubit + 1])
        
        # Get state and compute probabilities
        state = self.q_device.get_states_1d()
        probs = torch.abs(state) ** 2
        probs = probs.squeeze(0)
        
        return probs
    
    def get_n_quantum_params(self):
        return self.n_params