"""Quantum circuit - EXACT implementation from paper."""
import torch
import torch.nn as nn
import torchquantum as tq

class QuantumCircuit(nn.Module):
    """Quantum circuit using U3+CU3 gates (paper's actual implementation)."""
    
    def __init__(self, n_qubits, n_blocks, n_classical_params, device='cpu'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.n_classical_params = n_classical_params
        self.device_name = device
        
        # TorchQuantum layers
        self.u3_layers = tq.QuantumModuleList()
        self.cu3_layers = tq.QuantumModuleList()
        
        for _ in range(n_blocks):
            self.u3_layers.append(
                tq.Op1QAllLayer(
                    op=tq.U3,
                    n_wires=n_qubits,
                    has_params=True,
                    trainable=True
                )
            )
            self.cu3_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=n_qubits,
                    has_params=True,
                    trainable=True,
                    circular=True
                )
            )
    
    def forward(self):
        """Forward pass - generates scaled probabilities."""
        # Create quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=1,
            device=next(self.parameters()).device
        )
        
        # Apply U3 and CU3 layers
        for k in range(self.n_blocks):
            self.u3_layers[k](qdev)
            self.cu3_layers[k](qdev)
        
        # Get state and compute probabilities
        state_mag = qdev.get_states_1d().abs()[0]
        x = torch.abs(state_mag) ** 2
        x = x.reshape(2**self.n_qubits, 1)
        
        # CRITICAL: Apply scaling transformation from paper
        easy_scale_coeff = 2 ** (self.n_qubits - 1)
        gamma = 0.1
        beta = 0.8
        alpha = 0.3
        x = (beta * torch.tanh(gamma * easy_scale_coeff * x)) ** alpha
        
        # CRITICAL: Mean centering
        x = x - torch.mean(x)
        
        return x
    
    def get_n_quantum_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters())