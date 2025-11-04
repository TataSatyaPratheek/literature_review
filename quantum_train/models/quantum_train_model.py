"""Complete Quantum-Train model integrating all components."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumTrainModel(nn.Module):
    """
    Complete Quantum-Train framework with proper gradient flow.
    Uses functional approach - NO parameter assignment during forward pass.
    """
    
    def __init__(self, quantum_circuit, mapping_model, classical_nn, device):
        """Initialize with quantum on CPU, mapping on MPS."""
        super().__init__()
        self.quantum_circuit = quantum_circuit  # Stays on CPU
        self.mapping_model = mapping_model  # On MPS
        self.device = device
        
        # Store classical NN architecture info (NOT the module itself)
        self.classical_nn_config = classical_nn.config
        self.n_classical_params = classical_nn.count_parameters()
        
        # Pre-compute basis vectors (constant) - on mapping model's device
        self.basis_vectors = mapping_model.create_basis_vectors(device)
        
        # Store layer shapes for functional computation
        self.layer_shapes = []
        self.layer_names = []
        for name, param in classical_nn.named_parameters():
            self.layer_shapes.append(param.shape)
            self.layer_names.append(name)
        
        print(f"Classical NN needs: {self.n_classical_params} parameters")
        print(f"Quantum circuit generates: {2 ** quantum_circuit.n_qubits} values")
        
    def forward(self, x):
        """
        Complete forward pass with proper device handling.
        QNN and mapping on same device - no transfer needed!
        """
        # 1. Generate quantum measurement probabilities
        # Now probs is ALREADY on correct device (mps or cpu)
        probs = self.quantum_circuit()
        
        # 2. Map probabilities to classical parameters
        # No .to() needed - everything on same device!
        theta_full = self.mapping_model(self.basis_vectors, probs)
        
        # 3. Trim to exact number of parameters needed
        theta = theta_full[:self.n_classical_params]
        
        # 4. Split theta into layer parameters
        offset = 0
        params = {}
        for name, shape in zip(self.layer_names, self.layer_shapes):
            numel = torch.prod(torch.tensor(shape)).item()
            param = theta[offset:offset + numel].view(shape)
            
            params[name] = param
            offset += numel
        
        # 5. Functional forward pass through CNN
        output = self._functional_cnn_forward(x, params)
        
        return output
    
    def _functional_cnn_forward(self, x, params):
        """
        Functional CNN forward pass with explicit parameter usage.
        """
        # Conv1 + ReLU + Pool
        conv1_weight = params['conv1.weight']
        conv1_bias = params['conv1.bias']
        x = F.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=0)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Conv2 + ReLU + Pool
        conv2_weight = params['conv2.weight']
        conv2_bias = params['conv2.bias']
        x = F.conv2d(x, conv2_weight, conv2_bias, stride=1, padding=0)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC1 + ReLU
        fc1_weight = params['fc1.weight']
        fc1_bias = params['fc1.bias']
        x = F.linear(x, fc1_weight, fc1_bias)
        x = F.relu(x)
        
        # FC2 (output logits)
        fc2_weight = params['fc2.weight']
        fc2_bias = params['fc2.bias']
        x = F.linear(x, fc2_weight, fc2_bias)
        
        return x
    
    def count_total_params(self):
        """Count trainable parameters (quantum + mapping only)."""
        qnn_params = self.quantum_circuit.get_n_quantum_params()
        mapping_params = sum(p.numel() for p in self.mapping_model.parameters())
        
        return {
            'quantum': qnn_params,
            'mapping': mapping_params,
            'classical_target': self.n_classical_params,
            'trainable': qnn_params + mapping_params
        }
