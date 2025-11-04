import pennylane as qml
import torch.nn as nn
import torch
import numpy as np

class HybridQuantumDQN(nn.Module):
    """Hybrid approach from paper: Classical encoder -> Quantum -> Classical decoder"""
    def __init__(self, input_dim, n_qubits, n_layers, n_actions):
        super(HybridQuantumDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical pre-processing (compression layer)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, n_qubits),
            nn.Tanh()
        )

        # Quantum device
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            print("  → Using lightning.qubit (faster)")
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            print("  → Using default.qubit")

        # Quantum circuit with data re-uploading
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            # Data re-uploading structure
            for l in range(n_layers):
                # *** FIX APPLIED: Use AngleEmbedding for batch-compatible encoding ***
                qml.AngleEmbedding(inputs * np.pi, wires=range(self.n_qubits), rotation='Y')

                # Variational layer
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], 
                           weights[l, i, 2], wires=i)

                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Final encoding layer
            # *** FIX APPLIED: Use AngleEmbedding here as well ***
            qml.AngleEmbedding(inputs * np.pi, wires=range(self.n_qubits), rotation='Y')

            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical post-processing (action decoder)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        q_out = self.qlayer(encoded)
        return self.decoder(q_out)