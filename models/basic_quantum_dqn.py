import pennylane as qml
import torch.nn as nn
import torch

class BasicQuantumDQN(nn.Module):
    """Basic quantum approach - direct encoding without preprocessing"""
    def __init__(self, n_qubits, n_layers, n_actions):
        super(BasicQuantumDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum device
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            print("  → Using lightning.qubit (faster)")
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            print("  → Using default.qubit")

        # Quantum weights
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            # *** FIX APPLIED: Use AngleEmbedding for batch-compatible encoding ***
            # This single line replaces the old for-loop. It correctly handles
            # inputs with shape (batch_size, n_qubits).
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')

            # Variational layers (this part was already fine)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], 
                           weights[l, i, 2], wires=i)
                # Entangling
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measure
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Simple post-processing
        self.fc = nn.Linear(n_qubits, n_actions)

    def forward(self, x):
        # This is now correct because the QNode handles the batch
        q_out = self.qlayer(x)
        return self.fc(q_out)