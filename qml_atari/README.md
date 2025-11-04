## QML Atari — TLDR Reproducibility Guide

**TLDR:** This repo replicates a quantum-classical hybrid RL paper on CartPole-v1. Three models: Classical DQN (baseline), Basic Quantum DQN (direct encoding), Hybrid Quantum DQN (encoder + quantum + decoder). Experiment: 500 episodes, lr=0.001, seeds=42. Results show hybrid matches classical performance but at ~172x slower speed.

### Models (Exact Params from Code)

- **ClassicalDQN**: `ClassicalDQN(input_dim=4, hidden_dim=16, output_dim=2)` — 2 hidden layers (16 units each), ReLU, no quantum.
- **BasicQuantumDQN**: `BasicQuantumDQN(n_qubits=4, n_layers=2, n_actions=2)` — Direct AngleEmbedding, Rot gates (3 params each), CZ entangling, Z measurements.
- **HybridQuantumDQN**: `HybridQuantumDQN(input_dim=4, n_qubits=4, n_layers=2, n_actions=2)` — Encoder (4→8→4, Tanh), data re-uploading (AngleEmbedding * π), Rot+CZ, Decoder (4→8→2).

### Experiment Setup (from run_experiment.py)

- **Environment**: CartPole-v1 (state_dim=4, action_dim=2).
- **Training**: 500 episodes, lr=0.001, target network updates, epsilon decay (from train.py).
- **Device**: MPS (Apple Silicon) if available, else CUDA, else CPU.
- **Seeds**: torch.manual_seed(42), np.random.seed(42), random.seed(42).
- **Metrics**: Episode rewards, training time, final avg reward (last 10 episodes).

### Reproducibility Notes

- **Dependencies**: PyTorch, PennyLane (with lightning.qubit fallback), Gymnasium, tqdm, matplotlib.
- **Run**: `python run_experiment.py` — outputs plot (quantum_rl_comparison.png) and console stats.
- **Key Constants**: n_qubits=4, n_layers=2, episodes=500, lr=0.001, hidden_dim=16 (classical), encoder/decoder width=8 (hybrid).
- **Quantum Config**: interface='torch', diff_method='adjoint', weight_shapes=(n_layers, n_qubits, 3).

### Research Methodology

- **Hypothesis**: Hybrid quantum-classical can match classical RL performance on low-dim tasks like CartPole.
- **Baselines**: Classical DQN (fast, reliable), Basic Quantum (naive encoding), Hybrid (paper's method with data re-uploading).
- **Evaluation**: Reward curves, training time, parameter counts.
- **Findings**: Hybrid achieves ~107 reward (vs classical 116), but 172x slower; basic quantum fails (47 reward, 47x slower).

### Precise Code Params (TLDR)

- **QNode Config**: `interface='torch'`, `diff_method='adjoint'`, `weight_shapes={"weights": (n_layers, n_qubits, 3)}`.
- **Devices**: Try `lightning.qubit`, fallback `default.qubit`.
- **Gates/Embeddings**: `AngleEmbedding(rotation='Y')` (basic: direct, hybrid: *π scaled), `Rot(3 params)`, `CZ` entangling, `expval(PauliZ)`.
- **Architectures**: Classical: 2x Linear(hidden_dim=16) + ReLU. Basic: Quantum + Linear(n_qubits→n_actions). Hybrid: Encoder(4→8→4, Tanh) + Quantum + Decoder(4→8→2).
- **Shapes**: Basic expects `(batch_size, n_qubits)`, Hybrid `(batch_size, input_dim)` → `(batch_size, n_qubits)` → `(batch_size, n_actions)`.
- **Constants**: Encoder/decoder width=8, Rot params=3, no defaults in constructors.

### Results Interpretation & Lit Review Plan (TLDR)

- **Results Breakdown**: Classical: 116 reward, 31s. Basic Quantum: 47 reward, 1487s (47x slower, worse). Hybrid: 107 reward, 5409s (172x slower, matches classical).
- **Why Hybrid Slower?**: Data re-uploading (encoding per layer) boosts expressivity but explodes simulation cost.
- **Conclusion**: Hybrid matches classical on CartPole but impractical on simulators.
- **Lit Review Questions**:
  1. **Performance**: Why hybrid worked? Search "data encoding in QML", "expressivity of PQCs", "data re-uploading", "quantum kernels".
  2. **Speed/Advantage**: Why slow? Search "barren plateaus in QNNs", "trainability of QNNs", "quantum advantage in ML".
  3. **Scaling**: Beyond CartPole? Search "QCNN", "hybrid QML for vision", "dim reduction in QML", "quantum RL for Atari".
- **Next Steps**: Review papers on barren plateaus/data encoding; start with "A review of data encoding for quantum machine learning".
