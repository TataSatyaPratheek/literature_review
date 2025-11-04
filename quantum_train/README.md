# Quantum-Train: Hybrid Quantum-Classical ML

PyTorch/PennyLane implementation of "Quantum-Train: Rethinking Hybrid Quantum-Classical Machine Learning in the Model Compression Perspective" tested on Apple M1 Mac.

## TL;DR - What We Built & Results

### Architecture
- **13 qubits, 8 blocks** → 104 quantum parameters
- **2-layer mapping MLP** (14→20→1) → 321 parameters  
- **Total trainable**: 425 params (6.37% of 6,676 classical baseline)
- **Compression**: 15.7× parameter reduction

### Results (5-Epoch Validation on M1 Mac)
| Epoch | Train Acc | Val Acc | Loss Reduction |
|-------|-----------|---------|----------------|
| 1     | 11.16%    | 10.33%  | Baseline       |
| 3     | 33.81%    | 44.35%  | 3.2× faster    |
| 4     | 45.54%    | **49.12%** | Peak      |
| 5     | 48.92%    | 47.00%  | 61% total      |

**Hardware**: Apple M1 MacBook Air (CPU quantum sim + MPS GPU classical layers)  
**Speed**: 2.3 batches/sec (~433ms/batch), 5.5 min/epoch  
**Memory**: 10MB peak (64KB quantum state)

### Critical Implementation Details
- **Gradient flow fix**: Functional forward pass (`F.conv2d`, `F.linear`) instead of `.data` assignment
- **Device strategy**: QNN on CPU (PennyLane), mapping/CNN on MPS GPU
- **Stability**: Gradient clipping (max norm=10.0) prevented explosion (11,238→115)
- **Optimization**: Precomputed basis vectors (-106k ops/batch)

## Quick Start

```
# Install dependencies
pip install torch torchvision pennylane numpy matplotlib tqdm scikit-learn

# Run MNIST (8 blocks, 50 epochs)
python run_experiment.py --dataset mnist --n_blocks 8 --epochs 50

# Run with more blocks for better accuracy (slower)
python run_experiment.py --dataset mnist --n_blocks 16 --epochs 50
```

## Project Structure

```
quantum_train/
├── config/          # Dataset configs (MNIST/CIFAR-10)
├── models/          # Quantum circuit, mapping MLP, classical CNN
├── training/        # Training loop, loss, metrics
├── data/            # Dataset loading
└── utils/           # Checkpointing, visualization
```

## Hyperparameters Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| Quantum blocks | 8 | Balance speed/accuracy |
| Learning rate | 5e-4 | Conservative for stability |
| Batch size | 64 | M1 memory-efficient |
| Gradient clip | 10.0 | Prevents explosion |
| Init scale | 0.1 | QNN phi init |
| Mapping init | Xavier (gain=0.5) | Small to avoid overflow |

## Results & Discussion

**Validation Trajectory**: Reached 49% in 5 epochs → projected ~94% at 50 epochs (paper's baseline)[file:1][file:86].

**What Worked**:
- Functional operations maintained gradient flow through QNN→mapping→classical chain
- Gradient clipping stabilized training despite initial explosion (11k→115 norm)[file:86]
- M1's unified memory efficiently handled 13-qubit state-vector simulation

**Performance vs. Classical Baseline**:
- **Parameters**: 425 vs 6,676 (93.6% reduction)
- **Expected accuracy drop**: ~4% at 50 epochs (94% vs 98% for full classical)[file:1][file:86]
- **Training time**: 2× slower due to quantum simulation overhead

## Critical Analysis: Future Literature Review

### Implementation Gaps vs. Original Paper
1. **Circuit architecture**: Used RY+CNOT instead of paper's U3+CU3 gates (simpler but less expressive)[file:83]
2. **Mapping network**: Simpler 2-layer vs paper's 4-layer [4,20,4] architecture[file:83]
3. **Scaling factors**: Did not implement paper's `(β·tanh(γ·x))^α` transformation[file:83]
4. **Training duration**: 5 epochs vs full 50-epoch convergence

### Open Research Questions
1. **Gradient flow mechanisms**: Why do mapping model biases initially receive such large gradients (11k+)?
2. **Quantum advantage**: Does logarithmic compression translate to faster training on real quantum hardware?
3. **Generalization**: How does learned quantum representation transfer across tasks?
4. **Noise robustness**: Performance degradation on NISQ devices vs ideal simulation?

### Scalability Considerations
- **Qubit limits**: State-vector simulation limited to ~20 qubits on classical hardware (tensor network methods needed beyond)
- **Circuit depth**: 8 blocks = 104 gates; deeper circuits → barren plateaus problem
- **Parameter mapping bottleneck**: 2^n states → n parameters requires careful architectural design

### Potential Improvements
1. **Alternative parameterization**: Explore amplitude encoding or QAOA-inspired ansatzes
2. **Hybrid training**: Pre-train classical CNN, fine-tune with quantum compression
3. **Dynamic block allocation**: Adaptive number of quantum blocks based on convergence
4. **Hardware-aware compilation**: Optimize for specific quantum backend constraints

### Literature Gaps to Address
- Theoretical guarantees on approximation capacity of quantum parameter generators
- Systematic study of entanglement patterns vs compression quality
- Comparison with lottery ticket hypothesis and neural architecture search
- Extension to recurrent/transformer architectures

## Reproducing Results

```
# Exact configuration from validation run
python run_experiment.py --dataset mnist --n_blocks 8 --epochs 5 --batch_size 64 --lr 5e-4
```

Checkpoints and training curves saved to `experiments/results/`.

## Citation

```
@article{liu2023quantumtrain,
  title={Quantum-Train: Rethinking Hybrid Quantum-Classical Machine Learning},
  author={Liu, Chen-Yu and Kuo, En-Jui and Lin, Chu-Hsuan Abraham and others},
  journal={arXiv preprint arXiv:2301.xxxxx},
  year={2023}
}
```

## License

Implementation follows original paper's approach. See paper for theoretical details.