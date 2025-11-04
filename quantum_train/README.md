# Quantum-Train: Hybrid Quantum-Classical ML

Implementation of "Quantum-Train: Rethinking Hybrid Quantum-Classical Machine Learning in the Model Compression Perspective"

## Quick Start

```
# Install dependencies
pip install -r requirements.txt

# Run MNIST experiment
python run_experiment.py --dataset mnist --n_blocks 16 --epochs 50

# Run CIFAR-10 experiment
python run_experiment.py --dataset cifar10 --n_blocks 19 --epochs 100
```

## Project Structure

- `config/`: Experiment configurations for different datasets
- `models/`: Quantum circuit, mapping model, and classical NN implementations
- `training/`: Training loop and optimization
- `data/`: Dataset loading and preprocessing
- `utils/`: Visualization and helper functions
- `experiments/`: Individual experiment scripts and results

## Key Features

- Logarithmic parameter compression via quantum encoding
- M1 Mac GPU acceleration (MPS backend)
- PennyLane quantum simulation
- Modular, researcher-friendly codebase

## Results

Results are saved in `experiments/results/` including:
- Training curves
- Model checkpoints
- Performance metrics
```

