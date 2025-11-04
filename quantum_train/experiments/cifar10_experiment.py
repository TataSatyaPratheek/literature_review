"""Standalone CIFAR-10 experiment script."""
import sys
sys.path.append('..')

import torch
from config.cifar10_config import CIFAR10Config
from data.dataset_loader import load_dataset
from models.quantum_circuit import QuantumCircuit
from models.mapping_model import MappingModel
from models.classical_target_nn import ClassicalTargetCNN
from models.quantum_train_model import QuantumTrainModel
from training.trainer import QuantumTrainTrainer
from utils.visualization import plot_training_curves, plot_parameter_comparison
from utils.helpers import set_seed, create_directories

def run_cifar10_experiment(n_blocks=19, epochs=100, seed=42):
    """
    Run CIFAR-10 experiment with specified parameters.
    
    Args:
        n_blocks: Number of quantum circuit blocks
        epochs: Number of training epochs
        seed: Random seed
    """
    set_seed(seed)
    
    config = CIFAR10Config()
    config.n_blocks = n_blocks
    config.n_epochs = epochs
    
    create_directories([config.checkpoint_dir, config.results_dir])
    
    print(f"=== CIFAR-10 Quantum-Train Experiment ===")
    print(f"Qubits: {config.n_qubits}, Blocks: {config.n_blocks}")
    print(f"Device: {config.device}")
    
    train_loader, val_loader, test_loader = load_dataset(config)
    
    quantum_circuit = QuantumCircuit(
        n_qubits=config.n_qubits,
        n_blocks=config.n_blocks
    ).to(config.device)
    
    mapping_model = MappingModel(
        n_qubits=config.n_qubits,
        layer_sizes=config.mapping_layers[1:]
    ).to(config.device)
    
    classical_nn = ClassicalTargetCNN(config).to(config.device)
    
    model = QuantumTrainModel(
        quantum_circuit=quantum_circuit,
        mapping_model=mapping_model,
        classical_nn=classical_nn,
        device=config.device
    )
    
    param_counts = model.count_total_params()
    print(f"\nParameter Counts:")
    print(f"  Quantum: {param_counts['quantum']}")
    print(f"  Mapping: {param_counts['mapping']}")
    print(f"  Trainable: {param_counts['trainable']}")
    print(f"  Compression: {param_counts['trainable']/param_counts['classical_target']:.2%}")
    
    trainer = QuantumTrainTrainer(model, config, train_loader, val_loader)
    results = trainer.train()
    
    plot_training_curves(results, config)
    plot_parameter_comparison(
        param_counts['quantum'],
        param_counts['mapping'],
        param_counts['classical_target'],
        f"{config.results_dir}/cifar10_parameter_comparison.png"
    )
    
    print("\n=== CIFAR-10 Experiment Complete ===")

if __name__ == '__main__':
    run_cifar10_experiment()
