"""Main experiment runner."""
import torch
from config.mnist_config import MNISTConfig
from config.cifar10_config import CIFAR10Config
from data.dataset_loader import load_dataset
from models.quantum_circuit import QuantumCircuit
from models.mapping_model import MappingModel
from models.classical_target_nn import ClassicalTargetCNN
from models.quantum_train_model import QuantumTrainModel
from training.trainer import QuantumTrainTrainer
from utils.visualization import plot_training_curves
import argparse
import torchquantum as tq

def main():
    parser = argparse.ArgumentParser(description='Quantum-Train Experiments')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--n_blocks', type=int, default=None,
                       help='Number of quantum circuit blocks')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Load configuration
    if args.dataset == 'mnist':
        config = MNISTConfig()
    else:
        config = CIFAR10Config()
    
    # Override config if specified
    if args.n_blocks is not None:
        config.n_blocks = args.n_blocks
    if args.epochs is not None:
        config.n_epochs = args.epochs
    
    # Force CPU for testing (remove this after confirming gradients work)
    config.device = torch.device("cpu")
    
    print(f"=== Quantum-Train Experiment: {config.dataset_name.upper()} ===")
    print(f"Device: {config.device}")
    print(f"Qubits: {config.n_qubits}, Blocks: {config.n_blocks}")
    
    # Load data
    train_loader, val_loader, test_loader = load_dataset(config)
    
        # Build models
    # Keep quantum circuit on CPU (PennyLane compatibility)
    quantum_circuit = QuantumCircuit(
        n_qubits=config.n_qubits,
        n_blocks=config.n_blocks,
        device=config.device  # Can now be 'mps'!
    )  # No .to(config.device)
    
    # Verify quantum parameters are trainable
    print(f"QNN parameters require grad: {next(quantum_circuit.parameters()).requires_grad}")
    print(f"QNN parameters device: {next(quantum_circuit.parameters()).device}")
    
    # Mapping model on MPS
    mapping_model = MappingModel(
        n_qubits=config.n_qubits,
        layer_sizes=config.mapping_layers[1:]  # Exclude input size
    ).to(config.device)
    
    classical_nn = ClassicalTargetCNN(config).to(config.device)
    
    # Combine into Quantum-Train model
    model = QuantumTrainModel(
        quantum_circuit=quantum_circuit,
        mapping_model=mapping_model,
        classical_nn=classical_nn,  # Used only for architecture info
        device=config.device
    )
    
    # Print parameter counts
    param_counts = model.count_total_params()
    print(f"\nParameter Counts:")
    print(f"  Quantum Circuit: {param_counts['quantum']}")
    print(f"  Mapping Model: {param_counts['mapping']}")
    print(f"  Classical Target: {param_counts['classical_target']}")
    print(f"  Total Trainable: {param_counts['trainable']}")
    print(f"  Compression Ratio: {param_counts['trainable']/param_counts['classical_target']:.2%}")
    
    # Train
    trainer = QuantumTrainTrainer(model, config, train_loader, val_loader)
    results = trainer.train()
    
    # Plot results
    plot_training_curves(results, config)
    
    print("\n=== Experiment Complete ===")

if __name__ == '__main__':
    main()
