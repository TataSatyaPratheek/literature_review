"""Visualization utilities for training results."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(results, config, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        results: Dictionary with training history
        config: Configuration object
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, results['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, results['val_accuracies'], 'g-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{config.results_dir}/{config.dataset_name}_training_curves.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_parameter_comparison(quantum_params, mapping_params, classical_params, save_path):
    """
    Plot parameter count comparison.
    
    Args:
        quantum_params: Number of quantum circuit parameters
        mapping_params: Number of mapping model parameters
        classical_params: Number of classical model parameters
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Quantum\nCircuit', 'Mapping\nModel', 'Total\nTrainable', 'Classical\nTarget']
    counts = [quantum_params, mapping_params, quantum_params + mapping_params, classical_params]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Parameter Count', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    compression_ratio = (quantum_params + mapping_params) / classical_params * 100
    ax.text(0.5, 0.95, f'Compression Ratio: {compression_ratio:.1f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parameter comparison saved to: {save_path}")
    plt.close()

def plot_quantum_circuit_probabilities(probs, n_qubits, save_path):
    """
    Plot quantum measurement probability distribution.
    
    Args:
        probs: Probability distribution from quantum circuit
        n_qubits: Number of qubits
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    basis_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    probs_np = probs.detach().cpu().numpy()
    
    ax.bar(range(len(probs_np)), probs_np, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Basis State', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Quantum Measurement Probabilities ({n_qubits} qubits)', fontsize=14, fontweight='bold')
    
    if len(basis_states) <= 32:
        ax.set_xticks(range(len(basis_states)))
        ax.set_xticklabels(basis_states, rotation=90, fontsize=8)
    else:
        ax.set_xlabel('Basis State Index', fontsize=12)
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Probability distribution saved to: {save_path}")
    plt.close()
