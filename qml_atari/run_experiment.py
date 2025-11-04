import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

# Import our new modules
from models.classical_dqn import ClassicalDQN
from models.basic_quantum_dqn import BasicQuantumDQN
from models.hybrid_quantum_dqn import HybridQuantumDQN
from agent.train import train_agent

# Apple Silicon optimization
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("QUANTUM-CLASSICAL HYBRID RL REPLICATION STUDY")
print("Paper: A quantum-classical reinforcement learning model")
print("=" * 60)

# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("\nInitializing CartPole environment...")
    env = gym.make('CartPole-v1')

    # Environment parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_qubits = 4
    n_layers = 2

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Quantum parameters: {n_qubits} qubits, {n_layers} layers\n")

    # Training parameters
    # *** FIX APPLIED ***
    # Increased episodes for a more realistic training run
    episodes = 500  

    results = {}
    timings = {}

    # ========================================================================
    # Experiment 1: Classical Baseline
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: CLASSICAL DQN BASELINE")
    print("=" * 60)

    # *** FIX APPLIED: Create model and target_model ***
    classical_model = ClassicalDQN(state_dim, 16, action_dim)
    classical_target_model = ClassicalDQN(state_dim, 16, action_dim)
    classical_target_model.load_state_dict(classical_model.state_dict())
    
    print(f"Model parameters: {sum(p.numel() for p in classical_model.parameters())}")

    start_time = time.time()
    classical_rewards = train_agent(classical_model, classical_target_model, 
                                   env, episodes, 
                                   model_name="Classical DQN", lr=0.001)
    timings['Classical DQN'] = time.time() - start_time
    results['Classical DQN'] = classical_rewards

    print(f"✓ Completed in {timings['Classical DQN']:.1f}s")

    # ========================================================================
    # Experiment 2: Basic Quantum Baseline
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: BASIC QUANTUM DQN (No preprocessing)")
    print("=" * 60)

    # *** FIX APPLIED: Create model and target_model ***
    basic_quantum_model = BasicQuantumDQN(n_qubits, n_layers, action_dim)
    basic_quantum_target_model = BasicQuantumDQN(n_qubits, n_layers, action_dim)
    basic_quantum_target_model.load_state_dict(basic_quantum_model.state_dict())
    
    print(f"Model parameters: {sum(p.numel() for p in basic_quantum_model.parameters())}")

    start_time = time.time()
    # *** FIX APPLIED: Changed lr from 0.01 to 0.001 ***
    basic_quantum_rewards = train_agent(basic_quantum_model, basic_quantum_target_model,
                                       env, episodes, 
                                       model_name="Basic Quantum", lr=0.001)
    timings['Basic Quantum'] = time.time() - start_time
    results['Basic Quantum'] = basic_quantum_rewards

    print(f"✓ Completed in {timings['Basic Quantum']:.1f}s")

    # ========================================================================
    # Experiment 3: Hybrid Approach (Paper's Method)
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: HYBRID QUANTUM-CLASSICAL (Paper's approach)")
    print("=" * 60)

    # *** FIX APPLIED: Create model and target_model ***
    hybrid_model = HybridQuantumDQN(state_dim, n_qubits, n_layers, action_dim)
    hybrid_target_model = HybridQuantumDQN(state_dim, n_qubits, n_layers, action_dim)
    hybrid_target_model.load_state_dict(hybrid_model.state_dict())
    
    print(f"Model parameters: {sum(p.numel() for p in hybrid_model.parameters())}")

    start_time = time.time()
    # *** FIX APPLIED: Changed lr from 0.01 to 0.001 ***
    hybrid_rewards = train_agent(hybrid_model, hybrid_target_model,
                                 env, episodes, 
                                 model_name="Hybrid (Paper)", lr=0.001)
    timings['Hybrid (Paper)'] = time.time() - start_time
    results['Hybrid (Paper)'] = hybrid_rewards

    print(f"✓ Completed in {timings['Hybrid (Paper)']:.1f}s")

    # ========================================================================
    # Plot Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("PLOTTING RESULTS")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Learning curves
    for name, rewards in results.items():
        # Smooth with rolling average
        window = 10
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(len(smoothed)), smoothed, label=name, linewidth=2)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward (10-episode window)', fontsize=12)
    ax1.set_title('Learning Curves: Quantum vs Classical RL', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training time comparison
    names = list(timings.keys())
    times = list(timings.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax2.bar(names, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('quantum_rl_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'quantum_rl_comparison.png'")
    plt.show()

    # Print final statistics
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Last 10 of {episodes} episodes avg)")
    print("=" * 60)
    print(f"{'Model':<25s} {'Avg Reward':>12s} {'Training Time':>15s}")
    print("-" * 60)
    for name in results.keys():
        final_avg = np.mean(results[name][-10:])
        time_str = f"{timings[name]:.1f}s"
        print(f"{name:<25s} {final_avg:>12.2f} {time_str:>15s}")

    env.close()
    print("\n" + "=" * 60)
    print("✓ EXPERIMENT COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()