# Quantum-Classical Hybrid RL for Simple Grid Environment
# Optimized for M1 Mac (Apple Silicon) with progress tracking
# Install: pip install pennylane pennylane-lightning torch gymnasium numpy matplotlib tqdm

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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
# 1. CLASSICAL BASELINE: Standard DQN
# ============================================================================

class ClassicalDQN(nn.Module):
    """Classical Deep Q-Network baseline"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ============================================================================
# 2. BASIC QUANTUM BASELINE: Simple Quantum DQN (No encoding layers)
# ============================================================================

class BasicQuantumDQN(nn.Module):
    """Basic quantum approach - direct encoding without preprocessing"""
    def __init__(self, n_qubits, n_layers, n_actions):
        super(BasicQuantumDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum device - use lightning.qubit for Apple Silicon optimization
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            print("  → Using lightning.qubit (faster)")
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            print("  → Using default.qubit")

        # Quantum weights
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # FIXED: Use 'adjoint' instead of 'backprop' for lightning.qubit
        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            # Simple encoding - just rotation by input values
            for i in range(n_qubits):
                if i < len(inputs):
                    qml.RY(inputs[i], wires=i)

            # Variational layers
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
        # Handle batching for quantum layer
        if x.dim() == 2:
            q_out_list = []
            for i in range(x.shape[0]):
                q_out_list.append(self.qlayer(x[i]))
            q_out = torch.stack(q_out_list)
        else:
            q_out = self.qlayer(x)
        return self.fc(q_out)

# ============================================================================
# 3. HYBRID APPROACH (Paper's Method): Pre + Quantum + Post processing
# ============================================================================

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
            nn.Tanh()  # Bound features for quantum encoding
        )

        # Quantum device - use lightning.qubit for optimization
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            print("  → Using lightning.qubit (faster)")
        except:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            print("  → Using default.qubit")

        # Quantum circuit with data re-uploading
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # FIXED: Use 'adjoint' instead of 'backprop' for lightning.qubit
        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            # Data re-uploading structure
            for l in range(n_layers):
                # Encode data
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)

                # Variational layer
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], 
                           weights[l, i, 2], wires=i)

                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Final encoding
            for i in range(n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)

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
        # Encode
        encoded = self.encoder(x)

        # Handle batching for quantum layer
        if encoded.dim() == 2:
            q_out_list = []
            for i in range(encoded.shape[0]):
                q_out_list.append(self.qlayer(encoded[i]))
            q_out = torch.stack(q_out_list)
        else:
            q_out = self.qlayer(encoded)

        # Decode to actions
        return self.decoder(q_out)

# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

def train_agent(model, env, episodes, model_name="Model", buffer_size=1000, 
                batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                epsilon_decay=0.995, lr=0.001, target_update=10):
    """Generic training function with progress tracking"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    episode_rewards = []

    # Progress bar for episodes
    pbar = tqdm(range(episodes), desc=f"Training {model_name}", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for episode in pbar:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_t)
                    action = q_values.argmax().item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)

                # Compute Q values
                current_q = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = model(next_states_t).max(1)[0]
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)

                # Loss and optimize
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)

        # Update progress bar with detailed stats
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            max_reward = np.max(episode_rewards[-10:])
            pbar.set_postfix({
                'Avg10': f'{avg_reward:.1f}',
                'Max10': f'{max_reward:.0f}',
                'Eps': f'{epsilon:.3f}',
                'Steps': steps
            })
        else:
            pbar.set_postfix({
                'Reward': f'{episode_reward:.1f}',
                'Eps': f'{epsilon:.3f}',
                'Steps': steps
            })

    pbar.close()
    return episode_rewards

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
    episodes = 150  # Increased for better convergence

    results = {}
    timings = {}

    # ========================================================================
    # Experiment 1: Classical Baseline
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: CLASSICAL DQN BASELINE")
    print("=" * 60)

    classical_model = ClassicalDQN(state_dim, 16, action_dim)
    print(f"Model parameters: {sum(p.numel() for p in classical_model.parameters())}")

    start_time = time.time()
    classical_rewards = train_agent(classical_model, env, episodes, 
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

    basic_quantum_model = BasicQuantumDQN(n_qubits, n_layers, action_dim)
    print(f"Model parameters: {sum(p.numel() for p in basic_quantum_model.parameters())}")

    start_time = time.time()
    basic_quantum_rewards = train_agent(basic_quantum_model, env, episodes, 
                                       model_name="Basic Quantum", lr=0.01)
    timings['Basic Quantum'] = time.time() - start_time
    results['Basic Quantum'] = basic_quantum_rewards

    print(f"✓ Completed in {timings['Basic Quantum']:.1f}s")

    # ========================================================================
    # Experiment 3: Hybrid Approach (Paper's Method)
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: HYBRID QUANTUM-CLASSICAL (Paper's approach)")
    print("=" * 60)

    hybrid_model = HybridQuantumDQN(state_dim, n_qubits, n_layers, action_dim)
    print(f"Model parameters: {sum(p.numel() for p in hybrid_model.parameters())}")

    start_time = time.time()
    hybrid_rewards = train_agent(hybrid_model, env, episodes, 
                                 model_name="Hybrid (Paper)", lr=0.01)
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
    print("FINAL RESULTS (Last 10 episodes average)")
    print("=" * 60)
    print(f"{'Model':<25s} {'Avg Reward':>12s} {'Training Time':>15s}")
    print("-" * 60)
    for name in results.keys():
        final_avg = np.mean(results[name][-10:])
        time_str = f"{timings[name]:.1f}s"
        print(f"{name:<25s} {final_avg:>12.2f} {time_str:>15s}")

    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE INSIGHTS")
    print("=" * 60)

    classical_final = np.mean(results['Classical DQN'][-10:])
    hybrid_final = np.mean(results['Hybrid (Paper)'][-10:])
    improvement = ((hybrid_final - classical_final) / classical_final) * 100

    print(f"Hybrid vs Classical: {improvement:+.1f}% reward change")
    print(f"Quantum speedup potential: {timings['Classical DQN']/timings['Hybrid (Paper)']:.2f}x")

    env.close()
    print("\n" + "=" * 60)
    print("✓ EXPERIMENT COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()
