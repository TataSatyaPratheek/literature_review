import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

# Import ReplayBuffer from the other file in this directory
from .replay_buffer import ReplayBuffer

def train_agent(model, target_model, env, episodes, model_name="Model", buffer_size=10000, 
                batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                epsilon_decay=0.995, lr=0.001, target_update=10):
    """
    Generic training function for a DQN agent, now with a Target Network.
    
    *** FIX APPLIED ***
    - Accepts both `model` (policy network) and `target_model`
    - Uses `target_model` to calculate stable target Q-values
    - Periodically copies `model` weights to `target_model`
    - Increased buffer_size for more stable learning
    """

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

                # --- TARGET NETWORK FIX ---
                with torch.no_grad():
                    # Calculate next Q-values using the STABLE target_model
                    next_q = target_model(next_states_t).max(1)[0]
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)
                # --- END OF FIX ---

                # Loss and optimize
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)

        # --- TARGET NETWORK UPDATE ---
        # Update the target network by copying weights from the policy network
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        # --- END OF UPDATE ---

        # Update progress bar
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_postfix({
                'Avg10': f'{avg_reward:.1f}',
                'Eps': f'{epsilon:.3f}'
            })
        else:
            pbar.set_postfix({
                'Reward': f'{episode_reward:.1f}',
                'Eps': f'{epsilon:.3f}'
            })

    pbar.close()
    return episode_rewards