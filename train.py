import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from parallel_audio_rl import ParallelAudioRL
import time
import os

# Training Configuration
NUM_ENVS = 64
OBS_DIM = 128
ACTION_DIM = 16
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPOCHS = 50
STEPS_PER_EPOCH = 500
BATCH_SIZE = 64
CLIP_RATIO = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
SAVE_DIR = "models"

# Make sure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Policy network (Actor-Critic architecture)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Action noise for exploration
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

    def get_action_and_value(self, obs, deterministic=False, noise_scale=None):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            mean, value = self(obs_tensor)
            
            if deterministic:
                if noise_scale is not None:
                    # Add controlled noise to deterministic action
                    noise = torch.randn_like(mean) * noise_scale
                    action = mean + noise
                    action = torch.clamp(action, -1.0, 1.0)
                    return action.numpy(), value.squeeze(-1).numpy()
                else:
                    # Completely deterministic
                    return mean.numpy(), value.squeeze(-1).numpy()
            
            # Standard stochastic sampling
            std = torch.exp(self.log_std)
            distribution = torch.distributions.Normal(mean, std)
            action = distribution.sample()
            
            # Clip actions to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            # Log probability of the action
            log_prob = distribution.log_prob(action).sum(-1)
            
            return action.numpy(), value.squeeze(-1).numpy(), log_prob.numpy()
    
    def evaluate_actions(self, obs, actions):
        mean, value = self(obs)
        std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(mean, std)
        
        log_prob = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)
        
        return value.squeeze(-1), log_prob, entropy

# PPO training function
def train_ppo():
    # Initialize environment
    env = ParallelAudioRL(num_envs=NUM_ENVS)
    
    # Initialize policy
    policy = PolicyNetwork(OBS_DIM, ACTION_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Initial reset
    obs = env.reset()
    
    # Training stats
    episode_rewards = []
    training_steps = 0
    start_time = time.time()
    
    print("Starting training...")
    
    # Main training loop
    for epoch in range(EPOCHS):
        # Collect experience
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_dones = []
        
        # Record starting time for performance measurement
        epoch_start_time = time.time()
        
        for step in range(STEPS_PER_EPOCH):
            # Get actions from policy
            actions, values, log_probs = policy.get_action_and_value(obs)
            
            # Step environments
            next_obs, rewards, dones, _ = env.step(actions)
            
            # Record experience
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(rewards)
            batch_values.append(values)
            batch_dones.append(dones)
            
            # Update observations
            obs = next_obs
            
            # Update tracking
            training_steps += NUM_ENVS
            
            # Track rewards for reporting
            episode_rewards.extend(rewards)
        
        # Calculate steps per second
        steps_per_second = STEPS_PER_EPOCH * NUM_ENVS / (time.time() - epoch_start_time)
        
        # Compute returns and advantages
        returns = np.zeros_like(batch_rewards)
        advantages = np.zeros_like(batch_rewards)
        
        last_value = policy.get_action_and_value(obs, deterministic=True)[1]
        
        # Calculate returns (discounted sum of rewards)
        last_gae_lam = 0
        for t in reversed(range(len(batch_rewards))):
            if t == len(batch_rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - np.array(batch_dones[t])
            else:
                next_value = batch_values[t + 1]
                next_non_terminal = 1.0 - np.array(batch_dones[t])
                
            delta = batch_rewards[t] + GAMMA * next_value * next_non_terminal - batch_values[t]
            last_gae_lam = delta + GAMMA * 0.95 * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + np.array(batch_values)

        b_obs = torch.FloatTensor(np.array(batch_obs).reshape(-1, OBS_DIM))
        b_actions = torch.FloatTensor(np.array(batch_actions).reshape(-1, ACTION_DIM))
        b_log_probs = torch.FloatTensor(np.array(batch_log_probs).reshape(-1))
        b_advantages = torch.FloatTensor(advantages.reshape(-1))
        b_returns = torch.FloatTensor(returns.reshape(-1))
        
        # Policy update
        for _ in range(4):  # Multiple epochs over the same data
            # Create mini-batches
            indices = np.random.permutation(len(b_obs))
            for start_idx in range(0, len(b_obs), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(b_obs))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                mb_obs = b_obs[batch_indices]
                mb_actions = b_actions[batch_indices]
                mb_old_log_probs = b_log_probs[batch_indices]
                mb_advantages = b_advantages[batch_indices]
                mb_returns = b_returns[batch_indices]
                
                # Evaluate actions
                values, log_probs, entropy = policy.evaluate_actions(mb_obs, mb_actions)
                
                # Calculate ratio (policy / old policy)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # PPO loss
                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = ((values - mb_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                
                # Update policy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        
        # Print stats
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
        else:
            mean_reward = min_reward = max_reward = 0.0
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Steps: {training_steps} | "
              f"Mean Reward: {mean_reward:.2f} | Min/Max: {min_reward:.2f}/{max_reward:.2f} | "
              f"Steps/s: {steps_per_second:.2f}")
        
        # Reset rewards tracking
        episode_rewards = []
        
        # Save model periodically
        # if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        if epoch == EPOCHS - 1:
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'steps': training_steps,
            }, os.path.join(SAVE_DIR, f"audio_rl_model_epoch_{epoch + 1}.pt"))
            
            # Export audio example from the best-performing environment
            # env.export_audio(0, f"audio_sample_epoch_{epoch + 1}.wav")
    
    # Final cleanup
    env.close()
    
    total_time = time.time() - start_time
    print(f"Training complete! {training_steps} total steps in {total_time:.2f} seconds")
    print(f"Average steps/s: {training_steps / total_time:.2f}")

if __name__ == "__main__":
    train_ppo()