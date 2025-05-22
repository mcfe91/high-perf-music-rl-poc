import numpy as np
import torch
from parallel_audio_rl import ParallelAudioRL
from train import PolicyNetwork
import os

ACTION_DIM = 80 + 4 * 16 * 4 # + Tracks * Steps * Params (336)

def test_model(model_path="models/audio_rl_model_epoch_1000.pt"):
    # Load model
    policy = PolicyNetwork(128, ACTION_DIM)
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    # Create environment (just one instance for testing)
    env = ParallelAudioRL(num_envs=1)
    
    # Create output directory
    os.makedirs("audio_samples", exist_ok=True)
    
    # Generate samples
    num_samples = 5
    print(f"Generating {num_samples} audio samples...")
    
    # Define reward threshold and maximum steps
    reward_threshold = 5.0
    max_steps = 1000
    
    for i in range(num_samples):
        # Reset environment for each sample
        obs = env.reset()
        
        print(f"Generating sample {i+1}...")
        step = 0
        rewards = []
        best_reward = -float('inf')
        best_audio = None
        
        # Run until we reach a good reward or hit max steps
        while step < max_steps:
            # Get action from policy (deterministic mode)
            action, _ = policy.get_action_and_value(obs, deterministic=True, noise_scale=0.05)
            
            # Step environment
            obs, reward, done, _ = env.step(action)
            rewards.append(reward[0])
            
            # print(f"  Step {step+1}/{max_steps} - Reward: {reward[0]:.4f}")
            
            # Save a copy if this is the best reward so far
            if reward[0] > best_reward:
                best_reward = reward[0]
                # We'd need to save a copy of the audio buffer here in a real implementation
                # For now, just noting that this is the best step
                best_step = step
            
            # Check if we've reached the reward threshold
            if reward[0] >= reward_threshold:
                print(f"  Reached reward threshold ({reward_threshold}) at step {step+1}")
                break
                
            step += 1
        
        # Export the audio from the last step (or we could export the best one)
        filename = f"audio_samples/sample_{i+1}_reward_{reward[0]:.4f}_steps_{step+1}.wav"
        env.export_audio(0, filename)
        
        print(f"  Saved to {filename}")
        print(f"  Best reward: {best_reward:.4f} at step {best_step+1}")
        print(f"  Average reward: {np.mean(rewards):.4f}")
    
    # Clean up
    env.close()
    print("Testing complete!")

if __name__ == "__main__":
    test_model()