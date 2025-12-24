from lending_environment import LendingEnvironment
from interest_rate_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import os

def train_rl_agent(num_episodes=1000):
    """Main training loop"""
    
    print("🤖 Training RL Agent for Interest Rate Optimization")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Initialize environment and agent
    env = LendingEnvironment(asset='ETH', initial_tvl=100000000)
    agent = DQNAgent()
    
    # Tracking
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, explore=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.save_experience(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            
            # Update tracking
            total_reward += reward
            
            # Next state
            state = next_state
            
        episode_rewards.append(total_reward)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
            
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"models/checkpoint_{episode+1}.pt")
            
    # Save final model
    agent.save("models/final_interest_rate_agent.pt")
    print("Training complete! Model saved to models/final_interest_rate_agent.pt")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_curve.png")
    print("Training curve saved to training_curve.png")

if __name__ == "__main__":
    train_rl_agent(num_episodes=1000)
