import matplotlib.pyplot as plt
import numpy as np
from train_ppo import train_ppo

def study_agent_effect():
    results = {}
    agent_counts = [2, 4, 6, 8]
    
    for count in agent_counts:
        print(f"Training with {count} agents...")
        rewards = train_ppo(num_agents=count)
        results[count] = rewards
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    for count, rewards in results.items():
        plt.plot(rewards, label=f'{count} Agents')
    
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Reward')
    plt.title('Effect of Number of Agents on Performance')
    plt.legend()
    plt.savefig('agents_effect.png')  # Save the plot as an image file
    plt.show()

if __name__ == "__main__":
    study_agent_effect()