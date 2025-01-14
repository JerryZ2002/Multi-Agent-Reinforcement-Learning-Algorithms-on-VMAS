from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from balance_scene import Balance
import numpy as np

def train_ppo(num_agents=2):
    config = (
        PPOConfig()
        .environment(env=Balance, env_config={"num_agents": num_agents})
        .rollouts(num_rollout_workers=8)
        .training(model={"fcnet_hiddens": [64, 64]}, lr=5e-4)
        .resources(num_gpus=0)
    )
    
    trainer = config.build()
    results = []
    
    for i in range(1000):
        result = trainer.train()
        avg_reward = result["episode_reward_mean"]
        results.append(avg_reward)
        print(pretty_print(result))
        
        if i % 100 == 0:
            checkpoint = trainer.save()
            print(f"Checkpoint saved at {checkpoint}")
            
    return results

if __name__ == "__main__":
    rewards = train_ppo()
    np.savetxt("ppo_rewards.txt", rewards)