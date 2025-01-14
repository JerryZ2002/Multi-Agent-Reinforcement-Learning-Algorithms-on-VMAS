from collections import deque
import random
import numpy as np
from ray.rllib.algorithms.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer


class ImprovedIPPO(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        # Initialize the superclass with provided parameters.
        super().__init__(config=config, env=env, logger_creator=logger_creator)

        # Initialize replay memory to store experiences.
        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step using experience replay."""
        if len(self.memory) < self.config["train_batch_size"]:
            return None

        minibatch = random.sample(self.memory, self.config["train_batch_size"])
        batch = self._convert_to_sample_batch(minibatch)

        # Perform one training step using the sampled batch.
        result = super().train()
        custom_loss = self._train_on_batch(batch)
        result.update({"custom_loss": custom_loss})
        return result

    def _convert_to_sample_batch(self, minibatch):
        """Convert a list of transitions into a SampleBatch object."""
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return SampleBatch({
            SampleBatch.CUR_OBS: np.array(states),
            SampleBatch.ACTIONS: np.array(actions),
            SampleBatch.REWARDS: np.array(rewards),
            SampleBatch.NEXT_OBS: np.array(next_states),
            SampleBatch.DONES: np.array(dones),
        })

    def _train_on_batch(self, batch):
        """Train on a single batch of data and return the loss."""
        policy = self.workers.local_worker().policy_map['default_policy']
        postprocessed_batch = policy.postprocess_trajectory(batch)
        loss_out = policy.learn_on_batch(postprocessed_batch)
        return loss_out['total_loss']



class DummyEnv:
    def reset(self):
        return np.zeros(4)

    def step(self, action):
        next_state = np.random.rand(4)
        reward = np.random.rand()
        done = False if np.random.rand() > 0.05 else True
        return next_state, reward, done, {}



if __name__ == "__main__":

    config = {
        "framework": "torch",
        "env": DummyEnv,
        "num_workers": 0,
        "train_batch_size": 64,

    }


    trainer = ImprovedIPPO(config=config)


    env = DummyEnv()
    state = env.reset()
    total_rewards = []

    for episode in range(10):
        episode_reward = 0
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            next_state, reward, done, _ = env.step(action)
            trainer.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward


            if len(trainer.memory) >= config["train_batch_size"]:
                trainer.train_step()

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")


        trainer.train_step()

    print("Training completed.")