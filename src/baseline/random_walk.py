import gym
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
seed = 0
np.random.seed(seed)

env = gym.make('CartPole-v1')
env.seed(seed)

num_episodes = 600
max_steps_per_episode = 200

episode_rewards = []
average_rewards = []

# Random walk baseline
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        action = env.action_space.sample()  # Choose a random action

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    average_rewards.append(np.mean(episode_rewards))
    print("episode: {}, score: {}, average score: {}".format(episode, total_reward, np.mean(episode_rewards)))

env.close()


# Plotting the rewards over episodes
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(average_rewards, label="Average Rewards")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Random Walk Baseline for CartPole')
plt.legend()
plt.show()
