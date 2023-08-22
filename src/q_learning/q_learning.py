import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt


_DEBUG = True


def bucketize_state_values(state_value):
    """Discretizes continuous values into fixed buckets"""
    bucket_indices = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:  # violates lower bound
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:  # violates upper bound
            bucket_index = no_buckets[i] - 1  # put in the last bucket
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * state_value[i] - offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)


def select_explore_rate(x):
    """Change the exploration rate over time"""
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()  # explore
    else:
        action = np.argmax(q_value_table[state_value])  # exploit
    return action


def select_learning_rate(x):
    """Change learning rate"""
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


env = gym.make("CartPole-v0")

no_buckets = (1, 1, 6, 3)
no_actions = env.action_space.n

state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_value_bounds[1] = (-0.5, 0.5)
state_value_bounds[3] = (-math.radians(50), math.radians(50))

print(state_value_bounds)
print(len(state_value_bounds))
print(np.shape(state_value_bounds))
print(state_value_bounds[0][0])

action_index = len(no_buckets)

# define q_value table
q_value_table = np.zeros(no_buckets + (no_actions,))

# user-defined parameters
min_explore_rate = 0.1
min_learning_rate = 0.1
max_episodes = 1000
max_time_steps = 250
streak_to_end = 100
solved_time = 199
discount = 0.99
no_streaks = 0

explore_rate_per_episode = []
learning_rate_per_episode = []
time_per_episode = []
avgtime_per_episode = []

# Training
random.seed(1)
total_time = 0
for episode_no in range(max_episodes):
    explore_rate = select_explore_rate(episode_no)
    learning_rate = select_learning_rate(episode_no)

    learning_rate_per_episode.append(learning_rate)
    explore_rate_per_episode.append(explore_rate)

    # Reset the environment while starting a new episode
    observation = env.reset()

    start_state_value = bucketize_state_values(observation)
    previous_state_value = start_state_value

    done = False
    time_step = 0

    while not done:
        action = select_action(previous_state_value, explore_rate)
        observation, reward_gain, done, info = env.step(action)
        state_value = bucketize_state_values(observation)
        best_q_value = np.max(q_value_table[state_value])

        # Update q_value table
        q_value_table[previous_state_value][action] += learning_rate * (reward_gain + discount * best_q_value -
                                                                        q_value_table[previous_state_value][action])

        previous_state_value = state_value

        if episode_no % 100 == 0 and _DEBUG == True:
            print("---------------------------------------------------------------------------------------------------")
            print("Episode number: {}".format(episode_no))
            print("Time step: {}".format(time_step))
            print("Previous State Value: {}".format(previous_state_value))
            print("Selected Action: {}".format(action))
            print("Current State: {}".format(str(state_value)))
            print("Reward Obtained: {}".format(reward_gain))
            print("Best Q Value: {}".format(best_q_value))
            print("Learning Rate: {}".format(learning_rate))
            print("Exploration Rate: {}".format(explore_rate))

        time_step += 1

    if time_step >= solved_time:
        no_streaks += 1
    else:
        no_streaks = 0

    if no_streaks > streak_to_end:
        print("CartPole problem is solved after {} episodes.".format(episode_no))
        break

    # Data log
    if episode_no % 100 == 0:
        print("Episode {} finished after {} time steps".format(episode_no, time_step))
    time_per_episode.append(time_step)
    total_time += time_step
    avgtime_per_episode.append(total_time / (episode_no + 1))

# Testing
scores = []
avg_scores = []
for episode in range(streak_to_end):
    # Reset the environment while starting a new episode
    state = env.reset()
    done = False
    time_step = 0
    total_reward = 0

    while not done:
        # Choose an action based on the Q-value table
        state_value = bucketize_state_values(state)

        action = np.argmax(q_value_table[state_value])

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            scores.append(total_reward)
            avg_scores = np.mean(scores)

        env.render()  # Display the environment

    print("episode: {}, score: {}, average score: {}".format(episode, total_reward, avg_scores))

env.close()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(range(len(time_per_episode)), time_per_episode)
ax1.plot(range(len(avgtime_per_episode)), avgtime_per_episode)
ax1.set_ylabel('Time per Episode')

ax2.plot(range(len(learning_rate_per_episode)), learning_rate_per_episode)
ax2.set_ylabel('Learning Rate')
ax2.set_xlabel('Num. Episodes')

plt.subplots_adjust(hspace=0.2)
plt.show()