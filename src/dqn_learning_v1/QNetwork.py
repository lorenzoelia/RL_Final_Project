import torch.nn as nn
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from src.dqn_learning_v1.Epsilon import Epsilon


def train_network(env, agent, n_episode, epsilon, dqn_parameters, train_id, verbose=False, render=False, plot=False, polyak_avg=False):
    """
    Deep Q_Learning training algorithm using double DQN, with experience replay
    :param env: Gym environment
    :param agent: instance of DQNAgent
    :param n_episode: number of episodes
    :param epsilon: object for epsilon_greedy
    """
    scores = []
    avg100_scores = []
    avg_scores = []
    last100_scores = deque(maxlen=100)
    memory = deque(maxlen=dqn_parameters.mem_size)
    average100_score = 0
    solved = False
    for episode in range(n_episode):
        total_reward_episode = 0

        if polyak_avg:
            agent.soft_copy_target(0.1)
        elif episode % dqn_parameters.target_update == 0:
            agent.copy_target()

        policy = agent.gen_epsilon_greedy_policy(epsilon.value, dqn_parameters.action_size)
        state = env.reset()
        done = False

        while not done:
            if render:  # Warning: render mode may slow the training process
                env.render()
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            total_reward_episode += reward
            memory.append((state, action, next_state, reward, done))

            if done:
                scores.append(total_reward_episode)
                avg_scores.append(np.mean(scores))
                last100_scores.append(total_reward_episode)
                avg100_scores.append(np.mean(last100_scores))
                break

            agent.replay(memory, dqn_parameters.replay_size, dqn_parameters.gamma)

            state = next_state

        epsilon.update()

        average100_score = np.mean(scores[-min(100, len(scores)):])

        if verbose:
            print("episode: {}, score: {}, memory length: {}, epsilon: {}, average score: {}".format(episode,
                                                                                                     total_reward_episode,
                                                                                                     len(memory),
                                                                                                     epsilon.value,
                                                                                                     average100_score))

        if average100_score >= (env.spec.max_episode_steps - 5):
            print("Training Phase: problem is solved in {} episodes".format(episode))
            solved = True
            agent.save_model(train_id, dqn_parameters.data_path)
            break

    if not solved:
        print("Training Phase: last 100 average Score: {}".format(average100_score))

    if plot:
        print_scores(scores, avg_scores, avg100_scores, train_id)


def print_scores(scores, avg_scores, avg100_scores, train_id):
    plt.plot(range(len(scores)), scores)
    plt.plot(range(len(avg_scores)), avg_scores)
    plt.plot(range(len(avg100_scores)), avg100_scores)
    plt.legend(["Actual", "Average", "Avg100Scores"])
    plt.title("Model: " + train_id)
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.show()


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, n_hidden):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, output_size)
        self.epsilon = Epsilon()

    def forward(self, state):
        out = self.act1(self.fc1(state))
        out = self.fc2(out)
        return out
