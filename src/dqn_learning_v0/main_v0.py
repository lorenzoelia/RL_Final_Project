from collections import deque
import gym
import torch
import random
import numpy as np

from src.dqn_learning_v0.DQNAgent import DQNAgent
from src.dqn_learning_v0.DQNParameters import DQNParameters
from src.dqn_learning_v0.Epsilon import Epsilon
from src.dqn_learning_v0.QNetwork import train_network, print_scores


def train_model():
    env = gym.envs.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episode = 600
    n_hidden_options = [30, 40, 50]
    lr_options = [0.001]
    replay_size_options = [20, 25]
    target_update_options = [30, 35]

    for n_hidden in n_hidden_options:
        for lr in lr_options:
            for replay_size in replay_size_options:
                for target_update in target_update_options:
                    train_id = str(n_hidden) + "_" + str(lr) + "_" + str(replay_size) + "_" + str(target_update)
                    print("Network params: hidden layers: {}, learning rate: {}, replay_size: {}, target_value {}".
                          format(n_hidden, lr, replay_size, target_update))
                    env.seed(1)
                    random.seed(1)
                    torch.manual_seed(1)
                    epsilon = Epsilon()
                    dqn_param = DQNParameters(action_size, lr=lr, mem_size=30000, replay_size=replay_size, gamma=0.9,
                                              n_hidden=n_hidden, target_update=target_update)
                    agent = DQNAgent(state_size, action_size, dqn_param)
                    train_network(env, agent, n_episode, epsilon, dqn_param, train_id, verbose=True, render=False)
    env.close()


def _test_model(model_id, n_episode, verbose):
    env = gym.envs.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    epsilon = Epsilon()
    dqn_param = DQNParameters(action_size, n_hidden=50)
    agent = DQNAgent(state_size, action_size, dqn_param)
    agent.load_model(model_id, dqn_param.data_path)
    scores = []
    avg100_scores = []
    avg_scores = []
    last100_scores = deque(maxlen=100)
    average100_score = 0
    solved = False
    for episode in range(n_episode):
        total_reward_episode = 0
        policy = agent.gen_greedy_policy(epsilon.value, dqn_param.action_size)
        state = env.reset()
        is_done = False

        while not is_done:
            env.render()
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode += reward
            if is_done:
                scores.append(total_reward_episode)
                avg_scores.append(np.mean(scores))
                last100_scores.append(total_reward_episode)
                avg100_scores.append(np.mean(last100_scores))
                break

            state = next_state

        average100_score = np.mean(scores[-min(100, len(scores)):])

        if verbose:
            print("episode: {}, score: {}, average score: {}".format(episode, total_reward_episode, average100_score))

    if average100_score >= (env.spec.max_episode_steps - 5):
        print("Test phase: problem is solved in {} episodes.".format(episode))
        solved = True

    if not solved:
        print("Test phase: last 100 average score: {}".format(average100_score))

    print_scores(scores, avg_scores, avg100_scores, model_id)
    env.close()


if __name__ == "__main__":
    random.seed(0)
    test = False
    if not test:
        train_model()
    else:
        model_id = "50_0.001_25_35"
        _test_model(model_id, 100, True)
