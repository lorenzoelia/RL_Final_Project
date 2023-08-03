import copy
import torch
from torch.autograd import Variable
import random
import os
from src.dqn_learning_v1.QNetwork import QNetwork


class DQNAgent():
    def __init__(self, input_size, output_size, dqn_parameter):
        self.criterion = torch.nn.MSELoss()
        self.model = QNetwork(input_size, output_size, dqn_parameter.n_hidden)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), dqn_parameter.learning_rate)

    def update(self, s, y):
        """
        Update the weight of the DQN given a training sample
        :param s:  state
        :param y: target value
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        :param s: input state
        :return: Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        """
        Compute the Q values of the state for all actions using the target network
        :param s: input state
        :return: targeted Q values of the state for all actions
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay with target network
        :param memory: a list of experience
        :param replay_size: the number of samples we use to update the model each time
        :param gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)

    def soft_copy_target(self, pa_tau=0.1):
        model_params = self.model.named_parameters()
        target_params = self.model_target.named_parameters()
        dict_target_params = dict(target_params)
        for name1, param1 in model_params:
            if name1 in dict_target_params:
                dict_target_params[name1].data.copy_(
                    pa_tau * param1.data + (1 - pa_tau) * dict_target_params[name1].data)

        self.model_target.load_state_dict(dict_target_params)

    def copy_target(self):
        self.model_target.load_state_dict((self.model.state_dict()))

    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        """
        Epsilon greedy policy for action selection
        :param epsilon: epsilon variable
        :param n_action: total number of actions available
        """
        def policy_function(state):
            if random.random() < epsilon:
                return random.randint(0, n_action - 1)
            else:
                q_values = self.predict(state)
                return torch.argmax(q_values).item()
        return policy_function

    def gen_greedy_policy(self, epsilon, n_action):
        """
        Epsilon greedy policy for action selection
        :param epsilon: epsilon variable
        :param n_action: total number of actions available
        """
        def policy_function(state):
            q_values = self.predict(state)
            return torch.argmax(q_values).item()
        return policy_function

    def save_model(self, model_id, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        torch.save(self.model.state_dict(), data_path + "/dqn_model_" + model_id + "_v1.pt")

    def load_model(self, model_id, data_path):
        self.model.load_state_dict(torch.load(data_path + "/dqn_model_" + model_id + "_v1.pt"))
