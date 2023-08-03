class DQNParameters():
    def __init__(self, action_size, lr=1e-3, mem_size=30000, replay_size=20, gamma=0.9, n_hidden=30, target_update=30):
        """
        :param action_size: number of actions
        :param lr: learning rate
        :param mem_size: size of replay memory
        :param replay_size: number of samples we use to update the model each time
        :param gamma: the discount factor
        :param n_hidden: number of neurons of hidden layer
        :param target_update: update interval of target network
        """
        super(DQNParameters, self).__init__()
        self.learning_rate = lr
        self.action_size = action_size
        self.mem_size = mem_size
        self.replay_size = replay_size
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.target_update = target_update
        self.data_path = "models_v1"
