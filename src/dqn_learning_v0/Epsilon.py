EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99


class Epsilon():
    def __init__(self):
        super(Epsilon, self).__init__()
        self.value = EPS_START

    def update(self):
        self.value = max(self.value * EPS_DECAY, EPS_END)
