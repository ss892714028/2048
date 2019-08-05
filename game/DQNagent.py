import numpy as np
from Network import Network
from collections import deque


class DQNAgent:

    def __init__(self, epsilon, gamma, epsilon_decay,epsilon_min,learning_rate):
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=5000)

    def store_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def experience_replay(self):
        pass

    def get_weight(self):
        return
