import numpy as np
from Network import Network
from collections import deque
from game import Game
import random
from keras.layers import Dense, Layer, Conv2D, MaxPool2D
import keras

class DQNAgent:

    def __init__(self, state_size=[4,4], epsilon = 1, gamma = 0.995, epsilon_decay = 0.995,
                 epsilon_min = 0.01, learning_rate=0.001, action_space=4):
        self.epsilon = epsilon
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.model = self.build()

        self.memory = deque(maxlen=5000)

    def build(self):
        model = keras.models.Sequential()
        model.add(Conv2D(32, (2, 2), padding='same', activation='relu',data_format='channels_last', input_shape=[4,4,1]))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dense(self.action_space, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        return model

    def store_memory(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = np.reshape(state, [1,4,4,1])
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def experience_replay(self, batch_size):
        try:
            mini_batch = random.sample(self.memory, batch_size)
        except:
            mini_batch = random.sample(self.memory, 32)

        for state, action, reward, next_state, game_over in mini_batch:
            target = reward
            if not game_over:
                state = np.reshape(state, [1,4,4,1])
                next_state = np.reshape(next_state, [1,4,4,1])

                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0][0][0])

                target_f = self.model.predict(state)

                target_f[0][0][0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    episodes = 5000
    agent = DQNAgent()
    stuck_each_epoch = []
    for i in range(episodes):
        g = Game()
        g.fill_cell()
        state = g.board
        stuck_counter = 0
        stuck = 0
        while True:
            action = agent.act(state)

            if not g.moved:
                stuck_counter+=1
            if stuck_counter>=2:
                g.main_loop(random.randint(0,3))
                stuck+=1
                stuck_counter = 0
            g.main_loop(action)
            next_state = g.board
            max_position = np.argmax(next_state)

            if max_position == np.array([0,3,12,15]).any():
                r_position = 0.7
            else:
                r_position = -0.5
            if g.moved:
                r_moved = 0.4
            else:
                r_moved = -0.5
            reward = g.empty / 16 + (max(g.joinable))/2 + r_position + r_moved

            game_over = g.game_over

            agent.store_memory(state, action, reward, next_state, game_over)
            state = next_state
            if game_over:
                print("episode: {}/{}, score: {}, max_cell: {}"
                      .format(i, episodes, g.score, np.max(g.board)))
                break
        stuck_each_epoch.append(stuck)
        if i%50 == 0:
            print(stuck_each_epoch)
        agent.experience_replay(256)

