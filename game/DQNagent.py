import numpy as np
from Network import Network
from collections import deque
from game import Game
import random
from keras.layers import Dense, Layer, Conv2D, MaxPool2D,Flatten,Dropout,BatchNormalization, Dropout
import keras
from keras.optimizers import SGD,Adam
from sklearn.preprocessing import StandardScaler
import keras.backend as K
class DQNAgent:

    def __init__(self, state_size=[1,4,4,1], epsilon = 1, gamma = 0.8, epsilon_decay = 0.99,
                 epsilon_min = 0.01, learning_rate=0.0002, action_space=4,c = 100):
        self.epsilon = epsilon
        self.action = 0
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.target = self.build()
        self.model = self.build()
        self.c = c
        self.memory = deque(maxlen=5000)


    def build(self):
        model = keras.models.Sequential()

        model.add(Conv2D(256, [2,2],strides=1,
                              padding='valid',
                              activation='relu',
                              input_shape=[4,4,1],
                              data_format='channels_last'))

        # Third convolutional layer
        model.add(Conv2D(128,[2,2], strides=1,
                              padding='valid',
                              activation='relu'))
        # Flatten the convolution output
        model.add(Flatten())

        # First dense layer
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # Output layer
        model.add(Dense(self.action_space))

        model.compile(loss=self.loss,
                      optimizer=Adam(),
                      metrics=['accuracy'])
        print(model.summary())
        return model

    def store_memory(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = self.scale(state).reshape(self.state_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def scale(self,d):
        d = d.flatten()
        d = [(np.math.log(i,2))/10 if i !=0 else 0 for i in d]
        d = np.array(d)
        return d

    def experience_replay(self, batch_size):
        try:
            mini_batch = random.sample(self.memory, batch_size)
        except:
            mini_batch = random.sample(self.memory, 32)
        loss = []
        accuracy = []
        for state, action, reward, next_state, game_over in mini_batch:
            target = reward
            self.action = action
            if not game_over:

                target = reward + self.gamma * np.amax(self.target.predict(next_state)[0])
            y = self.model.predict(state)[0]
            y[action] =target
            y = y.reshape(1,-1)
            hist = self.model.fit(state, y, epochs=1, verbose=0)
            loss.append(hist.history['loss'])
            accuracy.append(hist.history['acc'])

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        print('avg training loss {}:'.format(np.array(loss).mean()))
        print('training accuracy {}:'.format(np.array(accuracy).mean()))

    def loss(self, y_true, y_pred):
        return K.square(y_pred[self.action] - y_true[self.action])



if __name__ == "__main__":
    episodes = 10000
    agent = DQNAgent()
    stuck_each_epoch = []
    s = []
    score = []
    for i in range(episodes):
        g = Game()
        g.fill_cell()
        state = g.board
        stuck_counter = 0
        stuck = 0

        while True:
            action = agent.act(state)
            reward = 0

            if g.moved:
                stuck_counter = 0
            g.main_loop(action)
            next_state = g.board
            max_position = np.argmax(next_state)

            if max_position == np.array([0,3,12,15]).any():
                r_position = 1
            else:
                r_position = -1

            reward = g.empty / 4 + (max(g.joinable))/2 + r_position
            if not g.moved:
                stuck_counter+=1
                reward -= -5
            else:
                reward += 1

            if stuck_counter>=2:
                g.main_loop(random.randint(0,3))
                stuck+=1
                stuck_counter = 0
            game_over = g.game_over

            state = agent.scale(state).reshape(agent.state_size)
            next_state = agent.scale(next_state).reshape(agent.state_size)

            agent.store_memory(state, action, reward, next_state, game_over)
            state = next_state
            if game_over:
                print("episode: {}/{}, score: {}, max_cell: {}"
                      .format(i, episodes, g.score, np.max(g.board)))
                break
        stuck_each_epoch.append(stuck)
        score.append(g.score)
        agent.experience_replay(64)

        if i%agent.c == agent.c-1:
            agent.target.set_weights(agent.model.get_weights())
            s.append(np.array(score).mean())
            print(s)
            game = []
            for j in range(50):
                print(j)
                g = Game()
                g.fill_cell()
                state = g.board
                while not g.game_over:
                    act = agent.act(state)
                    g.main_loop(act)

                    if not g.moved:
                        for index,value in enumerate(sorted(
                                agent.model.predict(agent.scale(state).reshape(agent.state_size))[0])):
                            g.main_loop(index)
                            if g.moved:
                                break
                    state = g.board
                game.append(g.score)

            print(np.array(game).mean())
            score = []


