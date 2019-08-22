import numpy as np
from collections import deque
from game import Game
import random
from keras.layers import Dense, Layer, Conv2D, MaxPool2D,Flatten,Dropout,BatchNormalization, Dropout
import keras
from keras.optimizers import SGD,Adam,RMSprop
import keras.backend as K


class DQNAgent:

    def __init__(self, state_size=[1,4,4,1], epsilon = 0, gamma = 0.8, epsilon_decay = 0,
                 epsilon_min = 0, learning_rate=0.0001, action_space=4, c=100):
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
        self.memory = deque(maxlen=20000)


    def build(self):
        model = keras.models.Sequential()

        model.add(Conv2D(128, [2,2],strides=1,
                              padding='valid',
                              activation='relu',
                              input_shape=[4,4,1],
                              data_format='channels_last'))

        # Third convolutional layer
        model.add(Conv2D(256,[2,2], strides=1,
                              padding='valid',
                              activation='relu'))
        model.add(Conv2D(256,[2,2], strides=1,
                              padding='valid',
                              activation='relu'))
        # Flatten the convolution output
        model.add(Flatten())

        # First dense layer
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        # Output layer
        model.add(Dense(self.action_space))

        model.compile(loss=self.loss,
                      optimizer=RMSprop(lr=self.learning_rate),
                      metrics=['accuracy'])
        print(model.summary())
        return model

    def store_memory(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

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

    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    @staticmethod
    def calculate_reward(next_state, joined_cells, empty, joinable):
        result = np.where(next_state == np.max(next_state))
        listOfCoordinates = np.array(list(zip(result[0], result[1])))
        temp = np.array([list(i) in [[0, 0], [0, 3], [3, 0], [3, 3]] for i in listOfCoordinates])
        if temp.any():
            r_position = 10
        else:
            r_position = 0

        if joined_cells == 0:
            joined_cells = -1
        if empty == 0:
            empty = -5
        if joined_cells == 0:
            joined_cells = -2
        if joinable == 0 and joined_cells <= 1:
            joinable = -3
        r = r_position + joined_cells + empty / 3 + max(joinable)

        return r


if __name__ == "__main__":
    episodes = 100000
    agent = DQNAgent()
    # store all mean_score_cepochs
    s = []
    # average score of c epochs
    mean_score_cepochs = []
    Q_value = []
    for i in range(episodes):
        g = Game()
        g.fill_cell()
        state = agent.scale(g.board).reshape(agent.state_size)
        stuck_counter = 0
        stuck = 0
        total_r = []

        while True:
            # select action based on the current state
            action = agent.act(state)
            # initialize reward
            reward = 0
            # take action
            g.main_loop(action)

            # if not moved, select the largest available q as action
            if not g.moved:
                # store all q values and make a dictionary {qvalue: action_index}
                lst = agent.model.predict(state)[0]
                dict = {}
                for index, value in enumerate(lst):
                    dict[value] = index
                # reverse sort keys(q_values)
                # iterate through keys and get
                for key in reversed(sorted(dict.keys())):
                    g.main_loop(dict[key])
                    if g.moved:
                        break
            # gather information to calculate reward
            next_state = g.board
            max_position = np.argmax(next_state)
            empty = g.empty
            joinable = g.joinable
            joined_cells = g.joined_cells
            # calculate reward
            reward = agent.calculate_reward(next_state,joined_cells,empty,joinable)
            game_over = g.game_over
            # record reward
            total_r.append(reward)
            # preprocess next_state
            next_state = agent.scale(next_state).reshape(agent.state_size)
            # store everything to memory
            agent.store_memory(state, action, reward, next_state, game_over)

            state = next_state
            Q = np.array([value * (0.99 ** index) for index, value in enumerate(total_r)]).sum()
            if game_over:
                print("episode: {}/{}, score: {}, max_cell: {}, Q: {}"
                      .format(i, episodes, g.score, np.max(g.board), Q))
                print(g.board)
                break

        Q_value.append(Q)
        # store average score of agent.c epochs
        mean_score_cepochs.append(g.score)
        agent.experience_replay(128)

        if i%agent.c == agent.c-1:
            agent.target = keras.models.clone_model(agent.model)
            s.append(np.array(mean_score_cepochs).mean())
            print(np.array(s))

