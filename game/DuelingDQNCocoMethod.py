import numpy as np
from collections import deque
from game import Game
import random
from keras.layers import Dense, Layer, Conv2D, MaxPool2D,Flatten,Dropout,BatchNormalization, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import merge, Input

import keras
from keras.optimizers import SGD,Adam,RMSprop
import keras.backend as K


class DQNAgent:

    def __init__(self, state_size=[1, 4, 4, 1], epsilon=0, gamma=0.9, epsilon_decay=0,
                 epsilon_min=0, learning_rate=0.0001, action_space=4, c=100):
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
        input_layer = Input(shape=(4, 4, 1))
        conv1 = Convolution2D(64, 2, 2, activation='relu')(input_layer)
        conv2 = Convolution2D(128, 2, 2, activation='relu')(conv1)
        conv3 = Convolution2D(256, 2, 2, activation='relu')(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        advantage = Dense(self.action_space)(fc1)
        fc2 = Dense(512)(flatten)
        x = Dense(self.action_space + 1, activation='linear')(fc2)

        x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_space,))(x)
        # policy = Dense(NUM_ACTIONS)(merge_layer)

        model = keras.Model(input=[input_layer], output=[x])
        model.compile(loss=self.loss, optimizer=Adam(lr=0.00001),metrics=['accuracy'])

        return model

    def store_memory(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    @staticmethod
    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    @staticmethod
    def scale(d):
        d = d.flatten()
        d = [(np.math.log(i,2))/10 if i !=0 else 0 for i in d]
        d = np.array(d)
        return d

    @staticmethod
    def isMonotonic(A):

        return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
                all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

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

            y[action] = target
            y = y.reshape(1,-1)

            hist = self.model.fit(state, y, epochs=1, verbose=0)
            loss.append(hist.history['loss'])
            accuracy.append(hist.history['acc'])

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        print('avg training loss {}:'.format(np.array(loss).mean()))
        # print('training accuracy {}:'.format(np.array(accuracy).mean()))

    def loss(self, y_true, y_pred):
        return K.square(y_pred[self.action] - y_true[self.action])

    @staticmethod
    def minimax_scaler(x):
        return (2*x/30) - 1

    def calculate_reward(self, m, cells, empty, joinables):
        monotonic_row = []
        monotonic_col = []
        for i in m:
            monotonic_row.append(self.isMonotonic(i))
        for i in m.T:
            monotonic_col.append(self.isMonotonic(i))
        monotonic = 2 - max(monotonic_col.count(True), monotonic_row.count(True))

        weight_matrix = np.array([[20, 12, 8, 6],
                                  [12, 8, 6, 2],
                                  [8, 6, 2, 1],
                                  [6, 2, 1, 0]])
        score = 0
        for x in range(self.action_space):
            for y in range(self.action_space):
                score += weight_matrix[x,y] * m[x,y]

        score = score/(np.mean(m)*20)

        if cells == 0:
            cells = -1
        if empty == 0:
            empty = -5

        if joinables == 0 and cells <= 1:
            joinables = -3
        # return empty/2 + max(joinables) * 2
        return score


if __name__ == "__main__":
    episodes = 100000
    agent = DQNAgent()
    agent.reset_weights(agent.model)
    agent.reset_weights(agent.target)
    # store all mean_score_cepochs
    s = []
    # average score of c epochs
    mean_score_cepochs = []
    Q_value = []
    for i in range(episodes):
        g = Game()
        g.fill_cell()
        state = g.board
        stuck_counter = 0
        stuck = 0
        total_r = []
        board = g.board
        while True:
            # take action
            long_term_rewards = []
            counter = 0
            for k in range(-1, agent.action_space):
                long_term_rewards.append(counter)
                counter = 0
                g_temp = Game()
                g_temp.board = board
                while True:
                    temp_state = agent.scale(g_temp.board).reshape(agent.state_size)
                    action_temp = agent.act(temp_state)
                    try:
                        g_temp.main_loop(action_temp)
                    except:
                        print(g_temp.board)
                        print(action_temp)
                    # if not moved, select the largest available q as action
                    if not g_temp.moved:
                        # store all q values and make a dictionary {qvalue: action_index}
                        lst_temp = agent.model.predict(temp_state)[0]
                        dict_temp = {}
                        for index, value in enumerate(lst_temp):
                            dict_temp[value] = index
                        # reverse sort keys(q_values)
                        # iterate through keys and get
                        for key in reversed(sorted(dict_temp.keys())):
                            g_temp.main_loop(dict_temp[key])
                            if g_temp.moved:
                                break
                    counter += 1
                    if g_temp.game_over:
                        break
            # select action based on the current state
            action = np.argmax(long_term_rewards) -1

            state = agent.scale(g.board).reshape(agent.state_size)
            g.main_loop(action)
            # initialize reward
            reward = 0
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
            # reward = agent.calculate_reward(g.board, joined_cells, empty, joinable)
            reward = action
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

