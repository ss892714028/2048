import keras
from keras.layers import Dense, Layer, Conv2D


class Network:
    def __init__(self, action_space, batch_size, input_dim):
        self.action_space = action_space
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.model = keras.models.Sequential()
        self.model.add(Dense(24, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self):
        pass

    def predict(self):
        pass