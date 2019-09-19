from stable_baselines import DQN
import stable_baselines
from env import GameEnv
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy
import tensorflow as tf
import numpy as np


def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=64, filter_size=2, stride=1,  **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=128, filter_size=2, stride=1,  **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=256, filter_size=2, stride=1,  **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")
# env = stable_baselines.common.vec_env.DummyVecEnv(lambda: GameEnv())


env = GameEnv()
model = DQN(CustomPolicy, env,verbose=1).learn(total_timesteps=int(6e7))
model.save('model')