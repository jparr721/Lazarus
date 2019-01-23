import tensorflow as tf
import numpy as np
import pandas as pd


class DataModel:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._states = None
        self._actions = None
        self._logits = None
        self._optimizer = None
        self._var_init = None
        self._define_model()

    def _define_model(self):
        # All of our poassible states we can be in
        self._states = tf.placeholder(
                shape=[None, self._num_states], dtype=tf.float32)
        # Our Q applied to our state and action
        self.q_s_a = tf.placeholder(
                shape=[None, self._num_actions], dtype=tf.float32)

        # Create our neural network layers for prediction
        layer_1 = tf.layers.dense(
                self._states, 50, activation=tf.nn.relu, name='layer_1')
        layer_2 = tf.layers.dense(
                layer_1, 50, activation=tf.nn.relu, name='layer_2')

        # Our final, unactivated policy output which is our optimum
        # action given a particular state
        self._logits = tf.layers.dense(
                layer_2, self._num_actions, name='logits')

        # our cost function
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)

        # use the adam optimizer instead of gradient descent
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
