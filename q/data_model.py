# Copyright 2018 Grand Valley State University DEN Lab. All Rights Reserved
#==============================================================================

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
        self._initialize = None
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

        # our cost function defualt to a linear activation so our
        # neural network learns across all possible reals
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)

        # use the adam optimizer instead of gradient descent
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)

        # initialize vars into tensorflow
        self._initialize = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        '''
        predict_one returns the output of the neural network via
        our loits operation. This returns a single, optimum policy
        given te inputted state. The data was reshaped to:
        [1, self.num_states] to facilitate a one-dimensional feed
        dict. This allows the output data to be only one observation.

        Parameters
        ----------
        state - The state we will pass to the feed dict
        sess - The tensorflow session that is currently running
        '''
        return sess.run(self._logits, feed_dict={self.states:
                        state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        '''
        predict_batch predicts an entire batch of outputs when given a >1
        dimensional number of input states. This is used to perform batch
        evaluation of Q and Q' values during training.

        Paraeters
        ---------
        states - Our matrix of states
        sess - The tensorflow session that is currently running
        '''
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        '''
        train_batch takes a batch training step of the network and
        runs the optimizer over the projected policy output

        Parameters
        ----------
        sess - The tensorflow session variable
        x_batch - The input states to the optimizer function
        y_batch - The Q values with which we are training
        '''
        return sess.run(
                self._optimizer, feed_dict={
                    self._states: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def initialize(self):
        return self._initialize
