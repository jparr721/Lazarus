import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Predictor(object):
    def __init__(self, batch_size, gamma):
        self.batch_size = batch_size
        self._sess = tf.Session()
        self.gamma = gamma
        self.prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    def read_data(self, source):
        '''
        Takes the absolute path to some data
        and outputs a new pandas dataframe
        '''
        return pd.read_csv(source)

    def fit(self, X, y, b):
        distribution = tf.reduce_sum(tf.square(X), 1)
        distribution = tf.reshape(distribution, [-1, 1])

        squared_distributions = \
            tf.add(tf.sub(distribution,
                          tf.mul(2., tf.matmul(X, tf.transpose(X))),
                          tf.transpose(distribution)))

        kernel = tf.exp(tf.mul(self.gamma, tf.abs(squared_distributions)))

        model_output = tf.matmul(b, kernel)

        first_term = tf.reduce_sum(b)
        b_vec_cross = tf.matmul(tf.transpose(b), b)

        y_target_cross = tf.matmul(y, tf.transpose(y))
        second_term = tf.reduce_sum(tf.mul(kernel, tf.mul(b_vec_cross,
                                                          y_target_cross)))
        # maximize the negative loss function
        loss = tf.neg(tf.sub(first_term, second_term))

        return loss, model_output

    def predict(self, X, y, b):
        rA = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
        rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1, 1])

        pred_sq_dist = \
            tf.add(tf.sub(rA,
                          tf.mul(2.,
                                 tf.matmul(X,
                                           tf.transpose(self.prediction_grid))),
                          tf.transpose(rB)))

        # linear prediction kernel
        # pred_kernel = tf.matmul(X, tf.transpose(prediction_grid))
        pred_kernel = tf.exp(tf.mul(self.gamma, tf.abs(pred_sq_dist)))

        prediction_output = tf.matmul(tf.mul(tf.transpose(y), b), pred_kernel)

        prediction = tf.sign(prediction_output -
                             tf.reduce_mean(prediction_output))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction),
                                                   tf.squeeze(y)), tf.float32))

        return accuracy

    def optimize(self, X, y, b, loss, optimizer):
        '''
        Optimizer function to
        minimize the cost of the
        svm algorithm
        '''
        accuracy = self.predict(X, y, b)
        training_step = optimizer.minimize(loss)

        init = tf.initialize_all_variables()
        self._sess.run(init)

        # The training loop
        loss_vector = []
        batch_accuracy = []

        for i in range(self.batch_size):
            rand_index = np.random.choice(len(X), size=self.batch_size)
            rand_x = X[rand_index]
            rand_y = y[rand_index]

            temp_loss = self._sess.run(training_step,
                                       feed_dict={X: rand_x, y: rand_y})

            loss_vector.append(temp_loss)

            acc_temp = self._sess.run(accuracy,
                                      feed_dict={X: rand_x,
                                                 y: rand_y,
                                                 self.prediction_grid: rand_x})

            batch_accuracy.append(acc_temp)

            # Update output every 100 runs
            if (i+1) % 100 == 0:
                print('Step: {}'.format(str(i+1)))
                print('Loss = '.format(str(temp_loss)))
