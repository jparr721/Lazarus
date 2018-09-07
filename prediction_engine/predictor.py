import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Predictor(object):
    def __init__(self, batch_size=50, gamma=-10):
        self.batch_size = batch_size
        self._sess = tf.Session()
        self.gamma = gamma
