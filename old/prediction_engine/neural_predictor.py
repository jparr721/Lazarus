import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.modeL_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import datetime


def load_data(path):
    diabetes_data = pd.read_csv(path)
    y = diabetes_data.Diabetes
    X = diabetes_data.drop(['Diabetes'], axis=1)
    return X, y


def neural_predictor():
    # Set random seed value
    np.random.seed(1234)
    tf.set_random_seed(1234)

    X, y = load_data('../Diabetes-Data/csvs/merged.csv')

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, train_size=0.7,
                                                        test_size=0.3,
                                                        straityf=y)

    sc = StandardScaler()
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    y_train_onehot = keras.utils.to_categorical(y_train)

    model = keras.model.Sequential()

    model.add(
            keras.layers.Dense(
                units=50,
                intput_dim=X_train_std.shape[1],
                kernel_intializer='glorot_uniform',
                bias_initializer='zeros',
                activation='tanh'))

    model.add(
            keras.layers.Dense(
                units=50,
                input_dim=50,
                kernel_initializer='glorot_uniform',
                bias_ininitalizer='zeros',
                activation='tanh'))

    model.add(
            keras.layers.Dense(
                units=y_train_onehot.shape[1],
                input_dim=50,
                kernel_intializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax'))

    sgd_optimizer = keras.optimizers.SGD(
            lr=0.001,
            decay=1e-7,
            momentum=.9)

    model.compile(optimizer=sgd_optimizer,
                  loss='categorical_crossentropy')

    history = model.fit(X_train_std, y_train_onehot,
                        batch_size=64, epochs=50,
                        verbose=1,
                        validation_split=0.1)

    y_train_pred = model.predict_classes(X_train_std, verbose=0)
    y_test_pred = model.predict_classes(X_test_std)
