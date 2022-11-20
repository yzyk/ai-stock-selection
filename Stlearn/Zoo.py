from __future__ import annotations

from abc import ABC, abstractmethod

import IPython

import matplotlib.pyplot as plt
from Data import *
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


class Prototype:

    @classmethod
    def simple_nn(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(Constant.NUM_FEATURES + 1, name='dense_head')
        ])
        return model

    @classmethod
    def cnn(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_1'),
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_2'),
            tf.keras.layers.MaxPooling1D(pool_size=(2,)),
            tf.keras.layers.Conv1D(16, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_3'),
            tf.keras.layers.Conv1D(16, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_4'),
            tf.keras.layers.MaxPooling1D(pool_size=(2,)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(Constant.NUM_FEATURES + 1)
        ])
        return model

    @classmethod
    def lstm(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.LSTM(1, return_sequences=True)
            # tf.keras.layers.Dense(1, name='dense_head')
        ])
        return model

    @classmethod
    def cnn_lstm(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(25, padding='same', kernel_size=(1,),
                                   activation='relu', input_shape=(Constant.WIN_SIZE, Constant.NUM_FEATURES)),
            tf.keras.layers.Conv1D(50, padding='same', kernel_size=(1,),
                                   activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(1, return_sequences=True)
        ])


class AutoRegressor(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def warmup(self, inputs):
        return None

    @abstractmethod
    def call(self, inputs, training=None):
        return None


class LSTMAutoRegressor(AutoRegressor):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(Constant.NUM_FEATURES + 1)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction[:, :-1]
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions[:, :, -1]


class CNNAutoRegressor(AutoRegressor):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.cnn = Prototype.cnn()

    def warmup(self, inputs):
        prediction = self.cnn(inputs)
        return prediction

    def call(self, inputs, training=None):
        predictions = []
        prediction = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            new_X = prediction[:, :-1]
            new_X = tf.reshape(new_X, [-1, 1, Constant.NUM_FEATURES])

            inputs = tf.boolean_mask(inputs, np.array(
                [False] + [True] * (Constant.WIN_SIZE - 1)), axis=1
            )
            inputs = tf.concat([inputs, new_X], 1)
            prediction = self.cnn(inputs)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions[:, :, -1]
