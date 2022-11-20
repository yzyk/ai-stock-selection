from __future__ import annotations

from abc import ABC, abstractmethod

import IPython

import matplotlib.pyplot as plt

import Constant
from Data import *
from Zoo import *
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from utils import *


class Model(ABC):
    _model = None
    _params = None
    _name = None

    @abstractmethod
    def __init__(self, name) -> None:
        self._create_model()
        self._name = name
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def fit(self, data) -> None:
        pass

    def predict(self, X):
        return self._model.predict(X)

    @abstractmethod
    def evaluate(self, data, s) -> None:
        pass

    @abstractmethod
    def info(self) -> None:
        pass


class MlModel(Model):

    def __init__(self, name) -> None:
        super().__init__(name)
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, data) -> None:
        X_train, y_train = data.get_train()
        self._model.fit(X_train, y_train)
        pass

    def evaluate(self, data, s) -> None:
        X_test = None
        y_test = None
        if s == 'val':
            X_test, y_test = data.get_val()
        if s == 'test':
            X_test, y_test = data.get_test()
        y_pred = self.predict(X_test)
        performance_measure(y_test, y_pred)
        pass

    def info(self):
        print(self._model)
        pass


class DlModel(Model):
    _input_shape = None

    def __init__(self, name) -> None:
        self._input_shape = (Constant.WIN_SIZE, Constant.NUM_FEATURES)
        super().__init__(name)
        self._model.compile(loss=Constant.ERROR, metrics=[Constant.ERROR], weighted_metrics=[Constant.ERROR])
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, data) -> None:
        X_train, y_train = data.get_train()
        X_val, y_val = data.get_val()
        history_ = self._model.fit(X_train, y_train,
                                   epochs=Constant.MAX_EPOCHS,
                                   validation_data=(X_val, y_val)
                                   )
        fig, axs = plt.subplots(1, 2, figsize=(24, 10))
        self._plot_train(history=history_, axs=axs)
        pass

    def predict(self, X):
        pred = self._model.predict(X)
        if len(pred.shape) == 2:
            pred = pred[:, :, np.newaxis]
        return pred

    def evaluate(self, data, s) -> None:
        X_test = None
        y_test = None
        if s == 'val':
            X_test, y_test = data.get_val()
        if s == 'test':
            X_test, y_test = data.get_test()
        score_ = self._model.evaluate(X_test, y_test)
        print("{n:s}: Test loss: {l:3.5f}".format(
            n=self._name, l=score_[0])
        )
        pass

    def info(self):
        tf.keras.utils.plot_model(
            self._model, to_file=Constant.MODEL_IMAGE_PATH + '/' + self._name + '.png', show_shapes=True, dpi=100
        )
        print("Parameters number in model: ", self._model.count_params())
        return IPython.display.Image(Constant.MODEL_IMAGE_PATH + '/' + self._name + '.png')

    def _plot_train(self, history, axs):
        # Determine the name of the key that indexes into the accuracy metric
        acc_string = 'weighted_' + Constant.ERROR
        # Plot loss
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title(self._name + " " + 'model loss')
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')
        # Plot accuracy
        axs[1].plot(history.history[acc_string])
        axs[1].plot(history.history['val_' + acc_string])
        axs[1].set_title(self._name + ' ' + acc_string)
        axs[1].set_ylabel(acc_string)
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')
        pass


class LinearRegressionModel(MlModel):

    def _create_model(self):
        self._model = LinearRegression()
        pass


class RandomForestRegressorModel(MlModel):

    def _create_model(self):
        self._model = RandomForestRegressor()
        pass


class AdaBoostRegressorModel(MlModel):

    def _create_model(self):
        self._model = AdaBoostRegressor()
        pass


class SimpleNNModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self._input_shape),
            tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, name='dense_head')
        ])

    pass


class CNNModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', input_shape=self._input_shape, name='CNN_1'),
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
            tf.keras.layers.Dense(30,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([30, 1])
        ])
        pass


class LSTMModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=self._input_shape),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.LSTM(1, return_sequences=True)
            # tf.keras.layers.Dense(1, name='dense_head')
        ])
        pass


class LSTMAutoRegressorModel(DlModel):
    def _create_model(self):
        self._model = LSTMAutoRegressor(50, 30)
        pass


class CNNAutoRegressorModel(DlModel):
    def _create_model(self):
        self._model = CNNAutoRegressor(50, 30)
    pass
