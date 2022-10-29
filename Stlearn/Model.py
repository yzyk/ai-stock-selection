from __future__ import annotations

from abc import ABC, abstractmethod
from Data import *
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import DATA_PATH
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


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
    def evaluate(self, data) -> None:
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

    def evaluate(self, data) -> None:
        X_test, y_test = data.get_test()
        y_pred = self.predict(X_test)
        performance_measure(y_test, y_pred)
        pass


class DlModel(Model):
    _input_shape = None

    def __init__(self, name, input_shape) -> None:
        self._input_shape = input_shape
        super().__init__(name)
        self._model.compile(loss=ERROR, metrics=[ERROR], weighted_metrics=[ERROR])
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, data) -> None:
        max_epochs = 10
        X_train, y_train = data.get_train()
        X_val, y_val = data.get_val()
        history_ = self._model.fit(X_train, y_train,
                                   epochs=max_epochs,
                                   validation_data=(X_val, y_val)
                                   )
        pass

    def evaluate(self, data) -> None:
        X_test, y_test = data.get_test()
        score_ = self._model.evaluate(X_test, y_test)
        print("{n:s}: Test loss: {l:3.5f}".format(
            n=self._name, l=score_[0])
        )
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
            tf.keras.layers.Dense(1, name='dense_head')
        ])
        pass


class LSTMModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=self._input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, name='dense_head')
        ])
        pass
