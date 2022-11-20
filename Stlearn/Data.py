from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from utils import *


class Data(ABC):
    _data_window_size = None

    _data = None
    _data_train = None
    _data_val = None
    _data_test = None
    _X = None
    _y = None
    _ids = None
    _X_train = None
    _y_train = None
    _ids_train = None
    _X_val = None
    _y_val = None
    _ids_val = None
    _X_test = None
    _y_test = None
    _ids_test = None

    _shape = None

    _transformer = None

    _first_indexes = None
    _first_indexes_train = None
    _first_indexes_val = None
    _first_indexes_test = None

    @abstractmethod
    def __init__(self, train_start, val_start, test_start, test_end, three_d) -> None:
        self._define_transformer()
        self._generate_data(train_start, val_start, test_start, test_end, three_d)
        pass

    def get_train(self) -> tuple:
        return self._X_train, self._y_train

    def get_val(self) -> tuple:
        return self._X_val, self._y_val

    def get_test(self) -> tuple:
        return self._X_test, self._y_test

    def get_shape(self):
        self._shape = self._X_train.shape[1:]
        return self._shape

    def _log(self):
        print("X_train shape: " + str(self._X_train.shape))
        print("y_train shape: " + str(self._y_train.shape))
        print("X_valid shape: " + str(self._X_val.shape))
        print("y_valid shape: " + str(self._y_val.shape))
        print("X_test shape: " + str(self._X_test.shape))
        print("y_test shape: " + str(self._y_test.shape))

    def _define_transformer(self):
        self._transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        pass

    def _generate_data(self, train_start, val_start, test_start, test_end, three_d=False) -> None:
        self._data = pd.read_parquet(Constant.DATA_PATH)
        self._data = self._data[(self._data['Date'] >= train_start) &
                                (self._data['Date'] < test_end)].reset_index().drop('index', axis=1)

        d = self._data[['Ticker']].reset_index()
        lows = d.groupby(['Ticker'])['index'].min().rename('low')

        self._first_indexes = lows.values

        train_indexes = self._data[(self._data['Date'] >= train_start) &
                                   (self._data['Date'] < val_start)].index

        val_indexes = self._data[(self._data['Date'] >= val_start) &
                                 (self._data['Date'] < test_start)].index

        test_indexes = self._data[(self._data['Date'] >= test_start) &
                                  (self._data['Date'] < test_end)].index

        str_features = ['Date', 'Ticker']
        ids_features = ['Date', 'Ticker', 'Close', 'return', 'market return']

        self._data_train = self._data.loc[train_indexes, :].reset_index().drop('index', axis=1)
        self._data_val = self._data.loc[val_indexes, :].reset_index().drop('index', axis=1)
        self._data_test = self._data.loc[test_indexes, :].reset_index().drop('index', axis=1)

        self._X = self._data.drop(['er'] + str_features, axis=1).values
        self._y = self._data['er'].values
        self._ids = self._data[ids_features].values

        if three_d:

            if self._transformer is not None:
                self._transformer.fit(self._X[train_indexes])
                self._X[train_indexes] = self._transformer.transform(self._X[train_indexes])
                self._X[val_indexes] = self._transformer.transform(self._X[val_indexes])
                self._X[test_indexes] = self._transformer.transform(self._X[test_indexes])

            self._generate_data_window(self._X, self._y, self._ids, self._first_indexes,
                                       train_indexes, val_indexes, test_indexes)

        else:
            self._X_train = self._data_train.drop(['er'] + str_features, axis=1).values
            self._y_train = self._data_train['er'].values
            self._ids_train = self._data_train[ids_features].values

            self._X_val = self._data_val.drop(['er'] + str_features, axis=1).values
            self._y_val = self._data_val['er'].values
            self._ids_val = self._data_val[ids_features].values

            self._X_test = self._data_test.drop(['er'] + str_features, axis=1).values
            self._y_test = self._data_test['er'].values
            self._ids_test = self._data_test[ids_features].values

            if self._transformer is not None:
                self._transformer.fit(self._X_train)
                self._X_train = self._transformer.transform(self._X_train)
                self._X_val = self._transformer.transform(self._X_val)
                self._X_test = self._transformer.transform(self._X_test)

        pass

    def _generate_data_window(self, X, y, ids, first_indexes,
                              train_indexes, val_indexes, test_indexes):

        data_window_size = self._data_window_size

        if len(train_indexes) - data_window_size <= 0:
            raise ValueError("Do not have enough training samples!")

        if len(train_indexes) - data_window_size < len(val_indexes):
            raise ValueError("Fewer training samples than validation samples!")

        if len(train_indexes) - data_window_size < len(test_indexes):
            raise ValueError("Fewer training samples than test samples!")

        num_samples = X.shape[0] - data_window_size + 1
        start_index = data_window_size - 1
        num_features = X.shape[1]

        X_windowed = np.zeros(shape=(num_samples, data_window_size, num_features))

        for i in range(start_index, X.shape[0]):
            X_windowed[i - data_window_size + 1, :, :] = X[i - data_window_size + 1: i + 1, :]

        y_windowed = y[start_index:]
        ids_windowed = ids[start_index:]

        excluding_indexes_origin = []
        for index in first_indexes:
            excluding_indexes_origin += range(index, index + data_window_size - 1)
        excluding_indexes_new = [i - data_window_size + 1 for i in excluding_indexes_origin
                                 if ((i - data_window_size + 1) >= 0)]
        train_indexes = [i - data_window_size + 1 for i in train_indexes
                         if ((i - data_window_size + 1) >= 0)]
        val_indexes = [i - data_window_size + 1 for i in val_indexes
                       if ((i - data_window_size + 1) >= 0)]
        test_indexes = [i - data_window_size + 1 for i in test_indexes
                        if ((i - data_window_size + 1) >= 0)]
        train_indexes = [i for i in train_indexes if i not in excluding_indexes_new]
        val_indexes = [i for i in val_indexes if i not in excluding_indexes_new]
        test_indexes = [i for i in test_indexes if i not in excluding_indexes_new]

        X_train = X_windowed[train_indexes]
        y_train = y_windowed[train_indexes]
        ids_train = ids_windowed[train_indexes]

        X_val = X_windowed[val_indexes]
        y_val = y_windowed[val_indexes]
        ids_val = ids_windowed[val_indexes]

        X_test = X_windowed[test_indexes]
        y_test = y_windowed[test_indexes]
        ids_test = ids_windowed[test_indexes]

        self._X_train = X_train
        self._y_train = y_train
        self._ids_train = ids_train

        self._X_val = X_val
        self._y_val = y_val
        self._ids_val = ids_val

        self._X_test = X_test
        self._y_test = y_test
        self._ids_test = ids_test

        pass

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_val(self):
        return self._X_val

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_val(self):
        return self._y_val

    @property
    def y_test(self):
        return self._y_test

    @property
    def ids_train(self):
        return self._ids_train

    @property
    def ids_val(self):
        return self._ids_val

    @property
    def ids_test(self):
        return self._ids_test


class MlData(Data):

    def __init__(self, train_start, val_start, test_start, test_end) -> None:
        super().__init__(train_start, val_start, test_start, test_end, False)
        self._log()
        pass


class DlData(Data):

    def __init__(self, train_start, val_start, test_start, test_end, data_window_size) -> None:
        self._data_window_size = data_window_size
        super().__init__(train_start, val_start, test_start, test_end, True)
        self._log()
        pass
