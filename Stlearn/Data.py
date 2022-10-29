from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *


class Data(ABC):
    _data = None
    _data_train = None
    _data_val = None
    _data_test = None
    _X = None
    _y = None
    _X_train = None
    _y_train = None
    _X_val = None
    _y_val = None
    _X_test = None
    _y_test = None

    _shape = None

    _transformer = None

    _first_indexes = None
    _first_indexes_train = None
    _first_indexes_val = None
    _first_indexes_test = None

    @abstractmethod
    def __init__(self, capacity, train_portion, val_portion, test_portion) -> None:
        self._define_transformer()
        self._generate_data(capacity, train_portion, val_portion, test_portion)
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
        self._transformer = None
        pass

    def _generate_data(self, capacity, train_portion, val_portion, test_portion) -> None:
        self._data = pd.read_parquet(DATA_PATH)
        self._data = self._data.loc[:int(self._data.shape[0] * capacity), :]

        d = self._data[['Ticker']].reset_index()
        lows = d.groupby(['Ticker'])['index'].min().rename('low')
        highs = d.groupby(['Ticker'])['index'].max().rename('high')

        self._first_indexes = lows.values

        split_df = pd.merge(lows, highs, left_index=True, right_index=True)
        split_df['60pct_index'] = (split_df['high'] - split_df['low']) * train_portion + split_df['low']
        split_df['80pct_index'] = (split_df['high'] - split_df['low']) * (train_portion + val_portion) + split_df['low']

        train_indexes = []
        val_indexes = []
        test_indexes = []

        for ticker in split_df.index:
            train_indexes += range(int(split_df.loc[ticker, 'low']), int(split_df.loc[ticker, '60pct_index']) + 1)
            val_indexes += range(int(split_df.loc[ticker, '60pct_index']) + 1,
                                 int(split_df.loc[ticker, '80pct_index']) + 1)
            test_indexes += range(int(split_df.loc[ticker, '80pct_index']) + 1, int(split_df.loc[ticker, 'high']) + 1)

        self._data_train = self._data.loc[train_indexes, :].reset_index().drop('index', axis=1).reset_index()
        self._data_val = self._data.loc[val_indexes, :].reset_index().drop('index', axis=1).reset_index()
        self._data_test = self._data.loc[test_indexes, :].reset_index().drop('index', axis=1).reset_index()

        self._first_indexes_train = self._data_train.groupby(['Ticker'])['index'].min().values
        self._first_indexes_val = self._data_val.groupby(['Ticker'])['index'].min().values
        self._first_indexes_test = self._data_test.groupby(['Ticker'])['index'].min().values

        str_features = []
        for c in self._data.columns:
            if self._data[c].dtype == 'O':
                str_features.append(c)
        self._data = self._data.drop(str_features, axis=1)
        self._data_train = self._data_train.drop(str_features, axis=1)
        self._data_val = self._data_val.drop(str_features, axis=1)
        self._data_test = self._data_test.drop(str_features, axis=1)

        self._X = self._data.drop('return', axis=1)
        self._y = self._data['return']

        self._X_train = self._data_train.drop('return', axis=1).values
        self._y_train = self._data_train['return'].values

        self._X_val = self._data_val.drop('return', axis=1).values
        self._y_val = self._data_val['return'].values

        self._X_test = self._data_test.drop('return', axis=1).values
        self._y_test = self._data_test['return'].values

        if self._transformer is not None:
            self._transformer.fit(self._X_train)
            self._X_train = self._transformer.transform(self._X_train)
            self._X_val = self._transformer.transform(self._X_val)
            self._X_test = self._transformer.transform(self._X_test)

        pass


class MlData(Data):

    def __init__(self, capacity, train_portion, val_portion, test_portion) -> None:
        super().__init__(capacity, train_portion, val_portion, test_portion)
        self._log()
        pass


class DlData(Data):
    _data_window_size = None

    def __init__(self, capacity, train_portion, val_portion, test_portion, data_window_size) -> None:
        self._data_window_size = data_window_size
        super().__init__(capacity, train_portion, val_portion, test_portion)
        self._X_train, self._y_train = self._generate_data_window(self._X_train, self._y_train,
                                                                  self._first_indexes_train)
        self._X_val, self._y_val = self._generate_data_window(self._X_val, self._y_val,
                                                              self._first_indexes_val)
        self._X_test, self._y_test = self._generate_data_window(self._X_test, self._y_test,
                                                                self._first_indexes_test)
        self._log()
        pass

    def _generate_data_window(self, X_train, y_train, first_indexes_train):

        data_window_size = self._data_window_size

        num_samples = X_train.shape[0] - data_window_size + 1
        start_index = data_window_size - 1
        num_features = X_train.shape[1]

        X_train_windowed = np.zeros(shape=(num_samples, data_window_size, num_features))

        for i in range(start_index, X_train.shape[0]):
            X_train_windowed[i - data_window_size + 1, :, :] = X_train[i - data_window_size + 1: i + 1, :]

        y_train_windowed = y_train[start_index:]

        excluding_indexes_origin = []
        for index in first_indexes_train:
            excluding_indexes_origin += range(index, index + data_window_size - 1)
        excluding_indexes_new = [i - data_window_size + 1 for i in excluding_indexes_origin
                                 if ((i - data_window_size + 1) >= 0)]

        X_train_windowed = np.delete(X_train_windowed, excluding_indexes_new, axis=0)
        y_train_windowed = np.delete(y_train_windowed, excluding_indexes_new, axis=0)

        return X_train_windowed, y_train_windowed
