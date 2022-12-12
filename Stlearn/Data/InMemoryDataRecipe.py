from __future__ import annotations

from .DataInterface import *
from .DataStructure import ArrayGenerator
import Constant
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DiskData(InMemoryData):

    def __init__(self, train_start, val_start, test_start, test_end) -> None:
        super().__init__()
        self._read_data(train_start, val_start, test_start, test_end)
        self.generate()
        pass

    def _read_data(self, train_start, val_start, test_start, test_end) -> None:
        self._data = pd.read_parquet(Constant.DATA_PATH)

        # data slice

        self._data = self._data[(self._data['Date'] >= train_start) &
                                (self._data['Date'] < test_end)].reset_index().drop('index', axis=1)

        # data check

        count = self._data.groupby('Ticker')['Ticker'].count()
        Constant.STOCK_LIST = count[count == self._data['Date'].unique().shape[0]
                                    ].sort_index().index.tolist()
        Constant.NUM_STOCKS = len(Constant.STOCK_LIST)

        self._data = self._data[self._data['Ticker'].isin(Constant.STOCK_LIST)].reset_index(drop=True)

        d = self._data[['Ticker']].reset_index()
        lows = d.groupby(['Ticker'])['index'].min().rename('low')

        self._first_indexes = lows.values

        self._train_indexes = self._data[(self._data['Date'] >= train_start) &
                                         (self._data['Date'] < val_start)].index

        self._val_indexes = self._data[(self._data['Date'] >= val_start) &
                                       (self._data['Date'] < test_start)].index

        self._test_indexes = self._data[(self._data['Date'] >= test_start) &
                                        (self._data['Date'] < test_end)].index

        str_features = ['Date', 'Ticker']
        ids_features = ['Date', 'Ticker', 'Close', 'return', 'market return']

        self._X = self._data.drop(['er'] + str_features, axis=1).values
        self._y = self._data['er'].values
        self._ids = self._data[ids_features].values

        pass

    def generate(self):
        pass


'''
Decorators
'''


class DataDecorator(InMemoryData, ABC):

    def __init__(self, data):
        super().__init__()
        self.copy_constructor(data)
        self.generate()
        pass

    @abstractmethod
    def generate(self):
        pass


'''
Pre Split Decorators
'''


class PreSplitDataDecorator(DataDecorator, ABC):

    def generate(self):
        self.pre_split_data_process()
        pass

    @abstractmethod
    def pre_split_data_process(self):
        pass


class PreSplitScaler(PreSplitDataDecorator):

    def pre_split_data_process(self):
        transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformer.fit(self._X[self._train_indexes])
        self._X[self._train_indexes] = transformer.transform(self._X[self._train_indexes])
        self._X[self._val_indexes] = transformer.transform(self._X[self._val_indexes])
        self._X[self._test_indexes] = transformer.transform(self._X[self._test_indexes])
        pass


class WindowGenerator(PreSplitDataDecorator):
    _forward_steps_size = None
    _data_window_size = None

    def __init__(self, data, data_win_size, forward_steps_size):
        self._data_window_size = data_win_size
        self._forward_steps_size = forward_steps_size
        super().__init__(data)

    def pre_split_data_process(self):
        test_date_length = len(self._test_indexes.tolist()) / Constant.NUM_STOCKS
        if self._forward_steps_size > test_date_length:
            raise ValueError("Forward step size is larger than the test data size!")

        self._generate_data_window()

        pass

    def _generate_data_window(self):

        if len(self._train_indexes) / Constant.NUM_STOCKS - self._data_window_size <= 0:
            raise ValueError("Do not have enough training samples!")

        if len(self._train_indexes) / Constant.NUM_STOCKS - self._data_window_size < len(
                self._val_indexes) / Constant.NUM_STOCKS:
            raise ValueError("Fewer training samples than validation samples!")

        if len(self._train_indexes) / Constant.NUM_STOCKS - self._data_window_size < len(
                self._test_indexes) / Constant.NUM_STOCKS:
            raise ValueError("Fewer training samples than test samples!")

        num_samples = self._X.shape[0] - self._data_window_size + 1
        start_index = self._data_window_size - 1
        num_features = self._X.shape[1]

        X_windowed = np.zeros(shape=(num_samples, self._data_window_size, num_features))
        y_windowed = np.zeros(shape=(num_samples, self._forward_steps_size, 1))

        for i in range(start_index, self._X.shape[0]):
            X_windowed[i - self._data_window_size + 1, :, :] = self._X[i - self._data_window_size + 1: i + 1, :]

        for i in [j for j in range(start_index, self._X.shape[0]) if j not in self._test_indexes]:
            y_windowed[i - self._data_window_size + 1, :, :] = self._y[i: i + self._forward_steps_size].reshape(-1, 1)

        # y_windowed = y[start_index:]
        ids_windowed = self._ids[start_index:]

        excluding_indexes_origin = []
        for index in self._first_indexes:
            excluding_indexes_origin += range(index, index + self._data_window_size - 1)
        excluding_indexes_new = [i - self._data_window_size + 1 for i in excluding_indexes_origin
                                 if ((i - self._data_window_size + 1) >= 0)]
        self._train_indexes = [i - self._data_window_size + 1 for i in self._train_indexes
                               if ((i - self._data_window_size + 1) >= 0)]
        self._val_indexes = [i - self._data_window_size + 1 for i in self._val_indexes
                             if ((i - self._data_window_size + 1) >= 0)]
        self._test_indexes = [i - self._data_window_size + 1 for i in self._test_indexes
                              if ((i - self._data_window_size + 1) >= 0)]
        self._train_indexes = [i for i in self._train_indexes if i not in excluding_indexes_new]
        self._val_indexes = [i for i in self._val_indexes if i not in excluding_indexes_new]
        self._test_indexes = [i for i in self._test_indexes if i not in excluding_indexes_new]

        self._X = X_windowed
        self._y = y_windowed
        self._ids = ids_windowed

        pass


'''
Split Decorators
'''


class SplitDataDecorator(DataDecorator, ABC):

    def generate(self):
        self.split_data_process()
        pass

    @abstractmethod
    def split_data_process(self):
        pass


class StandardSplitProcessor(SplitDataDecorator):

    def split_data_process(self):
        self._X_train = self._X[self._train_indexes]
        self._y_train = self._y[self._train_indexes]
        self._ids_train = self._ids[self._train_indexes]

        self._X_val = self._X[self._val_indexes]
        self._y_val = self._y[self._val_indexes]
        self._ids_val = self._ids[self._val_indexes]

        self._X_test = self._X[self._test_indexes]
        self._y_test = self._y[self._test_indexes]
        self._ids_test = self._ids[self._test_indexes]
        pass


'''
Post Split Decorators
'''


class PostSplitDataDecorator(DataDecorator, ABC):

    def generate(self):
        self.post_split_data_process()
        pass

    @abstractmethod
    def post_split_data_process(self):
        pass


class FirstDimensionReshaper(PostSplitDataDecorator, ABC):
    @abstractmethod
    def post_split_data_process(self):
        pass

    @classmethod
    def arrange_by_split_date_stock(cls, data):
        d = data
        num_stocks = Constant.NUM_STOCKS
        num_dates = int(d.shape[0] / num_stocks)
        other_dims = d.shape[1:]
        d = d.reshape(num_stocks, num_dates, *other_dims)
        d = np.transpose(d, tuple([1, 0] + list(range(2, d.ndim))))
        return d


class FirstDimensionReshaperOnUnWindowed(FirstDimensionReshaper):

    def post_split_data_process(self):
        self._X_train = super().arrange_by_split_date_stock(self._X_train)
        self._y_train = super().arrange_by_split_date_stock(self._y_train.reshape(-1, 1))
        self._ids_train = super().arrange_by_split_date_stock(self._ids_train)

        self._X_val = super().arrange_by_split_date_stock(self._X_val)
        self._y_val = super().arrange_by_split_date_stock(self._y_val.reshape(-1, 1))
        self._ids_val = super().arrange_by_split_date_stock(self._ids_val)

        self._X_test = super().arrange_by_split_date_stock(self._X_test)
        self._y_test = super().arrange_by_split_date_stock(self._y_test.reshape(-1, 1))
        self._ids_test = super().arrange_by_split_date_stock(self._ids_test)
        pass


class FirstDimensionReshaperOnWindowed(FirstDimensionReshaper):

    def post_split_data_process(self):
        self._X_train = self.arrange_by_split_date_stock(self._X_train)
        self._y_train = self.arrange_by_split_date_stock(self._y_train)
        self._ids_train = super().arrange_by_split_date_stock(self._ids_train)

        self._X_val = self.arrange_by_split_date_stock(self._X_val)
        self._y_val = self.arrange_by_split_date_stock(self._y_val)
        self._ids_val = super().arrange_by_split_date_stock(self._ids_val)

        self._X_test = self.arrange_by_split_date_stock(self._X_test)
        self._y_test = self.arrange_by_split_date_stock(self._y_test)
        self._ids_test = super().arrange_by_split_date_stock(self._ids_test)
        pass

    @classmethod
    def arrange_by_split_date_stock(cls, data):
        data = super().arrange_by_split_date_stock(data)
        return np.transpose(data, (0, 2, 1, 3))


class TestDataHandlerOnFirstDimensionAsRecord(PostSplitDataDecorator):

    def post_split_data_process(self):
        t = self._ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        self._X_test = self._X_test[first_indexes]
        pass


class TestDataHandlerOnFirstDimensionAsDate(PostSplitDataDecorator):

    def post_split_data_process(self):
        self._X_test = self._X_test[[0]]
        self._ids_test = np.transpose(self._ids_test, (1, 0, 2))
        self._ids_test = np.reshape(self._ids_test, (-1, self._ids_test.shape[-1]))
        pass
