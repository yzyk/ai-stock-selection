from .DataInterface import *
from .DataStructure import ArrayGenerator
import Constant
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DiskData(GeneratedData):

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

        self._memory_X = self._data.drop(['er'] + str_features, axis=1).values
        self._memory_y = self._data['er'].values
        self._memory_ids = self._data[ids_features].values

        self._X = ArrayGenerator(
            self._memory_X,
            list(range(self._data.shape[0])),
            self._memory_X.shape
        )
        self._y = ArrayGenerator(
            self._memory_y,
            list(range(self._data.shape[0])),
            self._memory_y.shape
        )
        self._ids = ArrayGenerator(
            self._memory_ids,
            list(range(self._data.shape[0])),
            self._memory_ids.shape
        )
        pass

    def generate(self):
        pass


class DiskDataTest(GeneratedData):
    def __init__(self, path) -> None:
        super().__init__()
        self.generate(path, allow_pickle=True)
        pass

    def log(self):
        print('X_test shape:' + str(self._X_test.shape))
        print('ids_test shape: ' + str(self._ids_test.shape))

    def generate(self, path):
        d = np.load(path)
        X = d['x']
        ids = d['ids']
        self._X_test = X
        self._ids_test = ids
        pass

    @property
    def X_test(self):
        return self._X_test

    @property
    def ids_test(self):
        return self._ids_test


'''
Decorators
'''


class DataDecorator(GeneratedData, ABC):

    def __init__(self, data):
        super().__init__()
        self.copy_constructor(data)
        self.generate()
        pass

    @abstractmethod
    def generate(self):
        pass


'''
In Memory Data Decorators
'''


class InMemoryDataDecorator(DataDecorator, ABC):

    def generate(self):
        self.in_memory_data_process()
        pass

    @abstractmethod
    def in_memory_data_process(self):
        pass


class PreSplitScaler(InMemoryDataDecorator):

    def in_memory_data_process(self):
        transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformer.fit(self._memory_X[self._train_indexes])
        self._memory_X[self._train_indexes] = transformer.transform(self._memory_X[self._train_indexes])
        self._memory_X[self._val_indexes] = transformer.transform(self._memory_X[self._val_indexes])
        self._memory_X[self._test_indexes] = transformer.transform(self._memory_X[self._test_indexes])
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


class WindowGenerator(PreSplitDataDecorator):

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

        excluding_indexes_origin = []
        for index in self._first_indexes:
            excluding_indexes_origin += range(index, index + self._data_window_size - 1)

        self._train_indexes = [i for i in self._train_indexes if i not in excluding_indexes_origin]
        self._val_indexes = [i for i in self._val_indexes if i not in excluding_indexes_origin]
        self._test_indexes = [i for i in self._test_indexes if i not in excluding_indexes_origin]

        """
        """

        def pre_split_X_func(data, memory_data_index_range, cur_indexes_on_memory_data_index, steps):
            indexes = memory_data_index_range[
                cur_indexes_on_memory_data_index
            ]
            res = np.zeros(shape=(len(indexes), Constant.WIN_SIZE, Constant.NUM_FEATURES))
            for j in range(len(indexes)):
                j_index = indexes[j]
                res[j, :, :] = data[j_index - Constant.WIN_SIZE + 1: j_index + 1, :]
            return res

        def pre_split_y_func(data, memory_data_index_range, cur_indexes_on_memory_data_index, steps):
            indexes = memory_data_index_range[
                cur_indexes_on_memory_data_index
            ]
            res = np.zeros(shape=(len(indexes), Constant.FORWARD_STEPS_SIZE, 1))
            for j in range(len(indexes)):
                index = indexes[j]
                res[j, :, :] = data[index: index + Constant.FORWARD_STEPS_SIZE].reshape(-1, 1)
            return res

        def pre_split_ids_func(data, memory_data_index_range, cur_indexes_on_memory_data_index, steps):
            indexes = memory_data_index_range[
                cur_indexes_on_memory_data_index
            ]
            return data[indexes, :]

        new_indexes = list(np.sort(list(self._train_indexes) + list(self._val_indexes) + list(self._test_indexes)))
        self._X.update_index_range(new_indexes)
        self._y.update_index_range(new_indexes)
        self._ids.update_index_range(new_indexes)

        self._X.add_ops_funcs(pre_split_X_func)
        self._y.add_ops_funcs(pre_split_y_func)
        self._ids.add_ops_funcs(pre_split_ids_func)

        self._X.update_shape((len(new_indexes), Constant.WIN_SIZE, Constant.NUM_FEATURES))
        self._y.update_shape((len(new_indexes), Constant.FORWARD_STEPS_SIZE, 1))
        self._ids.update_shape((len(new_indexes), *self._ids.shape[1:]))

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
        self._X_train = ArrayGenerator(
            self._memory_X,
            self._train_indexes,
            (int(len(self._train_indexes) / (len(self._X.cur_indexes_on_memory_data_index) / self._X.batch_size)),
             *(self._X.shape[1:])),
            self._X.ops_funcs
        )

        self._y_train = ArrayGenerator(
            self._memory_y,
            self._train_indexes,
            (int(len(self._train_indexes) / (len(self._y.cur_indexes_on_memory_data_index) / self._y.batch_size)),
             *(self._y.shape[1:])),
            self._y.ops_funcs
        )

        self._ids_train = ArrayGenerator(
            self._memory_ids,
            self._train_indexes,
            (int(len(self._train_indexes) / (len(self._ids.cur_indexes_on_memory_data_index) / self._ids.batch_size)),
             *(self._ids.shape[1:])),
            self._ids.ops_funcs
        )

        self._X_val = ArrayGenerator(
            self._memory_X,
            self._val_indexes,
            (int(len(self._val_indexes) / (len(self._X.cur_indexes_on_memory_data_index) / self._X.batch_size)),
             *(self._X.shape[1:])),
            self._X.ops_funcs
        )

        self._y_val = ArrayGenerator(
            self._memory_y,
            self._val_indexes,
            (int(len(self._val_indexes) / (len(self._y.cur_indexes_on_memory_data_index) / self._y.batch_size)),
             *(self._y.shape[1:])),
            self._y.ops_funcs
        )

        self._ids_val = ArrayGenerator(
            self._memory_ids,
            self._val_indexes,
            (int(len(self._val_indexes) / (len(self._ids.cur_indexes_on_memory_data_index) / self._ids.batch_size)),
             *(self._ids.shape[1:])),
            self._ids.ops_funcs
        )

        self._X_test = ArrayGenerator(
            self._memory_X,
            self._test_indexes,
            (int(len(self._test_indexes) / (len(self._X.cur_indexes_on_memory_data_index) / self._X.batch_size)),
             *(self._X.shape[1:])),
            self._X.ops_funcs
        )

        self._y_test = ArrayGenerator(
            self._memory_y,
            self._test_indexes,
            (int(len(self._test_indexes) / (len(self._y.cur_indexes_on_memory_data_index) / self._y.batch_size)),
             *(self._y.shape[1:])),
            self._y.ops_funcs
        )

        self._ids_test = ArrayGenerator(
            self._memory_ids,
            self._test_indexes,
            (int(len(self._test_indexes) / (len(self._ids.cur_indexes_on_memory_data_index) / self._ids.batch_size)),
             *(self._ids.shape[1:])),
            self._ids.ops_funcs
        )
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
    def arrange_by_split_date_stock(cls, data, *args):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        d = data
        num_stocks = Constant.NUM_STOCKS
        num_dates = int(d.shape[0] / num_stocks)
        other_dims = d.shape[1:]
        d = d.reshape(num_stocks, num_dates, *other_dims)
        d = np.transpose(d, tuple([1, 0] + list(range(2, d.ndim))))
        return d


class FirstDimensionReshaperOnUnWindowed(FirstDimensionReshaper):

    def post_split_data_process(self):
        self._X_train.add_ops_funcs(super().arrange_by_split_date_stock)
        self._y_train.add_ops_funcs(super().arrange_by_split_date_stock)
        self._ids_train.add_ops_funcs(super().arrange_by_split_date_stock)

        self._X_val.add_ops_funcs(super().arrange_by_split_date_stock)
        self._y_val.add_ops_funcs(super().arrange_by_split_date_stock)
        self._ids_val.add_ops_funcs(super().arrange_by_split_date_stock)

        self._X_test.add_ops_funcs(super().arrange_by_split_date_stock)
        self._y_test.add_ops_funcs(super().arrange_by_split_date_stock)
        self._ids_test.add_ops_funcs(super().arrange_by_split_date_stock)

        num_days_train = len(self._X_train.memory_data_index_range) / Constant.NUM_STOCKS
        first_trains = [int(i * num_days_train + j) for i in range(Constant.NUM_STOCKS)
                        for j in range(self._X_train.batch_size)]

        num_days_val = len(self._X_val.memory_data_index_range) / Constant.NUM_STOCKS
        first_vals = [int(i * num_days_val + j) for i in range(Constant.NUM_STOCKS)
                      for j in range(self._X_val.batch_size)]

        num_days_test = len(self._X_test.memory_data_index_range) / Constant.NUM_STOCKS
        first_tests = [int(i * num_days_test + j) for i in range(Constant.NUM_STOCKS)
                       for j in range(self._X_test.batch_size)]

        self._X_train.update_indexes(first_trains)
        self._y_train.update_indexes(first_trains)
        self._ids_train.update_indexes(first_trains)

        self._X_val.update_indexes(first_vals)
        self._y_val.update_indexes(first_vals)
        self._ids_val.update_indexes(first_vals)

        self._X_test.update_indexes(first_tests)
        self._y_test.update_indexes(first_tests)
        self._ids_test.update_indexes(first_tests)

        self._X_train.update_shape((num_days_train, Constant.NUM_STOCKS, *self._X_train.shape[1:]))
        self._y_train.update_shape((num_days_train, Constant.NUM_STOCKS, 1))
        self._ids_train.update_shape((num_days_train, Constant.NUM_STOCKS, *self._ids_train.shape[1:]))

        self._X_val.update_shape((num_days_val, Constant.NUM_STOCKS, *self._X_val.shape[1:]))
        self._y_val.update_shape((num_days_val, Constant.NUM_STOCKS, 1))
        self._ids_val.update_shape((num_days_val, Constant.NUM_STOCKS, *self._ids_val.shape[1:]))

        self._X_test.update_shape((num_days_test, Constant.NUM_STOCKS, *self._X_test.shape[1:]))
        self._y_test.update_shape((num_days_test, Constant.NUM_STOCKS, 1))
        self._ids_test.update_shape((num_days_test, Constant.NUM_STOCKS, *self._ids_test.shape[1:]))

        pass


class FirstDimensionReshaperOnWindowed(FirstDimensionReshaper):

    def post_split_data_process(self):
        self._X_train.add_ops_funcs(self.arrange_by_split_date_stock)
        self._y_train.add_ops_funcs(self.arrange_by_split_date_stock)
        self._ids_train.add_ops_funcs(super().arrange_by_split_date_stock)

        self._X_val.add_ops_funcs(self.arrange_by_split_date_stock)
        self._y_val.add_ops_funcs(self.arrange_by_split_date_stock)
        self._ids_val.add_ops_funcs(super().arrange_by_split_date_stock)

        self._X_test.add_ops_funcs(self.arrange_by_split_date_stock)
        self._y_test.add_ops_funcs(self.arrange_by_split_date_stock)
        self._ids_test.add_ops_funcs(super().arrange_by_split_date_stock)

        num_days_train = len(self._X_train.memory_data_index_range) / Constant.NUM_STOCKS
        first_trains = [int(i * num_days_train + j) for i in range(Constant.NUM_STOCKS)
                        for j in range(self._X_train.batch_size)]

        num_days_val = len(self._X_val.memory_data_index_range) / Constant.NUM_STOCKS
        first_vals = [int(i * num_days_val + j) for i in range(Constant.NUM_STOCKS)
                      for j in range(self._X_val.batch_size)]

        num_days_test = len(self._X_test.memory_data_index_range) / Constant.NUM_STOCKS
        first_tests = [int(i * num_days_test + j) for i in range(Constant.NUM_STOCKS)
                       for j in range(self._X_test.batch_size)]

        self._X_train.update_indexes(first_trains)
        self._y_train.update_indexes(first_trains)
        self._ids_train.update_indexes(first_trains)

        self._X_val.update_indexes(first_vals)
        self._y_val.update_indexes(first_vals)
        self._ids_val.update_indexes(first_vals)

        self._X_test.update_indexes(first_tests)
        self._y_test.update_indexes(first_tests)
        self._ids_test.update_indexes(first_tests)

        self._X_train.update_shape((num_days_train, Constant.WIN_SIZE, Constant.NUM_STOCKS, self._X_train.shape[-1]))
        self._y_train.update_shape(
            (num_days_train, Constant.FORWARD_STEPS_SIZE, Constant.NUM_STOCKS, self._y_train.shape[-1]))
        self._ids_train.update_shape((num_days_train, Constant.NUM_STOCKS, self._ids_train.shape[-1]))

        self._X_val.update_shape((num_days_val, Constant.WIN_SIZE, Constant.NUM_STOCKS, self._X_val.shape[-1]))
        self._y_val.update_shape(
            (num_days_val, Constant.FORWARD_STEPS_SIZE, Constant.NUM_STOCKS, self._y_val.shape[-1]))
        self._ids_val.update_shape((num_days_val, Constant.NUM_STOCKS, self._ids_val.shape[-1]))

        self._X_test.update_shape((num_days_test, Constant.WIN_SIZE, Constant.NUM_STOCKS, self._X_test.shape[-1]))
        self._y_test.update_shape(
            (num_days_test, Constant.FORWARD_STEPS_SIZE, Constant.NUM_STOCKS, self._y_test.shape[-1]))
        self._ids_test.update_shape((num_days_test, Constant.NUM_STOCKS, self._ids_test.shape[-1]))

        pass

    @classmethod
    def arrange_by_split_date_stock(cls, data, *args):
        data = super().arrange_by_split_date_stock(data)
        return np.transpose(data, (0, 2, 1, 3))


class SpecialXConverter(PostSplitDataDecorator):
    def post_split_data_process(self):
        self._X_train = copy.deepcopy(self._y_train)
        self._X_val = copy.deepcopy(self._y_val)
        self._X_test = copy.deepcopy(self._y_test)


class TestDataHandlerOnFirstDimensionAsRecord(PostSplitDataDecorator):

    def post_split_data_process(self):
        num_days_test = len(self._X_test.memory_data_index_range) / Constant.NUM_STOCKS
        first_tests = [int(i * num_days_test) for i in range(Constant.NUM_STOCKS)]
        self._X_test.update_indexes(first_tests)
        self._X_test.update_shape((Constant.NUM_STOCKS, *self._X_test.shape[1:]))
        self._X_test.update_limit(1)
        self._X_test.update_batch_size(1)
        pass


class TestDataHandlerOnFirstDimensionAsDate(PostSplitDataDecorator):

    def post_split_data_process(self):
        num_days_test = len(self._X_test.memory_data_index_range) / Constant.NUM_STOCKS
        first_tests = [int(i * num_days_test) for i in range(Constant.NUM_STOCKS)]
        self._X_test.update_indexes(first_tests)
        self._X_test.update_limit(1)
        self._X_test.update_shape((1, *self._X_test.shape[1:]))
        self._X_test.update_batch_size(1)

        self._ids_test.clear_ops_funcs()
        self._ids_test.update_indexes(list(range(self._ids_test.batch_size)))
        self._ids_test.update_shape((self._ids_test.shape[0] * self._ids_test.shape[1], self._ids_test.shape[-1]))


class SpecialTrainDataWrapper(PostSplitDataDecorator):

    def _generator_wrapper(self, X, y):
        y_copy = copy.deepcopy(y)

        """
                def gen():
            X.reset()
            y_copy.reset()
            y.reset()
            while 1:
                try:
                    yield (X(), y_copy()), y()
                except StopIteration:
                    break
        """

        class gen(Generator):
            def __init__(self, X, y1, y2):
                self.X = X
                self.y1 = y1
                self.y2 = y2
                pass

            def __call__(self, *args, **kwargs):
                self.X.reset()
                self.y1.reset()
                self.y2.reset()
                return self

            def send(self, ignore_arg):
                try:
                    return (next(self.X), next(self.y1)), next(self.y2)
                except StopIteration:
                    self.throw()

            def throw(self, type=None, value=None, traceback=None):
                raise StopIteration

        data = self.tf_dataset.from_generator(
            gen(X, y_copy, y),
            output_types=((tf.float64, tf.float64), tf.float64),
            output_shapes=(
                ((None, *X.shape[1:]), (None, *y.shape[1:])),
                (None, *y.shape[1:]))
        )
        return data

    def post_split_data_process(self):
        pass
