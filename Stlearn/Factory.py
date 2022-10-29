from __future__ import annotations

from Data import *
from Model import *


class StlearnFactory(ABC):

    _data = None
    _model = None

    _train_portion = None
    _val_portion = None
    _test_portion = None

    _capacity = None

    def __init__(self, capacity, train_portion, val_portion, test_portion):
        self._train_portion = train_portion
        self._val_portion = val_portion
        self._test_portion = test_portion
        self._capacity = capacity
        self._load()
        pass

    def create_data(self) -> Data:
        return self._data

    def create_model(self) -> Model:
        return self._model

    @abstractmethod
    def _load(self):
        pass


class MlFactory(StlearnFactory):

    @abstractmethod
    def _load(self):
        pass


class DlFactory(StlearnFactory):

    def __init__(self, capacity, train_portion, val_portion, test_portion, data_window_size):
        self._data_window_size = data_window_size
        super().__init__(capacity, train_portion, val_portion, test_portion)

    @abstractmethod
    def _load(self):
        pass


class LinearRegressionFactory(MlFactory):

    def _load(self):
        self._data = MlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion)
        self._model = LinearRegressionModel('lr')
    pass


class RandomForestRegressorFactory(MlFactory):

    def _load(self):
        self._data = MlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion)
        self._model = RandomForestRegressorModel('rf')
    pass


class AdaBoostRegressorFactory(MlFactory):

    def _load(self):
        self._data = MlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion)
        self._model = AdaBoostRegressorModel('Ada')
        pass


class SimpleNNFactory(DlFactory):

    def _load(self):
        self._data = DlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion, data_window_size=self._data_window_size)
        self._model = SimpleNNModel('SimpleNN', self._data.get_shape())
        pass


class CNNFactory(DlFactory):

    def _load(self):
        self._data = DlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion, data_window_size=self._data_window_size)
        self._model = CNNModel('CNN', self._data.get_shape())
        pass


class LSTMFactory(DlFactory):

    def _load(self):
        self._data = DlData(capacity=self._capacity, train_portion=self._train_portion, val_portion=self._val_portion,
                            test_portion=self._test_portion, data_window_size=self._data_window_size)
        self._model = LSTMModel('LSTM', self._data.get_shape())
        pass

