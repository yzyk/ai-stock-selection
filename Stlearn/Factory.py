from __future__ import annotations

import Constant
from Data.DataProcessor import *
from Model import *


class StlearnFactory(ABC):
    _data = None
    _model = None

    _train_start = None
    _val_start = None
    _test_start = None
    _test_end = None

    _data_window_size = None

    def __init__(self, train_start, val_start, test_start, test_end):
        self._train_start = train_start
        self._val_start = val_start
        self._test_start = test_start
        self._test_end = test_end
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
        self._data = TwoDimUnWinDataByStockDateByFeatureProcessor(train_start=self._train_start,
                                                                  val_start=self._val_start,
                                                                  test_start=self._test_start,
                                                                  test_end=self._test_end).load_data()
        pass


class DlFactory(StlearnFactory):

    def __init__(self, train_start, val_start, test_start, test_end, data_window_size=30, forward_size=15,
                 generator=False):
        self._window_size = data_window_size
        self._forward_size = forward_size
        self._generator = generator
        super().__init__(train_start, val_start, test_start, test_end)

    @abstractmethod
    def _load(self):
        self._data = ThreeDimWinDataByStockDateByWinByFeatureProcessor(train_start=self._train_start,
                                                                       val_start=self._val_start,
                                                                       test_start=self._test_start,
                                                                       test_end=self._test_end,
                                                                       win_size=self._data_window_size,
                                                                       forward_size=self._forward_size,
                                                                       generator=self._generator).load_data()
        pass


class LinearRegressionFactory(MlFactory):

    def _load(self):
        super()._load()
        self._model = LinearRegressionModel('lr')

    pass


class RandomForestRegressorFactory(MlFactory):

    def _load(self):
        super()._load()
        self._model = RandomForestRegressorModel('rf')

    pass


class AdaBoostRegressorFactory(MlFactory):

    def _load(self):
        super()._load()
        self._model = AdaBoostRegressorModel('Ada')
        pass


class SimpleNNFactory(DlFactory):

    def _load(self):
        super()._load()
        self._model = SimpleNNModel('SimpleNN')
        pass


class CNNFactory(DlFactory):

    def _load(self):
        super()._load()
        self._model = CNNModel('CNN')
        pass


class LSTMFactory(DlFactory):

    def _load(self):
        super()._load()
        self._model = LSTMModel('LSTM')
        pass


class LSTMAutoRegressorFactory(DlFactory):

    def _load(self):
        super()._load()
        self._model = LSTMAutoRegressorModel('LSTMAutoRegressor')
        pass


class CNNAutoRegressorFactory(DlFactory):

    def _load(self):
        super()._load()
        self._model = CNNAutoRegressorModel('CNNAutoRegressor')
        pass


class StandardVariationAutoEncoderFactory(DlFactory):

    def _load(self):
        self._data = ThreeDimUnWinDataByDateByStockByFeatureProcessor(train_start=self._train_start,
                                                                      val_start=self._val_start,
                                                                      test_start=self._test_start,
                                                                      test_end=self._test_end).load_data()
        self._model = StandardVariationAutoEncoderModel('VAE')
        pass


class ConditionalVariationAutoEncoderFactory(DlFactory):

    def _load(self):
        self._data = FourDimWinDataByDateByWinByStockByFeatureProcessor(train_start=self._train_start,
                                                                        val_start=self._val_start,
                                                                        test_start=self._test_start,
                                                                        test_end=self._test_end,
                                                                        win_size=self._window_size,
                                                                        forward_size=self._forward_size,
                                                                        generator=self._generator).load_data()
        self._model = ConditionalVariationAutoEncoderModel('VAE')
        pass
