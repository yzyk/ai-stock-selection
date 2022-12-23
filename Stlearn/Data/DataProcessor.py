from . import GeneratedDataRecipe
from . import InMemoryDataRecipe
from .DataInterface import *


class DataProcessor(ABC):
    def __init__(self, train_start, val_start, test_start, test_end, generator=False, *args):
        if generator:
            self._module = GeneratedDataRecipe
        else:
            self._module = InMemoryDataRecipe
        self._train_start = train_start
        self._val_start = val_start
        self._test_start = test_start
        self._test_end = test_end
        pass

    def load_data(self, *args):
        data = self._load_data(*args)
        data.log()
        return data

    @abstractmethod
    def _load_data(self, *args):
        pass


class TwoDimUnWinDataByStockDateByFeatureProcessor(DataProcessor):

    def _load_data(self):
        data = self._module.DiskData(self._train_start, self._val_start, self._test_start, self._test_end)
        data = self._module.PreSplitScaler(data)
        data = self._module.StandardSplitProcessor(data)
        data = self._module.TestDataHandlerOnFirstDimensionAsRecord(data)
        return data


class ThreeDimUnWinDataByDateByStockByFeatureProcessor(DataProcessor):

    def _load_data(self):
        data = self._module.DiskData(self._train_start, self._val_start, self._test_start, self._test_end)
        data = self._module.PreSplitScaler(data)
        data = self._module.StandardSplitProcessor(data)
        data = self._module.FirstDimensionReshaperOnUnWindowed(data)
        data = self._module.TestDataHandlerOnFirstDimensionAsDate(data)
        return data


class ThreeDimWinDataByStockDateByWinByFeatureProcessor(DataProcessor):

    def __init__(self, train_start, val_start, test_start, test_end, win_size, forward_size, generator=False, *args):
        self._win_size = win_size
        self._forward_size = forward_size
        super().__init__(train_start, val_start, test_start, test_end, generator)

    def _load_data(self, *args):
        data = self._module.DiskData(self._train_start, self._val_start, self._test_start, self._test_end)
        data = self._module.PreSplitScaler(data)
        data = self._module.WindowGenerator(data, self._win_size, self._forward_size)
        data = self._module.StandardSplitProcessor(data)
        data = self._module.TestDataHandlerOnFirstDimensionAsRecord(data)
        return data


class FourDimWinDataByDateByWinByStockByFeatureProcessor(DataProcessor):

    def __init__(self, train_start, val_start, test_start, test_end, win_size, forward_size, generator=False, *args):
        self._win_size = win_size
        self._forward_size = forward_size
        super().__init__(train_start, val_start, test_start, test_end, generator)

    def _load_data(self):
        data = self._module.DiskData(self._train_start, self._val_start, self._test_start, self._test_end)
        data = self._module.PreSplitScaler(data)
        data = self._module.WindowGenerator(data, self._win_size, self._forward_size)
        data = self._module.StandardSplitProcessor(data)
        data = self._module.FirstDimensionReshaperOnWindowed(data)
        data = self._module.TestDataHandlerOnFirstDimensionAsDate(data)
        data = self._module.SpecialTrainDataWrapper(data)
        return data


class FourDimWinDataByDateByWinByStockByFeatureSetXToYProcessor(DataProcessor):

    def __init__(self, train_start, val_start, test_start, test_end, win_size, forward_size, generator=False, *args):
        self._win_size = win_size
        self._forward_size = forward_size
        super().__init__(train_start, val_start, test_start, test_end, generator)

    def _load_data(self):
        data = self._module.DiskData(self._train_start, self._val_start, self._test_start, self._test_end)
        data = self._module.PreSplitScaler(data)
        data = self._module.WindowGenerator(data, self._win_size, self._forward_size)
        data = self._module.StandardSplitProcessor(data)
        data = self._module.FirstDimensionReshaperOnWindowed(data)
        data = self._module.SpecialXConverter(data)
        data = self._module.TestDataHandlerOnFirstDimensionAsDate(data)
        return data


class DiskDataTestProcessor(DataProcessor):

    def __init__(self, path, generator=False, *args):
        super().__init__(None, None, None, None, generator)
        self.path = path

    def _load_data(self, *args):
        data = self._module.DiskDataTest(self.path)
        return data
