import unittest
import numpy as np
import sys
sys.path.append('../Stlearn')
sys.path.append('../Stlearn/Data')

from Data.DataProcessor import *


class MyTestCase(unittest.TestCase):

    @classmethod
    def load(cls, processor, generator):
        if generator:
            data = processor.load_data()
            X_train = data._X_train.generate_all()
            X_val = data._X_val.generate_all()
            X_test = data._X_test.generate_all()

            y_train = data._y_train.generate_all()
            y_val = data._y_val.generate_all()

            ids_test = data._ids_test.generate_all()
        else:
            data = processor.load_data()
            X_train = data._X_train
            X_val = data._X_val
            X_test = data._X_test

            y_train = data._y_train
            y_val = data._y_val

            ids_test = data._ids_test

        return X_train, X_val, X_test, y_train, y_val, ids_test

    def test_1(self):
        a = self.load(TwoDimUnWinDataByStockDateByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', generator=True), True)
        b = self.load(TwoDimUnWinDataByStockDateByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', generator=False), False)
        for i in range(6):
            np.testing.assert_array_equal(a[i], b[i])
            print(str(i) + "passed")

    def test_2(self):
        a = self.load(ThreeDimUnWinDataByDateByStockByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', generator=True), True)
        b = self.load(ThreeDimUnWinDataByDateByStockByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', generator=False), False)
        print(a[0][-1])
        for i in range(6):
            np.testing.assert_array_equal(a[i], b[i])
            print(str(i) + "passed")

    def test_3(self):
        a = self.load(ThreeDimWinDataByStockDateByWinByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=True), True)
        b = self.load(ThreeDimWinDataByStockDateByWinByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=False), False)
        for i in range(6):
            np.testing.assert_array_equal(a[i], b[i])
            print(str(i) + "passed")

    def test_4(self):
        a = self.load(FourDimWinDataByDateByWinByStockByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=True), True)
        b = self.load(FourDimWinDataByDateByWinByStockByFeatureProcessor(
            '2019-07-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=False), False)
        for i in range(6):
            np.testing.assert_array_equal(a[i], b[i])
            print(str(i) + "passed")


if __name__ == '__main__':
    unittest.main()
