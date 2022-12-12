import unittest
from Data.DataProcessor import *
import sys
import numpy as np
from Zoo import *

sys.path.append('./Stlearn')


class MyTestCase(unittest.TestCase):

    @classmethod
    def load(cls, processor, generator):

        data = processor.load_data()
        X_train = data.X_train
        X_val = data.X_val
        X_test = data.X_test

        y_train = data.y_train
        y_val = data.y_val

        ids_test = data.ids_test

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
            '2019-01-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=True), True)
        b = self.load(FourDimWinDataByDateByWinByStockByFeatureProcessor(
            '2019-01-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=False), False)
        for i in range(6):
            np.testing.assert_array_equal(a[i], b[i])
            print(str(i) + "passed")
        print(a[-1])
        print(b[-1])

    def test_5(self):
        import tracemalloc
        import psutil
        tracemalloc.start()

        snapshot1 = tracemalloc.take_snapshot()

        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        processor = FourDimWinDataByDateByWinByStockByFeatureProcessor('2019-09-01', '2020-01-01', '2020-02-01',
                                                                       '2020-03-01', 30, 15, True)
        data = processor.load_data()
        data_train = data.get_train()
        data_val = data.get_val()

        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        model = ConditionalVariationalAutoEncoder()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=Constant.ERROR,
                      metrics=[Constant.ERROR], weighted_metrics=[Constant.ERROR])
        h = model.fit(data_train,
                      epochs=Constant.MAX_EPOCHS,
                      validation_data=data_val,
                      callbacks=[]
                      )

        id1 = id(data._memory_X)
        id2 = id(data._data)

        del data, model, data_train, data_val, processor
        import gc
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        topn = 20
        print("[ Top {} ]".format(topn))
        for stat in top_stats[:topn]:
            print(stat)

        import ctypes
        import gc
        l = gc.get_referrers(ctypes.cast(id1, ctypes.py_object).value)
        ll = gc.get_referrers(ctypes.cast(id2, ctypes.py_object).value)

        print(len(l))
        print(len(ll))

        for il in l:
            print(il)
        print('------------------------------')
        for ill in ll:
            print(ill)

        pass



if __name__ == '__main__':
    unittest.main()
