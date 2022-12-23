import unittest
from Data.DataProcessor import *
import sys
import numpy as np
from Zoo import *
from utils import GarbageCollectorCallback
import sys

from Factory import *
from Portfolio import *
from Data.DataStructure import DateGenerator, Date
from tqdm import tqdm
import psutil
import tracemalloc
import Constant
import gc
import tensorflow as tf
from utils import *

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
        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        print(data._memory_X)
        print(data._X_val)

        print('--------------------')

        id1 = id(data._memory_X)
        id2 = id(data._X_val)

        del model, data_train, data_val, processor
        data.clean_tf_dataset()
        del data
        import gc
        gc.collect()

        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

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

        for ill in ll:
            print(ill)

        pass

    def test_6(self):
        import tracemalloc
        import psutil

        tracemalloc.start()

        for i in range(5):
            print('-----------------------------------------------')

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
                          callbacks=[GarbageCollectorCallback()]
                          )
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            del model, data_train, data_val, processor
            data.clean_tf_dataset()
            del data
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)


    def test_7(self):
        import sys
        sys.path.append('/content/drive/MyDrive/capstone/Deliverable/project')
        sys.path.append('/content/drive/MyDrive/capstone/Deliverable/project/Stlearn')
        from Data.DataProcessor import ThreeDimWinDataByStockDateByWinByFeatureProcessor, \
            FourDimWinDataByDateByWinByStockByFeatureProcessor
        from Model import LSTMAutoRegressorModel, ConditionalVariationAutoEncoderModel
        import numpy as np
        from Zoo import ConditionalVariationalAutoEncoder
        from utils import display_top
        import tracemalloc
        import psutil
        import gc
        import tensorflow as tf
        import Constant

        class GarbageCollectorCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()

        tracemalloc.start()

        flag = True

        for i in range(5):
            print('-----------------------------------------------')

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            processor = FourDimWinDataByDateByWinByStockByFeatureProcessor('2019-06-01', '2020-01-01', '2020-02-01',
                                                                           '2020-03-01', 30, 15, True)
            data = processor.load_data()
            data_train = data.get_train()
            data_val = data.get_val()

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            model = ConditionalVariationAutoEncoderModel('cvae')
            '''
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=Constant.ERROR,
                          metrics=[Constant.ERROR], weighted_metrics=[Constant.ERROR])
            '''
            h = model._model.fit(data_train, None,
                                 epochs=Constant.MAX_EPOCHS,
                                 validation_data=data_val,
                                 callbacks=[GarbageCollectorCallback()]
                                 )
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            # print(id(data))
            # print(id(data_train))
            # print(id(data_val))
            # print(id(model))

            del h, model, data_train, data_val, processor
            if flag:
                data.clean_tf_dataset()
            del data

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            tf.keras.backend.clear_session()
            gc.collect()

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)
            pass


    def test_8(self):
        import sys
        sys.path.append('/content/drive/MyDrive/capstone/Deliverable/project')
        sys.path.append('/content/drive/MyDrive/capstone/Deliverable/project/Stlearn')
        from Data.DataProcessor import ThreeDimWinDataByStockDateByWinByFeatureProcessor, \
            FourDimWinDataByDateByWinByStockByFeatureProcessor
        from Model import LSTMAutoRegressorModel, ConditionalVariationAutoEncoderGraphModel
        import numpy as np
        from Zoo import ConditionalVariationalAutoEncoder
        from utils import display_top
        import tracemalloc
        import psutil
        import gc
        import tensorflow as tf
        import Constant

        class GarbageCollectorCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()

        tracemalloc.start()

        flag = True

        for i in range(5):
            print('-----------------------------------------------')

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            processor = FourDimWinDataByDateByWinByStockByFeatureProcessor('2019-06-01', '2020-01-01', '2020-02-01',
                                                                           '2020-03-01', 30, 15, True)
            data = processor.load_data()
            data_train = data.get_train()
            data_val = data.get_val()

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            model = ConditionalVariationAutoEncoderGraphModel('cvae')
            '''
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=Constant.ERROR,
                          metrics=[Constant.ERROR], weighted_metrics=[Constant.ERROR])
            '''
            h = model._model.fit(data_train, None,
                                 epochs=Constant.MAX_EPOCHS,
                                 validation_data=data_val,
                                 callbacks=[GarbageCollectorCallback()]
                                 )
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            # print(id(data))
            # print(id(data_train))
            # print(id(data_val))
            # print(id(model))

            del h, model, data_train, data_val, processor
            if flag:
                data.clean_tf_dataset()
            del data

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            tf.keras.backend.clear_session()
            gc.collect()

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)
            pass


    def test_9(self):
        print(__file__)
        from Data.DataStructure import DateGenerator, Date
        from tqdm import tqdm
        import psutil
        import tracemalloc

        train_start_date_s = '2019-06-01'
        val_start_date_s = '2019-12-01'
        test_start_date_s = '2020-01-01'
        test_end_date_s = '2020-02-01'

        freq = 'month'
        offset = 1
        limit = 1

        train_start_date_generator = DateGenerator(Date.create_from_str(train_start_date_s), freq, offset, limit)
        val_start_date_generator = DateGenerator(Date.create_from_str(val_start_date_s), freq, offset, limit)
        test_start_date_generator = DateGenerator(Date.create_from_str(test_start_date_s), freq, offset, limit)
        test_end_date_generator = DateGenerator(Date.create_from_str(test_end_date_s), freq, offset, limit)

        tracemalloc.start()

        for i in tqdm(range(limit)):
            train_start_date = train_start_date_generator.next()
            val_start_date = val_start_date_generator.next()
            test_start_date = test_start_date_generator.next()
            test_end_date = test_end_date_generator.next()

            lg = 'Train{} To {}_Validation{} To {}_Test{} To {}'.format(
                str(train_start_date), str(val_start_date),
                str(val_start_date), str(test_start_date),
                str(test_start_date), str(test_end_date))
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(lg)
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            f = StandardVariationAutoEncoderFactory(str(train_start_date), str(val_start_date),
                            str(test_start_date), str(test_end_date), Constant.WIN_SIZE, Constant.FORWARD_STEPS_SIZE,
                            True)
            data = f.create_data()
            model = f.create_model()
            print(model._model._encoder.summary())
            print(model._model._decoder.summary())
            model.fit(data)

            portfolio = LongRandomPortfolio(data, model)
            print("Long Randomly Picked Stock Portfolio Performance: " + str(portfolio.performance))
            portfolio = LongBestPortfolio(data, model)
            print("Long Best Predicted Stock Portfolio Performance: " + str(portfolio.performance))
            portfolio = LongBestThreePortfolio(data, model)
            print("Long Three Highest Predicted Stock Portfolio Performance: " + str(portfolio.performance))
            portfolio = LongShortBestShotPortfolio(data, model)
            print("Long Short Best Shot Portfolio Performance: " + str(portfolio.performance))
            portfolio = LongMarketPortfolio(data, model)
            print("Long Market Portfolio Performance: " + str(portfolio.performance))

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            data.clean_tf_dataset()
            del f, model, data

            tf.keras.backend.clear_session()
            gc.collect()

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)


    def test_10(self):
        p1 = ThreeDimWinDataByStockDateByWinByFeatureProcessor(
            '2019-06-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=True)
        d1 = p1.load_data().get_train()
        X_gen = None
        y_gen = None
        for element in d1.as_numpy_iterator():
            x, y = element
            if X_gen is None:
                X_gen = x
                y_gen = y
            else:
                X_gen = np.concatenate([X_gen, x], axis=0)
                y_gen = np.concatenate([y_gen, y], axis=0)

        p2 = ThreeDimWinDataByStockDateByWinByFeatureProcessor(
            '2019-06-01', '2020-01-01', '2020-02-01', '2020-03-01', 30, 15, generator=False)
        d2 = p2.load_data().get_train()
        X_mem, y_mem = d2

        np.testing.assert_allclose(X_gen, X_mem, rtol=1e-5)
        np.testing.assert_allclose(y_gen, y_mem, rtol=1e-5)

        print(X_mem.dtype)


    def test_11(self):

        train_start_date_s = '2019-07-01'
        val_start_date_s = '2019-12-01'
        test_start_date_s = '2020-01-01'
        test_end_date_s = '2020-02-01'

        freq = 'month'
        offset = 1
        limit = 5

        train_start_date_generator = DateGenerator(Date.create_from_str(train_start_date_s), freq, offset, limit)
        val_start_date_generator = DateGenerator(Date.create_from_str(val_start_date_s), freq, offset, limit)
        test_start_date_generator = DateGenerator(Date.create_from_str(test_start_date_s), freq, offset, limit)
        test_end_date_generator = DateGenerator(Date.create_from_str(test_end_date_s), freq, offset, limit)

        tracemalloc.start()

        for i in tqdm(range(limit)):
            train_start_date = train_start_date_generator.next()
            val_start_date = val_start_date_generator.next()
            test_start_date = test_start_date_generator.next()
            test_end_date = test_end_date_generator.next()

            lg = 'Train{} To {}_Validation{} To {}_Test{} To {}'.format(
                str(train_start_date), str(val_start_date),
                str(val_start_date), str(test_start_date),
                str(test_start_date), str(test_end_date))
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(lg)
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            f = ConditionalVariationAutoEncoderFactory(str(train_start_date), str(val_start_date),
                                         str(test_start_date), str(test_end_date), Constant.WIN_SIZE,
                                         Constant.FORWARD_STEPS_SIZE, True)
            data = f.create_data()
            model = f.create_model()
            model.fit(data)

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

            data.clean_tf_dataset()
            del f, model, data

            tf.keras.backend.clear_session()
            gc.collect()

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)


if __name__ == '__main__':
    unittest.main()
