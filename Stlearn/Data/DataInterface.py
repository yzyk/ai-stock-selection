from __future__ import annotations

from abc import ABC, abstractmethod
import tensorflow as tf
import copy


class Data(ABC):

    def __init__(self) -> None:
        pass

    def copy_constructor(self, data):
        for i in data.__dict__.keys():
            self.__dict__[i] = data.__dict__[i]
        pass

    @abstractmethod
    def get_train(self):
        return None

    @abstractmethod
    def get_val(self):
        return None

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def generate(self, *args):
        pass

    def clone(self):
        return copy.deepcopy(self)

    @abstractmethod
    def X_test(self):
        return None

    @abstractmethod
    def ids_test(self):
        return None


class InMemoryData(Data, ABC):
    _data = None

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

    _first_indexes = None
    _train_indexes = None
    _val_indexes = None
    _test_indexes = None

    def __init__(self) -> None:
        super().__init__()
        pass

    def get_train(self) -> tuple:
        return self._X_train, self._y_train

    def get_val(self) -> tuple:
        return self._X_val, self._y_val

    def log(self):
        print("X_train shape: " + str(self._X_train.shape))
        print("y_train shape: " + str(self._y_train.shape))
        print("X_valid shape: " + str(self._X_val.shape))
        print("y_valid shape: " + str(self._y_val.shape))

    @abstractmethod
    def generate(self, *args):
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


class GeneratedData(Data, ABC):
    _data = None

    _memory_X = None
    _memory_y = None
    _memory_ids = None

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

    _first_indexes = None
    _train_indexes = None
    _val_indexes = None
    _test_indexes = None

    def __init__(self) -> None:
        super().__init__()
        pass

    def get_train(self):
        return self._generator_wrapper(self._X_train, self._y_train)

    def get_val(self):
        return self._generator_wrapper(self._X_val, self._y_val)

    def log(self):
        print("X_train shape: " + str(self._X_train.shape))
        print("y_train shape: " + str(self._y_train.shape))
        print("X_valid shape: " + str(self._X_val.shape))
        print("y_valid shape: " + str(self._y_val.shape))

    @classmethod
    def _generator_wrapper(cls, X, y):
        y_copy = copy.deepcopy(y)

        def gen():
            X.reset()
            y_copy.reset()
            y.reset()
            while 1:
                try:
                    yield (X(), y_copy()), y()
                except StopIteration:
                    break

        data = tf.data.Dataset.from_generator(
            gen,
            output_types=((tf.float32, tf.float32), tf.float32),
            output_shapes=(
                ((None, *X.shape[1:]), (None, *y.shape[1:])),
                (None, *y.shape[1:]))
        )
        return data

    @abstractmethod
    def generate(self, *args):
        pass

    @classmethod
    def tf_data(cls, data):
        ds = tf.data.Dataset.from_generator(
            data,
            output_types=tf.float32,
            output_shapes=data.shape
        )
        return ds

    @property
    def X_train(self):
        return self._X_train.generate_all()

    @property
    def X_val(self):
        return self._X_val.generate_all()

    @property
    def X_test(self):
        return self._X_test.generate_all()

    @property
    def y_train(self):
        return self._y_train.generate_all()

    @property
    def y_val(self):
        return self._y_val.generate_all()

    @property
    def y_test(self):
        return self._y_test.generate_all()

    @property
    def ids_test(self):
        return self._ids_test.generate_all()

    @property
    def X_train_gen(self):
        return self._X_train

    @property
    def X_val_gen(self):
        return self._X_val

    @property
    def X_test_gen(self):
        return self._X_test

    @property
    def y_train_gen(self):
        return self._y_train

    @property
    def y_val_gen(self):
        return self._y_val

    @property
    def y_test_gen(self):
        return self._y_test

    @property
    def ids_test_gen(self):
        return self._ids_test

