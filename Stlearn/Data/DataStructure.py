import copy
from collections import Generator
import numpy as np

import Constant


class Date:
    _year = None
    _month = None
    _day = None

    _month_dict = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }

    def __init__(self, year, month, day):
        self._year = year
        self._month = month
        self._day = day
        self._validate()
        pass

    @classmethod
    def create_from_str(cls, s):
        l = s.split('-')
        year = int(l[0])
        month = int(l[1])
        day = int(l[2])
        return Date(year, month, day)

    def __str__(self):
        return str(self._year) + '-' + (
            '0' + str(self._month) if len(str(self._month)) == 1 else str(self._month)
        ) + '-' + (
                   '0' + str(self._day) if len(str(self._day)) == 1 else str(self._day)
               )

    def clone(self):
        return Date(self._year, self._month, self._day)

    def _validate(self):
        if self._year % 4 == 0:
            self._month_dict[2] = 29
        else:
            self._month_dict[2] = 28

        months_limit = 12

        while self._month > months_limit:
            self._month -= months_limit
            self._year += 1
            if self._year % 4 == 0:
                self._month_dict[2] = 29
            else:
                self._month_dict[2] = 28

        days_limit = self._month_dict[self._month]

        while self._day > days_limit:
            self._day -= days_limit
            self._month += 1

            while self._month > months_limit:
                self._month -= months_limit
                self._year += 1
                if self._year % 4 == 0:
                    self._month_dict[2] = 29
                else:
                    self._month_dict[2] = 28

            days_limit = self._month_dict[self._month]

        pass

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def day(self):
        return self._day


class DateGenerator:
    _next_date = None
    _freq = None
    _limit = None

    _year_offset = 0
    _month_offset = 0
    _day_offset = 0

    def __init__(self, start_date, freq, offset, limit):
        self._next_date = start_date
        self._freq = freq
        self._limit = limit

        if freq == 'year':
            self._year_offset = offset

        if freq == 'month':
            self._month_offset = offset

        if freq == 'day':
            self._day_offset = offset

        pass

    def next(self):
        if not self.has_next():
            return None
        res = self._next_date.clone()
        self._next_date = Date(self._next_date.year + self._year_offset,
                               self._next_date.month + self._month_offset,
                               self._next_date.day + self._day_offset)
        self._limit -= 1
        return res

    def has_next(self):
        return self._limit != 0


class ArrayGenerator(Generator):

    def __init__(self, memory_data,
                 memory_data_index_range=None, data_shape=None,
                 ops_funcs=None, start_indexes_on_index=None):
        self.batch_size = Constant.BATCH_SIZE
        self._memory_data = memory_data
        self._memory_data_index_range = np.array(copy.deepcopy(memory_data_index_range))
        self._data_shape = tuple([int(i) for i in data_shape])
        if start_indexes_on_index is not None:
            self._cur_indexes_on_memory_data_index = np.array(copy.deepcopy(start_indexes_on_index))
        else:
            self._cur_indexes_on_memory_data_index = np.array(list(range(self.batch_size)))
        self._steps = self.batch_size
        self._count = 0
        self._limit = int(len(self._memory_data_index_range) / len(self._cur_indexes_on_memory_data_index)) + (
                len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) > 0
        )
        self._last_mod = int(len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) / (
                len(self._cur_indexes_on_memory_data_index) / self.batch_size))

        self._ops_funcs = []
        if ops_funcs is not None:
            self._ops_funcs += copy.deepcopy(ops_funcs)
        super().__init__()

        self._start_indexes_on_indexes = copy.deepcopy(self._cur_indexes_on_memory_data_index)
        pass

    def __call__(self, *args, **kwargs):
        return self.__next__()

    def _compute(self, data):
        for func in self._ops_funcs:
            data = func(data, self._memory_data_index_range, self._cur_indexes_on_memory_data_index, self._steps)
        return data

    def reset(self):
        self._cur_indexes_on_memory_data_index = copy.deepcopy(self._start_indexes_on_indexes)
        self._count = 0
        pass

    def send(self, ignored_arg):
        if self._count == self._limit:
            self.throw()
        if self._count == (self._limit - 1):
            if self._last_mod > 0:
                self._cur_indexes_on_memory_data_index = np.array(
                    [(i * (self.batch_size * self._count + self._last_mod) + self._count * self.batch_size + j) for i in
                     range(
                         int(len(self._cur_indexes_on_memory_data_index) / self.batch_size)) for j in
                     range(self._last_mod)]
                )
        indexes = self._memory_data_index_range[self._cur_indexes_on_memory_data_index]
        if len(self._ops_funcs) > 0 and 'pre_split' in self._ops_funcs[0].__name__:
            data = self._memory_data
        else:
            data = self._memory_data[indexes]
        data = self._compute(data)
        self._cur_indexes_on_memory_data_index += self._steps
        self._count += 1
        return data

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def update_memory_data(self, memory_data):
        self._memory_data = memory_data
        pass

    def update_indexes(self, indexes):
        self._cur_indexes_on_memory_data_index = np.array(copy.deepcopy(indexes))
        self._limit = int(len(self._memory_data_index_range) / len(self._cur_indexes_on_memory_data_index)) + (
                len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) > 0
        )
        self._last_mod = int(len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) / (
                len(self._cur_indexes_on_memory_data_index) / self.batch_size))
        self._start_indexes_on_indexes = copy.deepcopy(self._cur_indexes_on_memory_data_index)
        pass

    def update_index_range(self, index_range):
        self._memory_data_index_range = np.array(copy.deepcopy(index_range))
        self._limit = int(len(self._memory_data_index_range) / len(self._cur_indexes_on_memory_data_index)) + (
                len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) > 0
        )
        self._last_mod = int(len(self._memory_data_index_range) % len(self._cur_indexes_on_memory_data_index) / (
                len(self._cur_indexes_on_memory_data_index) / self.batch_size))
        pass

    def add_ops_funcs(self, func):
        self._ops_funcs.append(copy.deepcopy(func))
        pass

    def update_shape(self, shape):
        self._data_shape = tuple([int(i) for i in shape])
        pass

    def update_limit(self, limit):
        self._limit = limit
        pass

    def clear_ops_funcs(self):
        self._ops_funcs = []
        pass

    def update_batch_size(self, size):
        self.batch_size = size
        self._steps = size
        pass

    @property
    def shape(self):
        return self._data_shape

    @property
    def memory_data_index_range(self):
        return self._memory_data_index_range

    @property
    def cur_indexes_on_memory_data_index(self):
        return self._cur_indexes_on_memory_data_index

    @property
    def ops_funcs(self):
        return self._ops_funcs

    def generate_all(self):
        self.reset()
        res = np.empty(self._data_shape, dtype=object)
        cur = 0
        while 1:
            try:
                a = self.send(None)
                res[cur: cur + a.shape[0]] = a
                cur += a.shape[0]
            except StopIteration:
                break

        try:
            res = res.astype('float32')
        except ValueError:
            pass
        return res
