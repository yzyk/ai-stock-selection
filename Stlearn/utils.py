import numpy as np
from sklearn.metrics import *
import os

DATA_PATH = "./data/data.parquet.gzip"

MODEL_IMAGE_PATH = os.path.realpath('./plot')

ERROR = 'mean_squared_error'

RF = 0.03

N_DAYS = 252


def performance_measure(y_true, y_pred):
    m_ = eval(ERROR)
    print('weighted_{}: {:.5f}'.format(
        ERROR, m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))
    )
    )
    return m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))


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
