from __future__ import annotations

from abc import ABC, abstractmethod
from Data import *
from Model import *


class Performance:

    daily_return = None

    mean_return = None
    volatility = None
    sharpe_ratio = None

    def __init__(self):
        pass

    def update(self, daily_return):
        self.daily_return = daily_return
        self._cal()

    def _cal(self):
        self.mean_return = self.daily_return.mean()
        self.volatility = self.daily_return.std()
        self.sharpe_ratio = (self.mean_return - RF) / self.volatility
        pass


class Portfolio(ABC):
    _data = None
    _model = None
    _long_stocks = None
    _short_stocks = None

    _performance = None

    def __init__(self, data, model):
        self._data = data
        self._model = model
        pass

    def construct(self):
        pred = self._model.predict(self._data._X_test)
        df = pd.DataFrame([])
        df['pred'] = pd.DataFrame(pred)
        df[['Date', 'Ticker', 'Close', 'Return', 'Market Return']] = pd.DataFrame(self._data._ids_test)
        df['er'] = pd.DataFrame(self._data._y_test)
        self._strategy(df)
        daily_returns = df[df['Ticker'] == self._long_stocks][['Date', 'Return']].reset_index().drop('index', axis=1)
        self._update_performance(daily_returns)
        pass

    def _update_performance(self, daily_returns):
        self._performance.update(daily_returns)

    @abstractmethod
    def _strategy(self, df):
        return 0


class LongPortfolio(Portfolio):
    def _strategy(self, df):
        ds = df.groupby('Ticker')['pred'].aggregate(mean='mean', count='count')
        ds = ds[ds['count'] == ds['count'].max()].sort_values('mean')
        self._long_stocks = ds.index[ds.shape[0] - 1]
        pass

