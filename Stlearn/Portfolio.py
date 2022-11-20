from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import pandas as pd

from Data import *
from Model import *


class Performance:
    _daily_returns = None

    _mean_return = None
    _volatility = None
    _sharpe_ratio = None

    _portfolio_value = None

    def __init__(self):
        pass

    def update(self, daily_returns, portfolio_value=None):
        daily_returns.index = pd.to_datetime(daily_returns.index)
        self._daily_returns = daily_returns
        if portfolio_value is not None:
            portfolio_value.index = pd.to_datetime(portfolio_value.index)
            self._portfolio_value = portfolio_value
        self._cal()

    def add(self, daily_returns, portfolio_value=None):
        daily_returns.index = pd.to_datetime(daily_returns.index)
        self._daily_returns = pd.concat([self._daily_returns, daily_returns], ignore_index=False).sort_index()
        if portfolio_value is not None:
            self._portfolio_value = pd.concat([self._portfolio_value, portfolio_value], ignore_index=False).sort_index()
        self._cal()
        pass

    def _cal(self):
        if self._portfolio_value is None or self._portfolio_value.shape[0] != self._daily_returns.shape[0]:
            self._portfolio_value = (1 + self._daily_returns).cumprod()
        self._mean_return = (1 + self._daily_returns.mean()) ** Constant.N_DAYS - 1
        self._volatility = self._daily_returns.std() * np.sqrt(Constant.N_DAYS)
        self._sharpe_ratio = (self._mean_return - Constant.RF) / self._volatility

        pass

    def __str__(self):
        s = '[%s - %s] mean return: %.2f, volatility: %.2f, sharpe ratio: %.2f' % (
            self._daily_returns.index[0], self._daily_returns.index[-1],
            self._mean_return, self._volatility, self._sharpe_ratio)
        return s

    __repr__ = __str__

    def plot_portfolio_value(self):
        _ = self._portfolio_value.plot()
        pass

    @property
    def daily_returns(self):
        return self._daily_returns

    @property
    def mean_return(self):
        return self._mean_return

    @property
    def volatility(self):
        return self._volatility

    @property
    def sharpe_ratio(self):
        return self._sharpe_ratio

    @property
    def portfolio_value(self):
        return self._portfolio_value


class Portfolio(ABC):
    # object variable
    _scheme = None

    _performance = None

    def __init__(self, data=None, model=None):
        self._scheme = {}
        self._performance = Performance()
        if data is not None and model is not None:
            self.construct(data, model)
        pass

    @abstractmethod
    def construct(self, data, model):
        pass

    def add_portfolio(self, portfolio):
        p_scheme = portfolio.scheme
        for key in p_scheme.keys():
            self._scheme[key] = p_scheme[key]
        self._performance.add(portfolio.performance.daily_returns)
        pass

    @property
    def scheme(self):
        return self._scheme

    @property
    def performance(self):
        return self._performance

    def clone(self):
        return copy.deepcopy(self)


class MultiStrategyPortfolio(Portfolio):
    def construct(self, data, model):
        pass

    def add_portfolio(self, portfolio):
        pass


class AlgoTradePortfolio(Portfolio):
    global_portfolio = None

    def __init__(self, data=None, model=None):
        super().__init__(data, model)
        if self.__class__.global_portfolio is None:
            self.__class__.global_portfolio = self.clone()
        else:
            self.__class__.global_portfolio.add_portfolio(self)

    def construct(self, data, model):
        df = pd.DataFrame([])
        df[['Date', 'Ticker', 'Close', 'Return', 'Market Return']] = pd.DataFrame(data.ids_test)
        df['er'] = pd.DataFrame(data.y_test)
        df = df.set_index('Date')
        period = df.sort_index().index.drop_duplicates().tolist()
        long_stocks, long_weights, short_stocks, short_weights = self._strategy(data, model)
        self._apply_strategy(long_stocks, long_weights, short_stocks, short_weights, period, df)
        self._scheme[period[0] + '-' + period[-1]] = {
            'long': [long_stocks],
            'short': [short_stocks]
        }
        pass

    @abstractmethod
    def _apply_strategy(self, long_stocks, long_weights, short_stocks, short_weights, period, df):
        pass

    @abstractmethod
    def _strategy(self, data, model):
        pass


class LongPortfolio(AlgoTradePortfolio):

    def _apply_strategy(self, long_stocks, long_weights, short_stocks, short_weights, period, df):
        if sum(long_weights) != 1:
            long_weights = (np.array(long_weights) / sum(long_weights)).tolist()
        daily_returns = pd.DataFrame([], index=period)
        daily_returns['Return'] = 0
        for i in range(len(long_stocks)):
            long_stock = long_stocks[i]
            long_weight = long_weights[i]
            daily_returns['Return'] += df[df['Ticker'] == long_stock]['Return'] * long_weight
        self._performance.update(daily_returns['Return'])
        pass

    @abstractmethod
    def _strategy(self, data, model):
        pass


class LongMarketPortfolio(LongPortfolio):

    def construct(self, data, model):
        df = pd.DataFrame([])
        df[['Date', 'Ticker', 'Close', 'Return', 'Market Return']] = pd.DataFrame(data.ids_test)
        daily_returns = df[['Date', 'Market Return']].drop_duplicates(
        ).set_index('Date').rename(columns={'Market Return': 'Return'})
        self._performance.update(daily_returns)
        period = df['Date'].values
        self._scheme[period[0] + '-' + period[-1]] = {
            'long': ['Market Portfolio'],
            'short': []
        }
        pass

    def _strategy(self, data, model):
        pass


class LongRandomPortfolio(LongPortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        count = pd.DataFrame(t).reset_index().groupby(1)['index'].count()
        stock_list = count[count == count.max()].index
        return [np.random.choice(stock_list)], [1], [], []


class LongBestPortfolio(LongPortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        count = pd.DataFrame(t).reset_index().groupby(1)['index'].count()
        first_indexes = first_indexes.drop(count[count != count.max()].index)
        pred = model.predict(data.X_test[first_indexes, :])[:, :, 0]
        pred_df = pd.DataFrame(pred).T
        pred_df.columns = first_indexes.index
        long_stocks = first_indexes.index[pred_df.loc[:30, :].mean().argmax()]
        return [long_stocks], [1], [], []


class LongBestThreePortfolio(LongPortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        count = pd.DataFrame(t).reset_index().groupby(1)['index'].count()
        first_indexes = first_indexes.drop(count[count != count.max()].index)
        pred = model.predict(data.X_test[first_indexes, :])[:, :, 0]
        pred_df = pd.DataFrame(pred).T
        pred_df.columns = first_indexes.index
        long_stocks = pred_df.loc[:30, :].mean().sort_values(ascending=False).index[:3]
        return long_stocks, [1, 1, 1], [], []


class DollarNeutralLongShortPortfolio(AlgoTradePortfolio):
    _long_performance = None
    _short_performance = None

    def __init__(self, data=None, model=None):
        self._long_performance = Performance()
        self._short_performance = Performance()
        super().__init__(data, model)

    def _apply_strategy(self, long_stocks, long_weights, short_stocks, short_weights, period, df):
        if sum(long_weights) != 1:
            long_weights = (np.array(long_weights) / sum(long_weights)).tolist()
        daily_prices = pd.DataFrame([], index=period)
        daily_prices['long'] = 0
        daily_prices['short'] = 0

        daily_returns = pd.DataFrame([], index=period)
        daily_returns['long'] = 0
        daily_returns['short'] = 0
        for i in range(len(long_stocks)):
            long_stock = long_stocks[i]
            long_weight = long_weights[i]
            daily_prices['long'] += df[df['Ticker'] == long_stock]['Close'] * long_weight
            daily_returns['long'] += df[df['Ticker'] == long_stock]['Return'] * long_weight
        for i in range(len(short_stocks)):
            short_stock = short_stocks[i]
            short_weight = short_weights[i]
            daily_prices['short'] += df[df['Ticker'] == short_stock]['Close'] * short_weight
            daily_returns['short'] += df[df['Ticker'] == short_stock]['Return'] * short_weight

        daily_prices['long_vol'] = 1
        daily_prices['short_vol'] = 1 * daily_prices['long'].iloc[0] / daily_prices['short'].iloc[0]

        daily_prices['long_value'] = daily_prices['long_vol'] * daily_prices['long']
        daily_prices['short_value'] = -daily_prices['short_vol'] * daily_prices['short']
        daily_prices['net_value'] = daily_prices['long_value'] + daily_prices['short_value']

        daily_prices['return'] = (daily_prices['net_value'].diff() / daily_prices['long_value'].shift(1))

        self._long_performance.update(daily_returns['long'], daily_prices['long_value'])
        self._short_performance.update(daily_returns['short'] * -1, daily_prices['short_value'])

        self._performance.update(daily_prices['return'], daily_prices['net_value'])

        pass

    @abstractmethod
    def _strategy(self, data, model):
        pass

    def add_portfolio(self, portfolio):
        p_scheme = portfolio.scheme
        for key in p_scheme.keys():
            self._scheme[key] = p_scheme[key]

        last_carry = self._performance.portfolio_value.iloc[-1]

        if last_carry >= 0:
            new_long_returns = copy.deepcopy(portfolio.long_performance.daily_returns)
            new_long_returns.iloc[0] = 0
            new_portfolio_value = portfolio.performance.portfolio_value + (
                    1 + new_long_returns).cumprod() * last_carry
            new_daily_returns = new_portfolio_value.diff() / (
                portfolio.long_performance.portfolio_value + (
                    1 + new_long_returns).cumprod() * last_carry
            ).shift(1)
            self._performance.add(new_daily_returns, new_portfolio_value)
        else:
            new_portfolio_value = portfolio.performance.portfolio_value + last_carry
            self._performance.add(portfolio.performance.daily_returns, new_portfolio_value)

        self._long_performance.add(portfolio.long_performance.daily_returns,
                                   portfolio.long_performance.portfolio_value
                                   )
        self._short_performance.add(portfolio.short_performance.daily_returns,
                                    portfolio.short_performance.portfolio_value)

        pass

    @property
    def long_performance(self):
        return self._long_performance

    @property
    def short_performance(self):
        return self._short_performance


class LongShortBestShotPortfolio(DollarNeutralLongShortPortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        count = pd.DataFrame(t).reset_index().groupby(1)['index'].count()
        first_indexes = first_indexes.drop(count[count != count.max()].index)
        pred = model.predict(data.X_test[first_indexes, :])[:, :, 0]
        pred_df = pd.DataFrame(pred).T
        pred_df.columns = first_indexes.index
        long_stocks = first_indexes.index[pred_df.loc[:30, :].mean().argmax()]
        short_stocks = first_indexes.index[pred_df.loc[:30, :].mean().argmin()]
        return [long_stocks], [1], [short_stocks], [1]

