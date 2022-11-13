from __future__ import annotations

from abc import ABC, abstractmethod
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

    def update(self, daily_returns):
        daily_returns.index = pd.to_datetime(daily_returns.index)
        self._daily_returns = daily_returns
        self._cal()

    def add(self, daily_returns):
        daily_returns.index = pd.to_datetime(daily_returns.index)
        self._daily_returns = pd.concat([self._daily_returns, daily_returns], ignore_index=False).sort_index()
        self._cal()
        pass

    def _cal(self):
        self._mean_return = (1 + self._daily_returns.mean()) ** N_DAYS - 1
        self._volatility = self._daily_returns.std() * np.sqrt(N_DAYS)
        self._sharpe_ratio = (self._mean_return - RF) / self._volatility
        self._portfolio_value = (1 + self._daily_returns).cumprod()
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

    @abstractmethod
    def _strategy(self, data, model):
        pass


class GlobalPortfolio(Portfolio):
    def construct(self, data, model):
        pass

    def _strategy(self, data, model):
        pass


class AlgoTradePortfolio(Portfolio):
    global_portfolio = None

    def __init__(self, data=None, model=None):
        super().__init__(data, model)
        if self.__class__.global_portfolio is None:
            self.__class__.global_portfolio = GlobalPortfolio()
        self.__class__.global_portfolio.add_portfolio(self)

    def construct(self, data, model):
        df = pd.DataFrame([])
        df[['Date', 'Ticker', 'Close', 'Return', 'Market Return']] = pd.DataFrame(data.ids_test)
        df['er'] = pd.DataFrame(data.y_test)
        df = df.set_index('Date')
        period = df.sort_index().index.drop_duplicates().tolist()
        long_stocks, long_weights, short_stocks, short_weights = self._strategy(data, model)
        if sum(long_weights) != 1:
            long_weights = (np.array(long_weights) / sum(long_weights)).tolist()
        daily_returns = pd.DataFrame([], index=period)
        daily_returns['Return'] = 0
        for i in range(len(long_stocks)):
            long_stock = long_stocks[i]
            long_weight = long_weights[i]
            daily_returns += df[df['Ticker'] == long_stock][['Return']] * long_weight
        self._performance.update(daily_returns)

        self._scheme[period[0] + '-' + period[-1]] = {
            'long': [long_stocks],
            'short': [short_stocks]
        }
        pass

    @abstractmethod
    def _strategy(self, data, model):
        pass


class LongMarketPortfolio(AlgoTradePortfolio):
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


class LongRandomPortfolio(AlgoTradePortfolio):
    def _strategy(self, data, model):
        stock_list = np.unique(data.ids_test[:, 1].ravel())
        return [np.random.choice(stock_list)], [1], [], []


class LongBestPortfolio(AlgoTradePortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        pred = model.predict(data.X_test[first_indexes, :])[:, :, 0]
        pred_df = pd.DataFrame(pred).T
        pred_df.columns = first_indexes.index
        long_stocks = first_indexes.index[pred_df.loc[:30, :].mean().argmax()]
        return [long_stocks], [1], [], []


class LongBestThreePortfolio(AlgoTradePortfolio):
    def _strategy(self, data, model):
        t = data.ids_test
        first_indexes = pd.DataFrame(t).reset_index().groupby(1)['index'].min()
        pred = model.predict(data.X_test[first_indexes, :])[:, :, 0]
        pred_df = pd.DataFrame(pred).T
        pred_df.columns = first_indexes.index
        long_stocks = pred_df.loc[:30, :].mean().sort_values(ascending=False).index[:3]
        return long_stocks, [1, 1, 1], [], []