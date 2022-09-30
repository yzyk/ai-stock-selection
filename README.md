# AI stock selection

## Problem Identification
This project is going to further exploit machine learning’s role in stock selection to predict the future realized excess return of any stock given historical data. It is applied to all available stocks over all their life cycles until today.

## Data
This project uses `yfinance` to collect required data for all S&P 500 stocks. The current data dimensions we collect is as below:
|  Dimension  |  Format  |
|  ----  | ----  |
| Date | TimeStamp |
| Open | Numeric |
| High | Numeric |
| Low | Numeric |
| Close | Numeric |
| Volume | Numeric |
| Dividends | Numeric |
| Stock Splits | Numeric |
| Ticker | String |
Note:
* This is only original data dimension and does not include any feature engineering,
* This only covers price and volume which should be enough for the initial iterations of machine learning and deep learning but may be extended to cover fundamental data for future advanced iterations.