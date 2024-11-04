import yfinance as yf


def align_data_range(stockData):
    start = stockData.index.min()
    end = stockData.index.max()
    for stock in stockData.columns.levels[1]:
        stock_start = stockData["Close"][stock].dropna().index.min()
        if stock_start > start:
            start = stock_start
            print(f"{stock} data starts at {start:%Y-%m-%d}. Adjusting start date.")
        stock_end = stockData["Close"][stock].dropna().index.max()
        if stock_end < end:
            end = stock_end
            print(f"{stock} data ends at {end:%Y-%m-%d}. Adjusting end date.")
    return stockData.loc[(stockData.index >= start) & (stockData.index <= end)]


def download_data(stocks, **kwargs):
    stockData = yf.download(stocks, **kwargs)
    stockData["Close"] = stockData["Adj Close"]
    stockData = stockData.drop(columns=["Adj Close"])
    stockData = align_data_range(stockData)
    stockData = stockData.swaplevel(axis=1)

    return stockData
