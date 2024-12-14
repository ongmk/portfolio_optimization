import sys
from collections import defaultdict
from datetime import timedelta

import logzero
import matplotlib.pyplot as pl
import pandas as pd
from logzero import logger

from log_formatter import ColoredFormatter

logzero.formatter(ColoredFormatter())

from analyzer import PortfolioAnalyzer
from data import download_data
from observer import WeightObserver
from optimizer import find_optimal_portfolio
from plot import plot_with_plotly

pl.style.use("ggplot")  # default is also fine

import backtrader as bt

from base_strategy import BaseStrategy


class BuyAndHold(BaseStrategy):

    params = dict(threshold=0.025)

    def start(self):
        super(BuyAndHold, self).start()
        self.order = False

    def next(self):
        if self.order:
            return None
        for ticker_data in self.datas:
            self.order_target_value(
                ticker_data,
                target=self.broker.get_cash()
                * self.weights[ticker_data._name]
                * (1 - self.p.threshold),
            )
        self.order = True

        return None


def main(tickers, start=None, end=None, risk_free_rate=0.04, start_cash=10_000):
    cerebro = bt.Cerebro(stdstats=True)

    data = download_data(tickers, start=start, end=end)
    data = data / 1000

    for ticker in tickers:
        bt_data = bt.feeds.PandasData(dataname=data[ticker])
        cerebro.adddata(bt_data, name=ticker)
        cerebro.addobserver(WeightObserver, ticker=ticker)

    cerebro.addstrategy(BuyAndHold, threshold=0.0)

    cerebro.setbroker(bt.BackBroker(cash=start_cash, coc=True, fundmode=True))
    cerebro.addobserver(bt.observers.DrawDown, fund=True)

    cerebro.addanalyzer(PortfolioAnalyzer, riskfreerate=risk_free_rate)

    strategies = cerebro.run()
    for analyzer in strategies[0].analyzers:
        analyzer.print()

    pl.rcParams["figure.figsize"] = (16, (len(tickers) + 1) * 5)
    cerebro.plot(volume=False)
    # plot_with_plotly(cerebro)


if __name__ == "__main__":
    main(
        [
            "VTI",  # Total stock Market
            # "BND",  # Total Bond Market
            # "TLT",  # 20+ year treasury bond
            # "GLD",  # Gold
            # "VNQ",  # Real Estate
            # "QQQ",  # Nasdaq
        ],
        start="2015-01-01",
        end="2024-11-01",
        risk_free_rate=0.03,
        start_cash=10_000,
    )
