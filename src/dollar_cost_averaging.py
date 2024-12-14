import sys
from collections import defaultdict

import backtrader as bt
import matplotlib.pyplot as pl
import pandas as pd

from analyzer import PortfolioAnalyzer
from base_strategy import BaseStrategy
from data import download_data
from observer import WeightObserver
from optimizer import find_optimal_portfolio
from plot import plot_with_plotly


class DollarCostAveraging(BaseStrategy):

    params = dict(
        threshold=0.05, monthly_cash=1000, risk_free_rate=0.04, n_portfolios=10_000
    )

    def __init__(self):
        self.order = None
        self.purchases = defaultdict(int)
        self.sales = defaultdict(int)
        self.weights = {
            ticker_data._name: 1 / len(self.datas) for ticker_data in self.datas
        }

    def start(self):
        super(DollarCostAveraging, self).start()
        self.add_timer(
            when=bt.timer.SESSION_START,
            monthdays=[15],
            monthcarry=True,
            timername="buytimer",
        )

    def calculate_weights(self):
        data = pd.DataFrame(
            data={
                ticker_data._name: ticker_data.get(0, len(self))
                for ticker_data in self.datas
            },
            index=self.datetime.get(0, len(self)),
        )
        optimal_portfolio = find_optimal_portfolio(
            data, self.p.risk_free_rate, num_portfolios=self.p.n_portfolios
        )
        self.weights = optimal_portfolio["allocations"]
        return None

    def notify_timer(self, timer, when, *args, **kwargs):
        six_months = 180
        if len(self) < six_months:
            return None
        self.broker.add_cash(self.p.monthly_cash)
        self.total_contributions += self.p.monthly_cash
        total_value = self.broker.get_value() + self.p.monthly_cash

        self.calculate_weights()

        for ticker_data in self.datas:
            self.order_target_value(
                ticker_data,
                target=total_value
                * self.weights[ticker_data._name]
                * (1 - self.p.threshold),
            )

        return None


def main(
    tickers,
    start=None,
    end=None,
    monthly_cash=1250,
    risk_free_rate=0.04,
    n_portfolios=1_000,
):
    cerebro = bt.Cerebro(stdstats=True)

    data = download_data(tickers, start=start, end=end)

    for ticker in tickers:
        bt_data = bt.feeds.PandasData(dataname=data[ticker])
        cerebro.adddata(bt_data, name=ticker)
        cerebro.addobserver(WeightObserver, ticker=ticker)

    cerebro.addstrategy(
        DollarCostAveraging,
        monthly_cash=monthly_cash,
        risk_free_rate=risk_free_rate,
        n_portfolios=n_portfolios,
    )

    cerebro.setbroker(
        bt.BackBroker(cash=sys.float_info.epsilon, coc=True, fundmode=True)
    )
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
            "BND",  # Total Bond Market
            "VEA",  # International Developed Markets
            # "VB",  # Small Cap
            # "VTV",  # Value
        ],
        start="2015-01-01",
        end="2024-11-01",
        monthly_cash=1250,
        risk_free_rate=0.03,
        n_portfolios=1_000,
    )
