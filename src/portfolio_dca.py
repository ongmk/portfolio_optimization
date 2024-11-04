import sys
from collections import defaultdict
from datetime import timedelta

import matplotlib.pyplot as pl
import pandas as pd

from analyzer import PortfolioAnalyzer
from data import download_data
from observer import WeightObserver
from optimizer import find_optimal_portfolio
from plot import plot_with_plotly

pl.style.use("ggplot")  # default is also fine

import backtrader as bt


class PortfolioDCA(bt.Strategy):

    params = dict(threshold=0.025, monthly_cash=1000, risk_free_rate=0.04)

    def __init__(self):
        self.order = None
        self.purchases = defaultdict(int)
        self.sales = defaultdict(int)
        self.weights = {
            ticker_data._name: 1 / len(self.datas) for ticker_data in self.datas
        }

    def start(self):
        self.total_contributions = self.broker.get_cash()

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
            data, self.p.risk_free_rate, num_portfolios=10_000
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
                ticker_data, target=total_value * self.weights[ticker_data._name]
            )

        return None

    def log(self, txt):
        print(f"[{self.datetime.date()}]    {txt}")

    def notify_order(self, order):
        """Triggered upon changes to orders."""
        if order.status == order.Submitted:
            return None

        # Check if an order has been completed
        elif order.status in [order.Completed]:
            if order.isbuy():
                action = "BUY"
                cost = order.executed.value
            else:
                action = "SELL"
                cost = -order.executed.value
            weight = self.broker.get_value(datas=[order.data]) / self.broker.get_value()
            self.log(
                f"{action:<4} "
                f"{order.data._name:>6} "
                f"@ {order.executed.price:6.2f} "
                f" x {order.created.size:<3}    "
                f"Cost (Comm): ${cost:7.2f} "
                f"({order.executed.comm:4.2f})    "
                f"Position: {self.getposition(order.data).size:<2}    "
                f"Weight: {weight:.1%}"
            )
            if order.isbuy():
                self.purchases[order.data._name] += 1
            else:
                self.sales[order.data._name] += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order status = {order.Status[order.status]}")
        return None


def main(tickers, start=None, end=None):
    cerebro = bt.Cerebro(stdstats=True)

    data = download_data(tickers, start=start, end=end)

    for ticker in tickers:
        bt_data = bt.feeds.PandasData(dataname=data[ticker])
        cerebro.adddata(bt_data, name=ticker)
        cerebro.addobserver(WeightObserver, ticker=ticker)

    cerebro.addstrategy(PortfolioDCA, monthly_cash=1250, risk_free_rate=0.04)

    cerebro.setbroker(
        bt.BackBroker(cash=sys.float_info.epsilon, coc=True, fundmode=True)
    )
    cerebro.addobserver(bt.observers.DrawDown, fund=True)

    cerebro.addanalyzer(PortfolioAnalyzer, riskfreerate=0.04)

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
            "TLT",  # 20+ year treasury bond
            "GLD",  # Gold
            "VNQ",  # Real Estate
            "QQQ",  # Nasdaq
        ],
        start="2015-05-01",
        end="2024-09-30",
    )
