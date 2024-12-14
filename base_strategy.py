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


class BaseStrategy(bt.Strategy):

    def __init__(self):
        self.purchases = defaultdict(int)
        self.sales = defaultdict(int)
        self.weights = {
            ticker_data._name: 1 / len(self.datas) for ticker_data in self.datas
        }

    def start(self):
        self.total_contributions = self.broker.get_cash()
        return None

    def log(self, txt, warning=False):
        func = logger.warning if warning else logger.info
        func(f"[{self.datetime.date()}]    {txt}")

    def notify_order(self, order):
        if order.status == order.Submitted:
            return None

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
                f"x {order.created.size:<3}    "
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
            self.log(
                f"Order status = {order.Status[order.status]} "
                f"for {order.data._name} "
                f"@ {order.created.price:,.2f} "
                f"x {order.created.size} "
                f"= ${order.created.size*order.created.price:,.2f}    "
                f"Cash available = ${self.broker.get_cash():,.2f}    ",
                warning=True,
            )
        return None
