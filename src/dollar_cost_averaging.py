import math

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import GOOG

GOOG = GOOG * 10**-6


class DCA(Strategy):

    amount_to_invest = 10

    def init(self):

        self.day_of_week = self.I(
            lambda x: x,
            self.data.Close.s.index.dayofweek,
            plot=False,
        )

    def next(self):

        if self.day_of_week[-1] == 1:
            self.buy(size=math.floor(self.amount_to_invest / self.data.Close[-1]))
            try:
                if self.data.Close[-1] / self.data.Close[-30] < 0.95:
                    self.buy(
                        size=math.floor(self.amount_to_invest / self.data.Close[-1])
                    )
            except:
                pass


def main():
    print(GOOG)

    bt = Backtest(GOOG, DCA, trade_on_close=True, cash=10_000)

    stats = bt.run()

    print(stats)

    # Size  EntryBar  ExitBar  EntryPrice  ExitPrice

    trades = stats["_trades"]
    price_paid = trades["Size"] * trades["EntryPrice"]
    total_invested = price_paid.sum()

    current_shares = trades["Size"].sum()
    current_equity = current_shares * GOOG.Close.iloc[-1]

    print("Total investment:", total_invested)
    print("Current Shares:", current_shares / (10**6))
    print("current Equity:", current_equity)

    print("Return:", current_equity / total_invested)


if __name__ == "__main__":
    main()
