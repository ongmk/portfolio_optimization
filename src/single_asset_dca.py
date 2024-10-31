from collections import defaultdict

import backtrader as bt
import pandas as pd
import yfinance as yf


class SingleAssetDCA(bt.Strategy):
    params = dict(monthly_cash=100.0)

    def __init__(self):
        self.purchases = defaultdict(int)
        self.sales = defaultdict(int)
        self.weights = {ticker_data._name: 1 for ticker_data in self.datas}

    @staticmethod
    def __parse_date(dt):
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        return dt

    def start(self):
        self.start_dt = self.__parse_date(self.data.datetime[1])
        self.cash_start = self.broker.get_cash()
        self.total_contributions = self.broker.get_cash()

        self.add_timer(
            bt.timer.SESSION_START,  # when it will be called
            monthdays=[1],  # called on the 1st day of the month
            monthcarry=True,  # called on the 2nd day if the 1st is holiday
        )

    def __get_weight(self, ticker_data, extra_cash=0):
        total_value = self.broker.get_value() + extra_cash
        return self.broker.get_value(datas=[ticker_data]) / total_value

    def notify_timer(self, timer, when, *args, **kwargs):
        # Add the influx of monthly cash to the broker
        self.broker.add_cash(self.p.monthly_cash)
        self.total_contributions += self.p.monthly_cash

        # buy available cash
        target_value = self.broker.get_value() + self.p.monthly_cash
        self.order_target_value(target=target_value)

    def log(self, txt):
        print(f"[{self.datetime.date()}]    {txt}")

    def notify_order(self, order):
        """Triggered upon changes to orders."""
        if order.status == order.Submitted:
            return

        # Check if an order has been completed
        elif order.status in [order.Completed]:
            self.log(
                f"{('BUY' if order.isbuy() else 'SELL'):<4} "
                f"{order.data._name:>6} "
                f"@ {order.executed.price:6.2f} "
                f" x {order.created.size:<3}    "
                f"Cost (Comm): ${order.executed.value if order.isbuy() else -order.executed.value:7.2f} "
                f"({order.executed.comm:4.2f})    "
                f"Position: {self.getposition(order.data).size:<2}    "
                f"Weight: {self.__get_weight(order.data):.1%}"
            )
            if order.isbuy():
                self.purchases[order.data._name] += 1
            else:
                self.sales[order.data._name] += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order status = {order.Status[order.status]}")
        return

    @staticmethod
    def __get_transaction_string(transactions):
        transaction_string = " + ".join(
            [f"{times}({ticker})" for ticker, times in transactions.items()]
        )
        transaction_string = f"{transaction_string} = {sum(transactions.values())}"
        return transaction_string

    def stop(self):
        self.stop_dt = self.__parse_date(self.data.datetime[0])

        duration = self.stop_dt - self.start_dt
        number_of_years = duration.days / 365

        value = self.broker.get_value()
        gross_return = value - self.total_contributions
        gross_return_pct = (value / self.total_contributions - 1) * 100
        annualized_return_pct = (
            ((value / self.total_contributions) ** (365 / duration.days)) - 1
        ) * 100
        print("-" * 50)
        print("Portfolio DCA")
        print(f"Time in Market      : {number_of_years:.1f} years")
        print(f"Cash                : ${self.broker.get_cash():.1f}")
        print(f"Purchases           : {self.__get_transaction_string(self.purchases)}")
        print(f"Sales               : {self.__get_transaction_string(self.sales)}")
        print(f"Nominal Value       : ${value:,.2f}")
        print(f"Total Contributions : ${self.total_contributions:,.2f}")
        print(f"Gross Return        : ${gross_return:,.2f}")
        print(f"Gross Return %      : {gross_return_pct:.2f}%")
        print(f"Annualised Return % : {annualized_return_pct:.2f}%")
        print("-" * 50)


def run(ticker, **yf_kwargs):

    cerebro = bt.Cerebro()

    ticker = yf.Ticker(ticker)
    data = ticker.history(**yf_kwargs)
    # data = data / 1e3
    bt_data = bt.feeds.PandasData(dataname=data, name=ticker.ticker)
    cerebro.adddata(bt_data)

    cerebro.addstrategy(SingleAssetDCA, monthly_cash=100)
    cerebro.setbroker(
        bt.BackBroker(
            cash=1000,
            coc=True,
        )
    )

    cerebro.run()
    cerebro.plot()


if __name__ == "__main__":
    run(
        "VTI",
        start="2019-12-31",
        end="2024-09-30",
    )
