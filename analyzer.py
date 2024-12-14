from backtrader import Analyzer, TimeFrame, num2date
from backtrader.analyzers import DrawDown, Returns, SharpeRatio, TimeReturn
from backtrader.mathsupport import average, standarddev
from backtrader.utils.py3 import itervalues


class PortfolioAnalyzer(Analyzer):
    params = dict(riskfreerate=0.01)

    def __init__(
        self,
        timeframe=TimeFrame.Years,
    ):
        self.sharpe = SharpeRatio(
            timeframe=timeframe,
            annualize=True,
            fund=True,
            riskfreerate=self.p.riskfreerate,
            compression=None,
        )

        self.time_return = TimeReturn(
            timeframe=timeframe,
            compression=None,
            fund=True,
        )

        self.drawdown = DrawDown(fund=True)

        self.total_contributions = self.cash = self.value = 0.0

    @staticmethod
    def __parse_date(dt):
        if isinstance(dt, float):
            dt = num2date(dt)
        return dt

    def start(self):
        self.start_dt = self.__parse_date(self.data.datetime[1])

    @staticmethod
    def __get_transaction_string(transactions):
        total = sum(transactions.values())
        if total > 0:
            transaction_string = " + ".join(
                [f"{times}({ticker})" for ticker, times in transactions.items()]
            )
            return f"{transaction_string} = {total}"
        else:
            return str(total)

    def stop(self):
        super(PortfolioAnalyzer, self).stop()
        self.stop_dt = self.__parse_date(self.data.datetime[0])
        duration = self.stop_dt - self.start_dt
        number_of_years = duration.days / 365
        remaining_cash = self.strategy.broker.get_cash()

        nominal_value = self.strategy.broker.get_value()

        total_return = self.strategy.broker.get_fundvalue() / 100 - 1
        annual_returns = list(itervalues(self.time_return.get_analysis()))
        avg_annual_return = average(annual_returns)

        drawdown_analysis = self.drawdown.get_analysis()
        max_drawdown = drawdown_analysis["max"]["drawdown"]
        annual_returns_std = standarddev(
            annual_returns, avgx=avg_annual_return, bessel=self.sharpe.p.stddev_sample
        )
        sharpe_ratio = self.sharpe.get_analysis()["sharperatio"]

        self.rets["Info"] = {
            "Time in Market         ": f"{number_of_years:.1f} years",
            "Total Contributions    ": f"${self.strategy.total_contributions:,.2f}",
            "Remaining Cash         ": f"${remaining_cash:,.2f}",
            "Purchases              ": f"{self.__get_transaction_string(self.strategy.purchases)}",
            "Sales                  ": f"{self.__get_transaction_string(self.strategy.sales)}",
        }
        self.rets["Reward"] = {
            "Nominal Value          ": f"${nominal_value:,.2f}",
            "Total Return           ": f"{total_return:.2%}",
            "Mean Annualized Return ": f"{avg_annual_return:.2%}",
        }
        self.rets["Risk"] = {
            "Risk Free Rate         ": self.p.riskfreerate,
            "Sharpe Ratio           ": f"{sharpe_ratio:.2f}",
            "Annual Returns STD     ": f"{annual_returns_std:.2%}",
            "Max Drawdown %         ": f"{max_drawdown:.2f}%",
            "Max Drawdown Length    ": drawdown_analysis["max"]["len"],
        }
