from backtrader import Analyzer, TimeFrame, num2date
from backtrader.analyzers import DrawDown, Returns, SharpeRatio


class PortfolioAnalyzer(Analyzer):
    params = dict(riskfreerate=0.01)

    def __init__(
        self,
        sharpe_ratio_analyzer_kwargs=dict(
            timeframe=TimeFrame.Days, annualize=True, fund=True
        ),
        drawdown_analyzer_kwargs=dict(fund=True),
        returns_analyzer_kwargs=dict(timeframe=TimeFrame.Days, fund=True),
    ):
        sharpe_ratio_analyzer_kwargs["riskfreerate"] = self.p.riskfreerate
        self.sharpe = SharpeRatio(**sharpe_ratio_analyzer_kwargs)

        self.drawdown = DrawDown(**drawdown_analyzer_kwargs)

        self.returns = Returns(**returns_analyzer_kwargs)

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

        sharpe_ratio = self.sharpe.get_analysis()["sharperatio"]

        drawdown_analysis = self.drawdown.get_analysis()
        max_drawdown = drawdown_analysis["max"]["drawdown"]

        returns_analysis = self.returns.get_analysis()
        total_return = returns_analysis["rtot"]
        annualized_return = returns_analysis["rnorm"]
        avg_return = returns_analysis["ravg"]
        remaining_cash = self.strategy.broker.get_cash()
        nominal_value = self.strategy.broker.get_value()

        self.rets[1] = {
            "Time in Market         ": f"{number_of_years:.1f} years",
            "Total Contributions    ": f"${self.strategy.total_contributions}",
            "Remaining Cash         ": f"${remaining_cash:,.2f}",
            "Purchases              ": f"{self.__get_transaction_string(self.strategy.purchases)}",
            "Sales                  ": f"{self.__get_transaction_string(self.strategy.sales)}",
        }
        self.rets[2] = {
            "Nominal Value          ": f"${nominal_value:,.2f}",
            "Total Return %         ": f"{total_return:.2%}",
            "Annualized Return %    ": f"{annualized_return:.2%}",
            "Avg. Daily Return %    ": f"{avg_return:.2%}",
        }
        self.rets[3] = {
            "Risk Free Rate         ": self.p.riskfreerate,
            "Annualized Sharpe Ratio": f"{sharpe_ratio:.2f}",
        }
        self.rets[4] = {
            "Max Drawdown %         ": f"{max_drawdown:.2f}%",
            "Max Drawdown Length    ": drawdown_analysis["max"]["len"],
        }
