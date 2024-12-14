import math
from collections import Counter
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from scipy import signal
from scipy.interpolate import interpn
from tqdm.notebook import tqdm

pio.templates.default = "plotly_dark"
GREEN = "LimeGreen"
RED = "crimson"

plt.style.use("fivethirtyeight")
np.random.seed(777)


def display(x):
    print(x)


def plot_returns(returns):
    normalized_returns = returns / returns.iloc[0]

    fig = go.Figure()

    traces = []
    for column in normalized_returns.columns:
        traces.append(
            go.Scatter(
                x=normalized_returns.index,
                y=normalized_returns[column],
                mode="lines",
                name=column,
            )
        )

    layout = go.Layout(
        title="Normalized Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Returns",
        width=1000,
        height=500,
        hovermode="x",
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


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


def download_data(stocks, plot=False, **kwargs):

    stockData = yf.download(stocks, **kwargs)
    stockData["Close"] = stockData["Adj Close"]
    stockData = stockData.drop(columns=["Adj Close"])
    if isinstance(stockData.columns, pd.MultiIndex):
        stockData = align_data_range(stockData)
        stockData = stockData.swaplevel(axis=1)
    else:
        stockData.columns = pd.MultiIndex.from_product([[stocks[0]], stockData.columns])

    if plot:
        plot_returns(stockData)

    return stockData


def calculate_vwr(prices, n_trading_days=251.5, max_volatility=0.19, tau=2):
    log_returns = np.log(prices / prices.shift(1))
    n_periods = len(log_returns) - 1
    mean_return = np.log(prices.iloc[-1] / prices.iloc[0]) / n_periods

    log_returns.index
    time_step = pd.Series(range(len(log_returns)), index=log_returns.index)
    zero_variability_log_returns = np.exp(mean_return * time_step) * prices.iloc[0]

    diff = (prices - zero_variability_log_returns) / zero_variability_log_returns
    diff_std = diff.std()

    mean_return_normalized = (np.exp(mean_return * n_trading_days) - 1) * 100

    vwr = mean_return_normalized * (1 - pow(diff_std / max_volatility, tau))
    return vwr


def calculate_sharpe_ratio(prices, risk_free_rate=0.04, n_trading_days=251.5):
    risk_free_rate_per_day = pow(1 + risk_free_rate, 1 / n_trading_days) - 1
    risk_adjusted_returns = prices.pct_change() - risk_free_rate_per_day
    mean_return = risk_adjusted_returns.mean()
    std_dev = risk_adjusted_returns.std()
    sharpe_ratio = mean_return / std_dev
    return sharpe_ratio


def portfolio_annualised_performance(weights, stock_data, risk_free_rate):
    portfolio_weighted_prices = pd.Series(0, index=stock_data.index)
    for idx, stock in enumerate(stock_data.columns.levels[0]):
        portfolio_weighted_prices += stock_data[stock, "Close"] * weights[idx]
    sharpe_ratio = calculate_sharpe_ratio(portfolio_weighted_prices, risk_free_rate)
    vwr = calculate_vwr(portfolio_weighted_prices)
    portfolio_returns = portfolio_weighted_prices.pct_change()
    portfolio_std_dev = portfolio_returns.std()
    portfolio_return = portfolio_returns.mean()

    return portfolio_std_dev, portfolio_return, sharpe_ratio, vwr


def n_combinations(n, r):
    result = math.factorial(n + r - 1) // (math.factorial(r) * math.factorial(n - 1))
    return result


def get_n_weight_combinations(stocks, num_portfolios):
    if len(stocks) == 1:
        return [np.array([1.0])]
    n_parts = len(stocks)
    while n_combinations(len(stocks), n_parts) < num_portfolios:
        n_parts += 1
    n_parts += 1
    portfolio_weights = []
    for combination in combinations_with_replacement(stocks, n_parts):
        counts = Counter(combination)
        portfolio_weights.append(np.array([counts[s] / n_parts for s in stocks]))
    return portfolio_weights


def get_portfolio_performances(numPortfolios, stock_data, risk_free_rate=0):
    tickers = stock_data.columns.levels[0]
    portfolio_weights = get_n_weight_combinations(tickers, numPortfolios)
    actual_num_portfolios = len(portfolio_weights)
    data = []
    for i in tqdm(range(actual_num_portfolios)):
        portfolio_std_dev, portfolio_return, sharpe_ratio, vwr = (
            portfolio_annualised_performance(
                portfolio_weights[i], stock_data, risk_free_rate
            )
        )
        data.append(
            {
                "std_dev": portfolio_std_dev,
                "returns": portfolio_return,
                "sharpe_ratio": sharpe_ratio,
                "vwr": vwr,
                "allocations": {
                    asset: weight
                    for asset, weight in zip(tickers, portfolio_weights[i])
                },
            }
        )
    return pd.DataFrame(data)


def find_max_sharpe_portfolio(performances, verbose=False):
    selected = performances["sharpe_ratio"] == performances["sharpe_ratio"].max()

    if verbose:
        print("-" * 80)
        print(
            "Annualised Return:",
            round(performances.loc[selected, "returns"].iloc[0], 2),
        )
        print(
            "Annualised Volatility:",
            round(performances.loc[selected, "std_dev"].iloc[0], 2),
        )
        print("\n")
        allocations = {
            k: round(100 * v, 2)
            for k, v in performances.loc[selected, "allocations"].iloc[0].items()
        }
        display(pd.DataFrame(allocations, index=["Allocation"]))
    performances.loc[selected, "max_sharpe"] = True
    return performances


def find_min_volatility_portfolio(performances, verbose=False):
    selected = performances["std_dev"] == performances["std_dev"].min()

    if verbose:
        print("-" * 80)
        print("Minimum Volatility Portfolio Allocation\n")
        print(
            "Annualised Return:",
            round(performances.loc[selected, "returns"].iloc[0], 2),
        )
        print(
            "Annualised Volatility:",
            round(performances.loc[selected, "std_dev"].iloc[0], 2),
        )
        print("\n")
        allocations = {
            k: round(100 * v, 2)
            for k, v in performances.loc[selected, "allocations"].iloc[0].items()
        }
        display(pd.DataFrame(allocations, index=["Allocation"]))
    performances.loc[selected, "min_volatility"] = True
    return performances


def find_efficient_portfolios(performances, n_bins):
    min_std_dev = performances.loc[
        performances["min_volatility"] == True, "std_dev"
    ].min()
    next_smallest = performances.loc[
        performances["std_dev"] > min_std_dev, "std_dev"
    ].min()

    max_std_dev = performances.loc[performances["max_sharpe"] == True, "std_dev"].max()
    next_largest = performances.loc[
        performances["std_dev"] < max_std_dev, "std_dev"
    ].max()

    if (
        not np.isnan(next_smallest)
        and not np.isnan(next_largest)
        and not next_smallest >= next_largest
    ):
        bins = np.linspace(next_smallest, next_largest, n_bins + 1)
        performances["vol_bin"] = pd.cut(performances["std_dev"], bins)

        performances["efficient"] = performances.groupby("vol_bin", observed=True)[
            "returns"
        ].transform(lambda x: x == x.max())

    performances.loc[
        (performances["min_volatility"] == True) | (performances["max_sharpe"] == True),
        "efficient",
    ] = True

    return performances


def plot_allocations(results):
    efficient_portfolios = results.loc[results["efficient"] == True].copy()
    if "vol_bin" in efficient_portfolios.columns:
        efficient_portfolios["std_dev"] = (
            efficient_portfolios["vol_bin"]
            .apply(lambda x: x.mid)
            .astype(float)
            .fillna(efficient_portfolios["std_dev"])
        )
    efficient_portfolios["volatility_str"] = efficient_portfolios["std_dev"].apply(
        lambda x: f"{x:.2%}"
    )

    efficient_portfolios["allocations"] = efficient_portfolios["allocations"].apply(
        lambda x: list(x.items())
    )
    efficient_portfolios = efficient_portfolios.explode("allocations")
    efficient_portfolios["asset"] = efficient_portfolios["allocations"].str[0]
    efficient_portfolios["weight"] = efficient_portfolios["allocations"].str[1]

    allocations = px.bar(
        efficient_portfolios,
        x="volatility_str",
        y="weight",
        labels="std_dev",
        color="asset",
    )

    min_volatility = go.Scatter(
        x=efficient_portfolios.loc[
            efficient_portfolios["min_volatility"] == True, "volatility_str"
        ].iloc[:1],
        y=[1.1],
        mode="markers",
        marker=dict(symbol="star", color=GREEN, size=10),
        name="Min Volatility",
        hoverinfo="none",
    )
    max_sharpe = go.Scatter(
        x=efficient_portfolios.loc[
            efficient_portfolios["max_sharpe"] == True, "volatility_str"
        ].iloc[-1:],
        y=[1.1],
        mode="markers",
        marker=dict(symbol="star", color=RED, size=10),
        name="Max Sharpe Ratio",
        hoverinfo="none",
    )

    layout = go.Layout(
        title="Efficient Portfolio Allocations",
        xaxis=dict(
            title="std_dev",
            categoryorder="array",
            categoryarray=efficient_portfolios["volatility_str"],
        ),
        yaxis=dict(title="Allocations"),
        width=1000,
        height=400,
        barmode="stack",
        hovermode="x",
    )
    fig = go.Figure(
        data=[*allocations.data],
        layout=layout,
    )
    fig.update_traces(hovertemplate="%{y:.2%}")
    fig.add_trace(min_volatility)
    fig.add_trace(max_sharpe)

    fig.show()


def get_label(row):
    _return = row["returns"]
    volatility = row["std_dev"]
    sharpe_ratio = row["sharpe_ratio"]
    allocations = [
        f"&nbsp;<i>{k}</i> - {v*100:.0f}%"
        for k, v in row["allocations"].items()
        if v != 0
    ]
    allocations = "<br>".join(allocations)
    return (
        f"<b>Volatility : </b>{volatility*100:.2f}%<br>"
        f"<b>Return : </b>{_return*100:.2f}%<br>"
        f"<b>Sharpe ratio : </b>{sharpe_ratio*100:.2f}%<br>"
        "<b>Allocations : </b><br>"
        f"{allocations}"
    )


def get_kde(x, y, bins=20):
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )
    z[np.where(np.isnan(z))] = np.min(z)
    return z


def plot_efficient_frontier(results, risk_free_rate):
    if len(results) > 10_000:
        density = get_kde(results["std_dev"], results["returns"])
        weights = 1 / density
        sampled = results.sample(weights=weights, n=10_000)
        sampled_results = pd.concat(
            [results.loc[results["efficient"] == True], sampled]
        )
    else:
        sampled_results = results

    labels = sampled_results.apply(get_label, axis=1)
    simulated_portfolios = go.Scatter(
        x=sampled_results["std_dev"],
        y=sampled_results["returns"],
        mode="markers",
        text=labels,
        hoverinfo="text",
        marker=dict(
            color=sampled_results["sharpe_ratio"],
            colorscale="plasma",
            size=10,
            opacity=0.8,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Simulated Portfolios",
    )
    efficient_rows = results.loc[results["efficient"] == True]
    efficient_frontier = go.Scatter(
        x=efficient_rows["std_dev"],
        y=signal.savgol_filter(
            efficient_rows["returns"],
            len(efficient_rows),
            min(5, len(efficient_rows) - 1),
        ),
        line=dict(color="white", width=5),
        mode="lines",
        name="Efficient Frontier",
    )

    max_sharpe = go.Scatter(
        x=results.loc[results["max_sharpe"] == True, "std_dev"],
        y=results.loc[results["max_sharpe"] == True, "returns"],
        mode="markers",
        marker=dict(symbol="star", color=RED, size=20),
        name="Max Sharpe Ratio",
        hoverinfo="skip",
    )

    min_volatility = go.Scatter(
        x=results.loc[results["min_volatility"] == True, "std_dev"],
        y=results.loc[results["min_volatility"] == True, "returns"],
        mode="markers",
        marker=dict(symbol="star", color=GREEN, size=20),
        name="Min Volatility",
        hoverinfo="skip",
    )
    layout = go.Layout(
        title="Simulated Portfolio Performance",
        xaxis=dict(title="annualised volatility", tickformat=".2%"),
        yaxis=dict(title="annualised returns", tickformat=".2%"),
        width=1000,
        height=700,
    )

    fig = go.Figure(
        data=[simulated_portfolios, efficient_frontier, max_sharpe, min_volatility],
        layout=layout,
    )
    fig.add_hline(
        y=risk_free_rate,
        line_width=3,
        line_dash="dash",
        line_color="LightGrey",
        annotation_text="Risk Free Rate",
    )
    fig.update_layout(legend_orientation="h")

    fig.show()
    return None


def simulate_efficient_frontier(stock_data, num_portfolios, risk_free_rate, plot):
    performances = get_portfolio_performances(
        num_portfolios, stock_data, risk_free_rate
    )
    performances = performances.sort_values(by="std_dev")

    performances = find_max_sharpe_portfolio(performances, verbose=plot)
    performances = find_min_volatility_portfolio(performances, verbose=plot)
    performances = find_efficient_portfolios(
        performances,
        n_bins=30,
    )
    if plot:
        plot_efficient_frontier(performances)
        plot_allocations(performances)
    return performances


def main():
    num_portfolios = 25_0
    risk_free_rate = 0.04

    stocks = [
        "VTI",  # Total stock Market
        "BND",  # Total Bond Market
        "VEA",  # International Developed Markets
        "VB",  # Small Cap
        "VTV",  # Value
    ]

    stock_data = download_data(stocks, plot=False, start="2024-05-01", end="2024-10-31")
    results = simulate_efficient_frontier(
        stock_data, num_portfolios, risk_free_rate, plot=False
    )


if __name__ == "__main__":
    main()
