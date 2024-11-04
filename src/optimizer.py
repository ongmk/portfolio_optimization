import math
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from tqdm import tqdm


def n_combinations(n, r):
    result = math.factorial(n + r - 1) // (math.factorial(r) * math.factorial(n - 1))
    return result


def get_n_weight_combinations(stocks, num_portfolios):
    n_parts = len(stocks)
    while n_combinations(len(stocks), n_parts) < num_portfolios:
        n_parts += 1
    n_parts += 1
    portfolio_weights = []
    for combination in combinations_with_replacement(stocks, n_parts):
        counts = Counter(combination)
        portfolio_weights.append(np.array([counts[s] / n_parts for s in stocks]))
    return portfolio_weights


def portfolio_annualised_performance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return std, returns


def get_portfolio_performances(numPortfolios, meanReturns, covMatrix, riskFreeRate=0):
    portfolio_weights = get_n_weight_combinations(meanReturns.index, numPortfolios)
    actual_num_portfolios = len(portfolio_weights)
    results = np.zeros((3, actual_num_portfolios))
    allocations = []
    for i in tqdm(range(actual_num_portfolios)):
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            portfolio_weights[i], meanReturns, covMatrix
        )
        allocations.append(
            {
                asset: weight
                for asset, weight in zip(meanReturns.index, portfolio_weights[i])
            }
        )
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - riskFreeRate) / portfolio_std_dev
    return results, allocations


def find_max_sharpe_portfolio(results):
    selected = results["sharpe_ratio"] == results["sharpe_ratio"].max()
    return results.loc[selected].iloc[0]


def find_optimal_portfolio(data, risk_free_rate, num_portfolios=1000):
    returns = data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    results, allocations = get_portfolio_performances(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate
    )
    results = pd.DataFrame(results.T, columns=["volatility", "return", "sharpe_ratio"])
    results["allocations"] = allocations
    return find_max_sharpe_portfolio(results)
