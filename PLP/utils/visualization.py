import matplotlib.pyplot as plt
import numpy as np
from .optimization import optimize_portfolio

def plot_efficient_frontier(returns, num_portfolios=1000):
    n_assets = returns.shape[1]
    returns_mean = returns.mean()
    returns_cov = returns.cov()

    portfolio_returns = []
    portfolio_volatilities = []

    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(returns_mean * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_std_dev)

    optimal_weights = optimize_portfolio(returns)
    optimal_return = np.sum(returns_mean * optimal_weights) * 252
    optimal_std_dev = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_cov * 252, optimal_weights)))

    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=np.array(portfolio_returns)/np.array(portfolio_volatilities), marker='o')
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.scatter(optimal_std_dev, optimal_return, c='red', s=50, marker='*', label='Optimal Portfolio')
    plt.legend()
    plt.show()
