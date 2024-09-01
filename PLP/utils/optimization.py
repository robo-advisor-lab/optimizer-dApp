import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns):
    def objective(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return -portfolio_return / portfolio_std_dev

    n = returns.shape[1]
    initial_weights = np.array([1/n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(objective, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x
    
    optimal_weights = result.x
    optimal_returns = np.dot(returns, optimal_weights)

    return {
        'weights': optimal_weights,
        'returns': optimal_returns
    }
