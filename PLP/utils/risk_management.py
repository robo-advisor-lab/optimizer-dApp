import numpy as np
from scipy import stats

def calculate_greeks(portfolio):
    # Implementación simplificada de las griegas
    delta = np.sum(portfolio['weights'] * portfolio['returns'].mean())
    gamma = np.sum(portfolio['weights'] * (portfolio['returns'].std() ** 2))
    theta = -np.sum(portfolio['weights'] * portfolio['mu'])
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta}

def var_calculation(portfolio_returns, confidence_level=0.95):
    return np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def expected_shortfall(portfolio_returns, confidence_level=0.95):
    var = var_calculation(portfolio_returns, confidence_level)
    return portfolio_returns[portfolio_returns <= var].mean()
