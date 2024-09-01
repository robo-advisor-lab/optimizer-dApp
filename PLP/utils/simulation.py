import numpy as np
import pandas as pd

def simulate_loan(S0, mu, sigma, T, dt, M):
    """
    Simula M trayectorias de un préstamo usando GBM.
    """
    steps = max(int(T / dt), 1)  # Asegurar al menos un paso
    S = np.zeros((M, steps + 1))
    S[:, 0] = S0
    
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(M)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return S

def simulate_returns(df):
    """
    Simula los retornos para cada préstamo en el DataFrame.
    """
    returns = pd.DataFrame()
    for loan in df.itertuples():
        simulated_values = simulate_loan(loan.funded_assets_dollar, loan.mu, loan.sigma, loan.loan_duration_days/365, 1/365, 1000)
        returns[loan.loan_id] = simulated_values[:, -1] / loan.funded_assets_dollar - 1
    return returns

def backtest_strategy(df, train_size=0.8):
    train = df.iloc[:int(len(df) * train_size)]
    test = df.iloc[int(len(df) * train_size):]

    train_returns = pd.DataFrame({loan.loan_id: simulate_loan(loan.funded_assets_dollar, loan.mu, loan.sigma, loan.loan_duration_days/365, 1/365, 1000)[:, -1] / loan.funded_assets_dollar - 1 for loan in train.itertuples()})
    
    from .optimization import optimize_portfolio
    optimal_weights = optimize_portfolio(train_returns)

    test_returns = pd.DataFrame({loan.loan_id: simulate_loan(loan.funded_assets_dollar, loan.mu, loan.sigma, loan.loan_duration_days/365, 1/365, 1000)[:, -1] / loan.funded_assets_dollar - 1 for loan in test.itertuples()})
    portfolio_return = np.dot(optimal_weights, test_returns.mean())
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(test_returns.cov(), optimal_weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility

    return {
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }
