import unittest
import numpy as np
import pandas as pd
from PLP.utils.simulation import simulate_loan, simulate_returns, backtest_strategy

class TestSimulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.S0 = 100
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.dt = 1/252
        self.M = 1000
        
        self.df = pd.DataFrame({
            'loan_id': ['loan1', 'loan2'],
            'funded_assets_dollar': [100, 200],
            'mu': [0.05, 0.06],
            'sigma': [0.2, 0.3],
            'loan_duration_days': [365, 180]
        })

    def test_simulate_loan(self):
        loan_paths = simulate_loan(self.S0, self.mu, self.sigma, self.T, self.dt, self.M)
        self.assertEqual(loan_paths.shape, (self.M, int(self.T/self.dt) + 1))
        self.assertTrue((loan_paths >= 0).all())

    def test_simulate_returns(self):
        returns = simulate_returns(self.df)
        self.assertEqual(returns.shape, (1000, 2))
        self.assertTrue((returns.columns == ['loan1', 'loan2']).all())

    def test_backtest_strategy(self):
        result = backtest_strategy(self.df)
        self.assertIn('portfolio_return', result)
        self.assertIn('portfolio_volatility', result)
        self.assertIn('sharpe_ratio', result)

if __name__ == '__main__':
    unittest.main()
