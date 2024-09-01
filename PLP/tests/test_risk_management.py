import unittest
import numpy as np
from PLP.utils.risk_management import calculate_greeks, var_calculation, expected_shortfall

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.portfolio = {
            'weights': np.array([0.2, 0.3, 0.5]),
            'returns': np.random.normal(0.001, 0.02, (1000, 3)),
            'mu': np.array([0.001, 0.002, 0.003])
        }
        self.portfolio_returns = np.dot(self.portfolio['returns'], self.portfolio['weights'])

    def test_calculate_greeks(self):
        greeks = calculate_greeks(self.portfolio)
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)

    def test_var_calculation(self):
        var = var_calculation(self.portfolio_returns)
        self.assertIsInstance(var, float)
        self.assertTrue(var < 0)  # VaR should be negative

    def test_expected_shortfall(self):
        es = expected_shortfall(self.portfolio_returns)
        self.assertIsInstance(es, float)
        self.assertTrue(es < 0)  # Expected shortfall should be negative

if __name__ == '__main__':
    unittest.main()
