import unittest
import numpy as np
import pandas as pd
from PLP.utils.optimization import optimize_portfolio

class TestOptimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.returns = pd.DataFrame(np.random.rand(100, 5))

    def test_optimize_portfolio(self):
        weights = optimize_portfolio(self.returns)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(np.sum(weights), 1, places=6)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 1).all())

if __name__ == '__main__':
    unittest.main()
