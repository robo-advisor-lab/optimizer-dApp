import unittest
from PLP.utils.optimization import optimize_portfolio
import numpy as np

class TestOptimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.returns = np.random.rand(100, 5)

    def test_optimize_portfolio(self):
        weights = optimize_portfolio(self.returns)
        self.assertAlmostEqual(np.sum(weights), 1, places=2)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 1).all())
