import unittest
from PLP.utils.simulation import simulate_loan
import numpy as np

class TestSimulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.S0 = 100
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.dt = 1/252
        self.M = 1000

    def test_simulate_loan(self):
        loan_paths = simulate_loan(self.S0, self.mu, self.sigma, self.T, self.dt, self.M)
        self.assertEqual(loan_paths.shape[0], self.M)
        self.assertEqual(loan_paths.shape[1], self.T/self.dt + 1)
        self.assertTrue((loan_paths >= 0).all())
