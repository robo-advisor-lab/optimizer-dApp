import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PLP.utils.visualization import plot_efficient_frontier

class TestVisualization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.returns = pd.DataFrame(np.random.normal(0.001, 0.02, (1000, 5)))

    def test_plot_efficient_frontier(self):
        initial_figures = plt.get_fignums()
        plot_efficient_frontier(self.returns)
        self.assertTrue(len(plt.get_fignums()) > len(initial_figures))  # Check if a new figure was created
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
