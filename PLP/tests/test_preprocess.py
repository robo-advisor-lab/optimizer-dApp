import unittest
import pandas as pd
import numpy as np
from PLP.utils.preprocess import preprocess_data

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        data = {
            'funding_open_timestamp': ['2020-01-01'],
            'term_start_timestamp': ['2020-01-02'],
            'term_end_timestamp': ['2021-01-01'],
            'base_interest_rate': [5.0],
            'principal_paid_dollar': [100],
            'interest_paid_dollar': [5],
            'funded_assets_dollar': [100]
        }
        self.df = pd.DataFrame(data)
    
    def test_preprocess_data(self):
        df_processed = preprocess_data(self.df)
        self.assertEqual(df_processed['loan_duration_days'][0], 365)
        self.assertAlmostEqual(df_processed['normalized_cumulative_return'][0], 1.05, places=2)

if __name__ == '__main__':
    unittest.main()
