import unittest
import pandas as pd
import numpy as np
from PLP.utils.preprocess import preprocess_data, load_data

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        self.data = {
            'funding_open_timestamp': ['2020-01-01'],
            'term_start_timestamp': ['2020-01-02'],
            'term_end_timestamp': ['2021-01-01'],
            'base_interest_rate': [5.0],
            'principal_paid_dollar': [100],
            'interest_paid_dollar': [5],
            'funded_assets_dollar': [100]
        }
        self.df = pd.DataFrame(self.data)
    
    def test_preprocess_data(self):
        df_processed = preprocess_data(self.df)
        self.assertEqual(df_processed['loan_duration_days'].values[0], 365)
        self.assertAlmostEqual(df_processed['normalized_cumulative_return'].values[0], 1.05, places=2)
        self.assertTrue('mu' in df_processed.columns)
        self.assertTrue('sigma' in df_processed.columns)

    def test_load_data(self):
        # You might want to create a small test CSV file for this
        test_file_path = 'test_data.csv'
        pd.DataFrame(self.data).to_csv(test_file_path, index=False)
        
        loaded_df = load_data(test_file_path)
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 1)
        
        import os
        os.remove(test_file_path)

if __name__ == '__main__':
    unittest.main()
