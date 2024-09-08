# %% [markdown]
# # Imports

# %%
import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3 import PPO
# from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib
import random
# import cvxpy as cp
# import matplotlib.pyplot as plt
import datetime as dt
# from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch
from flipside import Flipside

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  # Import pytz if using timezones

from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch
import sys

# %%
from utils import flipside_api_results, set_random_seed, to_time, clean_prices, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../sql_scripts')))

print(os.getcwd())

# %%
from queries import prices, volume 

os.chdir('../python_scripts')

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

def data_processing(api=False):
    # %% [markdown]
    # # Data Collection

    # %% [markdown]
    # ## BTC Volume

    # %%
    btc = yf.Ticker('BTC-USD')
    btc_df = btc.history(period='max')
    btc_df = btc_df['Volume'].to_frame('BTC Volume')
    btc_df_hourly = btc_df.resample('H').ffill()
    btc_df_hourly

    # %% [markdown]
    # ## DEX Volume & Asset Prices (Onchain Data)

    # %%
    def pull_data(api=False):
        prices_path = '../data/prices.csv'
        volume_path = '../data/volume.csv'

        if api == True:
            print('Pulling Fresh Data...')
            prices_df = flipside_api_results(prices, flipside_api_key)
            prices_df.to_csv(prices_path, index=False)
            volume_df = flipside_api_results(volume, flipside_api_key)
            volume_df.to_csv(volume_path, index=False)
        else:
            prices_df = pd.read_csv(prices_path)
            volume_df = pd.read_csv(volume_path)

        return prices_df, volume_df 

    # %% [markdown]
    # prices_df = flipside_api_results(prices, flipside_api_key)
    # prices_path = '../data/prices.csv'
    # prices_df.to_csv(prices_path, index=False)

    # %% [markdown]
    # volume_df = flipside_api_results(volume, flipside_api_key)
    # volume_path = '../data/volume.csv'
    # volume_df.to_csv(volume_path, index=False)

    # %%
    prices_df, volume_df = pull_data(api=api)

    # %%
    clean_prices_df = clean_prices(prices_df)
    clean_prices_df = to_time(clean_prices_df)
    if '__row_index' in clean_prices_df.columns:
        clean_prices_df.drop(columns=['__row_index'], inplace=True)
    clean_prices_df

    clean_prices_df

    # %%
    current_hour_start = dt.datetime.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)

    # %%
    clean_prices_df = clean_prices_df[
        (clean_prices_df.index >= '2019-01-31 02:00:00+00:00') &
        (clean_prices_df.index < current_hour_start)
    ].fillna(method='ffill')

    # %%
    volume_df = to_time(volume_df)
    volume_df.rename(columns={"volume":"DEX Volume"}, inplace=True)
    if '__row_index' in volume_df.columns:
        volume_df.drop(columns=['__row_index'], inplace=True)
    volume_df

    # %%


    # %% [markdown]
    # ## RWA.xyz Data

    # %%
    rwa_commodities_path = '../data/rwa_commodities.csv'
    rwa_commodities_df = pd.read_csv(rwa_commodities_path)
    rwa_commodities_df.head()

    # %%
    rwa_bonds_path = '../data/rwa_global_bonds.csv'
    rwa_bonds_df = pd.read_csv(rwa_bonds_path)
    rwa_bonds_df.head()

    # %%
    rwa_credit_path = '../data/rwa_private_credit.csv'
    rwa_credit_df = pd.read_csv(rwa_credit_path)
    rwa_credit_df.head()

    # %%
    rwa_pools_path = '../data/rwa_credit_pools.csv'
    rwa_pools_df = pd.read_csv(rwa_pools_path)
    rwa_pools_df.head()

    rwa_pools_df = rwa_pools_df[['pool_id','pool_name']]

    # %%
    rwa_loans_path = '../data/rwa_private_loans.csv'
    rwa_loans_df = pd.read_csv(rwa_loans_path)
    rwa_loans_df.head()

    # %%
    rwa_loans_df = rwa_loans_df.merge(rwa_pools_df, how='left', on='pool_id')


    # %%
    rwa_loans_df.columns

    # %%
    # Ensure term_start_timestamp and term_end_timestamp are in datetime format
    rwa_loans_df['term_start_timestamp'] = pd.to_datetime(rwa_loans_df['term_start_timestamp'])
    rwa_loans_df['term_end_timestamp'] = pd.to_datetime(rwa_loans_df['term_end_timestamp'])

    # Calculate the duration of each loan in days
    rwa_loans_df['loan_duration_days'] = (rwa_loans_df['term_end_timestamp'] - rwa_loans_df['term_start_timestamp']).dt.days

    # Calculate the number of compounding periods
    rwa_loans_df['compounding_periods'] = np.ceil(rwa_loans_df['loan_duration_days'] / rwa_loans_df['payment_interval_days'])

    # Convert base_interest_rate from percentage to a decimal
    rwa_loans_df['base_interest_rate_decimal'] = rwa_loans_df['base_interest_rate'] / 100

    # Calculate the normalized cumulative return for each loan with compounding at specific intervals
    rwa_loans_df['normalized_cumulative_return'] = (1 + rwa_loans_df['base_interest_rate_decimal'] / rwa_loans_df['compounding_periods']) ** rwa_loans_df['compounding_periods']

    # Calculate the cumulative return
    rwa_loans_df['cumulative_return'] = rwa_loans_df['normalized_cumulative_return'] - 1

    # %%
    top_20_loans = rwa_loans_df[['loan_id','pool_name','base_interest_rate','loan_duration_days','compounding_periods','cumulative_return','normalized_cumulative_return']].sort_values(by='cumulative_return', ascending=False).dropna().head(20)
    top_20_loans.to_csv('../data/cleaned_loans.csv', index=False)

    # %%
    top_20_loans

    # %%
    rwa_loans_df['pool_name'].isna().sum()

    # %%
    rwa_loans_df.columns
    rwa_loans_df[rwa_loans_df['pool_id']=='0xfe119e9c24ab79f1bdd5dd884b86ceea2ee75d92'].sort_values(by='funding_open_timestamp')

    # %%
    rwa_stocks_path = '../data/rwa_stocks.csv'
    rwa_stocks_df = pd.read_csv(rwa_stocks_path)
    rwa_stocks_df.head()

    # %%
    rwa_tbills_path = '../data/rwa_treasuries.csv'
    rwa_tbills_df = pd.read_csv(rwa_tbills_path)
    rwa_tbills_df.head()

    # %% [markdown]
    # # Feature Engineering

    # %%
    ## Calculate temporals like 7d, 30d vol, price, lags, etc

    # %%
    combined = pd.merge(clean_prices_df, volume_df, left_index=True, right_index=True, how='inner')

    # %%
    combined = combined.merge(btc_df_hourly, left_index=True, right_index=True, how='inner')

    # %%
    def temporal(df, windows):
        df1 = df.copy()
        
        # Create rolling average columns
        for col in df1.columns:
            for window in windows:
                df1[f'{col} {window}d_rolling_avg'] = df1[col].rolling(window=window, min_periods=1).mean()
        
        # Create lag columns, skipping columns that are rolling averages
        for col in df1.columns:
            if not any(col.endswith(f'{window}d_rolling_avg') for window in windows):
                for window in windows:
                    df1[f'{col} {window}d_lag'] = df1[col].shift(window)
        
        return df1

    # %%
    # Example usage
    windows = [7, 30]
    # combined = rolling_avg(combined, windows)
    combined = temporal(combined, windows)
    combined['ETH Price Pct Change'] = combined['ETH Price'].pct_change().fillna(0)
    combined['BTC Price Pct Change'] = combined['BTC Price'].pct_change().fillna(0)

    # %%
    combined_corr = combined.corr()
    print(combined_corr['ETH Price'].sort_values(ascending=False))

    # %%
    print(combined_corr['BTC Price'].sort_values(ascending=False))

    # %%
    eth_high_corr = combined_corr['ETH Price'] > 0.5
    eth_high_corr_cols = eth_high_corr.index[eth_high_corr].tolist()

    # %%
    btc_high_corr = combined_corr['BTC Price'] > 0.5
    btc_high_corr_cols = btc_high_corr.index[btc_high_corr].tolist()


    # %%
    # Combine the lists
    eth_high_corr_cols 

    # %%
    btc_high_corr_cols

    # %%
    combined_cols = eth_high_corr_cols + btc_high_corr_cols
    combined_cols = list(set(combined_cols))
    print(combined_cols)

    # %%
    filtered_combined = combined[combined_cols]
    filtered_combined.fillna(0, inplace=True)

    # %%
    filtered_combined

    # %%
    # filtered_combined = filtered_combined[filtered_combined.index < '2020-01-01']

    # %%
    price_cols = ['ETH Price',
            'BTC Price']

    temporals = ['BTC Price 30d_rolling_avg','ETH Price 30d_rolling_avg']

    return price_cols, temporals, filtered_combined, 
