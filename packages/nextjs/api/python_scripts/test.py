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

# %%
# os.chdir('..')

# %%
from utils import set_random_seed, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data
from data_processing import data_processing
from model import Portfolio

# %%
# os.chdir('notebooks')

# %% [markdown]
# # Environment Variables

# %%
seed=20
set_random_seed(seed)

price_cols, temporals, filtered_combined = data_processing(api=False)

# %%
def set_global_seed(env, seed=20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env.seed(seed)
    env.action_space.seed(seed)



# %%
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# env = Portfolio(filtered_combined[price_cols], hold_cash=False, seed=seed, rebalance_frequency=1)
# set_global_seed(env)
# # env.seed(seed)
# # env.action_space.seed(seed)
# # env = DummyVecEnv([lambda: Portfolio(filtered_combined_1, hold_cash=False, seed=seed)])

# # Create PPO agent
# model = PPO('MlpPolicy', env, seed=seed, verbose=1)

# # Train the model
# model.learn(total_timesteps=10000)

# # Save the model
# model.save('eth_btc_model')

# %% [markdown]
# def test_gym_seed(seed):
#     env = Portfolio(filtered_combined_1[['ETH Price','BTC Price','ETH Price 30d_rolling_avg','BTC Price 30d_rolling_avg']], hold_cash=False, seed=seed)
#     env.seed(seed)
#     
#     random_number_python = random.random()
#     random_number_numpy = np.random.rand()
#     random_tensor_torch = torch.rand(1).item()
#     random_tensor_tf = tf.random.uniform([1], 0, 1).numpy()[0]
#     
#     # Testing the gym.utils seeding
#     np_random_test = env.np_random.random()
#     
#     print(random_number_python, random_number_numpy, random_tensor_torch, random_tensor_tf, np_random_test)
# 
# # Set a fixed seed
# seed = 20
# set_random_seed(seed)
# test_gym_seed(seed)

# %%
import matplotlib.pyplot as plt

# Load the trained model
model = PPO.load("eth_btc_model")

# Initialize the environment
env = Portfolio(filtered_combined[price_cols], hold_cash=False, seed=seed, rebalance_frequency=1)  # Make sure this is your Gym environment
set_global_seed(env)
# env.seed(seed)
# env.action_space.seed(seed)
# Initialize lists to store results
states = []
rewards = []
actions = []
portfolio_values = []
compositions = []
dates = []

# Reset the environment to get the initial state
state, _ = env.reset(seed=seed)  # Get the initial state
done = False

while not done:
    # Use the model to predict the action
    action, _states = model.predict(state)
    
    # Take a step in the environment
    next_state, reward, done, truncated, info = env.step(action)
    
    # Normalize the action to ensure it sums to 1
    # action = action / np.sum(action)
    
    # Store the results
    states.append(next_state.flatten())  # Ensure the state is flattened
    rewards.append(reward)
    actions.append(action.flatten())  # Ensure the action is flattened
    portfolio_values.append(env.get_portfolio_value())
    compositions.append(env.portfolio)  # Store the portfolio composition
    print(f'Action: {action}')


    # Update the state
    state = next_state

    # Print debug information
    print(f"Step: {env.current_step}")
    print(f"State: {next_state}")
    print(f'Action: {action}')
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print(f"Portfolio Value: {env.get_portfolio_value()}")

# %%
# Access the logged data as DataFrames
states_df = env.get_states_df()
rewards_df = env.get_rewards_df()
actions_df = env.get_actions_df()
portfolio_values_df = env.get_portfolio_values_df()
composition = env.get_portfolio_composition_df()

# Analyze the results
print("States:")
print(states_df.head())

print("Rewards:")
print(rewards_df.describe())

print("Actions:")
print(actions_df.describe())

print("Portfolio Values:")
print(portfolio_values_df.head())

# Plot the rewards over time
plt.plot(rewards_df['Date'], rewards_df['Reward'])
plt.xlabel('Date')
plt.ylabel('Reward')
plt.title('Rewards over Time')
plt.show()

# Plot the portfolio value over time
plt.plot(portfolio_values_df['Date'], portfolio_values_df['Portfolio_Value'])
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time')
plt.show()

# %%
actions_df

# %%
portfolio_values_df.set_index('Date', inplace=True)

# %%
portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')


# %%
def normalize_asset_returns(price_timeseries, start_date, end_date, normalize_value=1e4):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Adjust start_date to the latest available date in the price_timeseries
   
    # print(f'Adjusted start date: {start_date}')
    # print(f'End date: {end_date}')
    
    # Filter the data based on the adjusted start date and end date
    filtered_data = price_timeseries[(price_timeseries.index >= start_date) & (price_timeseries.index <= end_date)].copy()
    #print(f'Normalize function filtered data: {filtered_data}')
    
    if filtered_data.empty:
        print("Filtered data is empty after applying start date.")
        return pd.DataFrame()

    # Converting data to float64 to ensure compatibility with numpy functions
    prev_prices = filtered_data.iloc[0][['BTC Price', 'ETH Price']].astype(np.float64).values
    #print(f"Initial previous prices: {prev_prices}")

    normalized_values = {
        'BTC Price': [normalize_value],
        'ETH Price': [normalize_value],
    }
    dates = [start_date]  # Use the original start date for labeling
    
    for i in range(1, len(filtered_data)):
        current_prices = filtered_data.iloc[i][['BTC Price', 'ETH Price']].astype(np.float64).values
        #print(f"Iteration {i}, Current Prices: {current_prices}")

        # Calculate log returns safely
        price_ratio = current_prices / prev_prices
        log_returns = np.log(price_ratio)
        #print(f"Price ratio: {price_ratio}")
        #print(f"Log returns: {log_returns}")

        # Update the normalized values for each asset using the exponential of log returns
        for idx, asset in enumerate(['BTC Price', 'ETH Price']):
            normalized_values[asset].append(normalized_values[asset][-1] * np.exp(log_returns[idx]))
            #print(f"Updated normalized value for {asset}: {normalized_values[asset][-1]}")

        dates.append(filtered_data.index)
        prev_prices = current_prices
    
    normalized_returns_df = pd.DataFrame({
        'ds': filtered_data.index,
        'normalized_ETH': normalized_values['ETH Price'],
        'normalized_BTC': normalized_values['BTC Price'],
    })
    
    return normalized_returns_df

# %%
filtered_combined.reset_index()

# %%
norm_prices = normalize_asset_returns(filtered_combined, portfolio_values_df.index.min(),portfolio_values_df.index.max())

# %%
norm_prices.set_index('ds', inplace=True)

# %%
norm_prices

# %%
eth_return = calculate_cumulative_return(norm_prices, 'normalized_ETH')
btc_return = calculate_cumulative_return(norm_prices, 'normalized_BTC')

# %%
excess_return_btc = portfolio_return - btc_return
excess_return_eth = portfolio_return - eth_return


# %%
portfolio_values_df.index

# %%
norm_prices.index

# %%
portfolio_values_df

# %%
fig = go.Figure()

for col in norm_prices.columns:
    fig.add_trace(
        go.Scatter(
            x=norm_prices.index,
            y=norm_prices[col],
            name=col
        )
    )

fig.add_trace(
    go.Scatter(
            x=portfolio_values_df.index,
            y=portfolio_values_df['Portfolio_Value'],
            name='Portfolio Value'
        )
    
)

fig.show()

# %%
print(f'{portfolio_values_df.index.min().strftime("%d/%m/%Y")} through {portfolio_values_df.index.max().strftime("%d/%m/%Y")}')
print(f'Portfolio Cumulative Return: {portfolio_return*100:.2f}%')
print(f'ETH Cumulative Return: {eth_return*100:.2f}%')
print(f'BTC Cumulative Return: {btc_return*100:.2f}%')

# %%
print(f'Portfolio Excess Return Over ETH: {excess_return_eth*100:,.2f}%')
print(f'Portfolio Excess Return Over BTC: {excess_return_btc*100:.2f}%')

# %%
composition.rename(columns={"Asset_0":"ETH_Comp","Asset_1":"BTC_Comp"}, inplace=True)

# %%
composition

# %%
norm_prices.tail(20)

# %%
filtered_combined[['ETH Price','BTC Price']]

# %%
composition.set_index(
    'Date',
    inplace=True
)

# %%
combined_analysis = pd.merge(filtered_combined[['ETH Price','BTC Price']], composition, left_index=True, right_index=True, how='inner')

# %%
combined_analysis['ETH_Value'] = combined_analysis['ETH_Comp'] * combined_analysis['ETH Price']
combined_analysis['BTC_Value'] = combined_analysis['BTC_Comp'] * combined_analysis['BTC Price']

# Calculate total portfolio value
combined_analysis['Total_Value'] = combined_analysis['ETH_Value'] + combined_analysis['BTC_Value'] + combined_analysis['Cash']

# Calculate the percentage composition
combined_analysis['ETH_Percentage'] = (combined_analysis['ETH_Value'] / combined_analysis['Total_Value']) * 100
combined_analysis['BTC_Percentage'] = (combined_analysis['BTC_Value'] / combined_analysis['Total_Value']) * 100
combined_analysis['Cash_Percentage'] = (combined_analysis['Cash'] / combined_analysis['Total_Value']) * 100

# Display the resulting DataFrame
print(combined_analysis[['ETH_Percentage', 'BTC_Percentage', 'Cash_Percentage']])

# %%
combined_analysis.index = combined_analysis.index.tz_convert(None)

november = combined_analysis[(combined_analysis.index >= pd.to_datetime('2019-11-01')) & (combined_analysis.index <= pd.to_datetime('2019-11-30'))]
november

# %%
combined_analysis.tail(120).to_csv(
    '../data/weird_analysis.csv'
)

# %%
november['Total_Value'].plot()

# %%
fig = go.Figure()

# %%
november[['ETH Price','BTC Price']].plot()

# %%
fig = go.Figure()

for col in combined_analysis[['ETH_Percentage', 'BTC_Percentage', 'Cash_Percentage']].columns:
    fig.add_trace(
        go.Scatter(
            x=combined_analysis.index,
            y=combined_analysis[col],
            name=col,
            stackgroup='one',
        )
    )

fig.update_layout(
    barmode='stack'
)

fig.show()

# %%
composition.head(20)

# %%
actions_df

# %% [markdown]
# # Market Comparison

# %%
crypto_index = pd.read_excel('../data/PerformanceGraphExport.xls')


# %%
fred_api_key = os.getenv("FRED_API_KEY")
three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"

# %%
try:
    three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, fred_api_key, "observations", "date", "value")
    three_month_tbill['decimal'] = three_month_tbill['value'] / 100
    current_risk_free = three_month_tbill['decimal'].iloc[-1]
    print(f"3-month T-bill data fetched: {three_month_tbill.tail()}")
except Exception as e:
    print(f"Error in fetching tbill data: {e}")

# %%
crypto_index.columns

# %%
crypto_index['Effective date '] = pd.to_datetime(crypto_index['Effective date '])
crypto_index.set_index('Effective date ', inplace=True)

# %%
daily_portfolio = portfolio_values_df.resample('D').last()

# %%
daily_portfolio

# %%
index_cagr = calculate_cagr(crypto_index['S&P Cryptocurrency Broad Digital Market Index (USD)'])
index_cumulative_risk_premium = index_cagr - current_risk_free

# %%
index_cumulative_risk_premium

# %%
index_cagr

# %%
# filtered_portfolio = daily_portfolio[daily_portfolio.index >= crypto_index.index.min()]

# # %%
# filtered_portfolio

# # %%
# portfolio_cagr = calculate_cagr(filtered_portfolio['Portfolio_Value'])


# %%
daily_portfolio.index

# %%
crypto_index.index

# %%
daily_portfolio.index = daily_portfolio.index.tz_localize(None)


# %%
analysis_df = pd.merge(daily_portfolio, crypto_index, left_index=True, right_index=True, how='inner')

# %%
portfolio_cagr = calculate_cagr(analysis_df['Portfolio_Value'])

analysis_df

# %%
portfolio_beta = calculate_beta(analysis_df, 'S&P Cryptocurrency Broad Digital Market Index (USD)'
                                ,'Portfolio_Value')

# %%
portfolio_expected_return = current_risk_free + (portfolio_beta*index_cumulative_risk_premium)

# %%
print(f'From {analysis_df.index.min()} through {analysis_df.index.max()}')
print(f'Market Risk Premium: {index_cumulative_risk_premium}')
print(f'Index CAGR: {index_cagr}')
print(f'Portfolio CAGR: {portfolio_cagr}')
print(f'Portfolio Beta: {portfolio_beta}')
print(f'Risk Free Rate: {current_risk_free}')
print(f'Portfolio Expected Return: {portfolio_expected_return}')

# %%
from plotly.subplots import make_subplots

# %%
portfolio_values_df

# %%
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Portfolio_Value'],
        name='Portfolio Value'
    ), secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=analysis_df.index,
        y=analysis_df['S&P Cryptocurrency Broad Digital Market Index (USD)'],
        name='S&P Crypto Index'
    ), secondary_y=True
)

fig.update_layout(
    yaxis=dict(
        # tickfont=dict(size=18, family="IBM Plex Mono", color='black'),
        tickprefix = "$"
        
    ),
    yaxis2=dict(  # Corrected yaxis2 configuration
        # tickfont=dict(size=, family="IBM Plex Mono", color='black'),
        overlaying='y',
        tickprefix = "$"
    )

)

fig.show()

# %%



