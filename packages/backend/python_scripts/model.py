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

class Portfolio(gym.Env):
    def __init__(self, df,  seed, cash=1e4, hold_cash=False, rebalance_frequency=24):
        super(Portfolio, self).__init__()
        self.df = df
        self.current_step = 0
        self.total_assets = 2  # Number of assets
        self.hold_cash = hold_cash
        self.seed(seed)
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.total_assets,), dtype=np.float32)
        
        # Define observation space based on the number of features in the dataframe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)
        
        # Initialize portfolio and logs
        self.portfolio = np.zeros(self.total_assets)  # Initialize portfolio
        self.cash = cash  # Initial capital
        self.prev_portfolio_value = self.cash  # Initialize previous portfolio values
        self.rebalance_frequency = rebalance_frequency
        self.steps_since_last_rebalance = 0  # Initialize the counter for rebalancing

        # Logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.portfolio_composition_log = []  # Initialize log for portfolio composition

        # Get the columns related to asset prices
        self.price_columns = ["ETH Price","BTC Price"]

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio = np.zeros(self.total_assets)  # Reset portfolio
        self.cash = 1e4  # Reset capital
        self.prev_portfolio_value = self.cash  # Reset previous portfolio value
        self.steps_since_last_rebalance = 0  # Reset rebalance counter
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.portfolio_composition_log = []
        
        obs = self._get_observation()
        print(f"Reset environment. Initial observation: {obs}")
        return obs.astype(np.float32), {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        seed = int(seed % (2**32 - 1))
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_observation(self):
        # Ensure current_step does not go out-of-bounds
        if self.current_step >= len(self.df):
            raise IndexError(f"Current step {self.current_step} is out of bounds for the dataframe.")
        
        obs = self.df.iloc[self.current_step].values
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        return obs.astype(np.float32)

    def step(self, action):
        print(f"\n--- Step {self.current_step} ---")
        
        # Increment the counter for rebalancing
        self.steps_since_last_rebalance += 1
        print(f"Steps since last rebalance: {self.steps_since_last_rebalance}")

        # Clip actions to ensure valid values
        action = np.clip(action, 0, 1)
        print(f"Action after clipping: {action}")

        if np.isnan(action).any():
            raise ValueError("Action contains NaN values.")

        # Get current asset prices
        current_prices = self.df.iloc[self.current_step][self.price_columns].values
        print(f"Current prices: {current_prices}")
        
        # Calculate total portfolio value (cash + value of assets)
        portfolio_value_before = np.sum(self.portfolio * current_prices) + self.cash
        print(f"Portfolio value before action: {portfolio_value_before}")

        if self.steps_since_last_rebalance >= self.rebalance_frequency or self.current_step == 0:
            print("Rebalancing portfolio...")

            # Reset the counter after performing the rebalance
            self.steps_since_last_rebalance = 0
            print("Steps since last rebalance reset to 0")

            # Calculate action as percentage of portfolio
            total_value = portfolio_value_before
            print(f"Total portfolio value: {total_value}")

            total_action = np.sum(action)
            print(f"Total action: {total_action}")

            action_percentages = action / total_action if total_action > 0 else np.zeros_like(action)
            print(f"Action percentages: {action_percentages}")

            # Calculate desired allocations based on the current portfolio value
            desired_allocations = total_value * action_percentages
            print(f"Desired allocations: {desired_allocations}")

            current_allocations = self.portfolio * current_prices
            print(f"Current allocations: {current_allocations}")

            # Calculate total available cash
            cash_to_invest = total_value - np.sum(current_allocations)
            print(f"Cash to invest: {cash_to_invest}")

            # Buy/sell to rebalance portfolio
            for i in range(self.total_assets):
                asset_price = current_prices[i]
                print(f"Processing asset {i}, price: {asset_price}")

                if desired_allocations[i] > current_allocations[i]:
                    # Buy the asset to reach desired allocation
                    amount_to_buy = (desired_allocations[i] - current_allocations[i]) / asset_price
                    cost = amount_to_buy * asset_price
                    print(f"Amount to buy: {amount_to_buy}, cost: {cost}")

                    if cash_to_invest >= cost:
                        self.portfolio[i] += amount_to_buy
                        cash_to_invest -= cost
                        print(f"Bought asset {i}: {amount_to_buy}, remaining cash to invest: {cash_to_invest}")
                    else:
                        amount_to_buy = cash_to_invest / asset_price
                        self.portfolio[i] += amount_to_buy
                        cash_to_invest = 0
                        print(f"Bought as much as possible for asset {i}: {amount_to_buy}, cash depleted to 0")

                elif desired_allocations[i] < current_allocations[i]:
                    # Sell the asset to reach desired allocation
                    amount_to_sell = (current_allocations[i] - desired_allocations[i]) / asset_price
                    self.portfolio[i] -= amount_to_sell
                    cash_to_invest += amount_to_sell * asset_price
                    print(f"Sold asset {i}: {amount_to_sell}, cash to invest: {cash_to_invest}")

            # Ensure no negative values in portfolio
            self.portfolio = np.maximum(self.portfolio, 0)
            print(f"Portfolio after adjustment: {self.portfolio}")

       
            # Invest any remaining cash
            if cash_to_invest > 0:
                remaining_asset = np.argmax(action)
                amount_to_buy = cash_to_invest / current_prices[remaining_asset]
                self.portfolio[remaining_asset] += amount_to_buy
                print(f"Invested remaining cash in asset {remaining_asset}: {amount_to_buy}")

            self.cash = 0
            print(f"Cash after adjustment: {self.cash}")

        # Calculate portfolio value after rebalancing
        portfolio_value_after = np.sum(self.portfolio * current_prices) + self.cash
        print(f"Portfolio value after action: {portfolio_value_after}")

        # Reward is based on the percentage change in portfolio value
        reward = np.log(portfolio_value_after / portfolio_value_before) if portfolio_value_before else 0
        print(f"Reward: {reward}")

        # Move to the next step

        # if done:
        #     state = None  # Or you can return the last valid state
        # else:
        state = self._get_observation()
        print(f"State: {state}")
        self.states_log.append((state, self.df.index[self.current_step]))
        print(f"State log updated")

    # state = self._get_observation()
    
        # Update logs
        self.prev_portfolio_value = portfolio_value_after
        print(f"Previous portfolio value updated: {self.prev_portfolio_value}")

        self.rewards_log.append((reward, self.df.index[self.current_step]))
        print(f"Rewards log updated")

        self.actions_log.append((action_percentages, self.df.index[self.current_step]))
        print(f"Actions log updated")

        self.portfolio_values_log.append((portfolio_value_after, self.df.index[self.current_step]))
        print(f"Portfolio values log updated")

        self.portfolio_composition_log.append((self.portfolio.copy(), self.cash, self.df.index[self.current_step]))
        print(f"Portfolio composition log updated")

        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        print(f"Done: {done}")

        return (state.astype(np.float32), reward, done, False, {}) if state is not None else (None, reward, done, False, {})



    def render(self, mode='human'):
        pass  # Optional: Implement visualization of portfolio status

    def get_states_df(self):
        states, dates = zip(*self.states_log) if self.states_log else ([], [])
        return pd.DataFrame(states, columns=self.df.columns).assign(Date=dates)

    def get_rewards_df(self):
        rewards, dates = zip(*self.rewards_log) if self.rewards_log else ([], [])
        return pd.DataFrame(rewards, columns=['Reward']).assign(Date=dates)

    def get_actions_df(self):
        actions, dates = zip(*self.actions_log) if self.actions_log else ([], [])
        return pd.DataFrame(actions, columns=[f'Action_{i}' for i in range(self.total_assets)]).assign(Date=dates)

    def get_portfolio_values_df(self):
        portfolio_values, dates = zip(*self.portfolio_values_log) if self.portfolio_values_log else ([], [])
        return pd.DataFrame(portfolio_values, columns=['Portfolio_Value']).assign(Date=dates)

    def get_portfolio_composition_df(self):
        # Extract portfolio compositions, cash, and dates
        compositions, cash, dates = zip(*self.portfolio_composition_log) if self.portfolio_composition_log else ([], [], [])

        # Convert portfolio compositions to a DataFrame
        compositions_df = pd.DataFrame(compositions, columns=[f'Asset_{i}' for i in range(len(compositions[0]))])

        # Add the cash column
        compositions_df['Cash'] = cash

        # Convert dates to a DataFrame
        dates_df = pd.DataFrame(dates, columns=['Date'])

        # Combine compositions_df and dates_df
        portfolio_composition_df = pd.concat([dates_df, compositions_df], axis=1)

        return portfolio_composition_df
    def get_portfolio_value(self):
        # Get the current prices for all assets
        current_prices = self.df.iloc[self.current_step][self.price_columns].values
        
        # Calculate the total portfolio value
        portfolio_value = np.sum(self.portfolio * current_prices) + self.cash
        
        return portfolio_value