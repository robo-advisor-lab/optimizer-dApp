import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import random

import datetime as dt
from plotly.subplots import make_subplots

from sklearn.metrics import r2_score, mean_absolute_error

from flipside import Flipside

import os
from dotenv import load_dotenv

import datetime as dt
from datetime import timedelta
import pytz  
import asyncio

from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch

from python_scripts.utils import set_random_seed, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data, flipside_api_results, set_global_seed, normalize_asset_returns, prepare_data_for_simulation
# from python_scripts.data_processing import data_processing
from python_scripts.data_cleaning import data_cleaning
from python_scripts.model import Portfolio
from sql_scripts.queries import live_prices

from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import logging
from diskcache import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# price_cols, temporals = data_processing(api=False) # need to pass the new sql query for testing?  (not historical)
price_cols = ['ETH Price',
            'BTC Price']
cache = Cache('cache_dir')
# historical_data = pd.DataFrame()
historical_port_values = pd.DataFrame()
# model_actions = pd.DataFrame()
# last_rebalance_time = None

historical_data = cache.get('historical_data', pd.DataFrame())
historical_port_values = cache.get('historical_port_values', pd.DataFrame())
model_actions = cache.get('model_actions', pd.DataFrame())
last_rebalance_time = cache.get('last_rebalance_time', None)

if last_rebalance_time != None:
    print(f'last rebalance time: {last_rebalance_time}')

def update_historical_data(live_comp):
    global historical_data
    new_data = pd.DataFrame([live_comp])
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set('historical_data', historical_data)

def update_portfolio_data(values):
    global historical_port_values
    print(f'values: {values}')
    # new_data = pd.DataFrame([values])
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set('historical_port_values', historical_port_values)
    print(f"cache:{cache.get('historical_port_values')}")

def update_model_actions(actions):
    global model_actions
    print(f'model actions before update: {model_actions}')
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    print(f'model actions after update: {model_actions}')
    cache.set('model_actions', model_actions)  

print(f'historical Port vals: {historical_port_values}')

def create_app():
    app = Flask(__name__)

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        print('Clearing the cache...')
        cache.clear()
        return jsonify({"status": "Cache cleared successfully"})

    @app.route('/run-model', methods=['POST'])
    def run_model():
        seed=20
        today_utc = dt.datetime.now(dt.timezone.utc) 
        # today_utc = dt.datetime.now(dt.timezone.utc) - timedelta(hours=2)

    # Format the UTC time
        formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00') 
        print(f'today: {formatted_today_utc}')

        flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

        if 'Date' in historical_port_values.columns and not historical_port_values['Date'].empty:
            start_date = pd.to_datetime(historical_port_values['Date'].min()).strftime('%Y-%m-%d %H:00:00')
        else:
            start_date = formatted_today_utc
        print(f'histortical port values: {historical_port_values}')
        print(f'sql start date: {start_date}')

        prices_query = live_prices(start_date)
        prices_df = flipside_api_results(prices_query, flipside_api_key)
        prices_df = data_cleaning(prices_df)
        prices_df = prepare_data_for_simulation(prices_df, start_date, formatted_today_utc)
        print(f'prices df: {prices_df}')
        prices_df.set_index('hour', inplace=True)

        model = PPO.load("eth_btc_model")
        env = Portfolio(prices_df[price_cols], hold_cash=False, seed=seed, rebalance_frequency=1)
        set_global_seed(env)

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
            
            # Break the loop if done to avoid processing a None state
            if done:
                print("Episode done. Exiting the loop.")
                break
            
            # Normalize the action to ensure it sums to 1
            # action = action / np.sum(action)
            
            # Store the results
            if next_state is not None:
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


        states_df = env.get_states_df()
        rewards_df = env.get_rewards_df()
        actions_df = env.get_actions_df()
        portfolio_values_df = env.get_portfolio_values_df()
        composition = env.get_portfolio_composition_df()

        plt.plot(rewards_df['Date'], rewards_df['Reward'])
        plt.xlabel('Date')
        plt.ylabel('Reward')
        plt.title('Rewards over Time')
        plt.show()

        print(f"cache:{cache.get('historical_port_values')}")
        update_portfolio_data(portfolio_values_df)
        print(f"cache:{cache.get('historical_port_values')}")

        portfolio_values_df.set_index('Date', inplace=True)
        portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')

        # prices_df.index = pd.to_datetime(prices_df.index)
        print(f'price index: {prices_df.index}')

       

        # Assuming 'portfolio_values_df' is your DataFrame with a timezone-aware index
        print(f'portfolio index: {portfolio_values_df.index}')
        portfolio_values_df.index = pd.to_datetime(portfolio_values_df.index)
        # portfolio_values_df.index = portfolio_values_df.index.tz_localize(None)

        print(f'portfolio_values_df: {portfolio_values_df}')


        norm_prices = normalize_asset_returns(prices_df, portfolio_values_df.index.min(),portfolio_values_df.index.max())
        norm_prices.set_index('ds', inplace=True)

        norm_prices

        eth_return = calculate_cumulative_return(norm_prices, 'normalized_ETH')
        btc_return = calculate_cumulative_return(norm_prices, 'normalized_BTC')

        excess_return_btc = portfolio_return - btc_return
        excess_return_eth = portfolio_return - eth_return

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

        fig.update_layout(
            yaxis=dict(
                tickprefix = "$"
                
            ))

        fig.show()

        print(f'{portfolio_values_df.index.min().strftime("%d/%m/%Y")} through {portfolio_values_df.index.max().strftime("%d/%m/%Y")}')
        print(f'Portfolio Cumulative Return: {portfolio_return*100:.2f}%')
        print(f'ETH Cumulative Return: {eth_return*100:.2f}%')
        print(f'BTC Cumulative Return: {btc_return*100:.2f}%')

        # %%
        print(f'Portfolio Excess Return Over ETH: {excess_return_eth*100:,.2f}%')
        print(f'Portfolio Excess Return Over BTC: {excess_return_btc*100:.2f}%')

        crypto_index = pd.read_excel('data/PerformanceGraphExport.xls')

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

        crypto_index['Effective date '] = pd.to_datetime(crypto_index['Effective date '])
        crypto_index.set_index('Effective date ', inplace=True)

        daily_portfolio = portfolio_values_df.resample('D').last()

          
        daily_portfolio.index = daily_portfolio.index.tz_localize(None)


        analysis_df = pd.merge(daily_portfolio, crypto_index, left_index=True, right_index=True, how='left')
        index_cagr = calculate_cagr(analysis_df['S&P Cryptocurrency Broad Digital Market Index (USD)'])
        index_cumulative_risk_premium = index_cagr - current_risk_free
        portfolio_cagr = calculate_cagr(analysis_df['Portfolio_Value'])

        portfolio_beta = calculate_beta(analysis_df, 'S&P Cryptocurrency Broad Digital Market Index (USD)'
                                ,'Portfolio_Value')


        portfolio_expected_return = current_risk_free + (portfolio_beta*index_cumulative_risk_premium)

        print(f'From {analysis_df.index.min()} through {analysis_df.index.max()}')
        print(f'Market Risk Premium: {index_cumulative_risk_premium}')
        print(f'Index CAGR: {index_cagr}')
        print(f'Portfolio CAGR: {portfolio_cagr}')
        print(f'Portfolio Beta: {portfolio_beta}')
        print(f'Risk Free Rate: {current_risk_free}')
        print(f'Portfolio Expected Return: {portfolio_expected_return}')

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
        
        # json_action = jsonify(actions_df.iloc[-1])
        actions_df.rename(columns={"Action_0": "ETH", "Action_1": "BTC"}, inplace=True)
        print(f'latest action: {actions_df.iloc[-1]}')
        actions_df.to_csv('data/actions.csv')
        # print(f'json action: {json_action}')

        actions_df[['ETH','BTC']] = actions_df[['ETH','BTC']] * 10000

        actions_df[['ETH','BTC']] = actions_df[['ETH','BTC']].astype(int)

        resp = jsonify((actions_df[['ETH','BTC']].iloc[-1].values).tolist())

        print(f'actions: {resp}')

        print(f'historical port values variable: {historical_port_values}')

        cache.set('historical_port_values', historical_port_values)
        print(f"cache:{cache.get('historical_port_values')}")

        return resp

    return app

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app = create_app()
    print('Starting Flask app...')
    app.run(debug=True, use_reloader=False, port=5000)
    # Since app.run() is blocking, the following line will not execute until the app stops:
    logger.info("Flask app has stopped.")
    print('Flask app has stopped.')














        




