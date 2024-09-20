from flask import Flask, jsonify, request
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error
from flipside import Flipside
import os
from dotenv import load_dotenv
from datetime import timedelta
import pytz
import asyncio
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch
from python_scripts.utils import set_random_seed, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data, flipside_api_results, set_global_seed, normalize_asset_returns, prepare_data_for_simulation
from python_scripts.data_cleaning import data_cleaning
from python_scripts.model import Portfolio
from sql_scripts.queries import live_prices
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from diskcache import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

price_cols = ['ETH Price', 'BTC Price']
cache = Cache('cache_dir')

historical_data = cache.get('historical_data', pd.DataFrame())
historical_port_values = cache.get('historical_port_values', pd.DataFrame())
model_actions = cache.get('model_actions', pd.DataFrame())
last_rebalance_time = cache.get('last_rebalance_time', None)

def update_historical_data(live_comp):
    global historical_data
    new_data = pd.DataFrame([live_comp])
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set('historical_data', historical_data)

def update_portfolio_data(values):
    global historical_port_values
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set('historical_port_values', historical_port_values)

def update_model_actions(actions):
    global model_actions
    new_data = pd.DataFrame(actions)
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set('model_actions', model_actions)

app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    print('Clearing the cache...')
    cache.clear()
    return jsonify({"status": "Cache cleared successfully"})

@app.route('/api/run-model', methods=['POST'])
def run_model():
    seed = 20
    today_utc = dt.datetime.now(dt.timezone.utc)
    formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')
    
    flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

    if 'Date' in historical_port_values.columns and not historical_port_values['Date'].empty:
        start_date = pd.to_datetime(historical_port_values['Date'].min()).strftime('%Y-%m-%d %H:00:00')
    else:
        start_date = formatted_today_utc

    prices_query = live_prices(start_date)
    prices_df = flipside_api_results(prices_query, flipside_api_key)
    prices_df = data_cleaning(prices_df)
    prices_df = prepare_data_for_simulation(prices_df, start_date, formatted_today_utc)
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

    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        action, _states = model.predict(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        if done:
            print("Episode done. Exiting the loop.")
            break
        
        if next_state is not None:
            states.append(next_state.flatten())
        rewards.append(reward)
        actions.append(action.flatten())
        portfolio_values.append(env.get_portfolio_value())
        compositions.append(env.portfolio)
        
        state = next_state

    states_df = env.get_states_df()
    rewards_df = env.get_rewards_df()
    actions_df = env.get_actions_df()
    portfolio_values_df = env.get_portfolio_values_df()
    composition = env.get_portfolio_composition_df()

    update_portfolio_data(portfolio_values_df)

    portfolio_values_df.set_index('Date', inplace=True)
    portfolio_return = calculate_cumulative_return(portfolio_values_df, 'Portfolio_Value')

    portfolio_values_df.index = pd.to_datetime(portfolio_values_df.index)

    norm_prices = normalize_asset_returns(prices_df, portfolio_values_df.index.min(), portfolio_values_df.index.max())
    norm_prices.set_index('ds', inplace=True)

    eth_return = calculate_cumulative_return(norm_prices, 'normalized_ETH')
    btc_return = calculate_cumulative_return(norm_prices, 'normalized_BTC')

    excess_return_btc = portfolio_return - btc_return
    excess_return_eth = portfolio_return - eth_return

    actions_df.rename(columns={"Action_0": "ETH", "Action_1": "BTC"}, inplace=True)
    actions_df.to_csv('data/actions.csv')

    actions_df[['ETH','BTC']] = actions_df[['ETH','BTC']] * 10000
    actions_df[['ETH','BTC']] = actions_df[['ETH','BTC']].astype(int)

    resp = jsonify((actions_df[['ETH','BTC']].iloc[-1].values).tolist())

    cache.set('historical_port_values', historical_port_values)

    return resp

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    print('Starting Flask app...')
    app.run(debug=True, use_reloader=False, port=5000)
    logger.info("Flask app has stopped.")
    print('Flask app has stopped.')
