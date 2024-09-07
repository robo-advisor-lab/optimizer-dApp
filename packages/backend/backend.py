import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objs as go
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib
import random

import datetime as dt

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

from python_scripts.utils import set_random_seed, calculate_cumulative_return, calculate_cagr, calculate_beta, fetch_and_process_tbill_data, flipside_api_results
from python_scripts.data_processing import data_processing
from python_scripts.model import Portfolio
from sql_scripts.queries import live_prices

from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import logging
from diskcache import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

price_cols, temporals, filtered_combined = data_processing(api=False) # need to pass the new sql query for testing?  (not historical)

cache = Cache('cache_dir')
# historical_data = pd.DataFrame()
# historical_port_values = pd.DataFrame()
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
    new_data = pd.DataFrame([values])
    historical_port_values = pd.concat([historical_port_values, new_data]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set('historical_port_values', historical_port_values)

def update_model_actions(actions):
    global model_actions
    print(f'model actions before update: {model_actions}')
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    print(f'model actions after update: {model_actions}')
    cache.set('model_actions', model_actions)  

today = dt.datetime.today().strftime('%Y-%m-%d %H:00:00')
print(f'today: {today}')

flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

scheduler = BackgroundScheduler()

if 'date' in historical_data.columns and not historical_data['date'].empty:
    start_date = pd.to_datetime(historical_data['date'].min()).strftime('%Y-%m-%d %H:00:00')
else:
    start_date = today.strftime('%Y-%m-%d %H:00:00')
print(f'histortical data {historical_data}')
print(f'sql start date: {start_date}')

prices_query = live_prices(start_date)
prices_df = flipside_api_results(prices_query, flipside_api_key)



