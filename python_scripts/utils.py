import pandas as pd
from flipside import Flipside
import random
import numpy as np
import torch 
import tensorflow

def flipside_api_results(query, api_key):
  
  flipside_api_key = api_key
  flipside = Flipside(flipside_api_key, "https://api-v2.flipsidecrypto.xyz")

  query_result_set = flipside.query(query)
  # what page are we starting on?
  current_page_number = 1

  # How many records do we want to return in the page?
  page_size = 1000

  # set total pages to 1 higher than the `current_page_number` until
  # we receive the total pages from `get_query_results` given the 
  # provided `page_size` (total_pages is dynamically determined by the API 
  # based on the `page_size` you provide)

  total_pages = 2


  # we'll store all the page results in `all_rows`
  all_rows = []

  while current_page_number <= total_pages:
    results = flipside.get_query_results(
      query_result_set.query_id,
      page_number=current_page_number,
      page_size=page_size
    )

    total_pages = results.page.totalPages
    if results.records:
        all_rows = all_rows + results.records
    
    current_page_number += 1

  return pd.DataFrame(all_rows)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensorflow.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_time(df):
    time_cols = ['date','dt','hour','time','day','month','year','week','timestamp','date(utc)','block_timestamp']
    for col in df.columns:
        if col.lower() in time_cols and col.lower() != 'timestamp':
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
        elif col.lower() == 'timestamp':
            df[col] = pd.to_datetime(df[col], unit='ms')
            df.set_index(col, inplace=True)
    print(df.index)
    return df 

def clean_prices(prices_df):
    prices_df_pivot = prices_df.pivot(
        index='hour',
        columns='symbol',
        values='price'
        )
    prices_df_pivot = prices_df_pivot.reset_index()
    prices_df_pivot.columns = [f'{col[0]}_{col[1]}' for col in prices_df_pivot.columns]
    prices_df_pivot.rename(columns={"h_o":"dt","W_B":"BTC Price","W_E":"ETH Price"}, inplace=True)

    return prices_df_pivot

def calculate_cumulative_return(portfolio_values_df, col):
    """
    Calculate the cumulative return of the portfolio.
    
    Parameters:
    portfolio_values_df (pd.DataFrame): DataFrame with 'Portfolio_Value' column
    
    Returns:
    float: Cumulative return of the portfolio
    """
    initial_value = portfolio_values_df[col].iloc[0]
    final_value = portfolio_values_df[col].iloc[-1]
    cumulative_return = (final_value / initial_value) - 1
    return cumulative_return

def calculate_cagr(history):
    #print(f'cagr history {history}')
    initial_value = history.iloc[0]
    #print(f'cagr initial value {initial_value}')
    final_value = history.iloc[-1]
    #print(f'cagr final value {final_value}')
    number_of_hours = (history.index[-1] - history.index[0]).total_seconds() / 3600
    #print(f'cagr number of hours {number_of_hours}')
    number_of_years = number_of_hours / (365.25 * 24)  # Convert hours to years
    #print(f'cagr number of years {number_of_years}')

    if number_of_years == 0:
        return 0

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr