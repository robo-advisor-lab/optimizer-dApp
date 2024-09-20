import pandas as pd
from flipside import Flipside
import random
import numpy as np
import torch 
import tensorflow
from sklearn.linear_model import LinearRegression
import requests
import os

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
    print('cleaning prices')
    # Pivot the dataframe
    prices_df = prices_df.drop_duplicates(subset=['hour', 'symbol'])
    prices_df_pivot = prices_df.pivot(
        index='hour',
        columns='symbol',
        values='price'
    )
    prices_df_pivot = prices_df_pivot.reset_index()

    # Rename the columns by combining 'symbol' with a suffix
    prices_df_pivot.columns = ['hour'] + [f'{col}_Price' for col in prices_df_pivot.columns[1:]]
    
    print(f'cleaned prices: {prices_df_pivot}')
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
    print(f'cagr history: {history}')
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

def calculate_beta(data, columnx, columny):
    X = data[f'{columnx}'].pct_change().dropna().values.reshape(-1, 1)  
    Y = data[f'{columny}'].pct_change().dropna().values
  
    # Check if X and Y are not empty
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Input arrays X and Y must have at least one sample each.")
        return 0

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Output the beta
    beta = model.coef_[0]
    return beta

def fetch_and_process_tbill_data(api_url, api_key, data_key, date_column, value_column, date_format='datetime'):
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
    
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

def normalize_asset_returns(price_timeseries, start_date, end_date, normalize_value=1e4):
   # Ensure that the price_timeseries index is timezone-naive
    # price_timeseries.index = price_timeseries.index.tz_localize(None)
    
    # Ensure that start_date and end_date are timezone-naive
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)

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

def prepare_data_for_simulation(price_timeseries, start_date, end_date):
    """
    Ensure price_timeseries has entries for start_date and end_date.
    If not, fill in these dates using the last available data.
    """
    # Ensure 'ds' is in datetime format
    # price_timeseries['hour'] = pd.to_datetime(price_timeseries['hour'])
    
    # Set the index to 'ds' for easier manipulation
    # if price_timeseries.index.name != 'hour':
    #     price_timeseries.set_index('hour', inplace=True)

    print(f'price index: {price_timeseries.index}')

    price_timeseries.index = price_timeseries.index.tz_localize(None)
    
    # Check if start_date and end_date exist in the data
    required_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    all_dates = price_timeseries.index.union(required_dates)
    
    # Reindex the dataframe to ensure all dates from start to end are present
    price_timeseries = price_timeseries.reindex(all_dates)
    
    # Forward fill to handle NaN values if any dates were missing
    price_timeseries.fillna(method='ffill', inplace=True)

    # Reset index if necessary or keep the datetime index based on further needs
    price_timeseries.reset_index(inplace=True, drop=False)
    price_timeseries.rename(columns={'index': 'hour'}, inplace=True)
    
    return price_timeseries

