import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Implementar limpieza y preparación de datos
    date_columns = ['funding_open_timestamp', 'term_start_timestamp', 'term_end_timestamp']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calcular la duración del préstamo en días
    df['loan_duration_days'] = (df['term_end_timestamp'] - df['term_start_timestamp']).dt.days

    # Calcular el retorno normalizado acumulativo
    df['normalized_cumulative_return'] = (df['principal_paid_dollar'] + df['interest_paid_dollar']) / df['funded_assets_dollar']
    
    # Estimar parámetros
    df['mu'], df['sigma'] = zip(*df.apply(estimate_parameters, axis=1))
    
    return df

def estimate_parameters(loan):
    mu = loan['base_interest_rate'] / 100  # Drift
    duration = max(loan['loan_duration_days'], 1) / 365
    
    if loan['normalized_cumulative_return'] > 0:
        log_return = np.log(loan['normalized_cumulative_return'])
        sigma = abs(log_return) / np.sqrt(duration)
    else:
        sigma = 0.1  # Valor predeterminado si no podemos calcular

    return mu, sigma
