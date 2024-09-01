import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('rwa_private_loans.csv')

# Convertir las columnas de fecha a datetime
date_columns = ['funding_open_timestamp', 'term_start_timestamp', 'term_end_timestamp']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calcular la duración del préstamo en días
df['loan_duration_days'] = (df['term_end_timestamp'] - df['term_start_timestamp']).dt.days

# Calcular el retorno normalizado acumulativo
df['normalized_cumulative_return'] = (df['principal_paid_dollar'] + df['interest_paid_dollar']) / df['funded_assets_dollar']

# Función para estimar parámetros de manera más robusta
def estimate_parameters(loan):
    mu = loan['base_interest_rate'] / 100  # Drift
    
    # Usar un valor mínimo para evitar divisiones por cero
    duration = max(loan['loan_duration_days'], 1) / 365
    
    # Usar un método más robusto para estimar la volatilidad
    if loan['normalized_cumulative_return'] > 0:
        log_return = np.log(loan['normalized_cumulative_return'])
        sigma = abs(log_return) / np.sqrt(duration)
    else:
        sigma = 0.1  # Valor predeterminado si no podemos calcular

    return mu, sigma

# Aplicar la estimación de parámetros a cada préstamo
df['mu'], df['sigma'] = zip(*df.apply(estimate_parameters, axis=1))

# Filtrar préstamos con duración válida y parámetros finitos
df_valid = df[(df['loan_duration_days'] > 0) & (df['mu'].notna()) & (df['sigma'].notna()) & np.isfinite(df['mu']) & np.isfinite(df['sigma'])]

def simulate_loan(S0, mu, sigma, T, dt, M):
    """
    Simula M trayectorias de un préstamo usando GBM.
    """
    steps = max(int(T / dt), 1)  # Asegurar al menos un paso
    S = np.zeros((M, steps + 1))
    S[:, 0] = S0
    
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(M)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return S

# Ejemplo de uso para un préstamo
if not df_valid.empty:
    loan = df_valid.iloc[0]
    S0 = loan['funded_assets_dollar']
    mu = loan['mu']
    sigma = loan['sigma']
    T = max(loan['loan_duration_days'] / 365, 1/365)  # Al menos un día
    dt = 1/365  # paso diario
    M = 1000  # número de simulaciones

    trajectories = simulate_loan(S0, mu, sigma, T, dt, M)

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.plot(trajectories.T, alpha=0.1)
    plt.title(f"Simulaciones de valor del préstamo {loan['loan_id']}")
    plt.xlabel("Días")
    plt.ylabel("Valor del préstamo")
    plt.show()
else:
    print("No hay préstamos válidos para simular.")
    
# def backtest_strategy(df, train_size=0.8):
#     train = df.iloc[:int(len(df) * train_size)]
#     test = df.iloc[int(len(df) * train_size):]

#     # Entrenar el modelo y optimizar el portafolio con los datos de entrenamiento
#     train_returns = pd.DataFrame({loan['loan_id']: simulate_loan(loan['funded_assets_dollar'], loan['mu'], loan['sigma'], loan['loan_duration_days']/365, 1/365, 1000)[:, -1] / loan['funded_assets_dollar'] - 1 for loan in train.itertuples()})
#     optimal_weights = optimize_portfolio(train_returns)

#     # Evaluar el rendimiento en el conjunto de prueba
#     test_returns = pd.DataFrame({loan['loan_id']: simulate_loan(loan['funded_assets_dollar'], loan['mu'], loan['sigma'], loan['loan_duration_days']/365, 1/365, 1000)[:, -1] / loan['funded_assets_dollar'] - 1 for loan in test.itertuples()})
#     portfolio_return = np.dot(optimal_weights, test_returns.mean())
#     portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(test_returns.cov(), optimal_weights)))
#     sharpe_ratio = portfolio_return / portfolio_volatility

#     return {
#         'portfolio_return': portfolio_return,
#         'portfolio_volatility': portfolio_volatility,
#         'sharpe_ratio': sharpe_ratio
#     }
