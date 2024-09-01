import pandas as pd
from PLP.utils import preprocess, simulation, optimization, visualization, risk_management

def main():
    # Cargar los datos
    df = preprocess.load_data('ETHonline24/PLP/data/rwa_private_loans.csv')

    # Preprocesar los datos
    df = preprocess.preprocess_data(df)

    # Simular y construir el portafolio
    returns = simulation.simulate_returns(df)
    portfolio = optimization.optimize_portfolio(returns)

    # Backtesting
    backtest_results = simulation.backtest_strategy(df)

    # Análisis de riesgo
    greeks = risk_management.calculate_greeks(portfolio)
    var = risk_management.var_calculation(portfolio['returns'])

    # Visualizaciones
    visualization.plot_efficient_frontier(returns)
    
    print("Resultados del modelo:")
    print(backtest_results)
    print("VaR:", var)
    print("Greeks:", greeks)

if __name__ == "__main__":
    main()
