# RWA Financial Model

# Construction of Self-Financing Replicating Portfolio for Valuation of Tokenized Debt Obligations

## **Black-Scholes-Merton Model**

The asset pricing theory that uses Wiener processes and stochastic differential equations (SDEs) to model the returns of financial assets is the **Black-Scholes-Merton model**.

### Wiener Process and SDEs in Finance

- Diffusion processes, such as Brownian motion or Wiener process, play a prominent role in the mathematical theory of finance and are key to deriving the Black-Scholes equation for determining the price of certain financial assets.

### Black-Scholes Model

- The Black-Scholes model uses a Wiener process to model the dynamics of the underlying asset price. This leads to a partial differential equation whose solution gives the price of financial options.

### Implementation of GBM using the Euler-Maruyama Scheme

The Geometric Brownian Motion (GBM) we are using to model the value of loans is numerically implemented using the Euler-Maruyama scheme. This is a numerical approximation method for stochastic differential equations (SDEs).

The stochastic differential equation for GBM is:

$dS_t = \mu S_t dt + \sigma S_t dW_t$

Where:
- $S_t$ is the loan value at time t
- $\mu$ is the drift (expected rate of return)
- $\sigma$ is the volatility
- $W_t$ is a standard Wiener process

The Euler-Maruyama scheme for this SDE is expressed as:

$S_{t+\Delta t} = S_t + \mu S_t \Delta t + \sigma S_t \sqrt{\Delta t} Z_t$

Where:
- $\Delta t$ is the time step
- $Z_t \sim N(0,1)$ is a standard normal random variable

In practice, for simulation, we use the logarithmic form to ensure positivity and improve numerical stability:

$S_{t+\Delta t} = S_t \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$

**This implementation of the Euler-Maruyama scheme for GBM is what we will use in our model to simulate the trajectories of loan values over time.**

**In the context of on-chain implementation, this precise formulation of the Euler-Maruyama scheme could be directly coded into a smart contract or, more likely, implemented in an off-chain oracle that then feeds the results to the smart contract for decision-making.**

## Important characteristics of this implementation:

1. Time discretization: We divide the total period into discrete steps $\Delta t$. Loan payments and interest accrual typically occur at discrete intervals. A discrete-time model directly reflects this reality.
2. Numerical approximation: It is a first-order approximation of the exact solution.
3. Convergence: The convergence is of strong order 0.5 and weak order 1.0.
4. Positivity preservation: The exponential form ensures that simulated values are always positive.

---

# Modeling Self-Financing Replicating Portfolios Using DeFi Loan Data

## 1. Introduction

This technical note describes an approach to model self-financing replicating portfolios using loan data from DeFi platforms. The objective is to develop a model that allows the creation of sophisticated investment strategies and risk management in the context of decentralized finance.

## 2. Input Data

The main data comes from the `cleaned_loans.csv` file, which contains the following relevant information:

- `loan_id`: Unique loan identifier
- `base_interest_rate`: Base interest rate (in %)
- `loan_duration_days`: Loan duration in days
- `compounding_periods`: Number of compounding periods
- `normalized_cumulative_return`: Normalized cumulative return

## 3. Stochastic Process Modeling

We will assume that the value of each loan follows a Geometric Brownian Motion (GBM):

$dS = \mu S dt + \sigma S dW$

Where:
- $S$ is the loan value
- $\mu$ is the drift (estimated from `base_interest_rate`)
- $\sigma$ is the volatility (estimated from the variation in `normalized_cumulative_return`)
- $dW$ is a standard Wiener process

## 4. Parameter Estimation

### 4.1 Drift ($\mu$)

$\mu = \text{base\_interest\_rate} / 100$

### 4.2 Volatility ($\sigma$)

Volatility will be estimated using the maximum likelihood method on logarithmic returns derived from `normalized_cumulative_return`.

## 5. Replicating Portfolio Construction

We define a portfolio $\Pi$ as a linear combination of $n$ loans:

$\Pi = \sum_{i=1}^n w_i S_i$

Where $w_i$ are the weights of each loan in the portfolio.

## 6. Self-Financing Condition

For the portfolio to be self-financing, it must satisfy:

$d\Pi = \sum_{i=1}^n w_i dS_i$

## 7. Monte Carlo Simulation

We will implement a Monte Carlo simulation to estimate the future value of the portfolio:

1. Generate $M$ trajectories for each loan using the discretized GBM:

   $S_{i,t+1} = S_{i,t} \exp((\mu_i - \frac{1}{2}\sigma_i^2)\Delta t + \sigma_i \sqrt{\Delta t} Z_t)$

   Where $Z_t \sim N(0,1)$ and $\Delta t = \text{loan\_duration\_days} / 365$

2. Calculate the portfolio value for each trajectory and time.

## 8. Portfolio Optimization

We will use Markowitz's mean-variance optimization method to determine the optimal weights $w_i$:

$\max_{w} \quad w^T \mu - \frac{\lambda}{2} w^T \Sigma w$

Subject to: $\sum_{i=1}^n w_i = 1$, $w_i \geq 0$

Where:
- $\mu$ is the vector of estimated drifts
- $\Sigma$ is the covariance matrix of returns
- $\lambda$ is the risk aversion parameter

## 9. Sensitivity Analysis

We will calculate the following "Greeks" for the portfolio:

- Delta: $\Delta = \frac{\partial \Pi}{\partial S}$
- Gamma: $\Gamma = \frac{\partial^2 \Pi}{\partial S^2}$
- Theta: $\Theta = \frac{\partial \Pi}{\partial t}$

## 10. Backtesting

We will perform backtesting of the model using historical data:

1. Split the data into training and test sets.
2. Fit the model and optimize the portfolio using the training set.
3. Evaluate the performance of the optimized portfolio on the test set.
4. Calculate performance metrics such as the Sharpe ratio and maximum drawdown.

## 11. Implementation

The model will be implemented in Python, using the following libraries:

- `pandas` for data handling
- `numpy` for numerical calculations
- `scipy` for optimization
- `statsmodels` for parameter estimation
- `matplotlib` for visualization

## 12. Limitations and Future Considerations

- The model assumes normality and stationarity, which may not always be valid in DeFi markets.
- The model could be extended to include jumps and heavy tails in the return distribution.
- Liquidity and transaction costs in DeFi should be considered for practical implementation.
- **Accuracy vs. Simplicity:** While the Euler-Maruyama method is simple, more sophisticated discretization schemes (e.g., Milstein method) can improve accuracy.

This technical note provides a framework for modeling and optimizing self-financing replicating portfolios using DeFi loan data, laying the foundation for more sophisticated investment strategies and risk management in this emerging space.

---

# Model Results and On-Chain Actions

To determine specific buy or sell actions for on-chain assets, we must focus on the elements of the model that provide **actionable signals** and can be efficiently implemented in a blockchain environment. Here are the most relevant aspects:

1. Portfolio Optimization:
   - Markowitz's mean-variance optimization is crucial for determining the optimal weights of each asset in the portfolio.
   - These weights will directly translate into buy or sell orders to adjust current positions.

2. Sensitivity Analysis (Greeks):
   - Delta (Δ): Indicates how much the position in each asset should be adjusted to maintain the desired exposure.
   - Gamma (Γ): Helps anticipate when more frequent rebalancing will be needed.
   - Theta (Θ): Signals how the passage of time affects the portfolio value, which can trigger rebalancing actions.

3. Self-Financing Condition:
   - Ensures that buy/sell decisions are made without additional capital injection, only reallocating existing assets.

4. Monte Carlo Simulation:
   - Provides short-term risk estimates that can be used to dynamically adjust positions.

5. Parameter Estimation:
   - The estimated drift (μ) and volatility (σ) are fundamental for calculating return and risk expectations, directly influencing trading decisions.

6. Backtesting:
   - Backtesting results can be used to adjust and fine-tune trading strategies before implementing them on-chain.

To implement this on-chain, you would need:

1. A smart contract that:
   - Stores the model parameters (μ, σ, optimal weights).
   - Implements rebalancing logic based on deviations from optimal weights.
   - Has functions to execute buys/sells based on model calculations.

2. An oracle that:
   - Provides updated price and interest rate data.
   - Possibly executes complex calculations off-chain and sends results to the smart contract.

3. Governance mechanism to:
   - Adjust model parameters.
   - Approve significant changes in strategy.

4. Monitoring system that:
   - Detects when deviations from the optimal portfolio exceed certain thresholds.
   - Triggers rebalancing functions in the smart contract.

5. Integration with DeFi protocols:
   - To execute buy/sell transactions through DEXs or lending pools.

The typical flow would be:

1. The oracle updates market data.
2. The model recalculates optimal weights and risk metrics.
3. If deviations exceed thresholds, a rebalancing transaction is triggered.
4. The smart contract executes necessary buys/sells through integrated DeFi protocols.

This approach allows for dynamic on-chain portfolio management, based on the principles of the self-financing replicating portfolio model, adapted to the particularities of the blockchain and DeFi environment.
