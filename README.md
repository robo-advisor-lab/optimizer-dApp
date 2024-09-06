# Tokenomics

**Agents:**

- **DAO:** The Decentralized Autonomous Organization governing the fund and executing the issuance of new instruments.
- **Investors:** Individuals or organizations buying and selling tokens of the various fund investment vehicles.
- **Market Maker (MM):** Entity providing liquidity in the secondary market.
- **Smart Contract (SC):** The smart contract governing the issuance, exchange, and rebalancing of fund tokens.
- **Machine Learning Model (MLM):** Flask server trained with on-chain dataset to calculate optimal proportions based on BTC/USDC, ETH/USDC rates.

**Nomenclature:**

```
The terms instrument and investment vehicle are used interchangeably.
```

- `t`: Moment in time measured in Coordinated Universal Time Timestamp.
- `t'`: Ethereum block (block time unit of measurement).
- $S_i(t)$: Price of underlying asset `i` at block `t'` (e.g., WETH, BTC).
- $V(t)$: Net Asset Value (NAV) of the instrument at block `t'`.
- $N$: Number of instrument tokens.
- $P(t)$: Price of the fund token in the secondary market at period `t`.
- $w_i(t)$: Weight of asset `i` in the fund's portfolio at period `t`.
- $c$: Transaction cost (proportional to the transaction value).

**Algorithmic Mechanics:**

1. **Initial Issuance (t=0):**
    - The DAO determines initial weights $w_i(0)$ for each asset `i`.
    - Investors deposit USDC into the SC.
    - The SC uses USDC to purchase underlying assets in proportions $w_i(0)$.
    - The SC issues $N(0)$ tokens, where $N(0) = V(0) / P(0)$, and $P(0)$ is a predefined issuance price.
    - Tokens are distributed to investors in proportion to their USDC contribution.

2. **Rebalancing (t > 0):**
    - **Every $T_{rebalancing}$ or when a rebalancing criterion is met (e.g., deviation from target VaR) or predictably (daily):**
        - The SC obtains current prices $S_i(t)$ from oracles.
        - The SC calculates the current NAV: $V(t) = \sum w_i(t-1) \times S_i(t) \times (1 - c)$.
        - The SC uses the oracle model to consume optimal weights by executing a serverless function of the ML model to determine new optimal weights $w_i^*(t)$.
        - The SC sends orders to the MM to buy/sell assets in the secondary market to adjust weights to $w_i^*(t)$, incurring transaction costs $c$.

3. **Token Liquidation:**
    - **Underlying Liquidation:** If t' > T', the SC liquidates its token portfolio in the secondary market to acquire USDC and disperses (disperse.app) the USDC/N proportion or acts as an exchange house, depending on which is more convenient based on transaction costs.

**Market Interaction:**

- **Primary Market (SC):**
    - Initial token issuance and exchange/destruction of tokens at liquidation time.
- **Secondary Market (Exchange):**
    - Investors buy and sell tokens among themselves.
    - The MM provides liquidity and facilitates price discovery.
    - The SC interacts with the MM to execute rebalancing orders.

**Rebalancing Example:**

If the ML model predicts ETH will outperform BTC, the SC might send an order to the MM to sell a portion of BTC and buy more ETH, adjusting the weights $w_{WETH}(t)$ and $w_{BTC}(t)$ accordingly.

**Additional Considerations:**

- **Governance:** The DAO can adjust the rebalancing algorithm parameters and token creation/destruction criteria through voting.
- **Fees:** The SC may charge fees for fund management and rebalancing, which will be deducted from the NAV.
