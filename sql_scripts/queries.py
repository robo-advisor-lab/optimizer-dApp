prices = """

select
  hour,
  symbol,
  price
from
  ethereum.price.ez_prices_hourly
where
  symbol in('WBTC', 'WETH')
order by
  hour desc


"""

volume = """

select
  dt,
  sum(volume) as volume
from
  (
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      ethereum.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      arbitrum.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      optimism.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      base.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      polygon.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(SWAP_FROM_AMOUNT_USD) as volume
    from
      solana.defi.ez_dex_swaps
    where
      SWAP_FROM_AMOUNT_USD is not null
    group by
      1
    union
    all
    select
      date_trunc('hour', block_timestamp) as dt,
      sum(AMOUNT_IN_USD) as volume
    from
      avalanche.defi.ez_dex_swaps
    where
      AMOUNT_IN_USD is not null
    group by
      1
  )
group by
  1
order by
  dt desc


"""