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
  date_trunc('hour', block_timestamp) as dt,
  coalesce(sum(AMOUNT_IN_USD), 0) as volume
from
  crosschain.defi.ez_dex_swaps
group by
  date_trunc('hour', block_timestamp)
order by
  1 asc


"""