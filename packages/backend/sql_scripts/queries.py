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

def live_prices(today):
    beginning = today 
    print('beginning', beginning)
    prices_query = f"""
    select
      hour,
      symbol,
      price
    from
      ethereum.price.ez_prices_hourly
    where
      symbol in('WBTC', 'WETH')
    and hour >= date('{beginning}')
    order by
      hour desc
    """
    return prices_query



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