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