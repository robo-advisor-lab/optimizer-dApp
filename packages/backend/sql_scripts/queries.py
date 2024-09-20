def lst_portfolio_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0xae78736cd615f374d3085123a210448e74fc6393')),
        (LOWER('0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0')),
        (LOWER('0xac3e018457b222d93114458476f3e3416abbe38f')),
        (LOWER('0xd5F7838F5C461fefF7FE49ea5ebaF7728bB0ADfa')),--meth,
        (LOWER('0xbf5495Efe5DB9ce00f80364C8B423567e58d2110')),--ezeth,
        (LOWER('0xDa7Fa7248F62e051ccA4Af2522439A61d3976462')),--msol,
        (LOWER('0xA1290d69c65A6Fe4DF752f95823fae25cB99e5A7')),--rseth,
        (LOWER('0xBe9895146f7AF43049ca1c1AE358B0541Ea49704')),--cbeth,
        (LOWER('0xf1C9acDc66974dFB6dEcB12aA385b9cD01190E38')),--oseth,
        (LOWER('0x8236a87084f8B84306f72007F36F2618A5634494')),--lbtc
        (LOWER('0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee'))--weeth

    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def eth_btc_prices(today):

  beginning = f"'{today}'"
  print('beginning', beginning)
  
  prices_query =f"""

WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599')),
        (LOWER('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'))
    ) AS tokens(column1)
)

select hour,
       symbol,
       price
from ethereum.price.ez_prices_hourly
where token_address in (select token_address from addresses)
and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
order by hour desc, symbol 


"""
  return prices_query

def dao_advisor_portfolio(today):
    beginning = f"'{today}'"
    print('beginning', beginning)
    
    prices_query =f"""

  WITH addresses AS (
    SELECT column1 AS token_address 
    FROM (VALUES
        (LOWER('0x1494CA1F11D487c2bBe4543E90080AeBa4BA3C2b')),--dpi,
        (LOWER('0x45804880De22913dAFE09f4980848ECE6EcbAf78')),--paxg,
        (LOWER('0xdab396cCF3d84Cf2D07C4454e10C8A6F5b008D2b')),--gfi,
        (LOWER('0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984')),--UNI
        (LOWER('0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2')),--MKR
        (LOWER('0xc221b7E65FfC80DE234bbB6667aBDd46593D34F0')),--CFG
        (LOWER('0xD33526068D116cE69F19A9ee46F0bd304F21A51f')),--RPL
        (LOWER('0x320623b8E4fF03373931769A31Fc52A4E78B5d70')),--RSR
        (LOWER('0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9')),--AAVE
        (LOWER('0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0')),--FRAX
        (LOWER('0xB50721BCf8d664c30412Cfbc6cf7a15145234ad1')),--ARB
        (LOWER('0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72')),--ENS
        (LOWER('0xba100000625a3754423978a60c9317c58a424e3D')),--BAL
        (LOWER('0x4F9254C83EB525f9FCf346490bbb3ed28a81C667')),--CELR
        (LOWER('0xc5102fE9359FD9a28f877a67E36B0F050d81a3CC')),--HOP
        (LOWER('0x33349B282065b0284d756F0577FB39c158F935e6')),--MPL
        (LOWER('0xAf5191B0De278C7286d6C7CC6ab6BB8A73bA2Cd6')),--STG
        (LOWER('0x408e41876cCCDC0F92210600ef50372656052a38')),--REN
        (LOWER('0x83F20F44975D03b1b09e64809B757c47f942BEeA')),--SDAI
        (LOWER('0xFC4B8ED459e00e5400be803A9BB3954234FD50e3')),--aWBTC
        (LOWER('0xccf4429db6322d5c611ee964527d42e5d685dd6a')),--CWBTC
        (LOWER('0x467719ad09025fcc6cf6f8311755809d45a5e5f3')),--AXL
        (LOWER('0x44108f0223a3c3028f5fe7aec7f9bb2e66bef82f')),--ACX
        (LOWER('0x0f2d719407fdbeff09d87557abb7232601fd9f29')),--SYN
        (LOWER('0xa11bd36801d8fa4448f0ac4ea7a62e3634ce8c7c'))--ABR

    ) AS tokens(column1)
)

  select hour,
        symbol,
        price
  from ethereum.price.ez_prices_hourly
  where token_address in (select token_address from addresses)
  and hour >= date_trunc('hour', to_timestamp({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
  order by hour desc, symbol 


"""
    return prices_query
   

