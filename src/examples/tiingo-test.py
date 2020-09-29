#pip3 install tiingo
import asyncio
import asyncpg
import orjson
from dateutil import parser
from datetime import datetime, timezone
from time import time
from tiingo import TiingoClient

"""
create table hist_prices (
    sym text,
    date timestamp with time zone,
    close double precision,
    high double precision,
    low  double precision,
    open double precision,
    volume bigint,
    adjClose double precision,
    adjHigh double precision,
    adjLow double precision,
    adjOpen double precision,
    adjVolume bigint,
    divCash double precision,
    splitFactor double precision);
"""
table_name = 'hist_prices'

async def db_write(ticker, quotes):
    pool = await asyncpg.create_pool(database='radixdb', user='postgres', port=5432)
    records = []
    for q in quotes:
        records.append((ticker, parser.parse(q['date']), q['close'], q['high'], q['low'],
                        q['open'], q['volume'], q['adjClose'], q['adjHigh'], q['adjLow'], q['adjOpen'], q['adjVolume'], q['divCash'], q['splitFactor']))
    #print(records)
    if len(records) > 0:
        async with pool.acquire() as connection:
            #async with connection.transaction():
            #await connection.set_builtin_type_codec('hstore', codec_name='pg_contrib.hstore')
            result = await connection.copy_records_to_table(table_name, records=records)

def insert_db(sym, prices):
    asyncio.get_event_loop().run_until_complete(db_write(sym, prices))

config = {}
config['session'] = True
config['api_key'] = "0113a6c956d63b6fa48c95679627c423408159b6"
client = TiingoClient(config)

syms = ['TLT', 'TSLA', 'MSFT']

for sym in syms:
    print("get {0} metadata".format(sym))
    ticker_metadata = client.get_ticker_metadata(sym)
    print("get {0} daily hist price".format(sym))
    historical_prices = client.get_ticker_price(sym,
                                                fmt='json',
                                                startDate='2020-04-20',
                                                endDate='2020-04-23',
                                                frequency='daily')
    print("insert {0} daily hist price into db".format(sym))
    insert_db(sym, historical_prices)
