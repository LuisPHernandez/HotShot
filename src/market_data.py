from binance.client import Client
import pandas as pd
import os

api_key = os.getenv('67SazbbKRRx4xCl1QC52Spx6ym3DTger3Xh9CGH5a2i9uSb2owaEDj607uiYS7qM')
api_secret = os.getenv('Rxc7IGd141JO9UZcFHQyzJVntmnMpRnyMhwOJMXufW9SvwqXOVhNDQaeuuYBz7M8')

client = Client(api_key, api_secret)

def fetch_ohlcv(symbol='BTCUSDT', interval='1d', limit=500):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]