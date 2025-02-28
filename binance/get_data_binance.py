import ccxt
import pandas as pd

exchange = ccxt.binance()
symbol = 'EURUSDT'
timeframe = '1d'

ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

df.to_csv("binance/eur_usd_historico_binance.csv", index=False)
print(df.head())
