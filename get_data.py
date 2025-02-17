# datos yahoo finenace
import yfinance as yf

# Descargar datos históricos del EUR/USD
df = yf.download("EURUSD=X", start="2024-01-01", end="2025-02-14", interval="1h")

# Guardar en un CSV para análisis
df.to_csv("eur_usd_historico.csv")

print(df.head())