# datos yahoo finenace
import yfinance as yf
import pandas as pd

# Descargar datos históricos del EUR/USD
df = yf.download("EURUSD=X", start="2024-01-01", end="2025-01-01", interval="1d")

# Guardar en un CSV para análisis
df.to_csv("eur_usd_historico.csv")

print(df.head())

# # Cargar los datos
# df = pd.read_csv("eur_usd_historico.csv", index_col="Date", parse_dates=True)

# # # Eliminar valores nulos
# df.dropna(inplace=True)

# # # Filtrar solo la columna 'Close' (precio de cierre)
# df = df[['Close']]

# print(df.info())  # Verificar datos limpios
