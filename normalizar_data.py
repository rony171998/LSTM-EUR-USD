import pandas as pd

# Cargar los datos con el encabezado correcto
df = pd.read_csv("eur_usd_historico.csv", skiprows=2, header=0, index_col="Datetime", parse_dates=True)

# Eliminar valores nulos
df.dropna(inplace=True)

# Filtrar solo la columna 'Close' (precio de cierre)
df = df[['Close']]

print(df.info())  # Verificar datos limpios