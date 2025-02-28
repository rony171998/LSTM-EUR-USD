import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("eur_usd_historico.csv", header=0, index_col="Timestamp", parse_dates=True)

# Filtrar solo la columna 'Close' (precio de cierre)
df = df[['Close']]

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Crear secuencias para la LSTM
def create_sequences(data, seq_length=50):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Definir longitud de secuencia
seq_length = 50
X, y = create_sequences(df_scaled, seq_length)

# Dividir en train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Cargar modelo guardado
model = keras.models.load_model("lstm_eurusd_model.h5")

# Hacer predicciones
y_pred = model.predict(X_test)

# Desnormalizar los datos
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Graficar resultados
plt.figure(figsize=(12,6))
plt.plot(y_test, label="Real")
plt.plot(y_pred, label="Predicho")
plt.legend()
plt.title("Predicción EUR/USD con LSTM")
plt.show()