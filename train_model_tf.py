import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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

# Construir la LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar la LSTM
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo
model.save("lstm_eurusd_model.h5")