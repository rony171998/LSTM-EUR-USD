import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo
model.save("lstm_eurusd_model.h5")

# Cargar modelo guardado
model = keras.models.load_model("lstm_eurusd_model.h5")

# Hacer predicciones
y_pred = model.predict(X_test)

# Desnormalizar los datos
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Graficar resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test, label="Real")
plt.plot(y_pred, label="Predicho")
plt.legend()
plt.title("Predicción EUR/USD con LSTM")
plt.show()

