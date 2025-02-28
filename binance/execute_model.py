import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 📌 Configurar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Cargar los datos
df = pd.read_csv("binance/eur_usd_historico_binance.csv", header=0, index_col="Timestamp", parse_dates=True)

# 📌 Filtrar solo la columna 'Close' (precio de cierre)
df = df[['Close']]

# 📌 Normalizar los datos
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# 📌 Crear secuencias para la LSTM
def create_sequences(data, seq_length=50):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# 📌 Definir longitud de secuencia
seq_length = 50
X, y = create_sequences(df_scaled, seq_length)

# 📌 Convertir a tensores de PyTorch
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 📌 Dividir en train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 📌 Definir el modelo LSTM (debe coincidir con el que guardaste)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Tomar la última salida de la secuencia

# 📌 Cargar modelo guardado
model = LSTMModel().to(device)  # Instanciar modelo
model.load_state_dict(torch.load("binance/lstm_eurusd_model_pytorch.pth", map_location=device))
model.eval()  # Poner en modo evaluación

# 📌 Hacer predicciones
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()  # Mover a CPU para convertir a NumPy

# 📌 Desnormalizar los datos
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.cpu().numpy())

# 📌 Graficar resultados
plt.figure(figsize=(12,6))
plt.plot(y_test, label="Real", color="blue")
plt.plot(y_pred, label="Predicho", color="red")
plt.legend()
plt.title("Predicción EUR/USD con LSTM (PyTorch)")
plt.show()
