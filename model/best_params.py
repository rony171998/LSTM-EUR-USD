import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import optuna
import json
from datetime import datetime

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 1. Clase del Modelo TLS-LSTM (optimizada para tuning)
class TLS_LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, output_size=1, dropout_prob=0.2):
        super(TLS_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        return self.fc(lstm2_out[:, -1, :])

# 2. Función para cargar y preparar datos (simplificada)
def load_and_prepare_data(filepath, target_column):
    df = pd.read_csv(
        filepath,
        index_col="Fecha",
        parse_dates=True,
        dayfirst=True,
        decimal=",",
        thousands=".",
        converters={
            "Último": lambda x: float(str(x).replace(".", "").replace(",", "."))
        }
    )
    df = df.sort_index().dropna(subset=[target_column])
    return df[[target_column]]

# 3. Función para crear secuencias
def create_sequences(data, seq_length, forecast_horizon):
    sequences, labels = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(sequences), np.array(labels)

# 4. Función de entrenamiento para Optuna
def train_for_optuna(params, X_train, y_train, X_val, y_val):
    model = TLS_LSTMModel(
        input_size=1,
        hidden_size=params['hidden_size'],
        output_size=FORECAST_HORIZON,
        dropout_prob=params['dropout_prob']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    # Entrenamiento rápido (solo para tuning)
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X.to(device))
            loss = criterion(outputs, batch_y.to(device))
            loss.backward()
            optimizer.step()
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_loss = criterion(val_outputs, y_val.to(device))
    
    return val_loss.item()

# 5. Función objetivo para Optuna
def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'seq_length': trial.suggest_categorical('seq_length', [30, 60, 90, 120]),
    }
    
    # Recrear secuencias si es necesario
    if params['seq_length'] != SEQ_LENGTH:
        X_train_seq, y_train_seq = create_sequences(train_scaled, params['seq_length'], FORECAST_HORIZON)
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).squeeze(-1)
    else:
        X_train_t, y_train_t = X_train, y_train
    
    val_loss = train_for_optuna(params, X_train_t, y_train_t, X_val, y_val)
    return val_loss

# Configuración principal
FILEPATH = "EUR_USD_2010-2024.csv"
TARGET_COLUMN = "Último"
FORECAST_HORIZON = 1
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% val (para tuning)
SEQ_LENGTH = 60  # Valor inicial

if __name__ == "__main__":
    print("\n=== Iniciando Optimización de Hiperparámetros ===")
    
    # Cargar y preparar datos
    df = load_and_prepare_data(FILEPATH, TARGET_COLUMN)
    split_index = int(len(df) * TRAIN_SPLIT_RATIO)
    train_data = df.iloc[:split_index]
    val_data = df.iloc[split_index:]
    
    # Escalar datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    
    # Crear secuencias iniciales
    X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH, FORECAST_HORIZON)
    X_val, y_val = create_sequences(val_scaled, SEQ_LENGTH, FORECAST_HORIZON)
    
    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).squeeze(-1)
    
    # Configurar estudio de Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Guardar resultados
    best_params = study.best_params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"best_params_{timestamp}_{FILEPATH}.json"
    
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\n=== Resultados de la Optimización ===")
    print(f"Mejores parámetros encontrados:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    print(f"\nParámetros guardados en: {output_file}")
    print("=== Proceso completado ===")