# predecir_future.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from datetime import timedelta
from hurst import compute_Hc
from statsmodels.tsa.stattools import adfuller

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Carga y preparación de datos con análisis de Hurst
def load_and_prepare_data(filepath):
    # Cargar datos manteniendo el orden cronológico correcto
    df = pd.read_csv(
        filepath,
        index_col="Fecha",
        parse_dates=True,
        dayfirst=True,  # Interpretar fechas como día/mes/año
        decimal=",",    # Usar coma como separador decimal
        thousands=".",  # Punto como separador de miles
        converters={
            "Último": lambda x: float(x.replace(".", "").replace(",", ".")),
            "Apertura": lambda x: float(x.replace(".", "").replace(",", ".")),
            "Máximo": lambda x: float(x.replace(".", "").replace(",", ".")),
            "Mínimo": lambda x: float(x.replace(".", "").replace(",", ".")),
            "% var.": lambda x: float(x.replace("%", "").replace(",", ".")) / 100
        }
    )
    df = df.sort_index(ascending=True)
    df = df.dropna(subset=["Último"])
    
    # Calcular Hurst Exponent para validar predictibilidad
    H, c, data = compute_Hc(df["Último"].values, kind='price')
    print(f"\nHurst Exponent (H): {H:.4f}")
    if H > 0.55:
        print("✅ Serie temporal con tendencia persistente (buena para predicción)")
    else:
        print("⚠️ Serie puede ser aleatoria (H ≈ 0.5) o anti-persistente (H < 0.5)")
    
    # Test de estacionariedad
    adf_result = adfuller(df["Último"].values)
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    
    # Normalización
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[["Último"]])
    
    return df, df_scaled, scaler, H

# 2. Creación de secuencias
def create_sequences(data, seq_length=60, forecast_horizon=1):
    sequences, labels = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(sequences), np.array(labels)

# 3. Modelo Two-Layer Stacked LSTM (TLS-LSTM)
class TLS_LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, output_size=1):
        super(TLS_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Primera capa LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Segunda capa LSTM (stacked)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Capa de salida
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # Inicializar estado oculto si es None
        if hidden is None:
            h1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            h2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            hidden = ((h1, c1), (h2, c2))
        
        # Extraer estados ocultos
        hidden1, hidden2 = hidden
        
        # Primera capa LSTM
        lstm1_out, hidden1 = self.lstm1(x, hidden1)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Segunda capa LSTM
        lstm2_out, hidden2 = self.lstm2(lstm1_out, hidden2)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Solo tomamos la última salida para la predicción
        out = self.fc(lstm2_out[:, -1, :])
        return out, (hidden1, hidden2)

# 4. Entrenamiento con early stopping
def train_model(model, train_loader, epochs=100, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}')
        
        # Early stopping
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        scheduler.step(avg_train_loss)
    
    # Cargar el mejor modelo
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 5. Predicción Futura con TLS-LSTM
def predict_future(model, last_sequence, steps=90):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    hidden_state = None
    
    with torch.no_grad():
        for _ in range(steps):
            # Asegurar dimensiones correctas [batch_size, seq_len, features]
            if current_sequence.dim() == 2:
                current_sequence = current_sequence.unsqueeze(0).unsqueeze(-1)
            elif current_sequence.size(-1) != 1:
                current_sequence = current_sequence.unsqueeze(-1)
            
            # Predicción
            pred, hidden_state = model(current_sequence, hidden_state)
            predictions.append(pred.item())
            
            # Preparar siguiente input con dimensiones [1, 1, 1]
            next_input = torch.tensor([[[pred.item()]]], 
                                    dtype=torch.float32,
                                    device=current_sequence.device)
            
            # Actualizar secuencia (asegurar same dims)
            current_sequence = torch.cat(
                (current_sequence[:, 1:, :], next_input),
                dim=1
            )
    
    return np.array(predictions).reshape(-1, 1)

# 6. Análisis de correlación (opcional)
def analyze_correlations(df, main_pair="USD/COP", other_pairs=["USD/AUD", "EUR/USD", "USD/MXN", "USD/BRL"]):
    """
    Analiza correlaciones entre USD/COP y otros pares de divisas
    
    Args:
        df (DataFrame): DataFrame con los datos históricos
        main_pair (str): Par principal a analizar (default: "USD/COP")
        other_pairs (list): Lista de pares para comparar
    """
    print("\n=== Análisis de Correlación ===")
    print(f"Par principal: {main_pair}")
    
    # Verificar que el par principal existe en los datos
    if "Último" not in df.columns:
        raise ValueError("La columna 'Último' no existe en el DataFrame")
    
    # Calcular correlaciones
    correlations = {}
    for pair in other_pairs:
        try:
            # Cargar datos del par comparativo
            pair_df = pd.read_csv(
                f"{pair.replace('/', '_')}_2010-2024.csv",
                index_col="Fecha",
                parse_dates=True,
                dayfirst=True,
                decimal=",",
                thousands=".",
                converters={
                    "Último": lambda x: float(x.replace(".", "").replace(",", "."))
                }
            )
            
            # Alinear fechas
            aligned_df = df[["Último"]].join(pair_df[["Último"]], how='inner', lsuffix='_main', rsuffix='_pair')
            
            # Calcular correlación
            corr = aligned_df["Último_main"].corr(aligned_df["Último_pair"])
            correlations[pair] = corr
            
            # Interpretación
            print(f"\n{main_pair} vs {pair}:")
            print(f"Correlación: {corr:.4f}")
            
            if abs(corr) > 0.7:
                print("🔵 CORRELACIÓN FUERTE " + ("POSITIVA" if corr > 0 else "NEGATIVA"))
            elif abs(corr) > 0.4:
                print("🟢 Correlación moderada")
            else:
                print("🟡 Correlación débil o nula")
                
            print(f"Datos alineados: {len(aligned_df)} puntos temporales")
            
        except FileNotFoundError:
            print(f"\n⚠️ Datos no encontrados para {pair}")
            correlations[pair] = None
    
    # Visualización
    if len(correlations) > 0:
        plot_correlations(main_pair, correlations)

def plot_correlations(main_pair, correlations):
    """Visualiza las correlaciones encontradas"""
    # Filtrar pares con datos disponibles
    valid_pairs = {k: v for k, v in correlations.items() if v is not None}
    
    if not valid_pairs:
        print("\nNo hay datos suficientes para visualizar correlaciones")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in valid_pairs.values()]
    bars = plt.bar(valid_pairs.keys(), valid_pairs.values(), color=colors)
    
    plt.title(f"Correlación de {main_pair} con otros pares", fontsize=14)
    plt.ylabel("Coeficiente de correlación", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(0.7, color='blue', linestyle='--', linewidth=0.5)
    plt.axhline(-0.7, color='blue', linestyle='--', linewidth=0.5)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'correlaciones_{main_pair.replace("/", "_")}.png', dpi=300)
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---

# Parámetros configurables
SEQ_LENGTH = 30  # Ventana temporal (30 días)
FORECAST_HORIZON = 1  # Predecir 1 día adelante
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 15  # Para early stopping
FUTURE_STEPS_TO_PREDICT = 30  # Días a predecir (3 meses)

# 1. Cargar y preparar datos con análisis Hurst
print("Cargando datos y analizando predictibilidad...")
df, df_scaled, scaler, hurst_exp = load_and_prepare_data("EUR_USD_2010-2024.csv")

# Ejecutar análisis de correlación
analyze_correlations(
    df,
    main_pair="EUR/USD",
    other_pairs=["USD/AUD", "USD/COP", "GBP/USD" , "USD/CHF"]
)
