# train_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib  # Para guardar el scaler
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from sklearn.preprocessing import RobustScaler
import time
from datetime import timedelta
from model.config import DEFAULT_PARAMS
from model.modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 1. Carga y preparación de datos con análisis de Hurst
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Carga y preprocesa los datos del archivo CSV."""
    print(f"📂 Cargando datos desde: data/{filepath}")
    try:
        df = pd.read_csv(
            f"data/{filepath}",
            index_col="Fecha",
            parse_dates=True,
            dayfirst=True,
            decimal=",",
            thousands=".",
            converters={
                "Último": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan,
                "Apertura": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan,
                "Máximo": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan,
                "Mínimo": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan,
                "% var.": lambda x: float(str(x).replace("%", "").replace(",", ".")) if x else np.nan
            }
        )
        
        df = df.sort_index(ascending=True)
        df = df.dropna(subset=["Último"])
        
        print(f"✅ Datos cargados: {df.shape[0]} filas")
        print(f"📅 Periodo: {df.index.min()} a {df.index.max()}")

        # 🎯 Calcular Hurst
        try:
            ultimo_series = df["Último"].dropna().values
            if len(ultimo_series) > 100:
                H, c, data_hurst = compute_Hc(ultimo_series, kind='price', simplified=True)
                print(f"📈 Exponente de Hurst (H): {H:.4f}")
                if H > 0.55:
                    print("✨ Tendencia persistente (H > 0.5)")
                elif H < 0.45:
                    print("⚡ Anti-persistente (H < 0.5)")
                else:
                    print("🔮 Posible paseo aleatorio (H ≈ 0.5)")
            else:
                print("🙈 No suficientes datos para Hurst.")
        except Exception as e:
            print(f"🚨 Error en Hurst: {e}")

        # 🎯 Test ADF
        try:
            ultimo_series_adf = df["Último"].dropna().values
            if len(ultimo_series_adf) > 0:
                adf_result = adfuller(ultimo_series_adf)
                print(f"\n🧪 Test ADF:")
                print(f"  Estadístico ADF: {adf_result[0]:.4f}")
                print(f"  p-valor: {adf_result[1]:.4f}")
                if adf_result[1] <= 0.05:
                    print("🌟 Serie estacionaria (p <= 0.05)")
                else:
                    print("💤 Serie NO estacionaria (p > 0.05)")
            else:
                print("🚫 Sin datos para ADF.")
        except Exception as e:
            print(f"🚨 Error en ADF: {e}")

        return df
    
    except FileNotFoundError:
        print(f"🚫 Archivo no encontrado: data/{filepath}")
        return None
    except Exception as e:
        print(f"🚨 Error inesperado: {e}")
        return None

# 2. Creación de secuencias
def create_sequences(data, seq_length, forecast_horizon):
    """Crea secuencias de entrada (X) y etiquetas de salida (y)."""
    sequences, labels = [], []
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - seq_length - forecast_horizon + 1):
        sequences.append(data[i: i + seq_length])  # X: todas las características (Último + RSI)
        labels.append(data[i + seq_length: i + seq_length + forecast_horizon, 0])  # y: solo "Último" (columna 0)
    return np.array(sequences), np.array(labels)

# 3. Modelo Two-Layer Stacked LSTM (TLS-LSTM)
def get_model(input_size, hidden_size, output_size, dropout_prob):
    if(DEFAULT_PARAMS.MODELNAME == "BidirectionalDeepLSTM"):
        return BidirectionalDeepLSTMModel(input_size, hidden_size, output_size, dropout_prob)
    elif(DEFAULT_PARAMS.MODELNAME == "HybridLSTMAttention"):
        return HybridLSTMAttentionModel(input_size, hidden_size, output_size, dropout_prob)
    elif(DEFAULT_PARAMS.MODELNAME == "TemporalAutoencoderLSTM"):
        return TemporalAutoencoderLSTM(input_size, hidden_size, output_size, dropout_prob)
    elif(DEFAULT_PARAMS.MODELNAME == "GRU_Model"):  
        return GRU_Model(input_size, hidden_size, output_size, dropout_prob)
    elif(DEFAULT_PARAMS.MODELNAME == "TLS_LSTMModel"):  
        return TLS_LSTMModel(input_size, hidden_size, output_size, dropout_prob)
    else:
        return TLS_LSTMModel(input_size, hidden_size, output_size, dropout_prob)    
# 4. Entrenamiento con early stopping
def train_model(model, train_loader, epochs, patience, learning_rate):
    """Entrena el modelo y guarda el mejor estado basado en la pérdida de entrenamiento."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler para ajustar la tasa de aprendizaje si la pérdida no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=patience//2, factor=0.5, verbose=True)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = DEFAULT_PARAMS.MODELPATH

    print("\n--- Iniciando Entrenamiento ---")
    for epoch in range(epochs):
        model.train()  # Poner el modelo en modo entrenamiento
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Mover datos al dispositivo
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass y optimización
            loss.backward()
            # Gradiente clipping para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)  # Acumular pérdida total

        # Calcular pérdida promedio de la época
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(
            f'Época {epoch+1}/{epochs} - Pérdida de Entrenamiento: {avg_train_loss:.6f}')

        # Early stopping y guardado del mejor modelo
        if avg_train_loss < best_loss:
            # print(f'Pérdida mejorada ({best_loss:.6f} --> {avg_train_loss:.6f}). Guardando modelo...')
            best_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f'La pérdida no mejoró. Contador de paciencia: {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print(
                    f"\n¡Detención temprana (Early Stopping)! No hubo mejora en {patience} épocas.")
                break  # Detener el entrenamiento

        # Ajustar tasa de aprendizaje basado en la pérdida
        scheduler.step(avg_train_loss)

    print("--- Entrenamiento Finalizado ---")
    # Cargar el mejor modelo encontrado durante el entrenamiento
    print(f"Cargando el mejor modelo guardado en: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    return model

def add_indicator(data, rsi_window=14, sma_window=20):
    """Calcula múltiples indicadores técnicos y los devuelve en un diccionario"""
    indicators = {}
    
    # Cálculo del RSI
    delta = data["Último"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss
    indicators["RSI"] = 100 - (100 / (1 + rs)).fillna(50)
    
    # Cálculo de SMA
    indicators["SMA"] = data["Último"].rolling(window=sma_window).mean().fillna(data["Último"])
    
    # Aquí puedes añadir más indicadores...
    # indicators["NuevoIndicador"] = cálculo...
    
    return indicators
# --- EJECUCIÓN PRINCIPAL ---

def run_training(params=DEFAULT_PARAMS):
    start_time = time.time()
    # 1. Cargar datos
    df = load_and_prepare_data(params.FILEPATH)

    # Obtener todos los indicadores de forma dinámica
    indicators = add_indicator(df)

    # Asignar automáticamente todos los indicadores al DataFrame principal
    for indicator_name, values in indicators.items():
        df[indicator_name] = values 
    
    if df is not None and params.TARGET_COLUMN and 'RSI' and 'SMA' in df.columns:

        # 2. Dividir en Train/Test
        split_index = int(len(df) * params.TRAIN_SPLIT_RATIO)
        # Seleccionar múltiples columnas (Último y RSI)
        features = params.FEATURES  # Añade más si es necesario
        train_data = df[features].iloc[:split_index]
        test_data = df[features].iloc[split_index:]
        print(f"Datos de Entrenamiento: {len(train_data)} muestras")
        print(f"Datos de Prueba: {len(test_data)} muestras")

        # 3. Escalar datos (RobustScaler)
        scaler = RobustScaler(quantile_range=(5, 95))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Guardar el scaler para uso futuro
        scaler_filename = f"{params.FILEPATH}_scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler guardado en: {scaler_filename}")

        # 4. Crear secuencias para Train y Test
        print(f"\nCreando secuencias con longitud={params.SEQ_LENGTH}, horizonte={params.FORECAST_HORIZON}...")
        X_train, y_train = create_sequences( train_scaled, params.SEQ_LENGTH, params.FORECAST_HORIZON)
        X_test, y_test = create_sequences( test_scaled, params.SEQ_LENGTH, params.FORECAST_HORIZON)
        print(f"Forma X_train: {X_train.shape}, Forma y_train: {y_train.shape}")
        print(f"Forma X_test: {X_test.shape}, Forma y_test: {y_test.shape}")

        # 5. Convertir a Tensores PyTorch y mover al dispositivo
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Después de crear X_train, y_train:
        y_train = y_train.reshape(-1, params.FORECAST_HORIZON)  # Forma (3461, 1)
        y_test = y_test.reshape(-1, params.FORECAST_HORIZON)    # Forma (332, 1)

        # Vamos a asumir que create_sequences da (N, H, 1) y que el squeeze es necesario.
        print(f"Forma y_train ajustada: {y_train.shape}")
        print(f"Forma y_test ajustada: {y_test.shape}")

        # 6. Crear DataLoader para Entrenamiento
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True)  # Shuffle para entrenamiento

        # 7. Inicializar el Modelo TLS-LSTM
        model = get_model(input_size=len(features),
                            hidden_size=params.HIDDEN_SIZE,
                            output_size=params.FORECAST_HORIZON,
                            dropout_prob=params.DROPOUT_PROB).to(device)

        print("\nModelo TLS-LSTM Definido:")
        print(model)

        # 8. Entrenar el Modelo
        model = train_model(model=model,
                            train_loader=train_loader,
                            epochs=params.EPOCHS,
                            patience=params.PATIENCE,
                            learning_rate=params.LEARNING_RATE)

        # 9. (Opcional) Evaluar en el conjunto de Test
        model.eval()  # Poner el modelo en modo evaluación (desactiva dropout, etc.)
        test_loss = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():  # No calcular gradientes durante la evaluación
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        print(f"\n--- Evaluación en Datos de Prueba ---")
        print(f"Pérdida MSE en Test: {test_loss.item():.6f}")

        # Aquí podrías añadir más métricas (RMSE, MAE) y visualizaciones
        # Ejemplo: Calcular RMSE
        rmse = torch.sqrt(test_loss).item()
        print(f"RMSE en Test: {rmse:.6f}")

        # 10. Guardar el Modelo Final (opcional, ya que train_model guarda el mejor)
        print(f"El mejor modelo del entrenamiento se encuentra en: {params.MODELPATH}")

        # Calcular y mostrar duración del entrenamiento
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\nDuración total del entrenamiento: {str(timedelta(seconds=training_duration))}")

    else:
        print("\nNo se pudo cargar o procesar el archivo de datos. Saliendo.")

if __name__ == "__main__":
    run_training()
