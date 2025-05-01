# predict_future.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib  # Para cargar el scaler
from datetime import timedelta
from train_model import (
    add_indicator,
    load_and_prepare_data,
    # create_sequences,
    get_model,
    device
)
from config import DEFAULT_PARAMS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Parámetros Consistentes con el Entrenamiento ---
FILEPATH = DEFAULT_PARAMS.FILEPATH
TARGET_COLUMN = DEFAULT_PARAMS.TARGET_COLUMN
SEQ_LENGTH = DEFAULT_PARAMS.SEQ_LENGTH
FORECAST_HORIZON = DEFAULT_PARAMS.FORECAST_HORIZON
TRAIN_SPLIT_RATIO = DEFAULT_PARAMS.TRAIN_SPLIT_RATIO
HIDDEN_SIZE = DEFAULT_PARAMS.HIDDEN_SIZE
DROPOUT_PROB = DEFAULT_PARAMS.DROPOUT_PROB
FEATURES = DEFAULT_PARAMS.FEATURES
MODEL_PATH = DEFAULT_PARAMS.MODELPATH
SCALER_PATH = DEFAULT_PARAMS.SCALER_PATH
# --- Fin Parámetros ---

# --- Parámetros para la Predicción Futura ---
# ¿Cuántos días hacia el futuro predecir? (ej: 90 días ~ 3 meses)
FUTURE_STEPS_TO_PREDICT = 30
HISTORICAL_DAYS_TO_PLOT = 100  # ¿Cuántos días históricos mostrar en el gráfico?
# --- Fin Parámetros ---

# --- Ejecución Principal ---

if __name__ == "__main__":
    # 1. Cargar Scaler
    print(f"Cargando scaler desde: {SCALER_PATH}")
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Archivo scaler '{SCALER_PATH}' no encontrado.")
        exit()
    except Exception as e:
        print(f"Error al cargar el scaler: {e}")
        exit()

    # 2. Cargar Modelo Entrenado
    print(f"Cargando modelo entrenado desde: {MODEL_PATH}")
    model = get_model(input_size=len(FEATURES),
                      hidden_size=HIDDEN_SIZE,
                      output_size=FORECAST_HORIZON,  # Debe ser 1 para predicción iterativa
                      dropout_prob=DROPOUT_PROB).to(device)
    try:
        model.load_state_dict(torch.load(
            f"modelos/{MODEL_PATH}", map_location=device))
        model.eval()  # Poner en modo evaluación
        print("Modelo cargado y en modo evaluación.")
    except FileNotFoundError:
        print(f"Error: Archivo del modelo '{MODEL_PATH}' no encontrado.")
        exit()
    except Exception as e:
        print(f"Error al cargar el estado del modelo: {e}")
        exit()

    # 3. Cargar Datos Históricos
    df = load_and_prepare_data(FILEPATH)
    df2025 = load_and_prepare_data("EUR_USD_2025_01_01 - 2025_03_08.csv")
    # Después de cargar df, añade el RSI
    # Obtener todos los indicadores de forma dinámica
    indicators = add_indicator(df)

    # Asignar automáticamente todos los indicadores al DataFrame principal
    for indicator_name, values in indicators.items():
        df[indicator_name] = values

    if df is None:
        exit()

    if len(df) < SEQ_LENGTH:
        print(
            f"Error: No hay suficientes datos históricos ({len(df)}) para formar la secuencia inicial de longitud {SEQ_LENGTH}.")
        exit()

    # 4. Preparar la Última Secuencia Conocida (ahora con 2 características)
    last_known_sequence_original = df[FEATURES].iloc[-SEQ_LENGTH:].values
    last_known_sequence_scaled = scaler.transform(last_known_sequence_original)

    # Convertir a tensor y ajustar forma para el modelo: (batch_size, seq_len, features)
    current_sequence_tensor = torch.tensor(
        last_known_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    # Debería ser (1, 60, 2)
    print(f"Forma tensor secuencia inicial: {current_sequence_tensor.shape}")

    # 5. Realizar Predicción Iterativa
    future_predictions_scaled = []
    print(
        f"Iniciando predicción iterativa para {FUTURE_STEPS_TO_PREDICT} días futuros...")

    with torch.no_grad():
        for i in range(FUTURE_STEPS_TO_PREDICT):
            # Predecir el siguiente paso (solo el precio)
            next_pred_scaled_tensor = model(
                current_sequence_tensor)  # Salida shape (1, 1)

            # Crear valores para todos los features
            # 1. Precio predicho (ya escalado)
            # 2. RSI neutral (50 escalado)
            neutral_rsi_scaled = (50 - scaler.center_[1]) / scaler.scale_[1]
            # 3. SMA (usar el último valor conocido escalado)
            last_sma_scaled = (df["SMA"].iloc[-1] -
                               scaler.center_[2]) / scaler.scale_[2]

            # Crear tensor con TRES características
            next_step_scaled = torch.tensor([
                [next_pred_scaled_tensor.item(), neutral_rsi_scaled,
                 last_sma_scaled]
            ], dtype=torch.float32).unsqueeze(0).to(device)

            future_predictions_scaled.append(next_pred_scaled_tensor.item())

            # Preparar la siguiente secuencia de entrada (manteniendo 3 features)
            new_sequence_tensor = torch.cat(
                (current_sequence_tensor[:, 1:, :],  # Todos excepto el primero
                 next_step_scaled),  # Añadir la nueva predicción con 3 features
                dim=1
            )
            current_sequence_tensor = new_sequence_tensor
    print("Predicción iterativa finalizada.")

    # 6. Desnormalizar las Predicciones Futuras
    future_predictions_for_inverse = np.zeros(
        (len(future_predictions_scaled), len(FEATURES)))
    future_predictions_for_inverse[:, 0] = future_predictions_scaled  # Precio
    future_predictions_for_inverse[:, 1] = 50  # RSI neutral
    # Último SMA conocido
    future_predictions_for_inverse[:, 2] = df["SMA"].iloc[-1]

    future_predictions = scaler.inverse_transform(future_predictions_for_inverse)[
        :, 0]  # Tomar solo el precio

    # 7. Generar Fechas Futuras para el Gráfico
    last_historical_date = df.index[-1]
    future_dates = pd.date_range(start=last_historical_date + timedelta(days=1),
                                 periods=FUTURE_STEPS_TO_PREDICT,
                                 freq='B')  # 'B' para días hábiles, ajusta si es necesario ('D' para todos los días)

    # 8. Crear DataFrame de Predicciones Futuras
    df_future = pd.DataFrame(
        {'Predicción': future_predictions.flatten()}, index=future_dates)

    # 9. Graficar Resultados
    print("Generando gráfico con predicciones futuras...")
    plt.figure(figsize=(14, 7))

    # Graficar datos históricos recientes
    plt.plot(df.index[-HISTORICAL_DAYS_TO_PLOT:],
            df[TARGET_COLUMN].iloc[-HISTORICAL_DAYS_TO_PLOT:],
            label="Histórico Reciente", color="blue")

    # Graficar datos de 2025 (si existen)
    if df2025 is not None and len(df2025) > 0:
        plt.plot(df2025.index,
                df2025[TARGET_COLUMN],
                label="Datos Reales 2025", color="green")

    common_dates = df_future.index.intersection(df2025.index)

    if len(common_dates) > 0:
        y_true = df2025.loc[common_dates, TARGET_COLUMN].values
        y_pred = df_future.loc[common_dates, 'Predicción'].values

        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Mostrar métricas en consola
        print("\n🔍 Métricas de evaluación:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Agregar métricas al gráfico
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAPE: {mape:.2f}%"
        plt.gcf().text(0.75, 0.45, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    else:
        print("⚠️ No hay fechas comunes entre predicción y datos reales para evaluar.")
    # Graficar predicciones futuras
    plt.plot(df_future.index,
            df_future['Predicción'],
            label=f"Predicción Futura ({FUTURE_STEPS_TO_PREDICT} días)", color="red", linestyle='--')

    plt.title(f"Predicción Futura ({FILEPATH}) con TLS-LSTM")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Guardar la figura antes de mostrarla
    plt.savefig(f"images/prediccion/{MODEL_PATH}.png",
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\nEjecución de predicción futura completada.")
