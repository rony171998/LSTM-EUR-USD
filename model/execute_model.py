# execute_model.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib # Para cargar el scaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_model import (
    add_indicator,
    get_model,
    device,
    load_and_prepare_data
)
#from save_data import get_df
from config import DEFAULT_PARAMS

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

# --- Definición de Creación de Secuencias (Debe ser idéntica) ---
def create_sequences(data, seq_length, forecast_horizon):
    """Crea secuencias de entrada (X) y etiquetas de salida (y)."""
    sequences, labels = [], []
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - seq_length - forecast_horizon + 1):
        sequences.append(data[i : i + seq_length])
        labels.append(data[i + seq_length : i + seq_length + forecast_horizon])
    return np.array(sequences), np.array(labels)
# --- Fin Definición de Creación de Secuencias ---

if __name__ == "__main__":
    # 1. Cargar los datos (similar al script de entrenamiento)
    df = load_and_prepare_data(FILEPATH)
    #df = get_df(table_name="eur_usd")  # Cambia esto según necesites

    # Después de cargar df, añade el RSI
    # Obtener todos los indicadores de forma dinámica
    indicators = add_indicator(df)

    # Asignar automáticamente todos los indicadores al DataFrame principal
    for indicator_name, values in indicators.items():
        df[indicator_name] = values

    print("Features usados:", FEATURES)
    print("Número de features:", len(FEATURES))
    print("Dimensiones de los datos:", df[FEATURES].shape)

    # 2. Cargar el Scaler
    print(f"Cargando scaler desde: {SCALER_PATH}")
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Archivo scaler no encontrado en {SCALER_PATH}.")
        print("Asegúrate de haber ejecutado train_model.py primero y que haya guardado el scaler.")
        exit()
    except Exception as e:
        print(f"Error al cargar el scaler: {e}")
        exit()

    # 3. Dividir los datos en Train/Test (usando el mismo ratio)
    split_index = int(len(df) * TRAIN_SPLIT_RATIO)
    # No necesitamos train_data aquí, solo test_data
    test_data = df[FEATURES].iloc[split_index:]  # Correcto

    print(f"Número de muestras de prueba: {len(test_data)}")

    if len(test_data) < SEQ_LENGTH + FORECAST_HORIZON:
        print(f"Error: No hay suficientes datos de prueba ({len(test_data)}) para crear al menos una secuencia completa "
            f"(se requieren {SEQ_LENGTH + FORECAST_HORIZON}). Ajusta el split ratio o verifica los datos.")
        exit()

    # 4. Normalizar los datos de prueba (¡SOLO transform!)
    test_scaled = scaler.transform(test_data)

    # 5. Crear secuencias para la evaluación
    print(f"Creando secuencias de prueba con longitud={SEQ_LENGTH}, horizonte={FORECAST_HORIZON}...")
    X_test, y_test_scaled = create_sequences(test_scaled, SEQ_LENGTH, FORECAST_HORIZON)
    print(f"Forma X_test: {X_test.shape}")  # Debe ser (N, 60, 2)
    print(f"Forma y_test_scaled: {y_test_scaled.shape}")

    # Guardar los índices correspondientes a las etiquetas y_test para el gráfico
    # El índice de y_test[i] corresponde al índice de la secuencia X_test[i] + SEQ_LENGTH
    test_indices = test_data.index[SEQ_LENGTH + FORECAST_HORIZON -1 : ] # Ajuste para el índice final de y
    # print(len(test_indices), y_test_scaled.shape[0]) # Deberían coincidir

    # Asegurar que las longitudes coincidan
    if len(test_indices) != y_test_scaled.shape[0]:
        print("Advertencia: Discrepancia en longitud entre índices y datos de prueba. Recalculando índices.")
        # Si hay discrepancia, una forma más segura es calcular desde el final
        num_test_predictions = y_test_scaled.shape[0]
        test_indices = test_data.index[-num_test_predictions:]


    # 6. Convertir a tensores de PyTorch
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    # No necesitamos y_test_scaled como tensor para predecir, solo para comparar después

    # 7. Cargar el modelo entrenado
    print(f"Cargando modelo entrenado desde: {MODEL_PATH}")
    # output_size debe coincidir con FORECAST_HORIZON
    model = get_model(input_size=len(FEATURES),
                        hidden_size=HIDDEN_SIZE,
                        output_size=FORECAST_HORIZON,
                        dropout_prob=DROPOUT_PROB).to(device)
    try:
        model.load_state_dict(torch.load(f"{MODEL_PATH}", map_location=device), strict=True)
    except FileNotFoundError:
        print(f"Error: Archivo del modelo no encontrado en {MODEL_PATH}.")
        print("Asegúrate de haber ejecutado train_model.py primero y que haya guardado el modelo.")
        exit()
    except Exception as e:
        print(f"Error al cargar el estado del modelo: {e}")
        print("Verifica que la arquitectura del modelo definida aquí coincida exactamente con la del modelo guardado.")
        exit()

    model.eval() # ¡Importante! Poner el modelo en modo evaluación

    # 8. Hacer predicciones
    print("Realizando predicciones en el conjunto de prueba...")
    with torch.no_grad(): # No necesitamos calcular gradientes para la inferencia
        y_pred_scaled_tensor = model(X_test)

    # Mover predicciones a CPU y convertir a NumPy array
    y_pred_scaled = y_pred_scaled_tensor.cpu().numpy()
    # y_pred_scaled tendrá forma (num_samples, FORECAST_HORIZON)

    # 9. Desnormalizar las predicciones y valores reales CORRECTAMENTE

    # Para las predicciones (y_pred_scaled tiene forma (272,1))
    # Necesitamos crear un array con 3 columnas para el scaler (mismo número que features)
    y_pred_for_inverse = np.zeros((len(y_pred_scaled), len(FEATURES)))
    y_pred_for_inverse[:, 0] = y_pred_scaled.squeeze()  # Columna 0: predicciones del precio
    y_pred_for_inverse[:, 1] = 50  # Valor medio típico para RSI
    y_pred_for_inverse[:, 2] = df["SMA"].iloc[-len(y_pred_scaled):].values  # O usar valor reciente de SMA

    y_pred = scaler.inverse_transform(y_pred_for_inverse)[:, 0]  # Tomamos solo la columna de interés (precio)

    # Para y_test_scaled (forma (272,1,3))
    y_test_reshaped = y_test_scaled.reshape(-1, len(FEATURES))  # Reshape a (272,3)
    y_test_original = scaler.inverse_transform(y_test_reshaped)[:, 0]  # Tomamos solo la columna Último
    print("Forma y_test_scaled:", y_test_scaled.shape)  # Debería ser (332,1,2)
    print("Número de características del scaler:", scaler.n_features_in_)  # Debe ser 2
    # Calcular métricas
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)

    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

    print("\n--- Métricas de Evaluación ---")
    print(f"MAE (Error Absoluto Medio): {mae:.4f}")
    print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
    print(f"R² (Coeficiente de Determinación): {r2:.4f}")
    print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")

    # Podemos considerar una forma de "exactitud" como (100 - MAPE)
    accuracy_like = 100 - mape
    print(f"\nPrecisión aproximada: {accuracy_like:.2f}% (100 - MAPE)")

    # 10. Graficar resultados con RSI
    print("Generando gráfico de resultados")

   # Crear figura con solo el gráfico principal
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # --- Gráfico principal: Precio y predicciones ---
    metrics_text = (
        f"MAE: {mae:.4f}\n"
        f"MSE: {mse:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"R²: {r2:.4f}\n"
        f"MAPE: {mape:.2f}%"
    )

    # Gráfico de precios con las predicciones
    ax1.plot(test_indices, y_test_original, label="Valor Real (Test)", color="blue", alpha=0.7, linewidth=1.5)
    ax1.plot(test_indices, y_pred, label="Predicción LSTM (Test)", color="red", linestyle='--', alpha=0.8, linewidth=1.5)

    # Añadir cuadro con métricas
    ax1.annotate(
        metrics_text,
        xy=(0.02, 0.75),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round", alpha=0.5, facecolor="white")
    )

    # Configuración del gráfico
    ax1.set_title(f"Comparación Real vs. Predicción LSTM ({FILEPATH}) con {DEFAULT_PARAMS.MODELNAME}", fontsize=14, pad=20)
    ax1.set_xlabel("Fecha", fontsize=12)
    ax1.set_ylabel("Precio", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Rotar fechas para mejor visualización
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    plt.tight_layout()

    # Guardar la figura
    fig.savefig(f"images/test/{MODEL_PATH}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\nEjecución completada.")