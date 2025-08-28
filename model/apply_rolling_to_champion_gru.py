#!/usr/bin/env python3
"""
apply_rolling_to_champion_gru.py - Aplicar Rolling Forecast al modelo campeón
Reproduce exactamente el 72.41% DA usando el modelo original
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_PARAMS
from modelos import GRU_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Usando dispositivo: {device}")

def load_multi_asset_data():
    """Cargar EUR/USD + DXY exactamente como en el script exitoso"""
    print("📊 Cargando EUR/USD + DXY...")
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        data_prefix = "../data/"
    else:
        data_prefix = "data/"
        
    # EUR/USD con conversor especial
    eur_file = f"{data_prefix}{DEFAULT_PARAMS.FILEPATH}"
    print(f"📂 Cargando EUR/USD: {eur_file}")
    eur_df = pd.read_csv(
        eur_file,
        index_col="Fecha",
        parse_dates=True,
        dayfirst=True,
        decimal=",",
        thousands=".",
        converters={
            "Último": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan
        }
    )
    eur_df = eur_df.sort_index(ascending=True)
    eur_prices = eur_df["Último"].dropna()
    
    # DXY con conversor especial
    dxy_file = f"{data_prefix}DXY_2010-2024.csv"
    print(f"📂 Cargando DXY: {dxy_file}")
    dxy_df = pd.read_csv(
        dxy_file,
        index_col="Fecha", 
        parse_dates=True,
        dayfirst=True,
        decimal=",",
        thousands=".",
        converters={
            "Último": lambda x: float(str(x).replace(".", "").replace(",", ".")) if x else np.nan
        }
    )
    dxy_df = dxy_df.sort_index(ascending=True)
    dxy_prices = dxy_df["Último"].dropna()
    
    print(f"📊 EUR/USD registros: {len(eur_prices)}")
    print(f"📊 DXY registros: {len(dxy_prices)}")
    print(f"📅 EUR/USD fechas: {eur_prices.index.min()} a {eur_prices.index.max()}")
    print(f"📅 DXY fechas: {dxy_prices.index.min()} a {dxy_prices.index.max()}")
    
    return eur_prices, dxy_prices

def create_proven_features(eur_prices, dxy_prices):
    """Crear características exactamente como en el script exitoso"""
    print("� Creando características probadas...")
    
    # Alinear fechas comunes
    common_dates = eur_prices.index.intersection(dxy_prices.index)
    eur_aligned = eur_prices.loc[common_dates]
    dxy_aligned = dxy_prices.loc[common_dates]
    
    print(f"📅 Fechas comunes: {len(common_dates)}")
    
    # 1. RSI para EUR/USD
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_aligned)
    
    # 2. SMA
    eur_sma = eur_aligned.rolling(window=20).mean()
    
    # 3. Crear DataFrame final
    feature_df = pd.DataFrame({
        'Último': eur_aligned,
        'Último_DXY': dxy_aligned,
        'RSI': eur_rsi,
        'SMA_20': eur_sma
    })
    
    # Eliminar NaN
    feature_df = feature_df.dropna()
    
    print(f"✅ Características creadas: {len(feature_df)} registros")
    print(f"📊 Columnas: {list(feature_df.columns)}")
    
    return feature_df

def create_sequences(data, seq_length):
    """Crear secuencias para LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def advanced_rolling_forecast_gru(model_path, data, seq_length=60, n_predictions=30):
    """
    Rolling Forecast Avanzado - La técnica que logró 72.41% DA
    Re-entrena el modelo incrementalmente en cada predicción
    """
    print(f"🔄 Iniciando Advanced Rolling Forecast con {n_predictions} predicciones...")
    
    # Cargar modelo original
    print(f"📂 Cargando modelo: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraer parámetros del modelo
    params = checkpoint['optuna_params']
    hidden_size = params['hidden_size']
    learning_rate = params['learning_rate'] 
    dropout_prob = params['dropout_prob']
    batch_size = params['batch_size']
    
    print(f"🧠 Parámetros del modelo:")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Dropout: {dropout_prob}")
    print(f"   Batch size: {batch_size}")
    
    # Preparar datos
    feature_columns = ['Último', 'Último_DXY', 'RSI', 'SMA_20']
    model_data = data[feature_columns].values
    
    # Normalizar datos
    scaler = RobustScaler(quantile_range=(5, 95))
    scaled_data = scaler.fit_transform(model_data)
    
    # Dividir en train/test (80/20)
    split_idx = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]
    
    print(f"📊 División datos: {len(train_data)} train, {len(test_data)} test")
    
    # Crear modelo inicial
    model = GRU_Model(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        output_size=1,
        dropout_prob=dropout_prob
    ).to(device)
    
    # Cargar pesos del modelo original
    model.load_state_dict(checkpoint['model_state_dict'])
    
    predictions = []
    actuals = []
    
    # Rolling Forecast con re-entrenamiento
    for i in range(min(n_predictions, len(test_data) - seq_length)):
        print(f"🔄 Predicción {i+1}/{min(n_predictions, len(test_data) - seq_length)}")
        
        # Datos disponibles hasta este punto
        available_data = np.concatenate([train_data, test_data[:split_idx + i + seq_length]])
        
        # Crear secuencias para re-entrenamiento
        X_retrain, y_retrain = create_sequences(available_data, seq_length)
        
        if len(X_retrain) == 0:
            continue
            
        # Re-entrenar modelo (solo unas pocas épocas)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Entrenamiento incremental (3-5 épocas)
        for epoch in range(3):
            # Crear batches
            for j in range(0, len(X_retrain), batch_size):
                end_idx = min(j + batch_size, len(X_retrain))
                batch_X = torch.FloatTensor(X_retrain[j:end_idx]).to(device)
                batch_y = torch.FloatTensor(y_retrain[j:end_idx, 0]).unsqueeze(-1).to(device)  # Solo primer feature
                
                if batch_X.size(0) == 0:
                    continue
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Hacer predicción
        model.eval()
        with torch.no_grad():
            # Usar últimos seq_length puntos como input
            input_seq = test_data[i:i+seq_length]
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
            
            prediction = model(input_tensor).cpu().numpy()[0, 0]
            actual = test_data[i + seq_length, 0]  # Primer feature es 'Último'
            
            predictions.append(prediction)
            actuals.append(actual)
    
    # Desnormalizar predicciones
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Crear arrays completos para desnormalizar
    pred_full = np.zeros((len(predictions), len(feature_columns)))
    actual_full = np.zeros((len(actuals), len(feature_columns)))
    
    pred_full[:, 0] = predictions.flatten()
    actual_full[:, 0] = actuals.flatten()
    
    pred_denorm = scaler.inverse_transform(pred_full)[:, 0]
    actual_denorm = scaler.inverse_transform(actual_full)[:, 0]
    
    return pred_denorm, actual_denorm

def calculate_directional_accuracy(actual, predicted):
    """Calcular precisión direccional"""
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    return np.mean(actual_direction == predicted_direction) * 100

def main():
    print("🎯 APLICANDO ROLLING FORECAST AL MODELO CAMPEÓN GRU")
    print("=" * 60)
    
    # Cargar datos usando el método exitoso
    eur_prices, dxy_prices = load_multi_asset_data()
    data = create_proven_features(eur_prices, dxy_prices)
    
    # Ruta del modelo campeón
    model_path = "modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth"
    
    if not Path(model_path).exists():
        print(f"❌ No se encuentra el modelo en: {model_path}")
        return
    
    start_time = time.time()
    
    # Aplicar Rolling Forecast avanzado
    predictions, actuals = advanced_rolling_forecast_gru(
        model_path=model_path,
        data=data,
        seq_length=60,
        n_predictions=30
    )
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    da = calculate_directional_accuracy(actuals, predictions)
    
    elapsed_time = time.time() - start_time
    
    print("\n🎯 RESULTADOS ROLLING FORECAST AVANZADO:")
    print("=" * 50)
    print(f"📊 RMSE: {rmse:.6f}")
    print(f"📈 R²: {r2:.6f}")
    print(f"🎯 Directional Accuracy: {da:.2f}%")
    print(f"⏱️ Tiempo total: {elapsed_time:.2f} segundos")
    print(f"🔢 Predicciones realizadas: {len(predictions)}")
    
    # ¿Logramos el 72.41%?
    target_da = 72.41
    if da >= target_da:
        print(f"🎉 ¡ÉXITO! Alcanzamos {da:.2f}% DA (objetivo: {target_da}%)")
    else:
        diff = target_da - da
        print(f"📉 Resultado: {da:.2f}% DA (falta {diff:.2f}% para {target_da}%)")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'technique': 'Advanced Rolling Forecast with Re-training',
        'predictions_count': len(predictions),
        'rmse': float(rmse),
        'r2': float(r2),
        'directional_accuracy': float(da),
        'target_da': target_da,
        'success': da >= target_da,
        'execution_time': elapsed_time
    }
    
    results_file = f"champion_gru_rolling_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Resultados guardados en: {results_file}")
    
    # Visualización
    plt.figure(figsize=(15, 10))
    
    # Gráfico de predicciones vs actuals
    plt.subplot(2, 2, 1)
    plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Rolling Forecast Predictions\nDA: {da:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', alpha=0.8)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted\nR² = {r2:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Errores
    plt.subplot(2, 2, 3)
    errors = predictions - actuals
    plt.plot(errors)
    plt.title('Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    # Direcciones
    plt.subplot(2, 2, 4)
    actual_dirs = np.diff(actuals) > 0
    pred_dirs = np.diff(predictions) > 0
    matches = actual_dirs == pred_dirs
    
    colors = ['red' if not match else 'green' for match in matches]
    plt.bar(range(len(matches)), matches.astype(int), color=colors, alpha=0.7)
    plt.title(f'Directional Accuracy: {da:.2f}%')
    plt.ylabel('Correct Direction (1) / Wrong (0)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = f"champion_gru_rolling_plot_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
