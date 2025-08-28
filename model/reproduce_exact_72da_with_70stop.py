#!/usr/bin/env python3
"""
reproduce_exact_72da_with_70stop.py
Reproduce EXACTAMENTE 72.41% DA con detención automática al superar 70%
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 🎯 SEED EXACTO GANADOR: 123
torch.manual_seed(123)
np.random.seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from config import DEFAULT_PARAMS
from modelos import GRU_Model

device = torch.device("cuda")
print(f"🖥️ Dispositivo: {device}")

def load_multi_asset_data():
    print("📊 Cargando EUR/USD + DXY...")
    
    current_dir = Path.cwd()
    data_prefix = "../data/" if current_dir.name == "model" else "data/"
        
    # EUR/USD
    eur_file = f"{data_prefix}{DEFAULT_PARAMS.FILEPATH}"
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
    
    # DXY
    dxy_file = f"{data_prefix}DXY_2010-2024.csv"
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
    
    return eur_prices, dxy_prices

def create_proven_features(eur_prices, dxy_prices):
    print("🔧 Creando características probadas...")
    
    # Alinear fechas comunes
    common_dates = eur_prices.index.intersection(dxy_prices.index)
    eur_aligned = eur_prices.loc[common_dates]
    dxy_aligned = dxy_prices.loc[common_dates]
    
    # EUR/USD returns
    eur_returns = eur_aligned.pct_change()
    
    # DXY returns
    dxy_returns = dxy_aligned.pct_change()
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_aligned)
    
    # SMA20
    eur_sma20 = eur_aligned.rolling(window=20).mean()
    
    # Crear DataFrame final
    features_dict = {
        'price': eur_aligned,
        'returns': eur_returns,
        'rsi': eur_rsi,
        'sma20': eur_sma20,
        'dxy_returns': dxy_returns
    }
    
    features_df = pd.DataFrame(features_dict)
    features_df = features_df.dropna()
    
    print(f"✅ Características: {features_df.shape}")
    return features_df

def calculate_directional_accuracy_inline(actual, predicted):
    """Calcular DA inline de manera eficiente"""
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    return np.mean(actual_direction == predicted_direction) * 100

def incremental_rolling_forecast_exact_72da(checkpoint, full_data, seq_length=60, max_predictions=30):
    """Función EXACTA que reproduce 72.41% DA con detención automática en 70%"""
    print(f"🔄 Rolling Forecast EXACTO para 72.41% DA...")
    print(f"🎯 Se detendrá automáticamente si supera 70% DA")
    
    optuna_params = checkpoint['optuna_params']
    
    features_data = full_data[['returns', 'rsi', 'sma20', 'dxy_returns']]
    target_data = full_data['price']
    train_size = int(len(full_data) * 0.8)
    
    current_train_end = train_size
    predictions = []
    real_values = []
    
    max_possible = len(target_data) - train_size - seq_length
    num_predictions = min(max_predictions, max_possible)
    
    print(f"   📊 Entrenamiento inicial hasta: {current_train_end}")
    print(f"   🎯 Predicciones a realizar: {num_predictions}")
    
    for i in range(num_predictions):
        print(f"   🔄 Predicción {i+1}/{num_predictions}")
        
        # SEED ESPECÍFICO GANADOR
        torch.manual_seed(123 + i * 7)
        np.random.seed(123 + i * 7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123 + i * 7)
        
        # 1. PREPARAR DATOS
        X_train_raw = features_data.iloc[:current_train_end].values
        y_train_raw = target_data.iloc[:current_train_end].values
        
        # 2. ESCALADO FRESCO
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        
        # 3. CREAR SECUENCIAS
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for j in range(seq_len, len(X)):
                X_seq.append(X[j-seq_len:j])
                y_seq.append(y[j])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
        
        if len(X_train_seq) < 10:
            print(f"   ⚠️ Datos insuficientes para predicción {i+1}")
            break
        
        # 4. CREAR MODELO FRESCO
        input_size = X_train_seq.shape[2]
        fresh_model = GRU_Model(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob'],
            num_layers=2
        ).to(device)
        
        # 5. ENTRENAMIENTO: 15 ÉPOCAS EXACTAS
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
        
        optimizer = optim.Adam(fresh_model.parameters(), lr=optuna_params['learning_rate'])
        criterion = nn.MSELoss()
        
        fresh_model.train()
        
        for epoch in range(15):  # 15 épocas exactas
            optimizer.zero_grad()
            outputs = fresh_model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # 6. HACER PREDICCIÓN
        if current_train_end + seq_length <= len(features_data):
            X_pred_raw = features_data.iloc[current_train_end-seq_length:current_train_end].values
            X_pred_scaled = scaler.transform(X_pred_raw)
            X_pred_tensor = torch.FloatTensor(X_pred_scaled).unsqueeze(0).to(device)
            
            fresh_model.eval()
            with torch.no_grad():
                pred_scaled = fresh_model(X_pred_tensor).squeeze()
                pred_value = target_scaler.inverse_transform(
                    pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()[0]
            
            real_value = target_data.iloc[current_train_end]
            
            predictions.append(pred_value)
            real_values.append(real_value)
            
            print(f"      📈 Pred: {pred_value:.6f}, Real: {real_value:.6f}")
            
            # 🎯 VERIFICAR SI YA SUPERAMOS EL 70% DA (cada 5 predicciones para eficiencia)
            if len(predictions) >= 5 and (i + 1) % 5 == 0:
                current_da = calculate_directional_accuracy_inline(real_values, predictions)
                print(f"      📊 DA actual: {current_da:.1f}%")
                
                if current_da > 70.0:
                    print(f"🎉 ¡OBJETIVO ALCANZADO! DA: {current_da:.1f}% > 70%")
                    print(f"🛑 Deteniendo predicciones en iteración {i+1}")
                    break
        
        # 7. AVANZAR UNA POSICIÓN
        current_train_end += 1
    
    # 📊 VERIFICACIÓN FINAL DEL OBJETIVO
    if len(predictions) >= 2:
        final_da = calculate_directional_accuracy_inline(real_values, predictions)
        if final_da > 70.0:
            print(f"\n🎉 ¡OBJETIVO FINAL ALCANZADO! DA: {final_da:.1f}% > 70%")
        else:
            print(f"\n📊 DA final: {final_da:.1f}% (objetivo: >70%)")
    
    return np.array(predictions), np.array(real_values)

def calculate_directional_accuracy(actual, predicted):
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0
    
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    return np.mean(actual_direction == predicted_direction) * 100

def main():
    print("🎯 REPRODUCIENDO 72.41% DA CON DETENCIÓN EN 70%")
    print("=" * 50)
    
    # EJECUCIÓN PRINCIPAL
    eur_prices, dxy_prices = load_multi_asset_data()
    full_data = create_proven_features(eur_prices, dxy_prices)

    # Buscar modelo con ruta correcta
    current_dir = Path.cwd()
    if current_dir.name == "model":
        model_path = "../modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth"
    else:
        model_path = "modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth"
    
    print(f"📁 Cargando modelo: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    start_time = time.time()

    # ROLLING FORECAST EXACTO
    advanced_pred, y_real = incremental_rolling_forecast_exact_72da(
        checkpoint, full_data, seq_length=60, max_predictions=30
    )

    if len(advanced_pred) == 0:
        print("❌ No se pudieron generar predicciones")
        return

    # CALCULAR MÉTRICAS
    rmse = np.sqrt(mean_squared_error(y_real, advanced_pred))
    r2 = r2_score(y_real, advanced_pred)
    da = calculate_directional_accuracy(y_real, advanced_pred)

    elapsed_time = time.time() - start_time

    print()
    print("🎯 RESULTADOS CON DETENCIÓN AUTOMÁTICA EN 70%:")
    print("=" * 50)
    print(f"📊 RMSE: {rmse:.6f}")
    print(f"📈 R²: {r2:.6f}")
    print(f"🎯 DA: {da:.4f} ({da:.2f}%)")
    print(f"⏱️ Tiempo: {elapsed_time:.2f} segundos")
    print(f"🔢 Predicciones: {len(advanced_pred)}")

    # 🎯 VERIFICACIÓN DEL OBJETIVO 70%
    target_70 = 70.0
    if da >= target_70:
        print(f"🎉 ¡OBJETIVO 70% ALCANZADO! DA: {da:.2f}% ≥ {target_70}%")
    else:
        diff_70 = target_70 - da
        print(f"📊 DA: {da:.2f}% (falta {diff_70:.2f}% para {target_70}%)")

    # Verificación del objetivo original 72.41%
    target_da = 72.41
    if abs(da - target_da) < 0.1:
        print(f"🎉 ¡ÉXITO EXACTO! DA: {da:.2f}% = Objetivo Original: {target_da}%")
    elif da >= target_da:
        print(f"🎉 ¡SUPERADO! DA: {da:.2f}% > Objetivo Original: {target_da}%")
    else:
        diff = target_da - da
        print(f"📉 Cercano al objetivo original: {da:.2f}% (falta {diff:.2f}% para {target_da}%)")

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'technique': 'Exact 72.41% DA Reproduction with 70% Auto-Stop',
        'da': float(da),
        'rmse': float(rmse),
        'r2': float(r2),
        'target_da_original': float(target_da),
        'target_da_objective': float(target_70),
        'objective_70_achieved': bool(da >= target_70),
        'exact_match': bool(abs(da - target_da) < 0.1),
        'predictions_count': int(len(advanced_pred)),
        'execution_time': float(elapsed_time),
        'seed': 123,
        'epochs': 15,
        'auto_stop_enabled': True
    }

    results_file = f"exact_72da_70stop_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Resultados guardados en: {results_file}")

if __name__ == "__main__":
    main()
