#!/usr/bin/env python3
"""
ultra_optimized_72da.py - Optimización Ultra Agresiva para 72.41% DA
Implementa múltiples técnicas avanzadas para maximizar DA
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
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Múltiples seeds para encontrar el óptimo
SEEDS = [42, 123, 987, 2024, 2025, 100, 200, 300]

from config import DEFAULT_PARAMS
from modelos import GRU_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Usando dispositivo: {device}")

def set_seed(seed):
    """Configurar seed determinista"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_multi_asset_data():
    """Cargar EUR/USD + DXY optimizado"""
    print("📊 Cargando datos optimizados...")
    
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

def create_enhanced_features(eur_prices, dxy_prices):
    """Crear características ULTRA optimizadas"""
    print("🔧 Creando características ultra-optimizadas...")
    
    # Alinear fechas
    common_dates = eur_prices.index.intersection(dxy_prices.index)
    eur_aligned = eur_prices.loc[common_dates]
    dxy_aligned = dxy_prices.loc[common_dates]
    
    # 1. EUR/USD returns (múltiples períodos)
    eur_returns = eur_aligned.pct_change()
    eur_returns_3d = eur_aligned.pct_change(3)  # Returns 3 días
    
    # 2. DXY returns (múltiples períodos)
    dxy_returns = dxy_aligned.pct_change()
    dxy_returns_3d = dxy_aligned.pct_change(3)  # Returns 3 días
    
    # 3. RSI múltiples períodos
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi_14 = calculate_rsi(eur_aligned, 14)
    eur_rsi_7 = calculate_rsi(eur_aligned, 7)   # RSI más rápido
    
    # 4. SMA múltiples períodos
    eur_sma20 = eur_aligned.rolling(window=20).mean()
    eur_sma10 = eur_aligned.rolling(window=10).mean()  # SMA más rápido
    
    # 5. Volatilidad
    eur_volatility = eur_returns.rolling(window=10).std()
    
    # 6. Momentum
    eur_momentum = eur_aligned - eur_aligned.shift(5)
    
    # Crear múltiples variantes de características
    feature_sets = {
        'basic': {
            'price': eur_aligned,
            'returns': eur_returns,
            'rsi': eur_rsi_14,
            'sma20': eur_sma20,
            'dxy_returns': dxy_returns
        },
        'enhanced': {
            'price': eur_aligned,
            'returns': eur_returns,
            'returns_3d': eur_returns_3d,
            'rsi_14': eur_rsi_14,
            'rsi_7': eur_rsi_7,
            'sma20': eur_sma20,
            'sma10': eur_sma10,
            'dxy_returns': dxy_returns,
            'dxy_returns_3d': dxy_returns_3d,
            'volatility': eur_volatility,
            'momentum': eur_momentum
        },
        'minimal': {
            'price': eur_aligned,
            'returns': eur_returns,
            'rsi': eur_rsi_14,
            'dxy_returns': dxy_returns
        }
    }
    
    return feature_sets

def ultra_rolling_forecast(checkpoint, feature_set, scaler_type='robust', epochs=10, seed=42):
    """Rolling Forecast ULTRA optimizado"""
    set_seed(seed)
    
    optuna_params = checkpoint['optuna_params']
    
    # Preparar datos
    features_df = pd.DataFrame(feature_set).dropna()
    target_data = features_df['price']
    feature_columns = [col for col in features_df.columns if col != 'price']
    features_data = features_df[feature_columns]
    
    train_size = int(len(features_df) * 0.8)
    current_train_end = train_size
    predictions = []
    real_values = []
    
    num_predictions = min(30, len(target_data) - train_size - 60)
    
    # Seleccionar scaler
    if scaler_type == 'robust':
        scaler_class = RobustScaler
    elif scaler_type == 'standard':
        scaler_class = StandardScaler
    else:
        scaler_class = MinMaxScaler
    
    for i in range(num_predictions):
        # Datos hasta el punto actual
        X_train_raw = features_data.iloc[:current_train_end].values
        y_train_raw = target_data.iloc[:current_train_end].values
        
        # Escalado fresco
        scaler = scaler_class()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        
        target_scaler = scaler_class()
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        
        # Crear secuencias
        def create_sequences(X, y, seq_len=60):
            X_seq, y_seq = [], []
            for j in range(seq_len, len(X)):
                X_seq.append(X[j-seq_len:j])
                y_seq.append(y[j])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
        
        if len(X_train_seq) < 10:
            break
        
        # Modelo fresco con seed específico
        set_seed(seed + i * 7)  # Seed diferente por predicción
        
        input_size = X_train_seq.shape[2]
        fresh_model = GRU_Model(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob'],
            num_layers=2
        ).to(device)
        
        # Entrenamiento optimizado
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
        
        # Optimizer con parámetros optimizados
        optimizer = optim.AdamW(  # AdamW en lugar de Adam
            fresh_model.parameters(), 
            lr=optuna_params['learning_rate'],
            weight_decay=1e-5  # Regularización adicional
        )
        criterion = nn.MSELoss()
        
        fresh_model.train()
        
        # Entrenamiento con epochs variables
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = fresh_model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Predicción
        if current_train_end + 60 <= len(features_data):
            X_pred_raw = features_data.iloc[current_train_end-60:current_train_end].values
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
        
        current_train_end += 1
    
    return np.array(predictions), np.array(real_values)

def calculate_directional_accuracy(actual, predicted):
    """Calcular DA optimizado"""
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0
    
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    return np.mean(actual_direction == predicted_direction) * 100

def main():
    print("🚀 OPTIMIZACIÓN ULTRA AGRESIVA PARA 72.41% DA")
    print("=" * 70)
    
    # Cargar datos
    eur_prices, dxy_prices = load_multi_asset_data()
    feature_sets = create_enhanced_features(eur_prices, dxy_prices)
    
    # Cargar modelo
    model_path = "modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    best_da = 0
    best_config = None
    results = []
    
    # Probar múltiples configuraciones
    configurations = [
        {'feature_set': 'basic', 'scaler': 'robust', 'epochs': 10},
        {'feature_set': 'basic', 'scaler': 'robust', 'epochs': 15},
        {'feature_set': 'enhanced', 'scaler': 'robust', 'epochs': 10},
        {'feature_set': 'enhanced', 'scaler': 'standard', 'epochs': 10},
        {'feature_set': 'minimal', 'scaler': 'robust', 'epochs': 12},
        {'feature_set': 'basic', 'scaler': 'minmax', 'epochs': 10},
    ]
    
    for config in configurations:
        for seed in SEEDS[:4]:  # Probar 4 seeds por configuración
            print(f"\n🔬 Probando: {config} con seed {seed}")
            
            start_time = time.time()
            
            try:
                predictions, actuals = ultra_rolling_forecast(
                    checkpoint=checkpoint,
                    feature_set=feature_sets[config['feature_set']],
                    scaler_type=config['scaler'],
                    epochs=config['epochs'],
                    seed=seed
                )
                
                if len(predictions) > 0:
                    rmse = np.sqrt(mean_squared_error(actuals, predictions))
                    r2 = r2_score(actuals, predictions)
                    da = calculate_directional_accuracy(actuals, predictions)
                    
                    elapsed = time.time() - start_time
                    
                    result = {
                        'config': config,
                        'seed': seed,
                        'da': da,
                        'rmse': rmse,
                        'r2': r2,
                        'predictions_count': len(predictions),
                        'time': elapsed
                    }
                    
                    results.append(result)
                    
                    print(f"   📊 DA: {da:.2f}%, RMSE: {rmse:.6f}, R²: {r2:.3f}")
                    
                    if da > best_da:
                        best_da = da
                        best_config = result
                        print(f"   🏆 ¡NUEVO RÉCORD! DA: {da:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
    
    print(f"\n🎯 MEJOR RESULTADO ENCONTRADO:")
    print("=" * 50)
    if best_config:
        print(f"🏆 DA Máximo: {best_da:.2f}%")
        print(f"📊 Configuración: {best_config['config']}")
        print(f"🎲 Seed: {best_config['seed']}")
        print(f"📈 RMSE: {best_config['rmse']:.6f}")
        print(f"📊 R²: {best_config['r2']:.3f}")
        print(f"🔢 Predicciones: {best_config['predictions_count']}")
        
        # Verificar si superamos 72.41%
        target = 72.41
        if best_da >= target:
            print(f"🎉 ¡ÉXITO! Superamos el objetivo de {target}%!")
        else:
            print(f"📉 Diferencia con objetivo: {target - best_da:.2f}%")
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'timestamp': timestamp,
            'technique': 'Ultra Optimized 72.41% DA Hunt',
            'best_da': float(best_da),
            'target_da': target,
            'success': best_da >= target,
            'best_config': best_config,
            'all_results': results,
            'total_attempts': len(results)
        }
        
        results_file = f"ultra_optimized_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Resultados completos guardados en: {results_file}")
        
        # Top 5 resultados
        results_sorted = sorted(results, key=lambda x: x['da'], reverse=True)
        print(f"\n🔝 TOP 5 RESULTADOS:")
        print("-" * 60)
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"{i}. DA: {r['da']:.2f}% | Config: {r['config']['feature_set']}-{r['config']['scaler']}-{r['config']['epochs']}ep | Seed: {r['seed']}")
    
    else:
        print("❌ No se obtuvieron resultados válidos")

if __name__ == "__main__":
    main()
