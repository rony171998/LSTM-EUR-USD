#!/usr/bin/env python3
"""
evaluate_optuna_models_rolling.py - Evaluación con Rolling Forecast
Implementa la técnica de rolling forecast que mejoró ARIMA de 50% → 54.5% DA
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
    ContextualLSTMTransformerFlexible
)

device = torch.device("cuda")

def load_multi_asset_data():
    """Cargar EUR/USD + DXY (igual que en entrenamiento)"""
    print("📊 Cargando EUR/USD + DXY...")
    
    # Detectar directorio y ajustar rutas
    current_dir = Path.cwd()
    if current_dir.name == "model":
        data_prefix = "../data/"
    else:
        data_prefix = "data/"
        
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
    dxy_prices = None
    dxy_file = f"{data_prefix}DXY_2010-2024.csv"
    if Path(dxy_file).exists():
        try:
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
            print(f"   ✅ DXY cargado: {len(dxy_prices)} registros")
        except:
            print("   ⚠️ DXY no disponible")
    
    print(f"   ✅ EUR/USD: {len(eur_prices)} registros")
    
    return eur_prices, dxy_prices

def create_proven_features(eur_prices, dxy_prices=None):
    """Crear características probadas (igual que en entrenamiento)"""
    print("🔧 Creando características probadas...")
    
    # 1. EUR/USD returns
    eur_returns = eur_prices.pct_change()
    
    # 2. RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_prices)
    
    # 3. SMA20
    eur_sma20 = eur_prices.rolling(window=20).mean()
    
    # Crear DataFrame base
    features_dict = {
        'price': eur_prices,
        'returns': eur_returns,
        'rsi': eur_rsi,
        'sma20': eur_sma20
    }
    
    # 4. DXY returns (si está disponible)
    if dxy_prices is not None:
        common_dates = eur_prices.index.intersection(dxy_prices.index)
        if len(common_dates) > 1000:
            dxy_aligned = dxy_prices.reindex(common_dates)
            dxy_returns = dxy_aligned.pct_change()
            
            eur_aligned = eur_prices.reindex(common_dates)
            eur_returns_aligned = eur_aligned.pct_change()
            eur_rsi_aligned = calculate_rsi(eur_aligned)
            eur_sma20_aligned = eur_aligned.rolling(window=20).mean()
            
            features_dict = {
                'price': eur_aligned,
                'returns': eur_returns_aligned,
                'rsi': eur_rsi_aligned,
                'sma20': eur_sma20_aligned,
                'dxy_returns': dxy_returns
            }
            print("   ✅ DXY incluido")
    
    features_df = pd.DataFrame(features_dict)
    features_df = features_df.dropna()
    
    print(f"✅ Características: {features_df.shape}")
    print(f"   Features: {list(features_df.columns)}")
    
    return features_df

def prepare_data_for_evaluation(seq_length):
    """Preparar datos para evaluación con seq_length específico"""
    eur_prices, dxy_prices = load_multi_asset_data()
    features_df = create_proven_features(eur_prices, dxy_prices)
    
    target_data = features_df['price']
    feature_columns = [col for col in features_df.columns if col != 'price']
    features_data = features_df[feature_columns]
    
    # Split temporal (80/20)
    train_size = int(len(features_data) * 0.8)
    
    X_train_raw = features_data.iloc[:train_size].values
    X_test_raw = features_data.iloc[train_size:].values
    y_train_raw = target_data.iloc[:train_size].values
    y_test_raw = target_data.iloc[train_size:].values
    
    # Escalado
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    # Tensores
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    y_test_tensor = torch.FloatTensor(y_test_seq).to(device)
    
    return {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train_seq': y_train_seq,
        'y_test_seq': y_test_seq,
        'target_scaler': target_scaler,
        'feature_scaler': scaler,
        'features_data': features_data,
        'target_data': target_data,
        'train_size': train_size,
        'seq_length': seq_length
    }

def rolling_forecast_lstm(model, data, max_predictions=100):
    """
    🚀 ROLLING FORECAST PARA LSTM 
    Técnica inspirada en el éxito de ARIMA (50% → 54.5% DA)
    """
    print(f"🔄 Aplicando Rolling Forecast (máximo {max_predictions} predicciones)...")
    
    model.eval()
    predictions = []
    
    # Datos iniciales
    X_test = data['X_test']
    y_test_seq = data['y_test_seq']
    target_scaler = data['target_scaler']
    feature_scaler = data['feature_scaler']
    
    # Limitar predicciones para evitar errores computacionales
    num_predictions = min(max_predictions, len(X_test))
    
    # Ventana inicial
    current_window = X_test[0:1].clone()  # Primera secuencia
    
    print(f"   📊 Ventana inicial shape: {current_window.shape}")
    
    for i in range(num_predictions):
        with torch.no_grad():
            # Predecir siguiente valor
            pred_scaled = model(current_window).squeeze()
            
            # Desnormalizar predicción
            pred_value = target_scaler.inverse_transform(
                pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()[0]
            
            predictions.append(pred_value)
            
            # 🎯 CLAVE: Actualizar ventana con información real
            if i < num_predictions - 1 and i < len(y_test_seq) - 1:
                # Obtener valor real del target (desnormalizado)
                real_target = target_scaler.inverse_transform(
                    y_test_seq[i].reshape(-1, 1)
                ).flatten()[0]
                
                # Construir nuevas características basadas en el valor real
                # Esto simula que en trading real tendríamos acceso al valor real
                
                if i + 1 < len(X_test):
                    # Tomar próxima secuencia original como base
                    next_window = X_test[i + 1:i + 2].clone()
                    current_window = next_window
                else:
                    # Si no hay más secuencias, crear una actualizando la ventana
                    # Deslizar ventana: quitar primer valor, agregar uno nuevo
                    if current_window.shape[1] > 1:
                        # Usar últimas características de la ventana actual
                        last_features = current_window[:, -1:, :].clone()
                        
                        # Deslizar ventana
                        current_window = torch.cat([
                            current_window[:, 1:, :],  # Quitar primer timestep
                            last_features  # Agregar último (simplificado)
                        ], dim=1)
        
        # Progreso cada 20 predicciones
        if (i + 1) % 20 == 0:
            print(f"   📈 Predicciones completadas: {i + 1}/{num_predictions}")
    
    predictions = np.array(predictions)
    
    # Valores reales correspondientes
    y_real = target_scaler.inverse_transform(
        y_test_seq[:len(predictions)].reshape(-1, 1)
    ).flatten()
    
    print(f"   ✅ Rolling Forecast completado: {len(predictions)} predicciones")
    
    return predictions, y_real

def evaluate_model_with_rolling(model_path):
    """Evaluar modelo con Rolling Forecast"""
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint['model_class']
        optuna_params = checkpoint['optuna_params']
        
        print(f"\n🔍 Evaluando {model_name} con Rolling Forecast")
        print(f"   📊 Parámetros Optuna: hidden={optuna_params['hidden_size']}, lr={optuna_params['learning_rate']:.6f}")
        
        # Preparar datos
        seq_length = checkpoint['seq_length']
        data = prepare_data_for_evaluation(seq_length)
        input_size = data['X_train'].shape[2]
        
        # Recrear modelo
        if model_name == "TLS_LSTMModel":
            model = TLS_LSTMModel(
                input_size=input_size,
                hidden_size=optuna_params['hidden_size'],
                output_size=1,
                dropout_prob=optuna_params['dropout_prob']
            ).to(device)
        elif model_name == "GRU_Model":
            model = GRU_Model(
                input_size=input_size,
                hidden_size=optuna_params['hidden_size'],
                output_size=1,
                dropout_prob=optuna_params['dropout_prob'],
                num_layers=2
            ).to(device)
        elif model_name == "HybridLSTMAttentionModel":
            model = HybridLSTMAttentionModel(
                input_size=input_size,
                hidden_size=optuna_params['hidden_size'],
                output_size=1,
                dropout_prob=optuna_params['dropout_prob']
            ).to(device)
        elif model_name == "BidirectionalDeepLSTMModel":
            model = BidirectionalDeepLSTMModel(
                input_size=input_size,
                hidden_size=optuna_params['hidden_size'],
                output_size=1,
                dropout_prob=optuna_params['dropout_prob']
            ).to(device)
        else:
            print(f"⚠️ Modelo {model_name} no soportado para rolling forecast")
            return None
        
        # Cargar pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 🚀 ROLLING FORECAST (nueva técnica)
        rolling_pred, y_real_rolling = rolling_forecast_lstm(model, data)
        
        # 📊 PREDICCIÓN ESTÁNDAR (para comparar)
        model.eval()
        with torch.no_grad():
            standard_pred_scaled = model(data['X_test'][:len(rolling_pred)]).squeeze()
            standard_pred = data['target_scaler'].inverse_transform(
                standard_pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()
        
        # Métricas para Rolling Forecast
        rolling_rmse = np.sqrt(mean_squared_error(y_real_rolling, rolling_pred))
        rolling_r2 = r2_score(y_real_rolling, rolling_pred)
        
        # Métricas para Estándar
        standard_rmse = np.sqrt(mean_squared_error(y_real_rolling, standard_pred))
        standard_r2 = r2_score(y_real_rolling, standard_pred)
        
        # Directional Accuracy
        def directional_accuracy(y_true, y_pred):
            if len(y_true) <= 1:
                return 0.5
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        
        rolling_da = directional_accuracy(y_real_rolling, rolling_pred)
        standard_da = directional_accuracy(y_real_rolling, standard_pred)
        
        # Mejora en DA
        da_improvement = rolling_da - standard_da
        
        print(f"   📊 RESULTADOS COMPARATIVOS:")
        print(f"   🔄 Rolling  - RMSE: {rolling_rmse:.6f}, R²: {rolling_r2:.6f}, DA: {rolling_da:.4f}")
        print(f"   📏 Estándar - RMSE: {standard_rmse:.6f}, R²: {standard_r2:.6f}, DA: {standard_da:.4f}")
        print(f"   🎯 Mejora DA: {da_improvement:.4f} ({da_improvement*100:.1f}%)")
        
        return {
            'model_name': model_name,
            'rolling_rmse': rolling_rmse,
            'rolling_r2': rolling_r2,
            'rolling_da': rolling_da,
            'standard_rmse': standard_rmse,
            'standard_r2': standard_r2,
            'standard_da': standard_da,
            'da_improvement': da_improvement,
            'optuna_params': optuna_params,
            'predictions_count': len(rolling_pred)
        }
        
    except Exception as e:
        print(f"❌ Error evaluando {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal para probar Rolling Forecast"""
    print("🚀 EVALUACIÓN CON ROLLING FORECAST")
    print("=" * 60)
    print("🎯 Probando técnica que mejoró ARIMA: 50% → 54.5% DA")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Usando CPU")
    
    # Buscar modelos Optuna
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    optuna_models = list(models_dir.glob("*_optuna_*.pth"))
    
    print(f"\n🔍 Modelos encontrados: {len(optuna_models)}")
    for model_path in optuna_models:
        print(f"   📁 {model_path.name}")
    
    if not optuna_models:
        print("❌ No se encontraron modelos Optuna")
        return
    
    # Evaluar modelos con rolling forecast
    results = []
    
    for i, model_path in enumerate(optuna_models, 1):
        print(f"\n🔄 [{i}/{len(optuna_models)}] Probando Rolling Forecast...")
        
        result = evaluate_model_with_rolling(model_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No se pudieron evaluar modelos")
        return
    
    # Análisis de resultados
    print(f"\n🏆 RESUMEN DE ROLLING FORECAST")
    print("=" * 80)
    print(f"{'Modelo':<35} {'DA Est.':<8} {'DA Roll.':<8} {'Mejora':<8} {'Status':<10}")
    print("-" * 80)
    
    improvements = []
    
    for result in results:
        improvement = result['da_improvement']
        improvements.append(improvement)
        
        status = "✅ Mejor" if improvement > 0 else "❌ Peor" if improvement < 0 else "➖ Igual"
        
        print(f"{result['model_name']:<35} "
              f"{result['standard_da']:<8.4f} "
              f"{result['rolling_da']:<8.4f} "
              f"{improvement:<8.4f} "
              f"{status:<10}")
    
    # Estadísticas generales
    avg_improvement = np.mean(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   🎯 Mejora promedio DA: {avg_improvement:.4f} ({avg_improvement*100:.1f}%)")
    print(f"   ✅ Modelos mejorados: {positive_improvements}/{len(results)}")
    print(f"   🏆 Mejor mejora: {max(improvements):.4f} ({max(improvements)*100:.1f}%)")
    
    if avg_improvement > 0:
        print(f"\n🎉 ¡Rolling Forecast es EXITOSO! Mejora promedio del {avg_improvement*100:.1f}%")
    else:
        print(f"\n⚠️ Rolling Forecast no mostró mejoras consistentes")
    
    # Guardar resultados
    current_dir = Path.cwd()
    if current_dir.name == "model":
        results_path = Path("../modelos") / DEFAULT_PARAMS.TABLENAME / f"rolling_forecast_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        results_path = Path("modelos") / DEFAULT_PARAMS.TABLENAME / f"rolling_forecast_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    results_path.parent.mkdir(exist_ok=True)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'technique': 'Rolling Forecast',
        'inspiration': 'ARIMA improvement 50% → 54.5% DA',
        'avg_improvement': float(avg_improvement),
        'models_improved': positive_improvements,
        'total_models': len(results),
        'results': results
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Resultados guardados: {results_path}")

if __name__ == "__main__":
    main()
