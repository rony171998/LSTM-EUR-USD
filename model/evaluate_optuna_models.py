#!/usr/bin/env python3
"""
evaluate_optuna_models.py - Evaluar modelos entrenados con parámetros optimizados por Optuna
Incluye comparación con baselines (Naive y ARIMA) y análisis de rendimiento mejorado.
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
    NaiveForecastModel,
    ARIMAModel
)

device = torch.device("cuda")

def load_multi_asset_data():
    """Cargar EUR/USD + DXY (igual que en entrenamiento)"""
    print("📊 Cargando EUR/USD + DXY...")
    
    # Detectar directorio y ajustar rutas
    current_dir = Path.cwd()
    if current_dir.name == "model":
        data_prefix = "../data/"  # Desde model/
    else:
        data_prefix = "data/"  # Desde raíz del proyecto
        
    # EUR/USD - ruta adaptativa
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
    
    # DXY - ruta adaptativa
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
    # Cargar y preparar datos igual que en entrenamiento
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
        'features_data': features_data,
        'target_data': target_data,
        'train_size': train_size,
        'seq_length': seq_length
    }

def load_and_evaluate_optuna_model(model_path):
    """Cargar y evaluar un modelo entrenado con parámetros Optuna"""
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint['model_class']
        optuna_params = checkpoint['optuna_params']
        
        print(f"\n🔍 Evaluando {model_name} (Optuna optimizado)")
        print(f"   📊 Parámetros Optuna: hidden={optuna_params['hidden_size']}, lr={optuna_params['learning_rate']:.6f}")
        
        # Preparar datos con seq_length del modelo
        seq_length = checkpoint['seq_length']
        data = prepare_data_for_evaluation(seq_length)
        input_size = data['X_train'].shape[2]
        
        # Recrear modelo con parámetros exactos
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
            print(f"⚠️ Modelo {model_name} no reconocido")
            return None
        
        # Cargar pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluar
        with torch.no_grad():
            test_pred_scaled = model(data['X_test']).squeeze()
            train_pred_scaled = model(data['X_train']).squeeze()
            
            # Desnormalizar
            test_pred = data['target_scaler'].inverse_transform(test_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
            train_pred = data['target_scaler'].inverse_transform(train_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
            
            y_test_real = data['target_scaler'].inverse_transform(data['y_test_seq'].reshape(-1, 1)).flatten()
            y_train_real = data['target_scaler'].inverse_transform(data['y_train_seq'].reshape(-1, 1)).flatten()
            
            # Métricas
            test_rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            test_r2 = r2_score(y_test_real, test_pred)
            
            # Directional Accuracy
            def directional_accuracy(y_true, y_pred):
                if len(y_true) <= 1:
                    return 0.5
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                return np.mean(true_direction == pred_direction)
            
            test_da = directional_accuracy(y_test_real, test_pred)
            
            return {
                'model_name': model_name,
                'model_type': 'Optuna Optimized',
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_da': test_da,
                'parameters': sum(p.numel() for p in model.parameters()),
                'optuna_params': optuna_params,
                'y_test_real': y_test_real,
                'test_pred': test_pred,
                'model_path': str(model_path)
            }
    
    except Exception as e:
        print(f"❌ Error evaluando {model_path}: {e}")
        return None

def evaluate_naive_baseline(data):
    """Evaluar baseline Naive con datos preparados"""
    print("\n🔍 Evaluando Naive Baseline...")
    
    # Para el baseline Naive: el último valor conocido predice el siguiente
    y_test_real = data['target_scaler'].inverse_transform(data['y_test_seq'].reshape(-1, 1)).flatten()
    
    # Crear predicciones naive: cada valor predice el siguiente
    if len(y_test_real) > 1:
        naive_pred = y_test_real[:-1]  # Último valor conocido
        y_test_for_naive = y_test_real[1:]  # Valor a predecir
    else:
        naive_pred = [y_test_real[0]]
        y_test_for_naive = y_test_real
    
    # Métricas
    test_rmse = np.sqrt(mean_squared_error(y_test_for_naive, naive_pred))
    test_r2 = r2_score(y_test_for_naive, naive_pred)
    
    # Directional Accuracy
    def directional_accuracy(y_true, y_pred):
        if len(y_true) <= 2:
            return 0.5
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction)
    
    test_da = directional_accuracy(y_test_for_naive, naive_pred)
    
    print(f"   📊 RMSE: {test_rmse:.6f}")
    print(f"   📊 R²: {test_r2:.6f}")
    print(f"   📊 DA: {test_da:.4f}")
    
    return {
        'model_name': 'Naive Baseline',
        'model_type': 'Baseline',
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_da': test_da,
        'parameters': 0,
        'optuna_params': {},
        'y_test_real': y_test_for_naive,
        'test_pred': naive_pred,
        'model_path': 'baseline'
    }

def evaluate_arima_baseline(data):
    """Evaluar baseline ARIMA con datos preparados"""
    print("\n🔍 Evaluando ARIMA Baseline...")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Usar datos originales sin escalar para ARIMA
        target_series = data['target_data']
        train_size = data['train_size']
        
        train_data = target_series.iloc[:train_size].dropna()
        test_data = target_series.iloc[train_size:].dropna()
        
        print(f"   📊 Train data: {len(train_data)} | Test data: {len(test_data)}")
        
        # Verificar estacionariedad
        adf_result = adfuller(train_data)
        is_stationary = adf_result[1] < 0.05
        print(f"   📊 Estacionariedad (ADF p-value): {adf_result[1]:.6f} ({'Estacionaria' if is_stationary else 'No estacionaria'})")
        
        # Para datos financieros, usar configuración más simple: ARIMA(1,1,0) o (0,1,1)
        # Probar diferentes órdenes hasta encontrar uno que funcione
        orders_to_try = [(1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2)]
        fitted_model = None
        best_aic = float('inf')
        best_order = None
        
        for order in orders_to_try:
            try:
                model = ARIMA(train_data, order=order)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    fitted_model = fitted
                    best_order = order
            except:
                continue
        
        if fitted_model is None:
            raise Exception("No se pudo ajustar ningún modelo ARIMA")
        
        print(f"   📊 Mejor ARIMA{best_order} - AIC: {best_aic:.2f}")
        
        # Rolling forecast: predecir paso a paso (más realista)
        predictions = []
        
        # Usar solo una porción del test set para comparar con modelos de secuencias
        # Los modelos usan seq_length=30, así que alinear con eso
        seq_length = data['seq_length']
        test_subset = test_data.iloc[:len(data['y_test_seq'])]  # Alinear con datos de modelos
        
        print(f"   📊 Prediciendo {len(test_subset)} pasos...")
        
        # Historia inicial: todos los datos de entrenamiento
        history = train_data.tolist()
        
        for i in range(min(len(test_subset), 100)):  # Limitar a 100 predicciones para evitar errores
            try:
                # Re-ajustar modelo con historia actualizada (rolling window)
                model = ARIMA(history, order=best_order)
                model_fit = model.fit()
                
                # Predecir siguiente valor
                forecast = model_fit.forecast(steps=1)
                predictions.append(forecast[0])
                
                # Actualizar historia con valor real observado
                if i < len(test_subset):
                    history.append(test_subset.iloc[i])
                    
            except Exception as e:
                # Si falla, usar último valor (fallback a naive)
                predictions.append(history[-1] if history else test_subset.iloc[i])
        
        # Convertir a arrays
        arima_pred = np.array(predictions)
        y_test_for_arima = test_subset.iloc[:len(predictions)].values
        
        print(f"   📊 Predicciones generadas: {len(arima_pred)}")
        print(f"   📊 Valores reales: {len(y_test_for_arima)}")
        
        # Validar que tenemos datos válidos
        if len(arima_pred) == 0 or len(y_test_for_arima) == 0:
            raise Exception("No se generaron predicciones válidas")
        
        # Métricas
        test_rmse = np.sqrt(mean_squared_error(y_test_for_arima, arima_pred))
        test_r2 = r2_score(y_test_for_arima, arima_pred)
        
        # Directional Accuracy
        def directional_accuracy(y_true, y_pred):
            if len(y_true) <= 1:
                return 0.5
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        
        test_da = directional_accuracy(y_test_for_arima, arima_pred)
        
        print(f"   📊 RMSE: {test_rmse:.6f}")
        print(f"   📊 R²: {test_r2:.6f}")
        print(f"   📊 DA: {test_da:.4f}")
        
        return {
            'model_name': f'ARIMA{best_order}',
            'model_type': 'Baseline',
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_da': test_da,
            'parameters': sum(best_order),  # p + d + q
            'optuna_params': {},
            'y_test_real': y_test_for_arima,
            'test_pred': arima_pred,
            'model_path': 'arima_baseline'
        }
    
    except Exception as e:
        print(f"❌ Error evaluando ARIMA: {e}")
        print("   💡 Usando fallback a Moving Average...")
        
        # Fallback: Moving Average simple como proxy de ARIMA
        y_test_real = data['target_scaler'].inverse_transform(data['y_test_seq'].reshape(-1, 1)).flatten()
        
        # Simple moving average (ventana de 5 días)
        window = 5
        ma_pred = []
        
        # Usar datos históricos para inicializar
        target_series = data['target_data']
        train_size = data['train_size']
        
        # Historia inicial: últimos valores del training
        history = target_series.iloc[train_size-window:train_size].tolist()
        
        for i in range(len(y_test_real)):
            # Predicción: promedio de últimos 'window' valores
            ma_pred.append(np.mean(history[-window:]))
            
            # Actualizar historia con valor real
            if i < len(y_test_real):
                history.append(y_test_real[i])
        
        ma_pred = np.array(ma_pred)
        
        test_rmse = np.sqrt(mean_squared_error(y_test_real, ma_pred))
        test_r2 = r2_score(y_test_real, ma_pred)
        
        def directional_accuracy(y_true, y_pred):
            if len(y_true) <= 1:
                return 0.5
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        
        test_da = directional_accuracy(y_test_real, ma_pred)
        
        print(f"   📊 RMSE (MA-5): {test_rmse:.6f}")
        print(f"   📊 R² (MA-5): {test_r2:.6f}")
        print(f"   📊 DA (MA-5): {test_da:.4f}")
        
        return {
            'model_name': 'Moving Average (5)',
            'model_type': 'Baseline',
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_da': test_da,
            'parameters': 1,
            'optuna_params': {},
            'y_test_real': y_test_real,
            'test_pred': ma_pred,
            'model_path': 'moving_average_baseline'
        }

def create_comparison_charts(results):
    """Crear gráficos de comparación mejorados"""
    print("\n📊 Creando gráficos de comparación...")
    
    if not results:
        print("❌ No hay resultados para graficar")
        return
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 EVALUACIÓN MODELOS OPTUNA vs BASELINES', fontsize=16, fontweight='bold')
    
    # Preparar datos
    df_results = pd.DataFrame(results)
    
    # Separar por tipo
    optuna_results = df_results[df_results['model_type'] == 'Optuna Optimized']
    baseline_results = df_results[df_results['model_type'] == 'Baseline']
    
    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    
    # Baseline bars
    if not baseline_results.empty:
        baseline_bars = ax1.bar(
            baseline_results['model_name'], 
            baseline_results['test_rmse'],
            color=['red', 'orange'],
            alpha=0.7,
            label='Baselines'
        )
        
        # Agregar valores en las barras
        for bar in baseline_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Optuna bars
    if not optuna_results.empty:
        optuna_bars = ax1.bar(
            optuna_results['model_name'], 
            optuna_results['test_rmse'],
            color='green',
            alpha=0.8,
            label='Optuna Optimized'
        )
        
        # Agregar valores en las barras
        for bar in optuna_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('📊 Test RMSE (Menor es Mejor)', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. R² Comparison
    ax2 = axes[0, 1]
    
    if not baseline_results.empty:
        baseline_bars = ax2.bar(
            baseline_results['model_name'], 
            baseline_results['test_r2'],
            color=['red', 'orange'],
            alpha=0.7,
            label='Baselines'
        )
        
        for bar in baseline_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    if not optuna_results.empty:
        optuna_bars = ax2.bar(
            optuna_results['model_name'], 
            optuna_results['test_r2'],
            color='green',
            alpha=0.8,
            label='Optuna Optimized'
        )
        
        for bar in optuna_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('📊 Test R² (Mayor es Mejor)', fontweight='bold')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Directional Accuracy
    ax3 = axes[1, 0]
    
    if not baseline_results.empty:
        baseline_bars = ax3.bar(
            baseline_results['model_name'], 
            baseline_results['test_da'] * 100,
            color=['red', 'orange'],
            alpha=0.7,
            label='Baselines'
        )
        
        for bar in baseline_bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    if not optuna_results.empty:
        optuna_bars = ax3.bar(
            optuna_results['model_name'], 
            optuna_results['test_da'] * 100,
            color='green',
            alpha=0.8,
            label='Optuna Optimized'
        )
        
        for bar in optuna_bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('🎯 Directional Accuracy (Mayor es Mejor)', fontweight='bold')
    ax3.set_ylabel('DA (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    
    # 4. Parameters vs Performance (solo Optuna)
    ax4 = axes[1, 1]
    
    if not optuna_results.empty:
        scatter = ax4.scatter(
            optuna_results['parameters'], 
            optuna_results['test_rmse'],
            c=optuna_results['test_da'] * 100,
            s=100,
            alpha=0.8,
            cmap='RdYlGn'
        )
        
        # Anotar puntos
        for idx, row in optuna_results.iterrows():
            ax4.annotate(
                row['model_name'].replace('Model', '').replace('LSTM', ''),
                (row['parameters'], row['test_rmse']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold'
            )
        
        plt.colorbar(scatter, ax=ax4, label='DA (%)')
    
    ax4.set_title('🔍 Parámetros vs RMSE (Color = DA)', fontweight='bold')
    ax4.set_xlabel('Número de Parámetros')
    ax4.set_ylabel('Test RMSE')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar - ruta adaptativa
    current_dir = Path.cwd()
    if current_dir.name == "model":
        images_dir = Path("../images/evaluacion_optuna")
    else:
        images_dir = Path("images/evaluacion_optuna")
    images_dir.mkdir(exist_ok=True)
    
    chart_path = images_dir / f"optuna_vs_baselines_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Gráfico guardado: {chart_path}")
    
    return chart_path

def main():
    """Función principal para evaluar modelos Optuna"""
    print("🎯 EVALUACIÓN MODELOS ENTRENADOS CON PARÁMETROS OPTUNA")
    print("=" * 70)
    print("🧠 Evaluando modelos con hiperparámetros optimizados por Optuna")
    print("🎲 Comparación rigurosa con baselines Naive y ARIMA")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Usando CPU")
    
    # Buscar modelos entrenados con Optuna - ruta adaptativa
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    optuna_models = list(models_dir.glob("*_optuna_*.pth"))
    
    print(f"\n🔍 Modelos Optuna encontrados: {len(optuna_models)}")
    for model_path in optuna_models:
        print(f"   📁 {model_path.name}")
    
    if not optuna_models:
        print("❌ No se encontraron modelos entrenados con parámetros Optuna")
        print("💡 Ejecuta primero: python train_all_models_optuna.py")
        return
    
    # Evaluar todos los modelos
    results = []
    
    # 1. Evaluar modelos Optuna
    print(f"\n🔄 Evaluando {len(optuna_models)} modelos Optuna...")
    for model_path in optuna_models:
        result = load_and_evaluate_optuna_model(model_path)
        if result:
            results.append(result)
    
    # 2. Evaluar baselines (usar seq_length=30 como estándar)
    print(f"\n🔄 Evaluando baselines...")
    baseline_data = prepare_data_for_evaluation(30)
    
    # Naive Baseline
    naive_result = evaluate_naive_baseline(baseline_data)
    if naive_result:
        results.append(naive_result)
    
    # ARIMA Baseline
    arima_result = evaluate_arima_baseline(baseline_data)
    if arima_result:
        results.append(arima_result)
    
    if not results:
        print("❌ No se pudieron evaluar modelos")
        return
    
    # Ordenar resultados por RMSE
    results_sorted = sorted(results, key=lambda x: x['test_rmse'])
    
    # Mostrar tabla de resultados
    print(f"\n🏆 TABLA DE RESULTADOS - OPTUNA vs BASELINES")
    print("=" * 95)
    print(f"{'Rank':<4} {'Modelo':<35} {'Tipo':<15} {'RMSE':<8} {'R²':<8} {'DA':<6} {'Params':<8}")
    print("-" * 95)
    
    for i, result in enumerate(results_sorted, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        tipo_color = "🟢" if result['model_type'] == 'Optuna Optimized' else "🔴"
        
        print(f"{emoji}{i:<3} {result['model_name']:<35} {tipo_color}{result['model_type']:<14} {result['test_rmse']:<8.6f} {result['test_r2']:<8.4f} {result['test_da']:<6.3f} {result['parameters']:<8,}")
    
    # Análisis de mejoras
    print(f"\n📈 ANÁLISIS DE MEJORAS")
    print("=" * 50)
    
    optuna_results = [r for r in results if r['model_type'] == 'Optuna Optimized']
    baseline_results = [r for r in results if r['model_type'] == 'Baseline']
    
    if optuna_results and baseline_results:
        best_optuna = min(optuna_results, key=lambda x: x['test_rmse'])
        best_baseline = min(baseline_results, key=lambda x: x['test_rmse'])
        
        improvement = (best_baseline['test_rmse'] - best_optuna['test_rmse']) / best_baseline['test_rmse'] * 100
        
        print(f"🥇 Mejor Optuna: {best_optuna['model_name']} (RMSE: {best_optuna['test_rmse']:.6f})")
        print(f"🥇 Mejor Baseline: {best_baseline['model_name']} (RMSE: {best_baseline['test_rmse']:.6f})")
        print(f"📊 Mejora: {improvement:.2f}% {'✅' if improvement > 0 else '❌'}")
        
        # Contar modelos que superan baselines
        better_than_best_baseline = sum(1 for r in optuna_results if r['test_rmse'] < best_baseline['test_rmse'])
        print(f"🎯 Modelos Optuna que superan mejor baseline: {better_than_best_baseline}/{len(optuna_results)}")
    
    # Crear gráficos
    chart_path = create_comparison_charts(results)
    
    # Guardar resumen detallado - ruta adaptativa
    current_dir = Path.cwd()
    if current_dir.name == "model":
        results_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        results_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    results_dir.mkdir(exist_ok=True)
    
    summary_path = results_dir / f"optuna_evaluation_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    # Preparar datos para JSON (sin arrays numpy)
    json_results = []
    for result in results:
        json_result = {
            'model_name': result['model_name'],
            'model_type': result['model_type'],
            'test_rmse': float(result['test_rmse']),
            'test_r2': float(result['test_r2']),
            'test_da': float(result['test_da']),
            'parameters': int(result['parameters']),
            'optuna_params': result['optuna_params'],
            'model_path': result['model_path']
        }
        json_results.append(json_result)
    
    summary_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models_evaluated': len(results),
        'optuna_models': len(optuna_results),
        'baseline_models': len(baseline_results),
        'results': json_results,
        'best_model': results_sorted[0]['model_name'],
        'best_rmse': float(results_sorted[0]['test_rmse']),
        'chart_path': str(chart_path) if 'chart_path' in locals() else None
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resumen guardado: {summary_path}")
    print(f"📊 Gráficos en: images/evaluacion_optuna/")
    print(f"\n🎯 EVALUACIÓN COMPLETADA")

if __name__ == "__main__":
    main()
