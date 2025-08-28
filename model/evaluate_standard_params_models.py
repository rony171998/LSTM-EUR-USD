#!/usr/bin/env python3
"""
Evaluación de modelos entrenados con PARÁMETROS ESTÁNDAR IDÉNTICOS.
Este script evalúa los modelos entrenados con train_all_models2.py
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Imports locales
sys.path.append('.')
from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
    ContextualLSTMTransformerFlexible
)

device = torch.device("cuda")

# MISMOS PARÁMETROS ESTÁNDAR USADOS EN EL ENTRENAMIENTO
STANDARD_PARAMS = {
    'hidden_size': 64,      # Tamaño estándar balanceado
    'dropout_prob': 0.2,    # Dropout estándar
    'epochs': 120,          # Épocas suficientes para convergencia
    'batch_size': 32,       # Batch size estándar
    'learning_rate': 0.001, # Learning rate estándar
    'num_layers': 2,        # Capas estándar
    'seq_length': 30        # Longitud de secuencia estándar
}

def get_standard_params(model_name):
    """Retorna PARÁMETROS ESTÁNDAR IDÉNTICOS para todos los modelos"""
    base_params = {
        'seq_length': STANDARD_PARAMS['seq_length'],
        'input_size': 4,   # ['returns', 'rsi', 'sma20', 'dxy_returns']
        'output_size': 1,
        'hidden_size': STANDARD_PARAMS['hidden_size'],      # IDÉNTICO para todos
        'dropout_prob': STANDARD_PARAMS['dropout_prob'],    # IDÉNTICO para todos
        'num_layers': STANDARD_PARAMS['num_layers'],        # IDÉNTICO para todos
        'forecast_horizon': 1
    }
    
    # Parámetros específicos para ContextualLSTMTransformer pero usando valores estándar
    if model_name == "ContextualLSTMTransformerFlexible":
        base_params.update({
            'window_size': 6,
            'max_neighbors': 1, 
            'lstm_units': STANDARD_PARAMS['hidden_size'],    # Usar estándar
            'num_heads': 2,
            'embed_dim': STANDARD_PARAMS['hidden_size'],     # Usar estándar
            'dropout_rate': STANDARD_PARAMS['dropout_prob']  # Usar estándar
        })
    
    return base_params

def load_standard_model(model_name, params_dict):
    """Carga un modelo con PARÁMETROS ESTÁNDAR IDÉNTICOS."""
    input_size = params_dict['input_size']
    hidden_size = params_dict['hidden_size']
    dropout_prob = params_dict['dropout_prob']
    output_size = params_dict['output_size']
    
    print(f"🔧 Cargando {model_name} con parámetros estándar:")
    print(f"   📋 Hidden Size: {hidden_size}")
    print(f"   📋 Dropout: {dropout_prob}")
    print(f"   📋 Input Size: {input_size}")
    
    if model_name == "TLS_LSTMModel":
        model = TLS_LSTMModel(input_size, hidden_size, output_size, dropout_prob)
    elif model_name == "GRU_Model":
        model = GRU_Model(input_size, hidden_size, output_size, dropout_prob, params_dict['num_layers'])
    elif model_name == "HybridLSTMAttentionModel":
        model = HybridLSTMAttentionModel(input_size, hidden_size, output_size, dropout_prob)
    elif model_name == "BidirectionalDeepLSTMModel":
        model = BidirectionalDeepLSTMModel(input_size, hidden_size, output_size, dropout_prob)
    elif model_name == "ContextualLSTMTransformerFlexible":
        model = ContextualLSTMTransformerFlexible(
            seq_len=params_dict['seq_length'],
            feature_dim=input_size,
            output_size=output_size,
            window_size=params_dict['window_size'],
            max_neighbors=params_dict['max_neighbors'],
            lstm_units=params_dict['lstm_units'],
            num_heads=params_dict['num_heads'],
            embed_dim=params_dict['embed_dim'],
            dropout_rate=params_dict['dropout_rate'],
        )
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")
    
    # Construir ruta del modelo (debe existir de train_all_models2.py)
    model_path = f"modelos/{DEFAULT_PARAMS.TABLENAME}/{model_name}_{DEFAULT_PARAMS.FILEPATH}.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Archivo del modelo no encontrado: {model_path}")
        print("   ⚠️ Asegúrate de haber ejecutado train_all_models2.py primero")
        return None
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Verificar que es un modelo con parámetros estándar
        if 'standard_params' in checkpoint:
            print(f"   ✅ Modelo entrenado con parámetros estándar confirmado")
            saved_params = checkpoint['standard_params']
            print(f"   📋 Parámetros guardados: hidden_size={saved_params['hidden_size']}, dropout={saved_params['dropout_prob']}")
        else:
            print(f"   ⚠️ Modelo sin información de parámetros estándar")
        
        # Cargar estado del modelo
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"   ✅ Modelo {model_name} cargado correctamente con parámetros estándar")
        return model, checkpoint
        
    except Exception as e:
        print(f"   ❌ Error cargando {model_name}: {e}")
        print(f"   ⚠️ Verifica que el modelo fue entrenado con train_all_models2.py")
        return None

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
    
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        mape = np.nan
    
    # Directional Accuracy
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        da = np.mean(true_direction == pred_direction) * 100
    else:
        da = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'DA': da
    }

def create_standard_comparison_plots(df_comparison, save_dir):
    """Crear gráficos comparativos de modelos con parámetros estándar + baselines."""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear directorio si no existe
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Figura principal
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colores específicos: Baselines en grises, LSTM en colores
    baseline_count = sum(1 for model in df_comparison['Modelo'] if model in ['NaiveForecast', 'ARIMA'])
    lstm_count = len(df_comparison) - baseline_count
    colors = ['lightgray', 'gray'] + ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral'][:lstm_count]
    
    # 1. RMSE por modelo
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df_comparison)), df_comparison['RMSE_Eval'], color=colors)
    ax1.set_title('RMSE: Baselines vs LSTM (Parámetros Estándar)\nMenor es Mejor', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks(range(len(df_comparison)))
    # Nombres más cortos
    short_names = []
    for name in df_comparison['Modelo']:
        if name == 'NaiveForecast':
            short_names.append('Naive')
        elif name == 'ARIMA':
            short_names.append('ARIMA')
        else:
            short_names.append(name.replace('Model', '').replace('LSTMModel', 'LSTM'))
    ax1.set_xticklabels(short_names, rotation=45, ha='right')
    
    # Agregar valores
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Línea horizontal para separar baselines de LSTM
    if baseline_count > 0:
        ax1.axvline(x=baseline_count-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(baseline_count/2-0.5, ax1.get_ylim()[1]*0.9, 'Baselines', ha='center', fontweight='bold', color='red')
        ax1.text(baseline_count + lstm_count/2-0.5, ax1.get_ylim()[1]*0.9, 'LSTM', ha='center', fontweight='bold', color='blue')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. R² por modelo
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df_comparison)), df_comparison['R2_Eval']*100, color=colors)
    ax2.set_title('R²: Baselines vs LSTM (Parámetros Estándar)\nMayor es Mejor', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² (%)', fontsize=12)
    ax2.set_xticks(range(len(df_comparison)))
    ax2.set_xticklabels(short_names, rotation=45, ha='right')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Línea separadora
    if baseline_count > 0:
        ax2.axvline(x=baseline_count-0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Directional Accuracy
    ax3 = axes[1, 0]
    # Ordenar por DA
    df_da_sorted = df_comparison.sort_values('DA_Eval', ascending=False)
    color_map = dict(zip(df_comparison['Modelo'], colors))
    da_colors = [color_map[model] for model in df_da_sorted['Modelo']]
    
    bars3 = ax3.bar(range(len(df_da_sorted)), df_da_sorted['DA_Eval'], color=da_colors)
    ax3.set_title('Directional Accuracy\nMayor es Mejor', fontsize=14, fontweight='bold')
    ax3.set_ylabel('DA (%)', fontsize=12)
    ax3.set_xticks(range(len(df_da_sorted)))
    short_names_sorted = []
    for name in df_da_sorted['Modelo']:
        if name == 'NaiveForecast':
            short_names_sorted.append('Naive')
        elif name == 'ARIMA':
            short_names_sorted.append('ARIMA')
        else:
            short_names_sorted.append(name.replace('Model', '').replace('LSTMModel', 'LSTM'))
    ax3.set_xticklabels(short_names_sorted, rotation=45, ha='right')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Scatter RMSE vs R²
    ax4 = axes[1, 1]
    
    # Separar baselines de LSTM
    baseline_mask = df_comparison['Modelo'].isin(['NaiveForecast', 'ARIMA'])
    lstm_mask = ~baseline_mask
    
    # Plot baselines
    if baseline_mask.any():
        ax4.scatter(df_comparison[baseline_mask]['RMSE_Eval'], 
                   df_comparison[baseline_mask]['R2_Eval']*100, 
                   c=['gray', 'darkgray'], s=200, alpha=0.8, marker='s', 
                   edgecolors='black', linewidth=2, label='Baselines')
    
    # Plot LSTM models
    if lstm_mask.any():
        lstm_colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue', 'purple']
        ax4.scatter(df_comparison[lstm_mask]['RMSE_Eval'], 
                   df_comparison[lstm_mask]['R2_Eval']*100, 
                   c=lstm_colors[:sum(lstm_mask)], s=200, alpha=0.7, 
                   edgecolors='black', linewidth=2, label='LSTM Models')
    
    # Añadir etiquetas
    for i, model in enumerate(df_comparison['Modelo']):
        if model == 'NaiveForecast':
            short_name = 'Naive'
        elif model == 'ARIMA':
            short_name = 'ARIMA'
        else:
            short_name = model.replace('Model', '').replace('LSTMModel', 'LSTM')
        
        ax4.annotate(f"{short_name}", 
                    (df_comparison['RMSE_Eval'].iloc[i], df_comparison['R2_Eval'].iloc[i]*100),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    ax4.set_title('RMSE vs R² Score\nEsquina Superior Izquierda = Mejor', fontsize=14, fontweight='bold')
    ax4.set_xlabel('RMSE (menor mejor)', fontsize=12)
    ax4.set_ylabel('R² Score (%) (mayor mejor)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Añadir información de parámetros estándar
    fig.suptitle(f'BASELINES vs LSTM (PARÁMETROS ESTÁNDAR IDÉNTICOS)\n' + 
                 f'LSTM: Hidden Size: {STANDARD_PARAMS["hidden_size"]} | ' +
                 f'Dropout: {STANDARD_PARAMS["dropout_prob"]} | ' +
                 f'Epochs: {STANDARD_PARAMS["epochs"]}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Guardar
    plot_path = Path(save_dir) / 'standard_params_vs_baselines_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Gráfico comparativo guardado en: {plot_path}")
    
    # Mostrar gráfico
    plt.show()
    
    return plot_path

def main():
    print("🔍 EVALUACIÓN DE MODELOS CON PARÁMETROS ESTÁNDAR + BASELINES")
    print("=" * 70)
    print("📋 PARÁMETROS ESTÁNDAR UTILIZADOS EN EL ENTRENAMIENTO:")
    print(f"   Hidden Size: {STANDARD_PARAMS['hidden_size']}")
    print(f"   Dropout: {STANDARD_PARAMS['dropout_prob']}")
    print(f"   Epochs: {STANDARD_PARAMS['epochs']}")
    print(f"   Batch Size: {STANDARD_PARAMS['batch_size']}")
    print(f"   Learning Rate: {STANDARD_PARAMS['learning_rate']}")
    print(f"   Sequence Length: {STANDARD_PARAMS['seq_length']}")
    print("🎲 Evaluando modelos entrenados con train_all_models2.py")
    print("📊 Incluyendo baselines Naive y ARIMA para comparación completa")
    print("=" * 70)
    
    # Lista de modelos entrenados con parámetros estándar
    models_to_evaluate = [
        "TLS_LSTMModel",
        "GRU_Model",
        "HybridLSTMAttentionModel",
        "BidirectionalDeepLSTMModel",
        "ContextualLSTMTransformerFlexible"
    ]
    
    # Cargar y preparar datos exactamente como en train_all_models2.py
    print("📊 Cargando datos...")
    
    # Importar funciones del entrenamiento
    from train_all_models2 import load_multi_asset_data, create_proven_features
    
    # Cargar datos multi-asset (EUR/USD + DXY)
    eur_prices, dxy_prices = load_multi_asset_data()
    
    # Crear features exactamente como en entrenamiento
    features_dict = create_proven_features(eur_prices, dxy_prices)
    
    # Crear DataFrame con las features exactas
    df_features = pd.DataFrame(features_dict)
    df_features = df_features.dropna()
    
    print(f"✅ Datos cargados: {len(df_features)} muestras")
    print(f"📈 Features: {list(df_features.columns)}")
    
    # Usar target y features exactas del entrenamiento
    target_column = 'price'
    feature_columns = ['returns', 'rsi', 'sma20', 'dxy_returns']
    
    # Verificar features disponibles
    available_features = [col for col in feature_columns if col in df_features.columns]
    print(f"📊 Features disponibles: {available_features}")
    
    # Preparar datos finales
    X_data = df_features[available_features].values
    y_data = df_features[target_column].values
    
    # Split temporal (80/20) 
    split_index = int(len(df_features) * 0.8)
    
    X_train = X_data[:split_index]
    X_test = X_data[split_index:]
    y_train = y_data[:split_index]
    y_test = y_data[split_index:]
    
    print(f"✅ Split: Train {X_train.shape}, Test {X_test.shape}")
    
    # Escalado (misma configuración que entrenamiento)
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    
    # Escalador del target
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Crear secuencias con SEQ_LENGTH estándar
    def create_sequences_simple(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences_simple(train_scaled, y_train_scaled, STANDARD_PARAMS['seq_length'])
    X_test_seq, y_test_seq = create_sequences_simple(test_scaled, y_test_scaled, STANDARD_PARAMS['seq_length'])
    
    print(f"✅ Secuencias creadas: Train {X_train_seq.shape}, Test {X_test_seq.shape}")
    
    # Convertir a tensores
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    
    # Evaluar cada modelo
    results = []
    
    # Primero agregar baselines (Naive y ARIMA) con métricas conocidas
    print(f"\n📊 AGREGANDO BASELINES...")
    print("-" * 50)
    
    # Naive Baseline
    print("📈 Naive Forecast Baseline")
    naive_result = {
        'model_name': 'NaiveForecast',
        'metrics': {
            'MSE': 2.5e-05,
            'RMSE': 0.005025,
            'MAE': 0.00379,
            'R2': 0.976684,
            'MAPE': 0.35732,
            'DA': 50.0
        },
        'training_rmse': 0.005025,
        'training_r2': 0.976684,
        'training_da': 0.50,
        'parameters': 0  # No parámetros
    }
    results.append(naive_result)
    print(f"   ✅ RMSE: {naive_result['metrics']['RMSE']:.6f}")
    print(f"   ✅ R²: {naive_result['metrics']['R2']:.6f}")
    print(f"   ✅ DA: {naive_result['metrics']['DA']:.2f}%")
    
    # ARIMA Baseline
    print("\n📈 ARIMA Baseline")
    arima_result = {
        'model_name': 'ARIMA',
        'metrics': {
            'MSE': 2.6e-05,
            'RMSE': 0.005063,
            'MAE': 0.00382,
            'R2': 0.976333,
            'MAPE': 0.360156,
            'DA': 50.449102
        },
        'training_rmse': 0.005063,
        'training_r2': 0.976333,
        'training_da': 0.50449102,
        'parameters': 0  # No parámetros
    }
    results.append(arima_result)
    print(f"   ✅ RMSE: {arima_result['metrics']['RMSE']:.6f}")
    print(f"   ✅ R²: {arima_result['metrics']['R2']:.6f}")
    print(f"   ✅ DA: {arima_result['metrics']['DA']:.2f}%")
    
    # Ahora evaluar modelos LSTM con parámetros estándar
    
    
    for model_name in models_to_evaluate:
        print(f"\n🚀 Evaluando {model_name}...")
        print("-" * 50)
        
        # Obtener parámetros estándar
        params_dict = get_standard_params(model_name)
        
        # Cargar modelo con parámetros estándar
        model_result = load_standard_model(model_name, params_dict)
        if model_result is None:
            continue
            
        model, checkpoint = model_result
        
        # Información del entrenamiento
        if 'test_rmse' in checkpoint:
            print(f"   📈 RMSE entrenamiento: {checkpoint['test_rmse']:.6f}")
        if 'test_r2' in checkpoint:
            print(f"   📈 R² entrenamiento: {checkpoint['test_r2']:.6f}")
        if 'test_da' in checkpoint:
            print(f"   📈 DA entrenamiento: {checkpoint['test_da']:.4f}")
        
        # Verificar que el RMSE coincide (validación)
        if 'test_rmse' in checkpoint:
            training_rmse = checkpoint['test_rmse']
            print(f"   🔍 Verificando reproducibilidad...")
        
        # Predicción
        try:
            with torch.no_grad():
                predictions = model(X_test_tensor)
                if len(predictions.shape) > 1:
                    predictions = predictions.squeeze()
                predictions_np = predictions.cpu().numpy()
            
            # Desnormalizar
            y_test_original = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            predictions_original = target_scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()
            
            # Calcular métricas
            metrics = calculate_metrics(y_test_original, predictions_original)
            
            print("   📊 Métricas de evaluación:")
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    print(f"      {metric_name}: {value:.6f}")
                else:
                    print(f"      {metric_name}: N/A")
            
            # Verificar reproducibilidad
            if 'test_rmse' in checkpoint:
                rmse_diff = abs(metrics['RMSE'] - training_rmse)
                if rmse_diff < 1e-6:
                    print(f"   ✅ RMSE reproducible: {rmse_diff:.2e} diferencia")
                else:
                    print(f"   ⚠️ RMSE diferente: {rmse_diff:.6f} diferencia")
            
            # Guardar resultado
            result = {
                'model_name': model_name,
                'metrics': metrics,
                'training_rmse': checkpoint.get('test_rmse', 'N/A'),
                'training_r2': checkpoint.get('test_r2', 'N/A'),
                'training_da': checkpoint.get('test_da', 'N/A'),
                'parameters': sum(p.numel() for p in model.parameters())
            }
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error en evaluación: {e}")
            continue
    
    # Crear tabla comparativa
    print(f"\n🏆 RESUMEN COMPARATIVO - BASELINES vs LSTM (PARÁMETROS ESTÁNDAR)")
    print("=" * 80)
    
    if results:
        # Crear DataFrame
        df_results = []
        for result in results:
            metrics = result['metrics']
            df_results.append({
                'Modelo': result['model_name'],
                'RMSE_Eval': metrics['RMSE'],
                'R2_Eval': metrics['R2'],
                'DA_Eval': metrics['DA'],
                'RMSE_Train': result['training_rmse'],
                'R2_Train': result['training_r2'],
                'DA_Train': result['training_da'],
                'Parámetros': result['parameters']
            })
        
        df_comparison = pd.DataFrame(df_results)
        
        # Ordenar por RMSE de evaluación
        df_comparison = df_comparison.sort_values('RMSE_Eval')
        
        print(df_comparison.to_string(index=False, float_format='%.6f'))
        
        # Guardar resultados
        os.makedirs(f'../images/comparacion/{DEFAULT_PARAMS.TABLENAME}', exist_ok=True)
        csv_path = f'../images/comparacion/{DEFAULT_PARAMS.TABLENAME}/standard_params_vs_baselines_comparison.csv'
        df_comparison.to_csv(csv_path, index=False)
        
        print(f"\n💾 Resultados guardados en: {csv_path}")
        
        # Crear gráficos comparativos
        print(f"\n🎨 Generando gráficos comparativos...")
        plot_path = create_standard_comparison_plots(df_comparison, f'../images/comparacion/{DEFAULT_PARAMS.TABLENAME}')
        
        # Mostrar el mejor modelo
        best_model = df_comparison.iloc[0]
        print(f"\n🥇 MEJOR MODELO (Incluyendo Baselines): {best_model['Modelo']}")
        print(f"   📊 RMSE Evaluación: {best_model['RMSE_Eval']:.6f}")
        print(f"   📊 R² Evaluación: {best_model['R2_Eval']:.6f}")
        print(f"   📊 DA Evaluación: {best_model['DA_Eval']:.4f}%")
        print(f"   ⚙️  Parámetros: {best_model['Parámetros']:,}")
        
        # Mostrar mejor LSTM específicamente
        lstm_models = df_comparison[~df_comparison['Modelo'].isin(['NaiveForecast', 'ARIMA'])]
        if not lstm_models.empty:
            best_lstm = lstm_models.iloc[0]
            print(f"\n🥇 MEJOR MODELO LSTM (Parámetros Estándar): {best_lstm['Modelo']}")
            print(f"   📊 RMSE Evaluación: {best_lstm['RMSE_Eval']:.6f}")
            print(f"   📊 R² Evaluación: {best_lstm['R2_Eval']:.6f}")
            print(f"   📊 DA Evaluación: {best_lstm['DA_Eval']:.4f}%")
            print(f"   ⚙️  Parámetros: {best_lstm['Parámetros']:,}")
        
        print(f"\n✅ VENTAJA: Comparación completa - Baselines vs LSTM con parámetros idénticos")
        print(f"📋 LSTM Hidden Size: {STANDARD_PARAMS['hidden_size']} | Dropout: {STANDARD_PARAMS['dropout_prob']} | Epochs: {STANDARD_PARAMS['epochs']}")
    
    else:
        print("❌ No se pudo evaluar ningún modelo.")
        print("⚠️ Asegúrate de haber ejecutado train_all_models2.py primero")
    
    print(f"\n✅ Evaluación completa: Baselines vs LSTM con parámetros estándar completada!")

if __name__ == "__main__":
    main()
