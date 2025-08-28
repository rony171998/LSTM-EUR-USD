#!/usr/bin/env python3
"""
Evaluación simplificada de modelos reproducibles.
Evalúa solo los modelos entrenados con configuraciones exactas y semilla fija.
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
from train_model import load_and_prepare_data, create_sequences, add_indicator

device = torch.device("cuda")

def get_model_specific_params(model_name):
    """Retorna parámetros específicos de cada modelo usados en el entrenamiento reproducible"""
    base_params = {
        'seq_length': 30,  # Usado en entrenamiento reproducible
        'input_size': 4,   # ['returns', 'rsi', 'sma20', 'dxy_returns']
        'output_size': 1,
        'forecast_horizon': 1
    }
    
    if model_name == "TLS_LSTMModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2}
    elif model_name == "GRU_Model":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2, 'num_layers': 2}
    elif model_name == "HybridLSTMAttentionModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2}
    elif model_name == "BidirectionalDeepLSTMModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2, 'num_layers': 2}
    elif model_name == "ContextualLSTMTransformerFlexible":
        return {
            **base_params,
            'window_size': 6,
            'max_neighbors': 1, 
            'lstm_units': 32,
            'num_heads': 2,
            'embed_dim': 64,
            'dropout_rate': 0.2
        }
    else:
        return base_params

def load_reproducible_model(model_name, params_dict):
    """Carga un modelo con parámetros reproducibles específicos."""
    input_size = params_dict['input_size']
    
    if model_name == "TLS_LSTMModel":
        model = TLS_LSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    elif model_name == "TLS_LSTMModel_Optimizado":
        model = TLS_LSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    elif model_name == "GRU_Model":
        model = GRU_Model(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    elif model_name == "HybridLSTMAttentionModel":
        model = HybridLSTMAttentionModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    elif model_name == "BidirectionalDeepLSTMModel":
        model = BidirectionalDeepLSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    elif model_name == "ContextualLSTMTransformerFlexible":
        model = ContextualLSTMTransformerFlexible(
            seq_len=params_dict['seq_length'],
            feature_dim=input_size,
            output_size=params_dict['output_size'],
            window_size=params_dict['window_size'],
            max_neighbors=params_dict['max_neighbors'],
            lstm_units=params_dict['lstm_units'],
            num_heads=params_dict['num_heads'],
            embed_dim=params_dict['embed_dim'],
            dropout_rate=params_dict['dropout_rate'],
        )
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")
    
    # Construir ruta del modelo
    model_path = f"modelos/{DEFAULT_PARAMS.TABLENAME}/{model_name}_{DEFAULT_PARAMS.FILEPATH}.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Archivo del modelo no encontrado: {model_path}")
        return None
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Cargar estado del modelo
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Modelo {model_name} cargado correctamente")
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ Error cargando {model_name}: {e}")
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

def create_comparison_plots(df_comparison, save_dir):
    """Crear gráficos comparativos de los modelos reproducibles."""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear directorio si no existe
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Figura con múltiples subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Gráfico de barras RMSE
    plt.subplot(3, 3, 1)
    bars1 = plt.bar(range(len(df_comparison)), df_comparison['RMSE_Eval'], 
                    color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral'])
    plt.title('RMSE por Modelo\n(Menor es Mejor)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(range(len(df_comparison)), df_comparison['Modelo'], rotation=45, ha='right')
    
    # Agregar valores en las barras
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Destacar los mejores 3
    bars1[0].set_color('gold')
    bars1[1].set_color('silver') 
    bars1[2].set_color('#CD7F32')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 2. Gráfico de barras R²
    plt.subplot(3, 3, 2)
    bars2 = plt.bar(range(len(df_comparison)), df_comparison['R2_Eval']*100, 
                    color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral'])
    plt.title('R² Score por Modelo\n(Mayor es Mejor)', fontsize=14, fontweight='bold')
    plt.ylabel('R² (%)', fontsize=12)
    plt.xticks(range(len(df_comparison)), df_comparison['Modelo'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    bars2[0].set_color('gold')
    bars2[1].set_color('silver')
    bars2[2].set_color('#CD7F32')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 3. Gráfico de barras DA (Directional Accuracy)
    plt.subplot(3, 3, 3)
    # Ordenar por DA para este gráfico
    df_da_sorted = df_comparison.sort_values('DA_Eval', ascending=False)
    bars3 = plt.bar(range(len(df_da_sorted)), df_da_sorted['DA_Eval'], 
                    color=['darkgreen', 'green', 'limegreen', 'yellow', 'orange', 'red'])
    plt.title('Directional Accuracy por Modelo\n(Mayor es Mejor)', fontsize=14, fontweight='bold')
    plt.ylabel('DA (%)', fontsize=12)
    plt.xticks(range(len(df_da_sorted)), df_da_sorted['Modelo'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 4. Scatter plot RMSE vs R²
    plt.subplot(3, 3, 4)
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue', 'purple']
    scatter = plt.scatter(df_comparison['RMSE_Eval'], df_comparison['R2_Eval']*100, 
                         c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Añadir etiquetas a cada punto
    for i, model in enumerate(df_comparison['Modelo']):
        plt.annotate(f"{i+1}. {model.replace('Model', '')}", 
                    (df_comparison['RMSE_Eval'].iloc[i], df_comparison['R2_Eval'].iloc[i]*100),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    plt.title('RMSE vs R² Score\n(Esquina Superior Izquierda = Mejor)', fontsize=14, fontweight='bold')
    plt.xlabel('RMSE (menor mejor)', fontsize=12)
    plt.ylabel('R² Score (%) (mayor mejor)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 5. Gráfico de barras horizontal - Número de parámetros
    plt.subplot(3, 3, 5)
    y_pos = np.arange(len(df_comparison))
    bars5 = plt.barh(y_pos, df_comparison['Parámetros']/1000, 
                     color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Numero de Parametros por Modelo\n(Miles)', fontsize=14, fontweight='bold')
    plt.xlabel('Parametros (miles)', fontsize=12)
    plt.yticks(y_pos, df_comparison['Modelo'])
    
    for i, bar in enumerate(bars5):
        width = bar.get_width()
        plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(df_comparison["Parámetros"].iloc[i]/1000)}K', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    
    # 6. Comparación Entrenamiento vs Evaluación - RMSE
    plt.subplot(3, 3, 6)
    x = np.arange(len(df_comparison))
    width = 0.35
    
    bars_train = plt.bar(x - width/2, df_comparison['RMSE_Train'], width, 
                        label='Entrenamiento', color='lightblue', alpha=0.8)
    bars_eval = plt.bar(x + width/2, df_comparison['RMSE_Eval'], width,
                       label='Evaluación', color='darkblue', alpha=0.8)
    
    plt.title('RMSE: Entrenamiento vs Evaluacion', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(x, df_comparison['Modelo'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 7. Comparación Entrenamiento vs Evaluación - R²
    plt.subplot(3, 3, 7)
    bars_train_r2 = plt.bar(x - width/2, df_comparison['R2_Train']*100, width,
                           label='Entrenamiento', color='lightgreen', alpha=0.8)
    bars_eval_r2 = plt.bar(x + width/2, df_comparison['R2_Eval']*100, width,
                          label='Evaluación', color='darkgreen', alpha=0.8)
    
    plt.title('R²: Entrenamiento vs Evaluacion', fontsize=14, fontweight='bold')
    plt.ylabel('R² (%)', fontsize=12)
    plt.xticks(x, df_comparison['Modelo'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 8. Comparación Entrenamiento vs Evaluación - DA
    plt.subplot(3, 3, 8)
    bars_train_da = plt.bar(x - width/2, df_comparison['DA_Train']*100, width,
                           label='Entrenamiento', color='orange', alpha=0.8)
    bars_eval_da = plt.bar(x + width/2, df_comparison['DA_Eval'], width,
                          label='Evaluación', color='darkorange', alpha=0.8)
    
    plt.title('DA: Entrenamiento vs Evaluacion', fontsize=14, fontweight='bold')
    plt.ylabel('DA (%)', fontsize=12)
    plt.xticks(x, df_comparison['Modelo'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 9. Tabla resumen
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Crear tabla con los mejores modelos
    table_data = []
    medals = ['🥇', '🥈', '🥉']
    for idx, (i, row) in enumerate(df_comparison.head(3).iterrows()):
        medal = medals[idx]
        table_data.append([
            f"{medal} {row['Modelo'].replace('Model', '')}",
            f"{row['RMSE_Eval']:.6f}",
            f"{row['R2_Eval']*100:.2f}%",
            f"{row['DA_Eval']:.2f}%",
            f"{row['Parámetros']:,}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['TOP Modelo', 'RMSE', 'R²', 'DA', 'Params'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colorear filas
    for i in range(3):
        for j in range(5):
            if i == 0:  # Oro
                table[(i+1, j)].set_facecolor('#FFD700')
            elif i == 1:  # Plata
                table[(i+1, j)].set_facecolor('#C0C0C0')
            else:  # Bronce
                table[(i+1, j)].set_facecolor('#CD7F32')
    
    plt.title('TOP 3 MODELOS REPRODUCIBLES', fontsize=14, fontweight='bold', pad=20)
    
    # Ajustar layout general
    plt.tight_layout()
    
    # Guardar gráfico
    plot_path = Path(save_dir) / 'reproducible_models_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Gráfico comparativo guardado en: {plot_path}")
    
    # Mostrar gráfico
    plt.show()
    
    return plot_path

def main():
    print("🔍 EVALUACIÓN DE MODELOS REPRODUCIBLES")
    print("=" * 60)
    print("📋 Configuraciones exactas del entrenamiento aplicadas")
    print("🎲 Evaluando modelos entrenados con semilla fija (seed=42)")
    print("⚙️  Parámetros de arquitectura específicos por modelo")
    print("")
    
    # Lista de modelos entrenados con reproducibilidad
    models_to_evaluate = [
        "TLS_LSTMModel",
        "TLS_LSTMModel_Optimizado", 
        "GRU_Model",
        "HybridLSTMAttentionModel",
        "BidirectionalDeepLSTMModel",
        "ContextualLSTMTransformerFlexible"
    ]
    
    # Cargar y preparar datos exactamente como en el entrenamiento reproducible
    print("📊 Cargando datos...")
    
    # Importar funciones específicas del entrenamiento reproducible
    from train_all_models import load_multi_asset_data, create_proven_features
    
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
    target_column = 'price'  # Target original
    feature_columns = ['returns', 'rsi', 'sma20', 'dxy_returns']
    
    # Verificar que todas las features están disponibles
    available_features = [col for col in feature_columns if col in df_features.columns]
    print(f"📊 Features disponibles: {available_features}")
    
    # Preparar datos finales
    X_data = df_features[available_features].values
    y_data = df_features[target_column].values
    
    # Split temporal (80/20) usando los mismos datos del entrenamiento
    split_index = int(len(df_features) * DEFAULT_PARAMS.TRAIN_SPLIT_RATIO)
    
    # Datos de entrenamiento y test
    X_train = X_data[:split_index]
    X_test = X_data[split_index:]
    y_train = y_data[:split_index]
    y_test = y_data[split_index:]
    
    print(f"✅ Split: Train {X_train.shape}, Test {X_test.shape}")
    
    # Escalado (misma configuración que entrenamiento)
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    
    # Escalador del target para desnormalizar
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Crear secuencias (SEQ_LENGTH = 30 como en entrenamiento reproducible)
    REPRODUCIBLE_SEQ_LENGTH = 30
    
    def create_sequences_simple(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences_simple(train_scaled, y_train_scaled, REPRODUCIBLE_SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences_simple(test_scaled, y_test_scaled, REPRODUCIBLE_SEQ_LENGTH)
    
    print(f"✅ Secuencias creadas: Train {X_train_seq.shape}, Test {X_test_seq.shape}")
    
    # Convertir a tensores
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    
    # Evaluar cada modelo
    results = []
    
    for model_name in models_to_evaluate:
        print(f"\n🚀 Evaluando {model_name}...")
        print("-" * 40)
        
        # Obtener parámetros específicos
        params_dict = get_model_specific_params(model_name)
        print(f"📋 Configuración: {params_dict}")
        
        # Cargar modelo
        model_result = load_reproducible_model(model_name, params_dict)
        if model_result is None:
            continue
            
        model, checkpoint = model_result
        
        # Información del entrenamiento
        if 'test_rmse' in checkpoint:
            print(f"📈 RMSE entrenamiento: {checkpoint['test_rmse']:.6f}")
        if 'test_r2' in checkpoint:
            print(f"📈 R² entrenamiento: {checkpoint['test_r2']:.6f}")
        if 'test_da' in checkpoint:
            print(f"📈 DA entrenamiento: {checkpoint['test_da']:.4f}")
        
        # Predicción
        try:
            with torch.no_grad():
                predictions = model(X_test_tensor)
                if len(predictions.shape) > 1:
                    predictions = predictions.squeeze()
                predictions_np = predictions.cpu().numpy()
            
            # Desnormalizar usando el target_scaler ya ajustado
            y_test_original = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            predictions_original = target_scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()
            
            # Calcular métricas
            metrics = calculate_metrics(y_test_original, predictions_original)
            
            print("📊 Métricas de evaluación:")
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    print(f"   {metric_name}: {value:.6f}")
                else:
                    print(f"   {metric_name}: N/A")
            
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
            print(f"❌ Error en evaluación: {e}")
            continue
    
    # Crear tabla comparativa
    print(f"\n🏆 RESUMEN COMPARATIVO - MODELOS REPRODUCIBLES")
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
        os.makedirs(f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}', exist_ok=True)
        csv_path = f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}/reproducible_models_comparison.csv'
        df_comparison.to_csv(csv_path, index=False)
        
        print(f"\n💾 Resultados guardados en: {csv_path}")
        
        # Crear gráficos comparativos
        print(f"\n🎨 Generando gráficos comparativos...")
        plot_path = create_comparison_plots(df_comparison, f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}')
        
        # Mostrar el mejor modelo
        best_model = df_comparison.iloc[0]
        print(f"\n🥇 MEJOR MODELO: {best_model['Modelo']}")
        print(f"   📊 RMSE Evaluación: {best_model['RMSE_Eval']:.6f}")
        print(f"   📊 R² Evaluación: {best_model['R2_Eval']:.6f}")
        print(f"   📊 DA Evaluación: {best_model['DA_Eval']:.4f}%")
        print(f"   ⚙️  Parámetros: {best_model['Parámetros']:,}")
    
    else:
        print("❌ No se pudo evaluar ningún modelo.")
    
    print(f"\n✅ Evaluación de modelos reproducibles completada!")

if __name__ == "__main__":
    main()
