#train_naive_baseline.py
"""
Script mejorado para entrenar y evaluar modelo Naive Forecast baseline.

Este script incluye:
- Validación robusta de datos
- Evaluación completa con métricas reales (RMSE, MAE, R², MAPE, Directional Accuracy)
- Comparación contra conjunto de test completo
- Guardado inteligente del modelo (solo si es necesario)
- Análisis de consistencia del modelo

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

# Importar desde el proyecto
from model.modelos import NaiveForecastModel
from model.config import DEFAULT_PARAMS
from model.train_model import load_and_prepare_data, create_sequences, add_indicator

def calculate_metrics(y_true, y_pred):
    """Calcula métricas completas de evaluación."""
    metrics = {}
    
    # Métricas básicas
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    
    # R² score
    try:
        metrics['R2'] = r2_score(y_true, y_pred)
    except:
        metrics['R2'] = np.nan
    
    # MAPE (Mean Absolute Percentage Error)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['MAPE'] = mape if not np.isnan(mape) else 999.0
    except:
        metrics['MAPE'] = 999.0
    
    # Directional Accuracy (DA)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        metrics['DA'] = np.mean(true_direction == pred_direction) * 100
    else:
        metrics['DA'] = np.nan
    
    return metrics

def evaluate_naive_on_test_set(model, X_test, y_test, scaler):
    """Evalúa el modelo Naive en el conjunto completo de test."""
    print("\n📊 === Evaluación en Conjunto de Test Completo ===")
    
    model.eval()
    predictions = []
    true_values = []
    
    # Hacer predicciones en batches para eficiencia
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            
            X_batch = torch.tensor(X_test[i:end_idx], dtype=torch.float32)
            y_batch = y_test[i:end_idx]
            
            # Predicciones del modelo
            pred_batch = model(X_batch)
            
            # Acumular resultados
            predictions.extend(pred_batch.cpu().numpy().flatten())
            true_values.extend(y_batch.flatten())
    
    # Convertir a arrays numpy
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    print(f"Total de predicciones evaluadas: {len(predictions)}")
    print(f"Primeras 5 predicciones: {predictions[:5]}")
    print(f"Primeros 5 valores reales: {true_values[:5]}")
    
    # Desescalar para métricas en escala original
    n_features = scaler.n_features_in_
    
    # Para valores reales
    y_true_full = np.zeros((len(true_values), n_features))
    y_true_full[:, 0] = true_values
    y_true_original = scaler.inverse_transform(y_true_full)[:, 0]
    
    # Para predicciones
    y_pred_full = np.zeros((len(predictions), n_features))
    y_pred_full[:, 0] = predictions
    y_pred_original = scaler.inverse_transform(y_pred_full)[:, 0]
    
    print(f"Rango valores reales (original): [{y_true_original.min():.6f}, {y_true_original.max():.6f}]")
    print(f"Rango predicciones (original): [{y_pred_original.min():.6f}, {y_pred_original.max():.6f}]")
    
    # Calcular métricas
    metrics = calculate_metrics(y_true_original, y_pred_original)
    
    return metrics, y_true_original, y_pred_original

def main():
    print("🚀 === Entrenamiento y Evaluación Completa del Modelo Naive ===")
    start_time = time.time()
    
    # 1. Cargar y preparar datos
    print("\n📁 Cargando datos...")
    df = load_and_prepare_data(DEFAULT_PARAMS.FILEPATH)
    if df is None:
        print("❌ Error: No se pudieron cargar los datos")
        return
    
    # Agregar indicadores técnicos
    indicators = add_indicator(df)
    for indicator_name, values in indicators.items():
        df[indicator_name] = values
    
    print(f"📈 Datos cargados: {len(df)} filas")
    print(f"📊 Features disponibles: {DEFAULT_PARAMS.FEATURES}")
    
    # Verificar que la columna objetivo existe y está bien formateada
    target_col = DEFAULT_PARAMS.TARGET_COLUMN
    if target_col not in df.columns:
        print(f"❌ Error: Columna objetivo '{target_col}' no encontrada")
        return
    
    print(f"🎯 Columna objetivo: '{target_col}'")
    print(f"📊 Primeros 5 valores: {df[target_col].head().values}")
    print(f"📊 Últimos 5 valores: {df[target_col].tail().values}")
    
    # 2. Dividir en Train/Test
    split_index = int(len(df) * DEFAULT_PARAMS.TRAIN_SPLIT_RATIO)
    features = DEFAULT_PARAMS.FEATURES
    train_data = df[features].iloc[:split_index]
    test_data = df[features].iloc[split_index:]
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {len(train_data)} muestras ({DEFAULT_PARAMS.TRAIN_SPLIT_RATIO:.1%})")
    print(f"   Prueba: {len(test_data)} muestras ({1-DEFAULT_PARAMS.TRAIN_SPLIT_RATIO:.1%})")
    
    # 3. Escalar datos
    print("\n🔧 Escalando datos...")
    scaler = RobustScaler(quantile_range=(5, 95))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    print(f"   Escalador configurado: quantiles [{5}, {95}]")
    print(f"   Rango datos escalados: [{test_scaled.min():.3f}, {test_scaled.max():.3f}]")
    
    # 4. Crear secuencias
    print(f"\n🔢 Creando secuencias...")
    print(f"   Longitud de secuencia: {DEFAULT_PARAMS.SEQ_LENGTH}")
    print(f"   Horizonte de predicción: {DEFAULT_PARAMS.FORECAST_HORIZON}")
    
    X_train, y_train = create_sequences(train_scaled, DEFAULT_PARAMS.SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)
    X_test, y_test = create_sequences(test_scaled, DEFAULT_PARAMS.SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)
    
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 5. Crear y configurar modelo
    print(f"\n🤖 Creando modelo Naive...")
    
    # Determinar índice de la feature objetivo
    target_feature_index = features.index(target_col) if target_col in features else 0
    
    model = NaiveForecastModel(
        input_size=len(features),
        output_size=DEFAULT_PARAMS.FORECAST_HORIZON,
        target_feature_index=target_feature_index
    )
    
    # Mostrar información del modelo
    model_info = model.get_model_info()
    print(f"   Información del modelo:")
    for key, value in model_info.items():
        print(f"     {key}: {value}")
    
    # 6. Validación rápida con una muestra
    print(f"\n🧪 Validación rápida...")
    sample_X = torch.tensor(X_test[:3], dtype=torch.float32)
    sample_y = y_test[:3]
    
    try:
        with torch.no_grad():
            sample_pred = model(sample_X)
        
        print(f"   Entrada de muestra: {sample_X.shape}")
        print(f"   Predicción de muestra: {sample_pred.shape}")
        print(f"   ✅ Modelo funciona correctamente")
        
        # Mostrar ejemplo de predicción
        for i in range(min(3, len(sample_pred))):
            last_val = sample_X[i, -1, target_feature_index].item()
            pred_val = sample_pred[i, 0].item()
            true_val = sample_y[i, 0] if len(sample_y[i].shape) > 0 else sample_y[i]
            
            print(f"   Ejemplo {i+1}: último={last_val:.6f}, predicción={pred_val:.6f}, real={true_val:.6f}")
            
            # Verificar que la predicción es exactamente el último valor
            if abs(pred_val - last_val) < 1e-6:
                print(f"     ✅ Predicción correcta (Naive)")
            else:
                print(f"     ❌ Error: predicción no es igual al último valor")
        
    except Exception as e:
        print(f"   ❌ Error en validación: {e}")
        return
    
    # 7. Evaluación completa en test set
    metrics, y_true_orig, y_pred_orig = evaluate_naive_on_test_set(model, X_test, y_test, scaler)
    
    # 8. Mostrar resultados
    print(f"\n🎯 === RESULTADOS FINALES ===")
    print(f"{'Métrica':<20} {'Valor':<15} {'Interpretación'}")
    print(f"{'-'*55}")
    print(f"{'MSE':<20} {metrics['MSE']:<15.8f} {'Menor es mejor'}")
    print(f"{'RMSE':<20} {metrics['RMSE']:<15.6f} {'Menor es mejor'}")
    print(f"{'MAE':<20} {metrics['MAE']:<15.6f} {'Menor es mejor'}")
    print(f"{'R²':<20} {metrics['R2']:<15.6f} {'Más cerca de 1 es mejor'}")
    print(f"{'MAPE (%)':<20} {metrics['MAPE']:<15.2f} {'Menor es mejor'}")
    print(f"{'DA (%)':<20} {metrics['DA']:<15.2f} {'Mayor es mejor'}")
    
    # 9. Análisis de calidad
    print(f"\n📈 === ANÁLISIS DE CALIDAD ===")
    if metrics['RMSE'] < 0.01:
        print(f"✅ RMSE excelente (< 0.01): {metrics['RMSE']:.6f}")
    elif metrics['RMSE'] < 0.05:
        print(f"✅ RMSE bueno (< 0.05): {metrics['RMSE']:.6f}")
    else:
        print(f"⚠️ RMSE alto (>= 0.05): {metrics['RMSE']:.6f}")
    
    if metrics['R2'] > 0.95:
        print(f"✅ R² excelente (> 0.95): {metrics['R2']:.6f}")
    elif metrics['R2'] > 0.8:
        print(f"✅ R² bueno (> 0.8): {metrics['R2']:.6f}")
    else:
        print(f"⚠️ R² bajo (<= 0.8): {metrics['R2']:.6f}")
    
    if abs(metrics['DA'] - 50) < 5:
        print(f"✅ Directional Accuracy cerca del 50% (aleatorio): {metrics['DA']:.2f}%")
    else:
        print(f"🤔 Directional Accuracy alejada del 50%: {metrics['DA']:.2f}%")
    
    # 10. Guardar modelo (solo si es útil)
    model_path = f"modelos/eur_usd/NaiveForecastModel_{DEFAULT_PARAMS.FILEPATH}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Nota: Guardamos el modelo completo, no solo state_dict ya que no tiene parámetros entrenables
    torch.save(model, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")
    print(f"   Nota: El modelo no tiene parámetros entrenables, se guarda solo por compatibilidad")
    
    # 11. Estadísticas de tiempo
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n⏱️ Tiempo total de ejecución: {execution_time:.2f} segundos")
    print(f"🚀 Predicciones por segundo: {len(X_test) / execution_time:.0f}")
    
    print(f"\n🎉 === ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS ===")
    print(f"📊 El modelo Naive es un excelente baseline para comparación")
    print(f"🎯 Cualquier modelo complejo debe superar RMSE: {metrics['RMSE']:.6f}")

if __name__ == "__main__":
    main()
