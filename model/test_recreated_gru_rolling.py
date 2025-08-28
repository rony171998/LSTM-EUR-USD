#!/usr/bin/env python3
"""
test_recreated_gru_rolling.py - Aplicar Rolling Forecast al modelo GRU recreado
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import DEFAULT_PARAMS
from modelos import GRU_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_recreated_gru_model():
    """Cargar el modelo GRU recién recreado"""
    print("🔍 Buscando modelo GRU recreado...")
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    # Buscar el modelo recreado más reciente
    recreated_models = list(models_dir.glob("GRU_Model_RECREATED_72DA_*.pth"))
    
    if not recreated_models:
        print("❌ No se encontró modelo GRU recreado")
        return None
    
    # Tomar el más reciente
    latest_model = max(recreated_models, key=lambda x: x.stat().st_mtime)
    print(f"   📂 Cargando: {latest_model.name}")
    
    checkpoint = torch.load(latest_model, map_location=device)
    
    print(f"   📊 DA del modelo base: {checkpoint['direction_accuracy']:.4f}")
    
    return checkpoint, latest_model

def apply_rolling_forecast_to_recreated(checkpoint):
    """Aplicar Rolling Forecast al modelo recreado"""
    print("\n🚀 Aplicando Rolling Forecast Avanzado al modelo recreado...")
    
    # Recrear la estructura del modelo
    input_size = 4  # returns, rsi, sma20, dxy_returns
    model = GRU_Model(
        input_size=input_size,
        hidden_size=checkpoint['optuna_params']['hidden_size'],
        output_size=1,
        dropout_prob=checkpoint['optuna_params']['dropout_prob'],
        num_layers=2
    ).to(device)
    
    # Cargar pesos del modelo entrenado
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Usar la función de Rolling Forecast
    from evaluate_optuna_models_rolling_advanced import (
        prepare_full_data, 
        incremental_rolling_forecast_lstm
    )
    
    # Preparar datos
    full_data = prepare_full_data()
    
    # Aplicar Rolling Forecast
    advanced_pred, y_real = incremental_rolling_forecast_lstm(
        model, 
        full_data, 
        checkpoint['seq_length'], 
        checkpoint['optuna_params'], 
        max_predictions=30
    )
    
    if len(advanced_pred) == 0:
        print("❌ No se pudieron generar predicciones con Rolling Forecast")
        return None
    
    # Validar y limpiar datos
    nan_mask_pred = np.isnan(advanced_pred)
    nan_mask_real = np.isnan(y_real)
    valid_mask = ~(nan_mask_pred | nan_mask_real)
    
    if np.sum(valid_mask) == 0:
        print("❌ No hay datos válidos")
        return None
    
    advanced_pred_clean = advanced_pred[valid_mask]
    y_real_clean = y_real[valid_mask]
    
    print(f"   ✅ Datos válidos: {len(advanced_pred_clean)}/{len(advanced_pred)}")
    
    # Calcular métricas
    try:
        advanced_rmse = np.sqrt(mean_squared_error(y_real_clean, advanced_pred_clean))
        advanced_r2 = r2_score(y_real_clean, advanced_pred_clean)
        
        # Directional Accuracy
        def directional_accuracy(y_true, y_pred):
            if len(y_true) <= 1:
                return 0.5
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        
        advanced_da = directional_accuracy(y_real_clean, advanced_pred_clean)
        
        print(f"\n📊 RESULTADOS ROLLING FORECAST:")
        print(f"   🎯 RMSE: {advanced_rmse:.6f}")
        print(f"   🎯 R²: {advanced_r2:.6f}")
        print(f"   🎯 DA: {advanced_da:.4f} ({advanced_da*100:.1f}%)")
        
        # Comparar con baseline naive
        if len(y_real_clean) > 1:
            naive_pred_clean = y_real_clean[:-1]
            naive_da = directional_accuracy(y_real_clean[1:], naive_pred_clean)
            da_vs_naive = advanced_da - naive_da
            
            print(f"   📊 vs Naive DA: {naive_da:.4f}")
            print(f"   🎯 Mejora vs Naive: {da_vs_naive:.4f} ({da_vs_naive*100:.1f}%)")
        
        return {
            'advanced_rmse': advanced_rmse,
            'advanced_r2': advanced_r2,
            'advanced_da': advanced_da,
            'predictions_count': len(advanced_pred_clean)
        }
        
    except Exception as e:
        print(f"❌ Error calculando métricas: {e}")
        return None

def main():
    """Función principal"""
    print("🎯 TESTING ROLLING FORECAST EN MODELO GRU RECREADO")
    print("=" * 70)
    print("🔍 Verificando si Rolling Forecast es la clave del 72.41% DA")
    print("=" * 70)
    
    # Cargar modelo recreado
    checkpoint, model_path = load_recreated_gru_model()
    if checkpoint is None:
        return
    
    print(f"\n📊 MODELO BASE (Sin Rolling Forecast):")
    print(f"   🎯 DA: {checkpoint['direction_accuracy']:.4f} ({checkpoint['direction_accuracy']*100:.1f}%)")
    print(f"   📉 RMSE: {checkpoint['test_rmse']:.8f}")
    print(f"   📈 R²: {checkpoint['test_r2']:.6f}")
    
    # Aplicar Rolling Forecast
    rolling_results = apply_rolling_forecast_to_recreated(checkpoint)
    
    if rolling_results:
        base_da = checkpoint['direction_accuracy']
        rolling_da = rolling_results['advanced_da']
        improvement = rolling_da - base_da
        
        print(f"\n🎯 COMPARACIÓN FINAL:")
        print(f"   📊 DA Base: {base_da:.4f} ({base_da*100:.1f}%)")
        print(f"   🚀 DA Rolling: {rolling_da:.4f} ({rolling_da*100:.1f}%)")
        print(f"   📈 Mejora: +{improvement:.4f} (+{improvement*100:.1f}%)")
        
        if rolling_da > 0.70:
            print(f"\n🎉 ¡ÉXITO! Rolling Forecast logró > 70% DA")
            print(f"   ✅ Confirmado: Rolling Forecast ES la clave del éxito")
        elif rolling_da > 0.60:
            print(f"\n✅ Muy bueno! > 60% DA con Rolling Forecast")
        else:
            print(f"\n⚠️ Aún no alcanza 70% DA - puede necesitar más ajustes")
        
        # Guardar resultados
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        results_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_model_da': float(base_da),
            'rolling_forecast_da': float(rolling_da),
            'improvement': float(improvement),
            'target_da_72pct': 0.7241,
            'achieved_target': rolling_da >= 0.72,
            'rolling_results': rolling_results,
            'model_params': checkpoint['optuna_params']
        }
        
        current_dir = Path.cwd()
        if current_dir.name == "model":
            results_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
        else:
            results_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
        
        results_file = results_dir / f"rolling_forecast_verification_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Resultados guardados: {results_file}")

if __name__ == "__main__":
    main()
