#!/usr/bin/env python3
"""
simple_model_comparison.py - Comparación simple de modelos
"""

import torch
import numpy as np
from pathlib import Path

def simple_compare():
    """Comparación básica sin cargar scalers"""
    print("🔍 COMPARACIÓN SIMPLE DE MODELOS")
    print("=" * 50)
    
    # Cargar solo el original
    original_path = Path("../modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth")
    
    print(f"📂 Cargando: {original_path.name}")
    
    try:
        original = torch.load(original_path, map_location='cpu')
        
        print(f"\n📊 INFORMACIÓN DEL MODELO ORIGINAL:")
        print(f"   🧠 Hidden size: {original['optuna_params']['hidden_size']}")
        print(f"   📈 Learning rate: {original['optuna_params']['learning_rate']:.10f}")
        print(f"   🎲 Dropout: {original['optuna_params']['dropout_prob']:.6f}")
        print(f"   📦 Batch size: {original['optuna_params']['batch_size']}")
        print(f"   🔄 Seq length: {original['seq_length']}")
        
        print(f"\n📊 MÉTRICAS DEL MODELO:")
        if 'test_rmse' in original:
            print(f"   📉 Test RMSE: {original['test_rmse']:.8f}")
        if 'test_r2' in original:
            print(f"   📈 Test R²: {original['test_r2']:.6f}")
        if 'direction_accuracy' in original:
            print(f"   🎯 Direction Accuracy: {original['direction_accuracy']:.4f}")
        
        print(f"\n🔍 CAMPOS DISPONIBLES:")
        for key in original.keys():
            if key not in ['model_state_dict', 'scaler_state', 'target_scaler_state']:
                print(f"   📂 {key}: {type(original[key])}")
        
        print(f"\n🧠 ARQUITECTURA DEL MODELO:")
        state_dict = original['model_state_dict']
        total_params = 0
        
        for name, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            print(f"   🔧 {name}: {param.shape} ({param_count:,} params)")
        
        print(f"\n📊 Total parámetros: {total_params:,}")
        
        # Verificar si este modelo realmente puede dar 72% DA
        print(f"\n🤔 ANÁLISIS:")
        print(f"   📊 DA reportado en modelo: {original.get('direction_accuracy', 'N/A')}")
        print(f"   🎯 DA esperado del JSON: 72.41%")
        
        da_in_model = original.get('direction_accuracy', 0)
        if da_in_model and abs(da_in_model - 0.7241) > 0.1:
            print(f"   🚨 GRAN DIFERENCIA: {da_in_model:.4f} vs 0.7241")
            print(f"   💡 Esto sugiere que:")
            print(f"      1. El 72.41% NO viene de este modelo directamente")
            print(f"      2. El 72.41% es resultado de Rolling Forecast")
            print(f"      3. Rolling Forecast transforma ~50% DA en ~72% DA")
        else:
            print(f"   ✅ DA coincide aproximadamente")
        
        # Investigar timestamp
        if 'timestamp' in original:
            print(f"\n🕒 Timestamp modelo: {original['timestamp']}")
        
        # Buscar el JSON con 72.41%
        json_path = Path("../modelos/eur_usd/advanced_rolling_results_20250822_201010.json")
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            print(f"\n📄 ANÁLISIS DEL JSON EXITOSO:")
            print(f"   🕒 JSON timestamp: {json_data['timestamp']}")
            print(f"   🎯 Best DA: {json_data['best_da']:.4f}")
            print(f"   🔄 Técnica: {json_data['technique']}")
            print(f"   📊 Predicciones: {json_data['results'][0]['predictions_count']}")
            
            # Comparar parámetros
            json_params = json_data['results'][0]['optuna_params']
            model_params = original['optuna_params']
            
            print(f"\n🔗 COMPARACIÓN PARÁMETROS JSON vs MODELO:")
            all_match = True
            for key in ['hidden_size', 'learning_rate', 'dropout_prob', 'batch_size']:
                if key in json_params and key in model_params:
                    j_val = json_params[key]
                    m_val = model_params[key]
                    match = abs(j_val - m_val) < 1e-6
                    all_match = all_match and match
                    status = "✅" if match else "❌"
                    print(f"   {status} {key}: {j_val} vs {m_val}")
            
            if all_match:
                print(f"\n🎉 CONFIRMADO: Este ES el modelo que logró 72.41% DA")
                print(f"   🔑 La clave es: Rolling Forecast con re-entrenamiento")
                print(f"   📊 Transformación: {da_in_model:.1%} → {json_data['best_da']:.1%}")
            else:
                print(f"\n❌ Los parámetros NO coinciden exactamente")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    simple_compare()

if __name__ == "__main__":
    main()
