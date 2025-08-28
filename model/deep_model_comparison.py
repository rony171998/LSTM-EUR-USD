#!/usr/bin/env python3
"""
deep_model_comparison.py - Comparación profunda entre modelos
"""

import torch
import numpy as np
from pathlib import Path
import json

def compare_models_deeply():
    """Comparar el modelo original (72% DA) vs el recreado (49% DA)"""
    print("🔍 COMPARACIÓN PROFUNDA DE MODELOS")
    print("=" * 60)
    
    # Cargar modelo original (el que logró 72% DA)
    original_path = Path("../modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth")
    
    # Buscar modelo recreado
    recreated_models = list(Path("../modelos/eur_usd").glob("GRU_Model_RECREATED_72DA_*.pth"))
    if not recreated_models:
        print("❌ No se encontró modelo recreado")
        return
    
    recreated_path = max(recreated_models, key=lambda x: x.stat().st_mtime)
    
    print(f"📂 Original: {original_path.name}")
    print(f"📂 Recreado: {recreated_path.name}")
    
    # Cargar ambos modelos
    original = torch.load(original_path, map_location='cpu')
    recreated = torch.load(recreated_path, map_location='cpu')
    
    print(f"\n📊 MÉTRICAS CONOCIDAS:")
    print(f"   Original DA: 72.41% (del JSON)")
    print(f"   Recreado DA: {recreated['direction_accuracy']:.4f} ({recreated['direction_accuracy']*100:.1f}%)")
    
    print(f"\n🔧 COMPARACIÓN DE PARÁMETROS:")
    
    # Comparar parámetros Optuna
    orig_params = original['optuna_params']
    rec_params = recreated['optuna_params']
    
    for key in orig_params:
        orig_val = orig_params[key]
        rec_val = rec_params[key]
        match = "✅" if abs(orig_val - rec_val) < 1e-10 else "❌"
        print(f"   {match} {key}: {orig_val} vs {rec_val}")
    
    # Comparar seq_length
    orig_seq = original['seq_length']
    rec_seq = recreated['seq_length']
    match = "✅" if orig_seq == rec_seq else "❌"
    print(f"   {match} seq_length: {orig_seq} vs {rec_seq}")
    
    print(f"\n🧠 COMPARACIÓN DE PESOS DEL MODELO:")
    
    # Comparar state_dict
    orig_state = original['model_state_dict']
    rec_state = recreated['model_state_dict']
    
    total_params = 0
    different_params = 0
    max_diff = 0
    
    for key in orig_state:
        if key in rec_state:
            orig_tensor = orig_state[key]
            rec_tensor = rec_state[key]
            
            if orig_tensor.shape == rec_tensor.shape:
                diff = torch.abs(orig_tensor - rec_tensor)
                max_param_diff = torch.max(diff).item()
                max_diff = max(max_diff, max_param_diff)
                
                if max_param_diff > 1e-6:
                    different_params += 1
                    print(f"   ❌ {key}: max_diff = {max_param_diff:.8f}")
                else:
                    print(f"   ✅ {key}: idéntico")
                
                total_params += 1
            else:
                print(f"   ⚠️ {key}: shape diferente {orig_tensor.shape} vs {rec_tensor.shape}")
        else:
            print(f"   ❌ {key}: falta en recreado")
    
    print(f"\n📊 RESUMEN DE DIFERENCIAS:")
    print(f"   Total parámetros: {total_params}")
    print(f"   Parámetros diferentes: {different_params}")
    print(f"   Máxima diferencia: {max_diff:.8f}")
    
    # Comparar metadatos adicionales
    print(f"\n🕒 METADATOS:")
    
    print(f"   Original timestamp: {original.get('timestamp', 'N/A')}")
    print(f"   Recreado timestamp: {recreated.get('timestamp', 'N/A')}")
    
    # Comparar métricas de entrenamiento
    print(f"\n📊 MÉTRICAS DE ENTRENAMIENTO:")
    print(f"   Original RMSE: {original.get('test_rmse', 'N/A')}")
    print(f"   Recreado RMSE: {recreated.get('test_rmse', 'N/A')}")
    
    print(f"   Original R²: {original.get('test_r2', 'N/A')}")
    print(f"   Recreado R²: {recreated.get('test_r2', 'N/A')}")
    
    # Buscar diferencias clave
    print(f"\n🔍 ANÁLISIS DE DIFERENCIAS:")
    
    if different_params == 0:
        print("   🤔 Los modelos son IDÉNTICOS en parámetros")
        print("   💡 La diferencia debe estar en:")
        print("      - Datos de entrenamiento diferentes")
        print("      - Proceso de evaluación diferente")
        print("      - Seed de inicialización diferente")
    else:
        print(f"   ⚠️ Los modelos tienen {different_params} capas diferentes")
        print("   💡 Esto explica la diferencia en rendimiento")
    
    # Verificar si hay campos especiales
    print(f"\n🔧 CAMPOS ESPECIALES:")
    
    orig_keys = set(original.keys())
    rec_keys = set(recreated.keys())
    
    only_in_original = orig_keys - rec_keys
    only_in_recreated = rec_keys - orig_keys
    
    if only_in_original:
        print(f"   📂 Solo en original: {only_in_original}")
    
    if only_in_recreated:
        print(f"   📂 Solo en recreado: {only_in_recreated}")
    
    # Conclusión
    print(f"\n🎯 CONCLUSIÓN:")
    
    if different_params == 0 and abs(recreated['direction_accuracy'] - 0.7241) > 0.1:
        print("   🚨 MISTERIO: Modelos idénticos pero DA muy diferente")
        print("   💡 Posibles causas:")
        print("      1. El 72.41% DA viene de Rolling Forecast con OTRO modelo")
        print("      2. Hay datos adicionales no considerados")
        print("      3. El JSON se refiere a una evaluación diferente")
        
        # Investigar el JSON original
        print(f"\n🔍 INVESTIGANDO JSON ORIGINAL...")
        json_path = Path("../modelos/eur_usd/advanced_rolling_results_20250822_201010.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            print(f"   📊 JSON timestamp: {json_data['timestamp']}")
            print(f"   🎯 JSON best_da: {json_data['best_da']}")
            print(f"   📈 JSON modelo: {json_data['results'][0]['model_name']}")
            print(f"   🔧 JSON parámetros: {json_data['results'][0]['optuna_params']}")
            
            # Comparar parámetros del JSON
            json_params = json_data['results'][0]['optuna_params']
            print(f"\n🔗 COMPARACIÓN CON JSON:")
            for key in ['hidden_size', 'learning_rate', 'dropout_prob', 'batch_size']:
                if key in json_params and key in orig_params:
                    json_val = json_params[key]
                    orig_val = orig_params[key]
                    match = "✅" if abs(json_val - orig_val) < 1e-6 else "❌"
                    print(f"   {match} {key}: JSON={json_val} vs Original={orig_val}")

def main():
    compare_models_deeply()

if __name__ == "__main__":
    main()
