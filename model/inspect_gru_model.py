#!/usr/bin/env python3
"""
inspect_gru_model.py - Inspeccionar modelo GRU que logró 72% DA
"""

import torch
import json
from pathlib import Path

def inspect_gru_model():
    """Inspeccionar parámetros del modelo GRU exitoso"""
    print("🔍 INSPECCIÓN DEL MODELO GRU EXITOSO (72% DA)")
    print("=" * 60)
    
    # Ruta del modelo
    model_path = Path("../modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth")
    
    if not model_path.exists():
        print("❌ Modelo no encontrado")
        return
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("📊 PARÁMETROS DEL MODELO GRU:")
    print(f"   🧠 Hidden size: {checkpoint['optuna_params']['hidden_size']}")
    print(f"   📈 Learning rate: {checkpoint['optuna_params']['learning_rate']:.10f}")
    print(f"   🎲 Dropout: {checkpoint['optuna_params']['dropout_prob']:.6f}")
    print(f"   📦 Batch size: {checkpoint['optuna_params']['batch_size']}")
    print(f"   🔄 Sequence length: {checkpoint['seq_length']}")
    
    print(f"\n📊 MÉTRICAS DE ENTRENAMIENTO:")
    if 'test_rmse' in checkpoint:
        print(f"   📉 Test RMSE: {checkpoint['test_rmse']:.8f}")
    if 'test_r2' in checkpoint:
        print(f"   📊 Test R²: {checkpoint['test_r2']:.6f}")
    if 'direction_accuracy' in checkpoint:
        print(f"   🎯 Direction Accuracy: {checkpoint['direction_accuracy']:.4f}")
    
    print(f"\n🕒 INFORMACIÓN ADICIONAL:")
    if 'timestamp' in checkpoint:
        print(f"   📅 Timestamp: {checkpoint['timestamp']}")
    if 'model_class' in checkpoint:
        print(f"   🏗️ Clase: {checkpoint['model_class']}")
    
    # Comparar con los parámetros del JSON exitoso
    print(f"\n🔗 COMPARACIÓN CON RESULTADO 72% DA:")
    print("   Parámetros del JSON exitoso:")
    json_params = {
        "hidden_size": 128,
        "learning_rate": 0.0010059426888791,
        "dropout_prob": 0.3441023356173669,
        "batch_size": 16,
        "seq_length": 60
    }
    
    for key, value in json_params.items():
        if key == 'seq_length':
            model_value = checkpoint['seq_length']
        else:
            model_value = checkpoint['optuna_params'][key]
        
        match = "✅" if abs(model_value - value) < 1e-6 else "❌"
        print(f"   {match} {key}: {model_value} vs {value}")
    
    print(f"\n🎯 CONCLUSIÓN:")
    if all(abs(checkpoint['optuna_params'][k] - json_params[k]) < 1e-6 for k in json_params if k != 'seq_length'):
        if checkpoint['seq_length'] == json_params['seq_length']:
            print("   🎉 ¡ESTE ES EL MODELO EXACTO QUE LOGRÓ 72% DA!")
        else:
            print("   ⚠️ Parámetros coinciden pero seq_length diferente")
    else:
        print("   ❌ Este NO es el modelo exacto del 72% DA")
    
    return checkpoint

if __name__ == "__main__":
    inspect_gru_model()
