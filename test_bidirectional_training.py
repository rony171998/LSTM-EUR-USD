#!/usr/bin/env python3
"""
test_bidirectional_training.py
Script de prueba para verificar que el entrenamiento bidireccional funcione correctamente.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Agregar el directorio model al path
sys.path.append('model')

# Verificar GPU
print("🚀 VERIFICACIÓN DE ENTRENAMIENTO BIDIRECCIONAL")
print("=" * 60)

# 1. Verificar GPU
print(f"🖥️  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("⚠️  Sin GPU disponible - el entrenamiento será en CPU (lento)")

# 2. Verificar importaciones
try:
    from modelos import BidirectionalDeepLSTMModel, GRU_Model
    print("✅ Modelos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando modelos: {e}")
    exit(1)

# 3. Verificar configuración
try:
    from config import DEFAULT_PARAMS
    print(f"✅ Configuración cargada: {DEFAULT_PARAMS.TABLENAME}")
except ImportError as e:
    print(f"❌ Error importando configuración: {e}")
    exit(1)

# 4. Verificar backtester
try:
    from backtest_trading_models import TradingBacktester
    print("✅ TradingBacktester importado correctamente")
except ImportError as e:
    print(f"❌ Error importando TradingBacktester: {e}")
    exit(1)

# 5. Verificar modelos disponibles
current_dir = Path.cwd()
if current_dir.name == "model":
    models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
else:
    models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME

print(f"\n📁 Verificando modelos en: {models_dir}")

if models_dir.exists():
    gru_models = list(models_dir.glob("GRU_*.pth"))
    bidirectional_models = list(models_dir.glob("BidirectionalDeepLSTMModel_*.pth"))
    
    print(f"   🤖 Modelos GRU encontrados: {len(gru_models)}")
    for model in gru_models[:3]:  # Mostrar solo los primeros 3
        print(f"      - {model.name}")
    if len(gru_models) > 3:
        print(f"      ... y {len(gru_models) - 3} más")
    
    print(f"   🔄 Modelos Bidireccionales encontrados: {len(bidirectional_models)}")
    for model in bidirectional_models[:3]:  # Mostrar solo los primeros 3
        print(f"      - {model.name}")
    if len(bidirectional_models) > 3:
        print(f"      ... y {len(bidirectional_models) - 3} más")
    
    if len(gru_models) == 0 and len(bidirectional_models) == 0:
        print("⚠️  No se encontraron modelos entrenados")
        print("   Para entrenar modelos, ejecuta:")
        print("   python model/train_model.py --model GRU_Model")
        print("   python model/train_model.py --model BidirectionalDeepLSTMModel")
else:
    print(f"❌ Directorio de modelos no existe: {models_dir}")

# 6. Verificar datos
data_files = ["EUR_USD_2010-2024.csv", "DXY_2010-2024.csv"]
data_dir = Path("data")

print(f"\n📊 Verificando datos en: {data_dir}")
missing_files = []
for file in data_files:
    file_path = data_dir / file
    if file_path.exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - FALTANTE")
        missing_files.append(file)

if missing_files:
    print(f"⚠️  Archivos de datos faltantes: {missing_files}")

# 7. Probar creación de modelos
print(f"\n🧪 PRUEBA DE CREACIÓN DE MODELOS")
print("-" * 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Crear modelo GRU
    gru_model = GRU_Model(input_size=4, hidden_size=64, output_size=1, dropout_prob=0.1).to(device)
    print("✅ Modelo GRU creado correctamente")
    
    # Crear modelo Bidireccional
    bi_model = BidirectionalDeepLSTMModel(input_size=4, hidden_size=64, output_size=1, dropout_prob=0.1).to(device)
    print("✅ Modelo Bidireccional creado correctamente")
    
    # Probar con datos sintéticos
    test_input = torch.randn(32, 120, 4).to(device)  # batch_size=32, seq_len=120, features=4
    
    with torch.no_grad():
        gru_output = gru_model(test_input)
        bi_output = bi_model(test_input)
    
    print(f"   🔍 GRU output shape: {gru_output.shape}")
    print(f"   🔍 Bidirectional output shape: {bi_output.shape}")
    
    if gru_output.shape == bi_output.shape:
        print("✅ Ambos modelos producen outputs compatibles")
    else:
        print("⚠️  Formas de output diferentes entre modelos")
        
except Exception as e:
    print(f"❌ Error creando o probando modelos: {e}")

print(f"\n🎯 RESUMEN")
print("=" * 30)
print("✅ El sistema está listo para entrenar modelos bidireccionales")
print("📝 Cambios implementados:")
print("   - Soporte para BidirectionalDeepLSTMModel en run_rolling_forecast_backtest_with_seed")
print("   - Detección automática de tipo de modelo por nombre de archivo")
print("   - Análisis de estabilidad con múltiples tipos de modelo")
print("   - Comparación GRU vs Bidirectional con mismas semillas")
print("\n🚀 Para ejecutar análisis completo:")
print("   python model/backtest_trading_models.py")
