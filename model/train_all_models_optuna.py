#!/usr/bin/env python3
"""
train_all_models_optuna.py - Entrenar todos los modelos con PARÁMETROS OPTIMIZADOS por Optuna
Utiliza los mejores parámetros encontrados mediante optimización bayesiana para cada modelo.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import time
from datetime import timedelta
from pathlib import Path
import random
import json
import os
from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
)

device = torch.device("cuda")

def set_seed(seed=DEFAULT_PARAMS.SEED):
    """Fijar semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Semilla fijada: {seed}")

def load_optuna_params():
    """Cargar parámetros optimizados con Optuna para cada modelo"""
    # Detectar si estamos en directorio model/ o raíz del proyecto
    current_dir = Path.cwd()
    if current_dir.name == "model":
        params_dir = Path("../params/eur_usd")  # Desde model/
    else:
        params_dir = Path("params/eur_usd")  # Desde raíz del proyecto
    
    # Mapeo exacto de archivos a nombres de modelos
    param_files = {
        "TLS_LSTMModel": "best_params_TLS_LSTMModel_eur_usd.json",
        "GRU_Model": "best_params_GRU_Model.json",
        "HybridLSTMAttentionModel": "best_params_HybridLSTMAttention_eur_usd.json",
        "BidirectionalDeepLSTMModel": "best_params_BidirectionalDeepLSTM.json",
    }
    
    optuna_params = {}
    
    print("📋 CARGANDO PARÁMETROS OPTIMIZADOS CON OPTUNA:")
    print("=" * 60)
    
    for model_name, filename in param_files.items():
        filepath = params_dir / filename
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                optuna_params[model_name] = params
                
                print(f"✅ {model_name}:")
                print(f"   📊 Hidden Size: {params.get('hidden_size', 'N/A')}")
                print(f"   📊 Learning Rate: {params.get('learning_rate', 'N/A'):.6f}")
                print(f"   📊 Dropout: {params.get('dropout_prob', 'N/A'):.4f}")
                print(f"   📊 Batch Size: {params.get('batch_size', 'N/A')}")
                print(f"   📊 Seq Length: {params.get('seq_length', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
        else:
            print(f"⚠️ Archivo no encontrado: {filename}")
            
    print("=" * 60)
    return optuna_params

def load_multi_asset_data():
    """Cargar EUR/USD + DXY solo (las 2 más importantes)"""
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
    
    # DXY (si existe) - ruta adaptativa
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
    """Crear SOLO las características que sabemos que funcionan"""
    print("🔧 Creando características probadas...")
    
    # 1. EUR/USD returns (CRÍTICO)
    eur_returns = eur_prices.pct_change()
    
    # 2. EUR/USD RSI (PROBADO)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_prices)
    
    # 3. SMA20 (ESTABLE)
    eur_sma20 = eur_prices.rolling(window=20).mean()
    
    # Crear DataFrame base
    features_dict = {
        'price': eur_prices,
        'returns': eur_returns,
        'rsi': eur_rsi,
        'sma20': eur_sma20
    }
    
    # 4. DXY returns (solo si está disponible)
    if dxy_prices is not None:
        # Alinear fechas
        common_dates = eur_prices.index.intersection(dxy_prices.index)
        if len(common_dates) > 1000:  # Solo si hay suficientes datos
            dxy_aligned = dxy_prices.reindex(common_dates)
            dxy_returns = dxy_aligned.pct_change()
            
            # Alinear todas las series a fechas comunes
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
    
    # Crear DataFrame
    features_df = pd.DataFrame(features_dict)
    features_df = features_df.dropna()
    
    print(f"✅ Características: {features_df.shape}")
    print(f"   Features: {list(features_df.columns)}")
    
    return features_df

def prepare_data_with_custom_seq_length(seq_length):
    """Preparar datos para entrenamiento con seq_length personalizado"""
    # 1. Cargar datos
    eur_prices, dxy_prices = load_multi_asset_data()
    
    # 2. Crear características probadas
    features_df = create_proven_features(eur_prices, dxy_prices)
    
    # 3. Preparar datos
    target_data = features_df['price']
    feature_columns = [col for col in features_df.columns if col != 'price']
    features_data = features_df[feature_columns]
    
    print(f"📊 Features: {len(feature_columns)} | Muestras: {len(features_data)}")
    print(f"   Features: {feature_columns}")
    
    # 4. Split temporal
    train_size = int(len(features_data) * 0.8)
    
    X_train_raw = features_data.iloc[:train_size].values
    X_test_raw = features_data.iloc[train_size:].values
    y_train_raw = target_data.iloc[:train_size].values
    y_test_raw = target_data.iloc[train_size:].values
    
    print(f"📊 Split: {train_size} train, {len(X_test_raw)} test")
    
    # 5. Escalado
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # 6. Crear secuencias con seq_length personalizado
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    print(f"✅ Secuencias (seq_length={seq_length}): Train {X_train_seq.shape} | Test {X_test_seq.shape}")
    
    # 7. Tensores
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
        'feature_columns': feature_columns,
        'seq_length': seq_length
    }

def train_model_with_optuna_params(model, data, model_name, optuna_params):
    """Entrenar un modelo con PARÁMETROS OPTIMIZADOS por Optuna"""
    params = optuna_params[model_name]
    
    print(f"\n🚀 Entrenando {model_name} con parámetros Optuna...")
    print("=" * 60)
    print(f"🎯 PARÁMETROS OPTIMIZADOS:")
    print(f"   📊 Hidden Size: {params['hidden_size']}")
    print(f"   📊 Learning Rate: {params['learning_rate']:.6f}")
    print(f"   📊 Dropout: {params['dropout_prob']:.4f}")
    print(f"   📊 Batch Size: {params['batch_size']}")
    print(f"   📊 Seq Length: {params['seq_length']}")
    
    start_time = time.time()
    
    # Parámetros del modelo
    print(f"   ⚙️  Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configuración de entrenamiento con parámetros Optuna
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    # Epochs: usar 150 para dar tiempo a converger con parámetros optimizados
    epochs = 150
    print(f"🔥 Entrenando por {epochs} epochs con parámetros optimizados...")
    
    # Entrenamiento
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if (epoch + 1) % 30 == 0:  # Cada 30 epochs
            avg_loss = epoch_loss / batch_count
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    
    # Evaluación
    print("📊 Evaluando...")
    model.eval()
    
    with torch.no_grad():
        # Predicciones
        test_pred_scaled = model(data['X_test']).squeeze()
        train_pred_scaled = model(data['X_train']).squeeze()
        
        # Desnormalizar
        test_pred = data['target_scaler'].inverse_transform(test_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        train_pred = data['target_scaler'].inverse_transform(train_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        
        y_test_real = data['target_scaler'].inverse_transform(data['y_test_seq'].reshape(-1, 1)).flatten()
        y_train_real = data['target_scaler'].inverse_transform(data['y_train_seq'].reshape(-1, 1)).flatten()
        
        # Métricas
        train_mse = np.mean((train_pred - y_train_real) ** 2)
        test_mse = np.mean((test_pred - y_test_real) ** 2)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_var = np.var(y_train_real)
        test_var = np.var(y_test_real)
        
        train_r2 = 1 - (train_mse / train_var) if train_var > 0 else 0.0
        test_r2 = 1 - (test_mse / test_var) if test_var > 0 else 0.0
        
        # Directional Accuracy
        def directional_accuracy(y_true, y_pred):
            if len(y_true) <= 1:
                return 0.5
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
        
        train_da = directional_accuracy(y_train_real, train_pred)
        test_da = directional_accuracy(y_test_real, test_pred)
    
    # Guardar modelo - ruta adaptativa
    current_dir = Path.cwd()
    if current_dir.name == "model":
        model_dir = Path("../modelos")  # Desde model/
    else:
        model_dir = Path("modelos")  # Desde raíz del proyecto
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / DEFAULT_PARAMS.TABLENAME / f"{model_name}_optuna_{DEFAULT_PARAMS.FILEPATH}.pth"
    model_path.parent.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model_name,
        'optuna_params': params,
        'feature_columns': data['feature_columns'],
        'seq_length': data['seq_length'],
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_da': test_da,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_da': train_da,
        'training_time': training_time,
        'epochs': epochs,
        'optimization_source': 'Optuna Bayesian Optimization'
    }, model_path)
    
    # Resultados
    print(f"\n✅ {model_name} COMPLETADO (Optuna)")
    print("=" * 60)
    print(f"📊 Train RMSE: {train_rmse:.6f}")
    print(f"📊 Test RMSE: {test_rmse:.6f}")
    print(f"📊 Train R²: {train_r2:.6f}")
    print(f"📊 Test R²: {test_r2:.6f}")
    print(f"📊 Train DA: {train_da:.4f} ({train_da*100:.1f}%)")
    print(f"📊 Test DA: {test_da:.4f} ({test_da*100:.1f}%)")
    print(f"⏱️ Tiempo: {timedelta(seconds=training_time)}")
    print(f"💾 Guardado: {model_path}")
    
    return {
        'model_name': model_name,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_da': test_da,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_da': train_da,
        'training_time': training_time,
        'model_path': model_path,
        'parameters': sum(p.numel() for p in model.parameters()),
        'optuna_params': params
    }

def main():
    """Función principal para entrenar todos los modelos con parámetros optimizados por Optuna"""
    print("🎯 ENTRENAMIENTO CON PARÁMETROS OPTIMIZADOS POR OPTUNA")
    print("=" * 70)
    print("🧠 Utilizando optimización bayesiana para mejores hiperparámetros")
    print("🎲 Semillas fijas para reproducibilidad")
    print("=" * 70)
    
    # Fijar semilla para reproducibilidad
    set_seed(DEFAULT_PARAMS.SEED)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Usando CPU")
    
    # Cargar parámetros optimizados
    optuna_params = load_optuna_params()
    
    if not optuna_params:
        print("❌ No se encontraron parámetros optimizados con Optuna")
        return
    
    # Lista de modelos para entrenar
    models_to_train = []
    
    for model_name in optuna_params.keys():
        params = optuna_params[model_name]
        
        print(f"\n🔧 Preparando {model_name} con parámetros Optuna...")
        
        # Preparar datos con seq_length específico del modelo
        data = prepare_data_with_custom_seq_length(params['seq_length'])
        input_size = data['X_train'].shape[2]
        
        # Crear modelo con parámetros optimizados
        if model_name == "TLS_LSTMModel":
            model = TLS_LSTMModel(
                input_size=input_size, 
                hidden_size=params['hidden_size'],
                output_size=1, 
                dropout_prob=params['dropout_prob']
            ).to(device)
        elif model_name == "GRU_Model":
            model = GRU_Model(
                input_size=input_size, 
                hidden_size=params['hidden_size'],
                output_size=1, 
                dropout_prob=params['dropout_prob'],
                num_layers=2
            ).to(device)
        elif model_name == "HybridLSTMAttentionModel":
            model = HybridLSTMAttentionModel(
                input_size=input_size, 
                hidden_size=params['hidden_size'],
                output_size=1, 
                dropout_prob=params['dropout_prob']
            ).to(device)
        elif model_name == "BidirectionalDeepLSTMModel":
            model = BidirectionalDeepLSTMModel(
                input_size=input_size, 
                hidden_size=params['hidden_size'],
                output_size=1, 
                dropout_prob=params['dropout_prob']
            ).to(device)
        else:
            print(f"⚠️ Modelo {model_name} no reconocido")
            continue
        
        models_to_train.append({
            'model': model,
            'name': model_name,
            'data': data,
            'params': params
        })
    
    # Entrenar todos los modelos con parámetros Optuna
    results = []
    total_start_time = time.time()
    
    for i, model_info in enumerate(models_to_train, 1):
        print(f"\n🔄 [{i}/{len(models_to_train)}] Entrenando {model_info['name']}...")
        print(f"   🎯 Parámetros optimizados por Optuna aplicados")
        
        # Mantener semilla fija (DEFAULT_PARAMS.SEED) para comparación justa entre modelos
        set_seed(DEFAULT_PARAMS.SEED)
        
        try:
            result = train_model_with_optuna_params(
                model_info['model'], 
                model_info['data'], 
                model_info['name'], 
                optuna_params
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error entrenando {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Resumen final
    print("\n🏆 RESUMEN FINAL - MODELOS CON PARÁMETROS OPTUNA")
    print("=" * 85)
    print("🧠 TODOS los modelos entrenados con parámetros optimizados por Optuna")
    print("🎯 Optimización bayesiana para encontrar mejores hiperparámetros")
    print("🎲 Semillas fijas para reproducibilidad completa")
    print("-" * 85)
    
    # Ordenar por RMSE
    results_sorted = sorted(results, key=lambda x: x['test_rmse'])
    
    print(f"{'Rank':<4} {'Modelo':<35} {'RMSE':<8} {'R²':<8} {'DA':<6} {'Params':<8} {'Hidden':<7} {'LR':<10} {'Tiempo':<12}")
    print("-" * 95)
    
    for i, result in enumerate(results_sorted, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        optuna_p = result['optuna_params']
        print(f"{emoji}{i:<3} {result['model_name']:<35} {result['test_rmse']:<8.6f} {result['test_r2']:<8.4f} {result['test_da']:<6.3f} {result['parameters']:<8,} {optuna_p['hidden_size']:<7} {optuna_p['learning_rate']:<10.6f} {str(timedelta(seconds=result['training_time'])):<12}")
    
    print(f"\n⏱️ Tiempo total: {timedelta(seconds=total_time)}")
    print(f"💾 Todos los modelos guardados en: modelos/")
    print(f"🎯 Parámetros aplicados desde optimización Optuna")
    
    # Guardar resumen - ruta adaptativa
    current_dir = Path.cwd()
    if current_dir.name == "model":
        summary_path = Path("../modelos") / DEFAULT_PARAMS.TABLENAME / "training_summary_optuna_optimized.txt"
    else:
        summary_path = Path("modelos") / DEFAULT_PARAMS.TABLENAME / "training_summary_optuna_optimized.txt"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE ENTRENAMIENTO CON PARÁMETROS OPTUNA\n")
        f.write("=" * 60 + "\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Semilla fija: {DEFAULT_PARAMS.SEED} (reproducibilidad garantizada)\n")
        f.write("Parámetros optimizados mediante Optuna (optimización bayesiana)\n\n")
        
        f.write("RESULTADOS:\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(results_sorted, 1):
            f.write(f"{i}. {result['model_name']}\n")
            f.write(f"   RMSE: {result['test_rmse']:.6f}\n")
            f.write(f"   R²: {result['test_r2']:.6f}\n")
            f.write(f"   DA: {result['test_da']:.4f}\n")
            f.write(f"   Parámetros: {result['parameters']:,}\n")
            f.write(f"   Tiempo: {timedelta(seconds=result['training_time'])}\n")
            f.write(f"   Archivo: {result['model_path']}\n")
            
            # Parámetros Optuna específicos
            optuna_p = result['optuna_params']
            f.write(f"   Parámetros Optuna:\n")
            f.write(f"     Hidden Size: {optuna_p['hidden_size']}\n")
            f.write(f"     Learning Rate: {optuna_p['learning_rate']:.6f}\n")
            f.write(f"     Dropout: {optuna_p['dropout_prob']:.4f}\n")
            f.write(f"     Batch Size: {optuna_p['batch_size']}\n")
            f.write(f"     Seq Length: {optuna_p['seq_length']}\n\n")
        
        f.write("VENTAJAS DE PARÁMETROS OPTUNA:\n")
        f.write("-" * 40 + "\n")
        f.write("• Optimización bayesiana para encontrar mejores hiperparámetros\n")
        f.write("• Parámetros específicos para cada arquitectura\n")
        f.write("• Mejor rendimiento esperado vs parámetros estándar\n")
        f.write("• Búsqueda inteligente en espacio de hiperparámetros\n")
        f.write("• Balanceado entre exploración y explotación\n")
    
    print(f"📄 Resumen detallado guardado en: {summary_path}")

if __name__ == "__main__":
    main()
