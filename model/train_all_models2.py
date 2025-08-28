# train_all_models2.py - Entrenar todos los modelos con PARÁMETROS IDÉNTICOS para comparación justa
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
from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
    ContextualLSTMTransformerFlexible,
)

device = torch.device("cuda")

# PARÁMETROS ESTÁNDAR PARA TODOS LOS MODELOS
STANDARD_PARAMS = {
    'hidden_size': 64,      # Tamaño estándar balanceado
    'dropout_prob': 0.2,    # Dropout estándar
    'epochs': 120,          # Épocas suficientes para convergencia
    'batch_size': 32,       # Batch size estándar
    'learning_rate': 0.001, # Learning rate estándar
    'num_layers': 2,        # Capas estándar
    'seq_length': 30        # Longitud de secuencia estándar
}

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

def load_multi_asset_data():
    """Cargar EUR/USD + DXY solo (las 2 más importantes)"""
    print("📊 Cargando EUR/USD + DXY...")
        
    # EUR/USD
    eur_file = f"data/{DEFAULT_PARAMS.FILEPATH}"
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
    
    # DXY (si existe)
    dxy_prices = None
    dxy_file = "data/DXY_2010-2024.csv"
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

def prepare_data():
    """Preparar datos para entrenamiento"""
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
    
    # 6. Crear secuencias
    seq_length = STANDARD_PARAMS['seq_length']  # Usar parámetro estándar
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    print(f"✅ Secuencias: Train {X_train_seq.shape} | Test {X_test_seq.shape}")
    
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

def train_model(model, data, model_name):
    """Entrenar un modelo con PARÁMETROS ESTÁNDAR"""
    print(f"\n🚀 Entrenando {model_name} con parámetros estándar...")
    print("=" * 60)
    print(f"📋 Hidden Size: {STANDARD_PARAMS['hidden_size']}")
    print(f"📋 Dropout: {STANDARD_PARAMS['dropout_prob']}")
    print(f"📋 Epochs: {STANDARD_PARAMS['epochs']}")
    print(f"📋 Batch Size: {STANDARD_PARAMS['batch_size']}")
    print(f"📋 Learning Rate: {STANDARD_PARAMS['learning_rate']}")
    
    start_time = time.time()
    
    # Parámetros del modelo
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configuración de entrenamiento ESTÁNDAR
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=STANDARD_PARAMS['learning_rate'])
    
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=STANDARD_PARAMS['batch_size'], shuffle=True)
    
    print(f"🔥 Entrenando por {STANDARD_PARAMS['epochs']} epochs con parámetros estándar...")
    
    # Entrenamiento
    model.train()
    for epoch in range(STANDARD_PARAMS['epochs']):
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
    
    # Guardar modelo
    model_dir = Path("modelos")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / DEFAULT_PARAMS.TABLENAME / f"{model_name}_{DEFAULT_PARAMS.FILEPATH}.pth"

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model_name,
        'model_config': get_model_config(model, model_name),
        'feature_columns': data['feature_columns'],
        'seq_length': data['seq_length'],
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_da': test_da,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_da': train_da,
        'training_time': training_time,
        'epochs': STANDARD_PARAMS['epochs'],
        'standard_params': STANDARD_PARAMS
    }, model_path)
    
    # Resultados
    print(f"\n✅ {model_name} COMPLETADO")
    print("=" * 50)
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
        'parameters': sum(p.numel() for p in model.parameters())
    }

def get_model_config(model, model_name):
    """Obtener configuración ESTÁNDAR del modelo"""
    base_config = {
        'hidden_size': STANDARD_PARAMS['hidden_size'],
        'dropout_prob': STANDARD_PARAMS['dropout_prob'],
        'epochs': STANDARD_PARAMS['epochs'],
        'batch_size': STANDARD_PARAMS['batch_size'],
        'learning_rate': STANDARD_PARAMS['learning_rate'],
        'seq_length': STANDARD_PARAMS['seq_length'],
        'config_source': 'Parámetros estándar para comparación justa'
    }
    
    if model_name == "TLS_LSTMModel":
        base_config.update({
            'input_size': model.lstm1.input_size,
            'layers': 2,
            'model_type': 'TLS LSTM básico'
        })
    elif model_name == "TLS_LSTMModel_Optimizado":
        base_config.update({
            'input_size': model.lstm1.input_size,
            'layers': 2,
            'model_type': 'TLS LSTM optimizado (mismos parámetros)'
        })
    elif model_name == "GRU_Model":
        base_config.update({
            'input_size': model.gru1.input_size,
            'num_layers': model.num_layers,
            'model_type': 'GRU estándar'
        })
    elif model_name == "HybridLSTMAttentionModel":
        base_config.update({
            'input_size': model.lstm1.input_size,
            'layers': 2,
            'attention': True,
            'model_type': 'LSTM + Atención'
        })
    elif model_name == "BidirectionalDeepLSTMModel":
        base_config.update({
            'input_size': model.lstm.input_size,
            'bidirectional': True,
            'num_layers': 2,
            'model_type': 'LSTM Bidireccional'
        })
    elif model_name == "ContextualLSTMTransformerFlexible":
        base_config.update({
            'seq_len': model.seq_len,
            'feature_dim': model.feature_dim,
            'window_size': model.window_size,
            'lstm_units': STANDARD_PARAMS['hidden_size'],  # Usar estándar
            'num_heads': 2,
            'embed_dim': STANDARD_PARAMS['hidden_size'],   # Usar estándar
            'dropout_rate': STANDARD_PARAMS['dropout_prob'], # Usar estándar
            'model_type': 'LSTM + Transformer'
        })
    
    return base_config

def main():
    """Función principal para entrenar todos los modelos con PARÁMETROS IDÉNTICOS"""
    print("🎯 ENTRENAMIENTO CON PARÁMETROS ESTÁNDAR IDÉNTICOS")
    print("=" * 70)
    print("📋 PARÁMETROS ESTÁNDAR PARA TODOS LOS MODELOS:")
    print(f"   Hidden Size: {STANDARD_PARAMS['hidden_size']}")
    print(f"   Dropout: {STANDARD_PARAMS['dropout_prob']}")
    print(f"   Epochs: {STANDARD_PARAMS['epochs']}")
    print(f"   Batch Size: {STANDARD_PARAMS['batch_size']}")
    print(f"   Learning Rate: {STANDARD_PARAMS['learning_rate']}")
    print(f"   Sequence Length: {STANDARD_PARAMS['seq_length']}")
    print("=" * 70)
    print("🎲 Usando semillas fijas para reproducibilidad completa")
    print("=" * 70)
    
    # Fijar semilla para reproducibilidad
    set_seed(42)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Usando CPU")
    
    # Preparar datos una vez
    data = prepare_data()
    input_size = data['X_train'].shape[2]
    
    # Lista de modelos con PARÁMETROS IDÉNTICOS
    models_to_train = []
    
    # 1. TLS_LSTMModel - Con parámetros estándar
    print("\n🔧 TLS_LSTMModel - Parámetros estándar")
    models_to_train.append({
        'model': TLS_LSTMModel(
            input_size=input_size, 
            hidden_size=STANDARD_PARAMS['hidden_size'],
            output_size=1, 
            dropout_prob=STANDARD_PARAMS['dropout_prob']
        ).to(device),
        'name': 'TLS_LSTMModel'
    })
    
    # 2. GRU_Model - Con parámetros estándar
    print("🔧 GRU_Model - Parámetros estándar")
    models_to_train.append({
        'model': GRU_Model(
            input_size=input_size, 
            hidden_size=STANDARD_PARAMS['hidden_size'],
            output_size=1, 
            dropout_prob=STANDARD_PARAMS['dropout_prob'],
            num_layers=STANDARD_PARAMS['num_layers']
        ).to(device),
        'name': 'GRU_Model'
    })
    
    # 3. HybridLSTMAttentionModel - Con parámetros estándar
    print("🔧 HybridLSTMAttentionModel - Parámetros estándar")
    models_to_train.append({
        'model': HybridLSTMAttentionModel(
            input_size=input_size, 
            hidden_size=STANDARD_PARAMS['hidden_size'],
            output_size=1, 
            dropout_prob=STANDARD_PARAMS['dropout_prob']
        ).to(device),
        'name': 'HybridLSTMAttentionModel'
    })
    
    # 4. BidirectionalDeepLSTMModel - Con parámetros estándar
    print("🔧 BidirectionalDeepLSTMModel - Parámetros estándar")
    models_to_train.append({
        'model': BidirectionalDeepLSTMModel(
            input_size=input_size, 
            hidden_size=STANDARD_PARAMS['hidden_size'],
            output_size=1, 
            dropout_prob=STANDARD_PARAMS['dropout_prob']
        ).to(device),
        'name': 'BidirectionalDeepLSTMModel'
    })
    
    # 5. TLS_LSTMModel "Optimizado" - CON LOS MISMOS PARÁMETROS ESTÁNDAR
    print("🔧 TLS_LSTMModel_Optimizado - Mismos parámetros estándar")
    models_to_train.append({
        'model': TLS_LSTMModel(
            input_size=input_size, 
            hidden_size=STANDARD_PARAMS['hidden_size'],  # MISMO que básico
            output_size=1, 
            dropout_prob=STANDARD_PARAMS['dropout_prob']  # MISMO que básico
        ).to(device),
        'name': 'TLS_LSTMModel_Optimizado'
    })
    
    # 6. ContextualLSTMTransformerFlexible - Con parámetros estándar adaptados
    print("🔧 ContextualLSTMTransformerFlexible - Parámetros estándar adaptados")
    models_to_train.append({
        'model': ContextualLSTMTransformerFlexible(
            seq_len=STANDARD_PARAMS['seq_length'],
            feature_dim=input_size, 
            output_size=1,
            window_size=6,
            max_neighbors=1,
            lstm_units=STANDARD_PARAMS['hidden_size'],    # Usar estándar
            num_heads=2,
            embed_dim=STANDARD_PARAMS['hidden_size'],     # Usar estándar  
            dropout_rate=STANDARD_PARAMS['dropout_prob']  # Usar estándar
        ).to(device),
        'name': 'ContextualLSTMTransformerFlexible'
    })
    
    # Entrenar todos los modelos con parámetros IDÉNTICOS
    results = []
    total_start_time = time.time()
    
    for i, model_info in enumerate(models_to_train, 1):
        print(f"\n🔄 [{i}/{len(models_to_train)}] Entrenando {model_info['name']}...")
        print(f"   📋 MISMOS parámetros estándar para comparación justa")
        
        # Re-fijar semilla antes de cada modelo para mayor reproducibilidad
        set_seed(42 + i)
        
        try:
            result = train_model(
                model_info['model'], 
                data, 
                model_info['name']
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error entrenando {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Resumen final
    print("\n🏆 RESUMEN FINAL - COMPARACIÓN CON PARÁMETROS IDÉNTICOS")
    print("=" * 85)
    print("📋 TODOS los modelos entrenados con PARÁMETROS ESTÁNDAR IDÉNTICOS")
    print(f"   Hidden Size: {STANDARD_PARAMS['hidden_size']} | Dropout: {STANDARD_PARAMS['dropout_prob']} | Epochs: {STANDARD_PARAMS['epochs']}")
    print("🎲 Semillas fijas para reproducibilidad completa")
    print("-" * 85)
    
    # Ordenar por RMSE
    results_sorted = sorted(results, key=lambda x: x['test_rmse'])
    
    print(f"{'Rank':<4} {'Modelo':<35} {'RMSE':<8} {'R²':<8} {'DA':<6} {'Params':<8} {'Tiempo':<12} {'Config'}")
    print("-" * 85)
    
    for i, result in enumerate(results_sorted, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        config_info = "Estándar✓"
        print(f"{emoji}{i:<3} {result['model_name']:<35} {result['test_rmse']:<8.6f} {result['test_r2']:<8.4f} {result['test_da']:<6.3f} {result['parameters']:<8,} {str(timedelta(seconds=result['training_time'])):<12} {config_info}")
    
    print(f"\n⏱️ Tiempo total: {timedelta(seconds=total_time)}")
    print(f"💾 Todos los modelos guardados en: modelos/")
    print(f"📊 Comparación justa: TODOS con parámetros idénticos")
    
    # Guardar resumen
    summary_path = Path("modelos") / DEFAULT_PARAMS.TABLENAME / "training_summary_standard_params.txt"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE ENTRENAMIENTO CON PARÁMETROS ESTÁNDAR IDÉNTICOS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Semilla fija: 42 (reproducibilidad garantizada)\n")
        f.write(f"Features: {data['feature_columns']}\n")
        f.write(f"Secuencias: {data['seq_length']}\n")
        f.write("PARÁMETROS ESTÁNDAR UTILIZADOS PARA TODOS LOS MODELOS:\n")
        f.write(f"  Hidden Size: {STANDARD_PARAMS['hidden_size']}\n")
        f.write(f"  Dropout: {STANDARD_PARAMS['dropout_prob']}\n")
        f.write(f"  Epochs: {STANDARD_PARAMS['epochs']}\n")
        f.write(f"  Batch Size: {STANDARD_PARAMS['batch_size']}\n")
        f.write(f"  Learning Rate: {STANDARD_PARAMS['learning_rate']}\n")
        f.write(f"  Sequence Length: {STANDARD_PARAMS['seq_length']}\n\n")
        
        f.write("RESULTADOS (Comparación justa con parámetros idénticos):\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(results_sorted, 1):
            f.write(f"{i}. {result['model_name']}\n")
            f.write(f"   RMSE: {result['test_rmse']:.6f}\n")
            f.write(f"   R²: {result['test_r2']:.6f}\n")
            f.write(f"   DA: {result['test_da']:.4f}\n")
            f.write(f"   Parámetros: {result['parameters']:,}\n")
            f.write(f"   Tiempo: {timedelta(seconds=result['training_time'])}\n")
            f.write(f"   Archivo: {result['model_path']}\n\n")
        
        f.write("VENTAJAS DE ESTA COMPARACIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write("• Comparación justa: TODOS con parámetros idénticos\n")
        f.write("• Elimina sesgo por configuración diferente\n")
        f.write("• Evalúa arquitectura pura, no hiperparámetros\n")
        f.write("• Reproducibilidad completa con semillas fijas\n")
        f.write("• Misma capacidad de aprendizaje para todos\n")
    
    print(f"📄 Resumen detallado guardado en: {summary_path}")

if __name__ == "__main__":
    main()
