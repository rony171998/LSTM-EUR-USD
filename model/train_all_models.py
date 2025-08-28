# train_all_models.py - Entrenar todos los modelos y guardarlos con SEMILLAS FIJAS
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
    seq_length = 30
    
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

def train_model(model, data, model_name, epochs=100):
    """Entrenar un modelo específico"""
    print(f"\n🚀 Entrenando {model_name}...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Parámetros del modelo
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configuración de entrenamiento
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"🔥 Entrenando por {epochs} epochs...")
    
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
        
        if (epoch + 1) % 25 == 0:
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
        'epochs': epochs
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
    """Obtener configuración específica del modelo"""
    if model_name == "TLS_LSTMModel":
        return {
            'input_size': model.lstm1.input_size,
            'hidden_size': model.hidden_size,
            'layers': 2,
            'dropout_prob': 0.2,
            'config_source': 'Resumen - Configuración básica'
        }
    elif model_name == "TLS_LSTMModel_Optimizado":
        return {
            'input_size': model.lstm1.input_size,
            'hidden_size': model.hidden_size,
            'layers': 2,
            'dropout_prob': 0.1,
            'epochs': 150,
            'config_source': 'Resumen - Configuración optimizada'
        }
    elif model_name == "GRU_Model":
        return {
            'input_size': model.gru1.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout_prob': 0.2,
            'config_source': 'Resumen - Configuración estándar'
        }
    elif model_name == "HybridLSTMAttentionModel":
        return {
            'input_size': model.lstm1.input_size,
            'hidden_size': 50,
            'layers': 2,
            'attention': True,
            'dropout_prob': 0.2,
            'config_source': 'Resumen - LSTM + Atención'
        }
    elif model_name == "BidirectionalDeepLSTMModel":
        return {
            'input_size': model.lstm.input_size,
            'hidden_size': model.lstm.hidden_size,
            'bidirectional': True,
            'num_layers': 2,
            'dropout_prob': 0.2,
            'config_source': 'Resumen - LSTM Bidireccional'
        }
    elif model_name == "ContextualLSTMTransformerFlexible":
        return {
            'seq_len': model.seq_len,
            'feature_dim': model.feature_dim,
            'window_size': model.window_size,
            'lstm_units': model.lstm_units,
            'num_heads': model.num_heads,
            'embed_dim': model.embed_dim,
            'dropout_rate': model.dropout_rate,
            'config_source': 'Resumen - LSTM + Transformer'
        }
    else:
        return {'config_source': 'Configuración desconocida'}

def main():
    """Función principal para entrenar todos los modelos con configuraciones exactas del resumen"""
    print("🎯 ENTRENAMIENTO COMPLETO CON CONFIGURACIONES EXACTAS DEL RESUMEN")
    print("=" * 70)
    print("📝 Usando configuraciones documentadas + semillas fijas para reproducibilidad")
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
    
    # Lista de modelos con configuraciones EXACTAS del resumen
    models_to_train = []
    
    # 1. TLS_LSTMModel (Básico) - Configuración del resumen
    print("\n🔧 TLS_LSTMModel - Configuración básica del resumen")
    models_to_train.append({
        'model': TLS_LSTMModel(
            input_size=input_size, 
            hidden_size=50,  # Del resumen
            output_size=1, 
            dropout_prob=0.2  # Del resumen
        ).to(device),
        'name': 'TLS_LSTMModel',
        'epochs': 100  # Del resumen
    })
    
    # 2. GRU_Model - Configuración del resumen
    print("🔧 GRU_Model - Configuración del resumen")
    models_to_train.append({
        'model': GRU_Model(
            input_size=input_size, 
            hidden_size=50,  # Del resumen
            output_size=1, 
            dropout_prob=0.2,  # Del resumen
            num_layers=2  # Del resumen
        ).to(device),
        'name': 'GRU_Model',
        'epochs': 100  # Del resumen
    })
    
    # 3. HybridLSTMAttentionModel - Configuración del resumen  
    print("🔧 HybridLSTMAttentionModel - Configuración del resumen")
    models_to_train.append({
        'model': HybridLSTMAttentionModel(
            input_size=input_size, 
            hidden_size=50,  # Del resumen
            output_size=1, 
            dropout_prob=0.2  # Del resumen
        ).to(device),
        'name': 'HybridLSTMAttentionModel',
        'epochs': 100  # Del resumen
    })
    
    # 4. BidirectionalDeepLSTMModel - Configuración del resumen
    print("🔧 BidirectionalDeepLSTMModel - Configuración del resumen")
    models_to_train.append({
        'model': BidirectionalDeepLSTMModel(
            input_size=input_size, 
            hidden_size=50,  # Del resumen
            output_size=1, 
            dropout_prob=0.2  # Del resumen
        ).to(device),
        'name': 'BidirectionalDeepLSTMModel',
        'epochs': 100  # Del resumen
    })
    
    # 5. TLS_LSTMModel Optimizado - Configuración EXACTA del resumen
    print("🔧 TLS_LSTMModel Optimizado - Configuración avanzada del resumen")
    models_to_train.append({
        'model': TLS_LSTMModel(
            input_size=input_size, 
            hidden_size=128,  # Del resumen - OPTIMIZADO
            output_size=1, 
            dropout_prob=0.1  # Del resumen - REDUCIDO
        ).to(device),
        'name': 'TLS_LSTMModel_Optimizado',
        'epochs': 150  # Del resumen - AUMENTADAS
    })
    
    # 6. ContextualLSTMTransformerFlexible - Configuración del resumen
    print("🔧 ContextualLSTMTransformerFlexible - Configuración del resumen")
    models_to_train.append({
        'model': ContextualLSTMTransformerFlexible(
            seq_len=30,  # Del resumen
            feature_dim=input_size, 
            output_size=1,
            window_size=6,  # Del resumen
            max_neighbors=1,  # Del resumen
            lstm_units=32,  # Del resumen
            num_heads=2,  # Del resumen
            embed_dim=64,  # Del resumen
            dropout_rate=0.2  # Del resumen
        ).to(device),
        'name': 'ContextualLSTMTransformerFlexible',
        'epochs': 150  # Del resumen
    })
    
    # Entrenar todos los modelos
    results = []
    total_start_time = time.time()
    
    for i, model_info in enumerate(models_to_train, 1):
        print(f"\n🔄 [{i}/{len(models_to_train)}] Entrenando {model_info['name']}...")
        print(f"   📋 Configuración del resumen aplicada")
        
        # Re-fijar semilla antes de cada modelo para mayor reproducibilidad
        set_seed(42 + i)
        
        try:
            result = train_model(
                model_info['model'], 
                data, 
                model_info['name'], 
                model_info['epochs']
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error entrenando {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Resumen final
    print("\n🏆 RESUMEN FINAL - CONFIGURACIONES DEL RESUMEN APLICADAS")
    print("=" * 85)
    print("🎲 Semillas fijas utilizadas para reproducibilidad")
    print("📋 Configuraciones exactas del archivo de resumen aplicadas")
    print("-" * 85)
    
    # Ordenar por RMSE
    results_sorted = sorted(results, key=lambda x: x['test_rmse'])
    
    print(f"{'Rank':<4} {'Modelo':<35} {'RMSE':<8} {'R²':<8} {'DA':<6} {'Params':<8} {'Tiempo':<12} {'Config'}")
    print("-" * 85)
    
    for i, result in enumerate(results_sorted, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        config_info = "Resumen✓"
        print(f"{emoji}{i:<3} {result['model_name']:<35} {result['test_rmse']:<8.6f} {result['test_r2']:<8.4f} {result['test_da']:<6.3f} {result['parameters']:<8,} {str(timedelta(seconds=result['training_time'])):<12} {config_info}")
    
    print(f"\n⏱️ Tiempo total: {timedelta(seconds=total_time)}")
    print(f"💾 Todos los modelos guardados en: modelos/")
    print(f"🎯 Configuraciones aplicadas desde el resumen de modelos entrenados")
    
    # Guardar resumen
    summary_path = Path("modelos") / DEFAULT_PARAMS.TABLENAME / "training_summary_with_resume_configs.txt"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE ENTRENAMIENTO CON CONFIGURACIONES DEL RESUMEN\n")
        f.write("=" * 60 + "\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Semilla fija: 42 (reproducibilidad garantizada)\n")
        f.write(f"Features: {data['feature_columns']}\n")
        f.write(f"Secuencias: {data['seq_length']}\n")
        f.write("Configuraciones aplicadas desde resumen de modelos entrenados\n\n")
        
        f.write("RESULTADOS:\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(results_sorted, 1):
            f.write(f"{i}. {result['model_name']}\n")
            f.write(f"   RMSE: {result['test_rmse']:.6f}\n")
            f.write(f"   R²: {result['test_r2']:.6f}\n")
            f.write(f"   DA: {result['test_da']:.4f}\n")
            f.write(f"   Parámetros: {result['parameters']:,}\n")
            f.write(f"   Tiempo: {timedelta(seconds=result['training_time'])}\n")
            f.write(f"   Archivo: {result['model_path']}\n\n")
        
        f.write("CONFIGURACIONES ESPECÍFICAS APLICADAS:\n")
        f.write("-" * 40 + "\n")
        f.write("• TLS_LSTMModel: hidden_size=50, dropout=0.2, epochs=100\n")
        f.write("• GRU_Model: hidden_size=50, dropout=0.2, num_layers=2, epochs=100\n")
        f.write("• HybridLSTMAttentionModel: hidden_size=50, dropout=0.2, epochs=100\n") 
        f.write("• BidirectionalDeepLSTMModel: hidden_size=50, dropout=0.2, epochs=100\n")
        f.write("• TLS_LSTMModel_Optimizado: hidden_size=128, dropout=0.1, epochs=150\n")
        f.write("• ContextualLSTMTransformerFlexible: lstm_units=32, embed_dim=64, epochs=150\n")
    
    print(f"📄 Resumen detallado guardado en: {summary_path}")

if __name__ == "__main__":
    main()
