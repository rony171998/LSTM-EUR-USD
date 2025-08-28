#!/usr/bin/env python3
"""
recreate_72_da_gru.py - Recrear el modelo GRU exitoso con 72.41% DA
Basado en los parámetros exactos que lograron este resultado
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import DEFAULT_PARAMS
from modelos import GRU_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARÁMETROS EXACTOS DEL MODELO EXITOSO (72.41% DA)
WINNING_PARAMS = {
    "hidden_size": 128,
    "learning_rate": 0.0010059426888791,
    "dropout_prob": 0.3441023356173669,
    "batch_size": 16,
    "seq_length": 60
}

def recreate_winning_gru():
    """Recrear exactamente el modelo GRU que logró 72.41% DA"""
    print("🏆 RECREANDO MODELO GRU CAMPEÓN (72.41% DA)")
    print("=" * 60)
    print("🎯 Usando parámetros exactos del modelo exitoso")
    print("=" * 60)
    
    # Mostrar parámetros
    print("📊 PARÁMETROS DEL CAMPEÓN:")
    for key, value in WINNING_PARAMS.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Cargar datos (misma función que el modelo exitoso)
    data = load_winning_data()
    
    # Entrenar con parámetros exactos
    model = train_winning_gru(data)
    
    # Evaluar y verificar
    test_results = evaluate_winning_model(model, data)
    
    print(f"\n🎯 RESULTADOS DE LA RECREACIÓN:")
    print(f"   📊 Test DA: {test_results['da']:.4f} ({test_results['da']*100:.1f}%)")
    print(f"   📉 Test RMSE: {test_results['rmse']:.8f}")
    print(f"   📈 Test R²: {test_results['r2']:.6f}")
    
    # Guardar modelo recreado
    save_recreated_model(model, test_results)
    
    return model, test_results

def load_winning_data():
    """Cargar datos exactamente como el modelo ganador"""
    print("\n📊 Cargando datos (método del campeón)...")
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        data_prefix = "../data/"
    else:
        data_prefix = "data/"
    
    # EUR/USD
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
    
    # DXY
    dxy_file = f"{data_prefix}DXY_2010-2024.csv"
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
    
    # Crear características como el modelo ganador
    features_df = create_winning_features(eur_prices, dxy_prices)
    
    print(f"   ✅ Datos: {features_df.shape}")
    
    return features_df

def create_winning_features(eur_prices, dxy_prices):
    """Crear características exactamente como el modelo ganador"""
    print("   🔧 Creando características...")
    
    # Returns
    eur_returns = eur_prices.pct_change()
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_prices)
    
    # SMA20
    eur_sma20 = eur_prices.rolling(window=20).mean()
    
    # DXY returns (aligned)
    common_dates = eur_prices.index.intersection(dxy_prices.index)
    dxy_aligned = dxy_prices.reindex(common_dates)
    dxy_returns = dxy_aligned.pct_change()
    
    eur_aligned = eur_prices.reindex(common_dates)
    eur_returns_aligned = eur_aligned.pct_change()
    eur_rsi_aligned = calculate_rsi(eur_aligned)
    eur_sma20_aligned = eur_aligned.rolling(window=20).mean()
    
    features_df = pd.DataFrame({
        'price': eur_aligned,
        'returns': eur_returns_aligned,
        'rsi': eur_rsi_aligned,
        'sma20': eur_sma20_aligned,
        'dxy_returns': dxy_returns
    })
    
    features_df = features_df.dropna()
    return features_df

def train_winning_gru(data):
    """Entrenar con parámetros exactos del campeón"""
    print("\n🚀 Entrenando modelo campeón...")
    
    # Preparar datos
    target_data = data['price']
    feature_columns = [col for col in data.columns if col != 'price']
    features_data = data[feature_columns]
    
    # Split 80/20
    train_size = int(len(features_data) * 0.8)
    
    X_train_raw = features_data.iloc[:train_size].values
    y_train_raw = target_data.iloc[:train_size].values
    X_test_raw = features_data.iloc[train_size:].values
    y_test_raw = target_data.iloc[train_size:].values
    
    # Escalado
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    seq_length = WINNING_PARAMS['seq_length']
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    # Tensores
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    y_test_tensor = torch.FloatTensor(y_test_seq).to(device)
    
    # Crear modelo
    input_size = X_train_seq.shape[2]
    model = GRU_Model(
        input_size=input_size,
        hidden_size=WINNING_PARAMS['hidden_size'],
        output_size=1,
        dropout_prob=WINNING_PARAMS['dropout_prob'],
        num_layers=2
    ).to(device)
    
    # Optimizer y loss
    optimizer = torch.optim.Adam(model.parameters(), lr=WINNING_PARAMS['learning_rate'])
    criterion = nn.MSELoss()
    
    # Entrenamiento
    model.train()
    best_val_loss = float('inf')
    
    print(f"   📊 Entrenando por 200 épocas...")
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor).squeeze()
                val_loss = criterion(val_outputs, y_test_tensor)
            print(f"   Época {epoch}: Loss={loss:.6f}, Val_Loss={val_loss:.6f}")
            model.train()
    
    print("   ✅ Entrenamiento completado")
    
    # Guardar scalers en el modelo
    model.scaler = scaler
    model.target_scaler = target_scaler
    model.seq_length = seq_length
    
    return model

def evaluate_winning_model(model, data):
    """Evaluar modelo recreado"""
    print("\n📊 Evaluando modelo recreado...")
    
    target_data = data['price']
    feature_columns = [col for col in data.columns if col != 'price']
    features_data = data[feature_columns]
    
    train_size = int(len(features_data) * 0.8)
    X_test_raw = features_data.iloc[train_size:].values
    y_test_raw = target_data.iloc[train_size:].values
    
    # Usar scalers del modelo
    X_test_scaled = model.scaler.transform(X_test_raw)
    y_test_scaled = model.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, model.seq_length)
    
    # Predicción
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
        test_pred_scaled = model(X_test_tensor).squeeze()
        test_pred = model.target_scaler.inverse_transform(
            test_pred_scaled.cpu().numpy().reshape(-1, 1)
        ).flatten()
    
    y_test_actual = y_test_raw[model.seq_length:]
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    r2 = r2_score(y_test_actual, test_pred)
    
    # Directional Accuracy
    def directional_accuracy(y_true, y_pred):
        if len(y_true) <= 1:
            return 0.5
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction)
    
    da = directional_accuracy(y_test_actual, test_pred)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'da': da,
        'predictions': test_pred,
        'actual': y_test_actual
    }

def save_recreated_model(model, results):
    """Guardar modelo recreado"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    model_path = models_dir / f"GRU_Model_RECREATED_72DA_{timestamp}.pth"
    
    # Checkpoint completo
    checkpoint = {
        'model_class': 'GRU_Model',
        'model_state_dict': model.state_dict(),
        'optuna_params': WINNING_PARAMS.copy(),
        'seq_length': WINNING_PARAMS['seq_length'],
        'test_rmse': results['rmse'],
        'test_r2': results['r2'],
        'direction_accuracy': results['da'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'recreation_source': '72.41% DA original model',
        'scaler_state': model.scaler,
        'target_scaler_state': model.target_scaler
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n💾 Modelo recreado guardado: {model_path}")

def main():
    """Función principal"""
    print("🎯 RECREACIÓN DEL MODELO GRU CAMPEÓN")
    
    model, results = recreate_winning_gru()
    
    if results['da'] > 0.70:
        print(f"\n🎉 ¡ÉXITO! DA: {results['da']*100:.1f}% > 70%")
    elif results['da'] > 0.60:
        print(f"\n✅ Muy bueno! DA: {results['da']*100:.1f}% > 60%")
    else:
        print(f"\n⚠️ DA: {results['da']*100:.1f}% - Puede necesitar ajuste")

if __name__ == "__main__":
    main()
