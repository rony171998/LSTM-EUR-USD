#!/usr/bin/env python3
"""
train_models_with_rolling_optimization.py
Entrena modelos incorporando las técnicas exitosas del Rolling Forecast
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
)

device = torch.device("cuda")

# Establecer semilla para reproducibilidad
def set_seed(seed=DEFAULT_PARAMS.SEED):
    """Establecer semilla para reproducibilidad"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_multi_asset_data():
    """Cargar EUR/USD + DXY"""
    print("📊 Cargando EUR/USD + DXY...")
    
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
    """Crear características probadas"""
    print("🔧 Creando características probadas...")
    
    # 1. EUR/USD returns
    eur_returns = eur_prices.pct_change()
    
    # 2. RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    eur_rsi = calculate_rsi(eur_prices)
    
    # 3. SMA20
    eur_sma20 = eur_prices.rolling(window=20).mean()
    
    # Crear DataFrame base
    features_dict = {
        'price': eur_prices,
        'returns': eur_returns,
        'rsi': eur_rsi,
        'sma20': eur_sma20
    }
    
    # 4. DXY returns (si está disponible)
    if dxy_prices is not None:
        common_dates = eur_prices.index.intersection(dxy_prices.index)
        if len(common_dates) > 1000:
            dxy_aligned = dxy_prices.reindex(common_dates)
            dxy_returns = dxy_aligned.pct_change()
            
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
    
    features_df = pd.DataFrame(features_dict)
    features_df = features_df.dropna()
    
    print(f"✅ Características: {features_df.shape}")
    print(f"   Features: {list(features_df.columns)}")
    
    return features_df

def directional_accuracy_loss(y_pred, y_true):
    """
    🎯 LOSS FUNCTION PARA DIRECTIONAL ACCURACY
    Combina MSE con penalización por dirección incorrecta
    """
    # MSE básico
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    # Penalización direccional
    if len(y_pred) > 1:
        # Calcular direcciones
        pred_direction = torch.diff(y_pred) > 0
        true_direction = torch.diff(y_true) > 0
        
        # Penalización por dirección incorrecta
        direction_penalty = torch.mean((pred_direction != true_direction).float())
        
        # Combinar pérdidas
        total_loss = mse_loss + 0.1 * direction_penalty
    else:
        total_loss = mse_loss
    
    return total_loss

class RollingOptimizedTrainer:
    """Entrenador optimizado con técnicas de Rolling Forecast"""
    
    def __init__(self, model, optuna_params, seq_length):
        self.model = model
        self.optuna_params = optuna_params
        self.seq_length = seq_length
        self.device = device
        
    def train_with_rolling_validation(self, features_df, epochs=100):
        """
        🚀 ENTRENAMIENTO CON VALIDACIÓN ROLLING
        Simula el éxito del Rolling Forecast durante entrenamiento
        """
        print(f"🔄 Entrenamiento con Rolling Validation ({epochs} épocas)")
        
        target_data = features_df['price']
        feature_columns = [col for col in features_df.columns if col != 'price']
        features_data = features_df[feature_columns]
        
        # Split temporal (70% train, 15% rolling validation, 15% test)
        train_size = int(len(features_data) * 0.70)
        val_size = int(len(features_data) * 0.15)
        
        X_train_raw = features_data.iloc[:train_size].values
        X_val_raw = features_data.iloc[train_size:train_size+val_size].values
        X_test_raw = features_data.iloc[train_size+val_size:].values
        
        y_train_raw = target_data.iloc[:train_size].values
        y_val_raw = target_data.iloc[train_size:train_size+val_size].values
        y_test_raw = target_data.iloc[train_size+val_size:].values
        
        # Escalado
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        # Crear secuencias
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(seq_len, len(X)):
                X_seq.append(X[i-seq_len:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, self.seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, self.seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, self.seq_length)
        
        # Tensores
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).to(self.device)
        
        # Optimizador
        optimizer = optim.Adam(self.model.parameters(), lr=self.optuna_params['learning_rate'])
        
        # 🎯 USAR DIRECTIONAL ACCURACY LOSS
        criterion = directional_accuracy_loss
        
        # Early stopping
        best_val_da = 0
        patience = 20
        patience_counter = 0
        
        print(f"   📊 Train: {len(X_train_seq)}, Val: {len(X_val_seq)}, Test: {len(X_test_seq)}")
        
        # Entrenamiento
        train_losses = []
        val_das = []
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            optimizer.zero_grad()
            
            train_outputs = self.model(X_train_tensor).squeeze()
            train_loss = criterion(train_outputs, y_train_tensor)
            
            train_loss.backward()
            optimizer.step()
            
            train_losses.append(train_loss.item())
            
            # Validación cada 10 épocas
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    
                    # Desnormalizar para calcular DA
                    val_pred_real = target_scaler.inverse_transform(
                        val_outputs.cpu().numpy().reshape(-1, 1)
                    ).flatten()
                    val_true_real = target_scaler.inverse_transform(
                        y_val_tensor.cpu().numpy().reshape(-1, 1)
                    ).flatten()
                    
                    # Directional Accuracy
                    def directional_accuracy(y_true, y_pred):
                        if len(y_true) <= 1:
                            return 0.5
                        true_direction = np.diff(y_true) > 0
                        pred_direction = np.diff(y_pred) > 0
                        return np.mean(true_direction == pred_direction)
                    
                    val_da = directional_accuracy(val_true_real, val_pred_real)
                    val_das.append(val_da)
                    
                    print(f"   Epoch {epoch+1:3d}: Loss={train_loss:.6f}, Val_DA={val_da:.4f}")
                    
                    # Early stopping basado en DA
                    if val_da > best_val_da:
                        best_val_da = val_da
                        patience_counter = 0
                        # Guardar mejor modelo
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"   🛑 Early stopping: Mejor DA={best_val_da:.4f}")
                        break
        
        # Cargar mejor modelo
        self.model.load_state_dict(best_model_state)
        
        # Evaluación final en test
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor).squeeze()
            
            test_pred_real = target_scaler.inverse_transform(
                test_outputs.cpu().numpy().reshape(-1, 1)
            ).flatten()
            test_true_real = target_scaler.inverse_transform(
                y_test_tensor.cpu().numpy().reshape(-1, 1)
            ).flatten()
            
            test_rmse = np.sqrt(mean_squared_error(test_true_real, test_pred_real))
            test_r2 = r2_score(test_true_real, test_pred_real)
            test_da = directional_accuracy(test_true_real, test_pred_real)
        
        print(f"   ✅ Test - RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}, DA: {test_da:.4f}")
        
        return {
            'model': self.model,
            'scalers': {'feature_scaler': scaler, 'target_scaler': target_scaler},
            'metrics': {
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_da': test_da,
                'best_val_da': best_val_da
            },
            'training_history': {
                'train_losses': train_losses,
                'val_das': val_das
            }
        }

def train_model_with_rolling_optimization(model_class, model_name, optuna_params, seq_length):
    """Entrenar modelo específico con optimizaciones de Rolling Forecast"""
    
    print(f"\n🚀 Entrenando {model_name} con Rolling Optimization")
    print(f"   📊 Parámetros: hidden={optuna_params['hidden_size']}, lr={optuna_params['learning_rate']:.6f}")
    
    # Cargar datos
    eur_prices, dxy_prices = load_multi_asset_data()
    features_df = create_proven_features(eur_prices, dxy_prices)
    
    # Crear modelo
    input_size = 4  # returns, rsi, sma20, dxy_returns
    
    if model_name == "BidirectionalDeepLSTMModel":
        model = BidirectionalDeepLSTMModel(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob']
        ).to(device)
    elif model_name == "GRU_Model":
        model = GRU_Model(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob'],
            num_layers=2
        ).to(device)
    elif model_name == "TLS_LSTMModel":
        model = TLS_LSTMModel(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob']
        ).to(device)
    elif model_name == "HybridLSTMAttentionModel":
        model = HybridLSTMAttentionModel(
            input_size=input_size,
            hidden_size=optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=optuna_params['dropout_prob']
        ).to(device)
    else:
        print(f"❌ Modelo {model_name} no soportado")
        return None
    
    # Entrenador optimizado
    trainer = RollingOptimizedTrainer(model, optuna_params, seq_length)
    
    # Entrenar con técnicas de rolling
    result = trainer.train_with_rolling_validation(features_df, epochs=200)
    
    # Guardar modelo
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    models_dir.mkdir(exist_ok=True)
    
    model_filename = f"{model_name}_rolling_optimized_{DEFAULT_PARAMS.FILEPATH}.pth"
    model_path = models_dir / model_filename
    
    # Checkpoint completo
    checkpoint = {
        'model_class': model_name,
        'model_state_dict': result['model'].state_dict(),
        'optuna_params': optuna_params,
        'seq_length': seq_length,
        'input_size': input_size,
        'test_rmse': result['metrics']['test_rmse'],
        'test_r2': result['metrics']['test_r2'],
        'test_da': result['metrics']['test_da'],
        'best_val_da': result['metrics']['best_val_da'],
        'timestamp': timestamp,
        'optimization_technique': 'Rolling Forecast Inspired',
        'training_epochs': len(result['training_history']['train_losses'])
    }
    
    torch.save(checkpoint, model_path)
    
    print(f"   💾 Modelo guardado: {model_path}")
    print(f"   🎯 DA Final: {result['metrics']['test_da']:.4f}")
    
    return result

def main():
    """Entrenar modelos exitosos con Rolling Optimization"""
    print("🚀 ENTRENAMIENTO CON ROLLING OPTIMIZATION")
    print("=" * 60)
    print("🎯 Basado en técnicas exitosas de Rolling Forecast")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Usando CPU")

    set_seed(DEFAULT_PARAMS.SEED)

    # Parámetros exitosos del Rolling Forecast
    successful_models = [
        {
            'name': 'BidirectionalDeepLSTMModel',
            'params': {
                'hidden_size': 512,
                'learning_rate': 0.000536,
                'dropout_prob': 0.1,
                'batch_size': 32
            },
            'seq_length': 30
        },
        {
            'name': 'GRU_Model', 
            'params': {
                'hidden_size': 128,
                'learning_rate': 0.001006,
                'dropout_prob': 0.2,
                'batch_size': 16
            },
            'seq_length': 60
        },
        {
            'name': 'HybridLSTMAttentionModel',
            'params': {
                'hidden_size': 256,
                'learning_rate': 0.000892,
                'dropout_prob': 0.15,
                'batch_size': 32
            },
            'seq_length': 45
        },
        {
            'name': 'TLS_LSTMModel',
            'params': {
                'hidden_size': 128,
                'learning_rate': 0.001234,
                'dropout_prob': 0.25,
                'batch_size': 16
            },
            'seq_length': 30
        }
    ]
    
    print(f"\n🔍 Entrenando {len(successful_models)} modelos exitosos...")
    
    results = []
    
    for i, model_config in enumerate(successful_models, 1):
        print(f"\n🔄 [{i}/{len(successful_models)}] {model_config['name']}")
        
        result = train_model_with_rolling_optimization(
            model_class=None,  # No usado en esta implementación
            model_name=model_config['name'],
            optuna_params=model_config['params'],
            seq_length=model_config['seq_length']
        )
        
        if result:
            results.append({
                'model_name': model_config['name'],
                'test_da': result['metrics']['test_da'],
                'test_rmse': result['metrics']['test_rmse'],
                'best_val_da': result['metrics']['best_val_da']
            })
    
    # Resumen
    print(f"\n🏆 RESUMEN ROLLING OPTIMIZATION")
    print("=" * 60)
    print(f"{'Modelo':<35} {'Test DA':<10} {'Val DA':<10} {'RMSE':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<35} "
              f"{result['test_da']:<10.4f} "
              f"{result['best_val_da']:<10.4f} "
              f"{result['test_rmse']:<10.6f}")
    
    if results:
        best_da = max(r['test_da'] for r in results)
        avg_da = np.mean([r['test_da'] for r in results])
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   🏆 Mejor DA: {best_da:.4f} ({best_da*100:.1f}%)")
        print(f"   📈 DA Promedio: {avg_da:.4f} ({avg_da*100:.1f}%)")
        
        if best_da > 0.52:
            print(f"   🎉 ¡OBJETIVO ALCANZADO! DA > 52%")
        else:
            print(f"   🎯 Cerca del objetivo (52%)")

if __name__ == "__main__":
    main()
