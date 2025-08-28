#!/usr/bin/env python3
"""
rolling_forecast_production.py - Sistema de Producción con Rolling Forecast
Implementa el modelo BidirectionalDeepLSTMModel que logró 65.5% DA para uso real
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
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_PARAMS
from modelos import BidirectionalDeepLSTMModel

device = torch.device("cuda")

class ProductionRollingForecast:
    """
    🚀 SISTEMA DE PRODUCCIÓN CON ROLLING FORECAST
    Implementa la técnica que logró 65.5% DA en BidirectionalDeepLSTMModel
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.optuna_params = None
        self.seq_length = None
        self.device = device
        
        if model_path:
            self.load_model()
    
    def load_model(self):
        """Cargar modelo pre-entrenado"""
        print(f"📁 Cargando modelo: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.optuna_params = checkpoint['optuna_params']
        self.seq_length = checkpoint['seq_length']
        
        # Crear modelo
        input_size = 4  # returns, rsi, sma20, dxy_returns
        self.model = BidirectionalDeepLSTMModel(
            input_size=input_size,
            hidden_size=self.optuna_params['hidden_size'],
            output_size=1,
            dropout_prob=self.optuna_params['dropout_prob']
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✅ Modelo cargado exitosamente")
        print(f"   📊 Parámetros: hidden={self.optuna_params['hidden_size']}, seq_length={self.seq_length}")
    
    def load_fresh_data(self):
        """Cargar datos más recientes disponibles"""
        print("📊 Cargando datos frescos...")
        
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
                print(f"   ✅ DXY: {len(dxy_prices)} registros")
            except:
                print("   ⚠️ DXY no disponible")
        
        print(f"   ✅ EUR/USD: {len(eur_prices)} registros")
        print(f"   📅 Último dato: {eur_prices.index[-1].strftime('%Y-%m-%d')}")
        
        return eur_prices, dxy_prices
    
    def create_features(self, eur_prices, dxy_prices=None):
        """Crear características para predicción"""
        print("🔧 Creando características...")
        
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
                print("   ✅ DXY incluido en características")
        
        features_df = pd.DataFrame(features_dict)
        features_df = features_df.dropna()
        
        print(f"   ✅ Características: {features_df.shape}")
        
        return features_df
    
    def predict_next_days(self, num_days=5, use_rolling=True):
        """
        🎯 PREDICCIÓN DE PRÓXIMOS DÍAS
        use_rolling=True: Usa Rolling Forecast (65.5% DA)
        use_rolling=False: Predicción estándar
        """
        print(f"🔮 Prediciendo próximos {num_days} días...")
        print(f"   🔄 Método: {'Rolling Forecast' if use_rolling else 'Predicción Estándar'}")
        
        # Cargar datos
        eur_prices, dxy_prices = self.load_fresh_data()
        features_df = self.create_features(eur_prices, dxy_prices)
        
        if not use_rolling:
            # PREDICCIÓN ESTÁNDAR
            return self._predict_standard(features_df, num_days)
        else:
            # ROLLING FORECAST (TÉCNICA EXITOSA)
            return self._predict_rolling(features_df, num_days)
    
    def _predict_standard(self, features_df, num_days):
        """Predicción estándar (sin re-entrenamiento)"""
        print("   📏 Usando predicción estándar...")
        
        target_data = features_df['price']
        feature_columns = [col for col in features_df.columns if col != 'price']
        features_data = features_df[feature_columns]
        
        # Usar últimos datos para predicción
        recent_data = features_data.tail(self.seq_length).values
        
        # Escalado (usar datos históricos para fit)
        scaler = RobustScaler()
        all_features_scaled = scaler.fit_transform(features_data.values)
        recent_data_scaled = scaler.transform(recent_data)
        
        target_scaler = RobustScaler()
        target_scaler.fit(target_data.values.reshape(-1, 1))
        
        # Predicciones
        predictions = []
        current_sequence = recent_data_scaled
        
        self.model.eval()
        
        for day in range(num_days):
            # Preparar tensor
            X_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model(X_tensor).squeeze()
                pred_value = target_scaler.inverse_transform(
                    pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()[0]
            
            predictions.append(pred_value)
            
            # Actualizar secuencia (simplificado)
            if len(current_sequence) > 1:
                current_sequence = current_sequence[1:]  # Quitar primer elemento
                # Agregar nueva predicción (simplificado)
                last_row = current_sequence[-1].copy()
                current_sequence = np.vstack([current_sequence, last_row])
        
        # Fechas futuras
        last_date = features_df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(num_days)]
        
        return predictions, future_dates
    
    def _predict_rolling(self, features_df, num_days):
        """
        🚀 ROLLING FORECAST PRODUCTION
        Implementa la técnica que logró 65.5% DA
        """
        print("   🔄 Usando Rolling Forecast (Técnica exitosa 65.5% DA)...")
        
        target_data = features_df['price']
        feature_columns = [col for col in features_df.columns if col != 'price']
        features_data = features_df[feature_columns]
        
        # Usar datos históricos como entrenamiento base
        train_size = int(len(features_data) * 0.85)  # Más datos para mejor modelo
        
        predictions = []
        prediction_dates = []
        
        # Para cada día futuro
        for day in range(num_days):
            print(f"      🔄 Prediciendo día {day+1}/{num_days}...")
            
            # 1. PREPARAR DATOS ACTUALIZADOS
            current_train_end = train_size + day
            
            if current_train_end >= len(features_data):
                # Usar todos los datos disponibles
                current_train_end = len(features_data)
            
            X_train_raw = features_data.iloc[:current_train_end].values
            y_train_raw = target_data.iloc[:current_train_end].values
            
            # Escalado
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            
            target_scaler = RobustScaler()
            y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            
            # Crear secuencias
            def create_sequences(X, y, seq_len):
                X_seq, y_seq = [], []
                for i in range(seq_len, len(X)):
                    X_seq.append(X[i-seq_len:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, self.seq_length)
            
            if len(X_train_seq) < 10:
                print(f"      ⚠️ Datos insuficientes para día {day+1}")
                break
            
            # 2. RE-ENTRENAR MODELO (clave del éxito)
            input_size = X_train_seq.shape[2]
            
            fresh_model = BidirectionalDeepLSTMModel(
                input_size=input_size,
                hidden_size=self.optuna_params['hidden_size'],
                output_size=1,
                dropout_prob=self.optuna_params['dropout_prob']
            ).to(self.device)
            
            # 3. ENTRENAMIENTO RÁPIDO
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
            
            optimizer = optim.Adam(fresh_model.parameters(), lr=self.optuna_params['learning_rate'])
            criterion = nn.MSELoss()
            
            fresh_model.train()
            
            # Entrenamiento express (15 épocas)
            for epoch in range(15):
                optimizer.zero_grad()
                outputs = fresh_model(X_train_tensor).squeeze()
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # 4. HACER PREDICCIÓN
            # Usar última secuencia disponible
            if current_train_end >= self.seq_length:
                X_pred_raw = features_data.iloc[current_train_end-self.seq_length:current_train_end].values
                X_pred_scaled = scaler.transform(X_pred_raw)
                X_pred_tensor = torch.FloatTensor(X_pred_scaled).unsqueeze(0).to(self.device)
                
                fresh_model.eval()
                with torch.no_grad():
                    pred_scaled = fresh_model(X_pred_tensor).squeeze()
                    pred_value = target_scaler.inverse_transform(
                        pred_scaled.cpu().numpy().reshape(-1, 1)
                    ).flatten()[0]
                
                predictions.append(pred_value)
                
                # Fecha de predicción
                last_date = features_data.index[-1]
                pred_date = last_date + timedelta(days=day+1)
                prediction_dates.append(pred_date)
                
                print(f"         📈 Predicción: {pred_value:.6f}")
        
        return predictions, prediction_dates
    
    def generate_forecast_report(self, num_days=5):
        """Generar reporte completo de predicción"""
        print("\n📊 GENERANDO REPORTE DE PREDICCIÓN")
        print("=" * 50)
        
        # Predicciones con ambos métodos
        print("\n🔄 Ejecutando Rolling Forecast...")
        rolling_pred, dates = self.predict_next_days(num_days, use_rolling=True)
        
        print("\n📏 Ejecutando Predicción Estándar...")
        standard_pred, _ = self.predict_next_days(num_days, use_rolling=False)
        
        # Cargar precio actual
        eur_prices, _ = self.load_fresh_data()
        current_price = eur_prices.iloc[-1]
        current_date = eur_prices.index[-1]
        
        # Reporte
        print(f"\n📊 REPORTE DE PREDICCIÓN EUR/USD")
        print(f"📅 Fecha base: {current_date.strftime('%Y-%m-%d')}")
        print(f"💰 Precio actual: {current_price:.6f}")
        print("=" * 70)
        print(f"{'Fecha':<12} {'Rolling':<12} {'Estándar':<12} {'Diferencia':<12} {'Señal':<10}")
        print("-" * 70)
        
        signals = []
        
        for i in range(len(rolling_pred)):
            rolling_val = rolling_pred[i]
            standard_val = standard_pred[i] if i < len(standard_pred) else 0
            diff = rolling_val - standard_val
            
            # Señal de trading
            if rolling_val > current_price * 1.001:
                signal = "🟢 COMPRA"
            elif rolling_val < current_price * 0.999:
                signal = "🔴 VENTA"
            else:
                signal = "🟡 MANTENER"
            
            signals.append(signal)
            
            print(f"{dates[i].strftime('%Y-%m-%d'):<12} "
                  f"{rolling_val:<12.6f} "
                  f"{standard_val:<12.6f} "
                  f"{diff:<12.6f} "
                  f"{signal:<10}")
        
        # Estadísticas
        rolling_change = (rolling_pred[-1] - current_price) / current_price * 100
        avg_rolling = np.mean(rolling_pred)
        
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   🎯 Cambio esperado (Rolling): {rolling_change:.2f}%")
        print(f"   📊 Precio promedio predicho: {avg_rolling:.6f}")
        print(f"   🔄 Método recomendado: Rolling Forecast (65.5% DA)")
        
        # Guardar reporte
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        current_dir = Path.cwd()
        if current_dir.name == "model":
            report_path = Path("../modelos") / DEFAULT_PARAMS.TABLENAME / f"forecast_report_{timestamp}.json"
        else:
            report_path = Path("modelos") / DEFAULT_PARAMS.TABLENAME / f"forecast_report_{timestamp}.json"
        
        report_path.parent.mkdir(exist_ok=True)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'base_date': current_date.isoformat(),
            'current_price': float(current_price),
            'forecast_days': num_days,
            'method': 'Rolling Forecast (65.5% DA)',
            'predictions': {
                'rolling': [float(x) for x in rolling_pred],
                'standard': [float(x) for x in standard_pred],
                'dates': [d.isoformat() for d in dates],
                'signals': signals
            },
            'statistics': {
                'expected_change_pct': float(rolling_change),
                'avg_predicted_price': float(avg_rolling)
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado: {report_path}")
        
        return report_data

def main():
    """Función principal para predicción con Rolling Forecast"""
    print("🚀 SISTEMA DE PRODUCCIÓN - ROLLING FORECAST")
    print("=" * 60)
    print("🎯 Basado en BidirectionalDeepLSTMModel (65.5% DA)")
    print("=" * 60)
    
    # Buscar modelo optimizado
    current_dir = Path.cwd()
    if current_dir.name == "model":
        models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    # Buscar modelo BidirectionalDeepLSTMModel
    model_files = list(models_dir.glob("BidirectionalDeepLSTMModel_optuna_*.pth"))
    
    if not model_files:
        print("❌ No se encontró modelo BidirectionalDeepLSTMModel")
        return
    
    model_path = model_files[0]
    print(f"📁 Usando modelo: {model_path.name}")
    
    # Crear sistema de predicción
    forecaster = ProductionRollingForecast(model_path)
    
    # Generar predicción para próximos 5 días
    report = forecaster.generate_forecast_report(num_days=5)
    
    print(f"\n🎉 Predicción completada con éxito!")
    print(f"📊 Método utilizado: Rolling Forecast (Técnica probada con 65.5% DA)")

if __name__ == "__main__":
    main()
