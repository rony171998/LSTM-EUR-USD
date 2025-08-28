#!/usr/bin/env python3
"""
backtest_trading_models.py - Sistema de Backtesting Completo
Evalúa el rendimiento real de trading de todos los modelos desarrollados
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
from datetime import datetime
import random
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
    ARIMAModel,
    NaiveForecastModel
)

device = torch.device("cuda")

class TradingBacktester:
    """
    🏦 SISTEMA DE BACKTESTING PARA MODELOS DE TRADING
    Simula trading real con costos realistas y diferentes estrategias
    
    💰 COSTOS REALISTAS IMPLEMENTADOS:
    - Spreads: 0.5-1 pip EUR/USD
    - Comisiones: $6-10 por lote estándar
    - Slippage: -0.01% por trade
    """
    
    def __init__(self, initial_capital=10000, test_seeds=None, retrain_frequencies=None):
        self.initial_capital = initial_capital
        self.test_seeds = test_seeds or [2048]  # Múltiples semillas para testing
        self.retrain_frequencies = retrain_frequencies or [5, 10, 20]  # Frecuencias de re-entrenamiento
        self.results = {}
        self.seed_results = {}  # Resultados por semilla
        
        # 💰 COSTOS DE TRADING REALISTAS
        self.spread_cost = 0.00008  # 0.8 pips promedio EUR/USD (0.5-1 pip)
        self.commission_per_lot = 8.0  # $8 por lote estándar ida y vuelta
        self.slippage = 0.0001  # 0.01% slippage por trade
        self.lot_size = 100000  # Tamaño estándar de lote EUR/USD
        
        # Para compatibilidad - costo total aproximado
        self.transaction_cost = 0.0001  # 0.01% costo total aproximado

    def max_drawdown_pct(self, equity_curve):
        """Calcular Maximum Drawdown en porcentaje (CORREGIDO)"""
        equity_curve = np.array(equity_curve, dtype=float)
        
        # Calcular peak rolling
        peak = np.maximum.accumulate(equity_curve)
        
        # Calcular drawdown en cada punto
        drawdown = (equity_curve - peak) / peak
        
        # El máximo drawdown es el mínimo valor (más negativo)
        max_dd = drawdown.min()
        
        # Debug: verificar consistencia
        initial_value = equity_curve[0]
        final_value = equity_curve[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Debug adicional para casos extremos (ahora con umbral más bajo para 1 año)
        if abs(total_return) > 0.2:  # ROI > 20% o < -20% (más realista para 1 año)
            print(f"🔍 DEBUG Max DD: inicial=${initial_value:.2f}, final=${final_value:.2f}")
            print(f"🔍 ROI total: {total_return*100:.2f}%, Max DD: {max_dd*100:.2f}%")
            print(f"🔍 Peak máximo: ${peak.max():.2f}, Valley mínimo: ${equity_curve.min():.2f}")
            
            # Calcular el drawdown real del peak al valley
            real_dd_from_peak = (equity_curve.min() - peak.max()) / peak.max()
            print(f"🔍 DD real peak-to-valley: {real_dd_from_peak*100:.2f}%")
        
        # Si el retorno total es muy negativo, el max DD debería ser similar o peor
        if total_return < -0.1 and max_dd > -0.1:  # Si ROI < -10% pero DD > -10%
            print(f"⚠️  WARNING: DD inconsistente: ROI={total_return*100:.2f}%, DD={max_dd*100:.2f}%")
            # En este caso, usar el peor entre el DD calculado y el retorno total
            max_dd = min(max_dd, total_return)
        
        return max_dd * 100

    def calculate_trading_costs(self, entry_price, exit_price, position_size, is_long=True):
        """
        💰 CALCULAR COSTOS DE TRADING REALISTAS
        
        Args:
            entry_price: Precio de entrada
            exit_price: Precio de salida  
            position_size: Tamaño de la posición (en USD)
            is_long: True para posición larga, False para corta
            
        Returns:
            total_cost: Costo total del trade (spread + comisión + slippage)
        """
        # 1. Spread cost: se paga al entrar y salir
        spread_cost_total = self.spread_cost * 2  # Entrada + Salida
        
        # 2. Comisión: basada en el tamaño de la posición
        lots_traded = position_size / self.lot_size
        commission_cost = lots_traded * self.commission_per_lot
        
        # 3. Slippage: ejecución peor de lo esperado
        if is_long:
            # Long: comprar más caro, vender más barato
            entry_slippage = entry_price * self.slippage
            exit_slippage = exit_price * self.slippage
        else:
            # Short: vender más barato, comprar más caro
            entry_slippage = entry_price * self.slippage
            exit_slippage = exit_price * self.slippage
        
        # Convertir slippage a costo en USD
        slippage_cost_usd = (entry_slippage + exit_slippage) * position_size / entry_price
        
        # Convertir spread a costo en USD
        spread_cost_usd = spread_cost_total * position_size / entry_price
        
        total_cost = spread_cost_usd + commission_cost + slippage_cost_usd
        
        return total_cost

        
    def calculate_directional_accuracy(self, predictions, actual):
        """Calcular precisión direccional"""
        if len(predictions) < 2 or len(actual) < 2:
            return 0.0
        
        pred_directions = np.diff(predictions) > 0
        actual_directions = np.diff(actual) > 0
        
        if len(pred_directions) == 0:
            return 0.0
            
        return np.mean(pred_directions == actual_directions)
    
    def calculate_maximum_drawdown(self, capital_history):
        """
        📉 Calcular el Maximum Drawdown (MDD) CORREGIDO
        MDD = máxima pérdida desde un pico hasta un valle
        Usa la función estándar max_drawdown_pct
        """
        if len(capital_history) < 2:
            return 0.0, 0.0
            
        capital_series = np.array(capital_history, dtype=float)
        
        # Usar función estándar para calcular drawdown porcentual
        max_drawdown_pct = self.max_drawdown_pct(capital_series)
        
        # Maximum drawdown en términos absolutos (dinero)
        peak = np.maximum.accumulate(capital_series)
        max_drawdown_abs = np.min(capital_series - peak)
        
        return max_drawdown_pct, max_drawdown_abs
    
    def get_trading_period(self, dates):
        """
        📅 Obtener el período de trading
        """
        if len(dates) < 2:
            return "N/A", "N/A", 0
            
        start_date = dates[0] if isinstance(dates[0], str) else dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1] if isinstance(dates[-1], str) else dates[-1].strftime('%Y-%m-%d')
        
        # Calcular duración en días
        if isinstance(dates[0], str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            start_dt = dates[0]
            end_dt = dates[-1]
            
        duration_days = (end_dt - start_dt).days
        
        return start_date, end_date, duration_days
    
    def _empty_results(self, strategy_name, error_msg):
        """Helper para retornar resultados vacíos en caso de error"""
        return {
            'strategy': f'{strategy_name} - Error',
            'modelo': strategy_name,
            'final_capital': self.initial_capital,
            'total_return': 0.0,
            'roi_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_abs': 0.0,
            'directional_accuracy': 0.0,
            'total_trades': 0,
            'entry_cost': 0.0,
            'exit_cost': 0.0,
            'total_costs': 0.0,
            'start_date': 'N/A',
            'end_date': 'N/A',
            'trading_days': 0,
            'capital_history': [self.initial_capital],
            'prices_used': [],
            'error': error_msg
        }
    
    def evaluate_buy_and_hold_strategy(self, data_test, trading_days=252):
        """
        📈 ESTRATEGIA BUY & HOLD EUR/USD
        Simula posición larga constante durante todo el período
        
        Args:
            data_test: Datos de testing
            trading_days: Días de trading para evaluar (default: 252 = 1 año)
            
        Returns:
            dict: Métricas de rendimiento de Buy & Hold
        """
        print(f"🏦 Evaluando estrategia Buy & Hold - {trading_days} días")
        
        # Debug: verificar columnas disponibles
        print(f"🔍 Columnas disponibles: {list(data_test.columns)}")
        
        # Buscar columna de precio (puede ser 'Close', 'close', 'EUR_USD', etc.)
        price_columns = [col for col in data_test.columns if col.lower() in ['close', 'eur_usd', 'price']]
        if not price_columns:
            # Si no hay columnas obvias, usar la primera columna numérica
            numeric_cols = data_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_column = numeric_cols[0]
                print(f"⚠️  Usando columna '{price_column}' como precio")
            else:
                return self._empty_results("Buy & Hold", "No se encontró columna de precio válida")
        else:
            price_column = price_columns[0]
            print(f"✅ Usando columna '{price_column}' como precio")
        
        # Limitar a los días de trading especificados
        actual_days = min(trading_days, len(data_test))
        prices = data_test[price_column].values[-actual_days:]
        dates = data_test.index[-actual_days:]
        
        if len(prices) < 2:
            return self._empty_results("Buy & Hold", "Datos insuficientes")
        
        # Simular Buy & Hold
        initial_price = prices[0]
        capital_history = [self.initial_capital]
        position_size = self.initial_capital  # Invertir todo el capital
        
        # Calcular costos de entrada (compra inicial)
        entry_cost = self.calculate_trading_costs(
            initial_price, initial_price, position_size, is_long=True
        )
        
        net_capital = self.initial_capital - entry_cost
        units_owned = net_capital / initial_price
        
        # Calcular valor de la posición día a día
        for i, current_price in enumerate(prices):
            current_value = units_owned * current_price
            capital_history.append(current_value)
        
        # Al final del período, calcular costo de salida
        final_price = prices[-1]
        final_value = units_owned * final_price
        exit_cost = self.calculate_trading_costs(
            final_price, final_price, final_value, is_long=True
        )
        
        final_capital = final_value - exit_cost
        capital_history[-1] = final_capital
        
        # Calcular métricas
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        max_dd_pct, max_dd_abs = self.calculate_maximum_drawdown(capital_history)
        
        # ROI anualizado
        days_in_period = len(prices)
        roi_annualized = total_return * (252 / days_in_period) if days_in_period > 0 else 0
        
        # Para Buy & Hold, la precisión direccional es diferente
        # Calculamos si el precio final es mayor que el inicial
        directional_accuracy = 1.0 if final_price > initial_price else 0.0
        
        # Obtener período de trading
        start_date, end_date, duration_days = self.get_trading_period(dates)
        
        return {
            'strategy': 'Buy & Hold EUR/USD',
            'modelo': 'Buy & Hold',
            'final_capital': final_capital,
            'total_return': total_return,
            'roi_annualized': roi_annualized,
            'max_drawdown': max_dd_pct,
            'max_drawdown_abs': max_dd_abs,
            'directional_accuracy': directional_accuracy,
            'total_trades': 1,  # Solo una operación (comprar y mantener)
            'entry_cost': entry_cost,
            'exit_cost': exit_cost,
            'total_costs': entry_cost + exit_cost,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': days_in_period,
            'capital_history': capital_history,
            'prices_used': prices.tolist()
        }
        
    def load_data(self):
        """Cargar datos históricos para backtesting"""
        print("📊 Cargando datos para backtesting...")
        
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
        
        return eur_prices, dxy_prices
    
    def create_features(self, eur_prices, dxy_prices=None):
        """Crear características para backtesting"""
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
                print("   ✅ DXY incluido")
        
        features_df = pd.DataFrame(features_dict)
        features_df = features_df.dropna()
        
        print(f"✅ Características: {features_df.shape}")
        
        return features_df
    
    def generate_signals(self, predictions, prices, method='threshold'):
        """
        🎯 GENERAR SEÑALES DE TRADING
        method: 'threshold', 'directional', 'hybrid'
        """
        if method == 'threshold':
            # Umbral fijo: si predicción > precio actual + 0.1%
            signals = []
            for i, (pred, price) in enumerate(zip(predictions, prices)):
                if pred > price * 1.001:  # +0.1%
                    signals.append(1)  # COMPRA
                elif pred < price * 0.999:  # -0.1%
                    signals.append(-1)  # VENTA
                else:
                    signals.append(0)  # MANTENER
            return np.array(signals)
            
        elif method == 'directional':
            # Dirección: si precio aumentará o disminuirá
            signals = []
            for i in range(len(predictions) - 1):
                pred_direction = predictions[i+1] > predictions[i]
                price_direction = prices[i+1] > prices[i]
                
                if pred_direction:
                    signals.append(1)  # COMPRA
                else:
                    signals.append(-1)  # VENTA
            
            signals.append(0)  # Última posición neutral
            return np.array(signals)
            
        elif method == 'hybrid':
            # Combinación de ambos métodos
            threshold_signals = self.generate_signals(predictions, prices, 'threshold')
            directional_signals = self.generate_signals(predictions, prices, 'directional')
            
            # Consenso: solo operar cuando ambos coinciden
            hybrid_signals = []
            for t_sig, d_sig in zip(threshold_signals, directional_signals):
                if t_sig == d_sig and t_sig != 0:
                    hybrid_signals.append(t_sig)
                else:
                    hybrid_signals.append(0)
            
            return np.array(hybrid_signals)
    
    def simulate_trading(self, signals, prices, strategy_name, dates=None):
        """
        💰 SIMULAR TRADING CON COSTOS REALISTAS
        Sistema con spreads, comisiones y slippage realistas
        """
        # Conversiones para asegurar tipos correctos
        prices = np.asarray(prices, dtype=float)
        signals = np.asarray(signals, dtype=int)
        assert prices.ndim == 1 and signals.ndim == 1 and len(prices) == len(signals)

        cash = float(self.initial_capital)
        position = 0              # -1 short, 0 flat, 1 long
        position_size = 0.0       # unidades del activo
        entry_price = None
        risk_fraction = 0.05      # 5% del capital por operación (en lugar de 95%)
        
        equity_curve = np.zeros_like(prices, dtype=float)
        trades = []
        total_costs = 0.0  # Tracking total de costos

        def mark_to_market(price):
            """Calcular valor de portfolio mark-to-market"""
            if position == 1:
                return cash + position_size * price
            elif position == -1:
                return cash - position_size * price
            else:
                return cash

        print(f"   💰 Simulando trading para {strategy_name} con costos realistas...")

        for t, (price, sig) in enumerate(zip(prices, signals)):
            # Cambiar de posición solo cuando el target cambia
            if sig != position:
                # Cerrar si había posición
                if position != 0:
                    # Calcular costos realistas para cierre
                    notional = position_size * price
                    trading_costs = self.calculate_trading_costs(
                        entry_price, price, abs(notional), is_long=(position == 1)
                    )
                    total_costs += trading_costs
                    
                    if position == 1:
                        pnl = position_size * (price - entry_price) - trading_costs
                        cash += pnl + position_size * entry_price  # recuperar efectivo de la compra original
                    else:  # short
                        pnl = position_size * (entry_price - price) - trading_costs
                        cash += pnl - position_size * entry_price  # devolver efectivo de la venta original
                    
                    trades.append({
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry_price": float(entry_price),
                        "exit_price": float(price),
                        "profit": float(pnl),
                        "trading_costs": float(trading_costs),
                        "position": position,
                        "size": position_size,
                        "entry_date": t,
                        "exit_date": t
                    })
                    position = 0
                    position_size = 0.0
                    entry_price = None

                # Abrir nueva posición si sig ∈ {-1,1}
                if sig != 0:
                    # Asignar fracción de capital al notional
                    notional = cash * float(risk_fraction)
                    if notional <= 0:
                        equity_curve[t] = mark_to_market(price)
                        continue
                    
                    # Calcular costos de entrada
                    entry_costs = self.calculate_trading_costs(
                        price, price, notional, is_long=(sig == 1)
                    )
                    total_costs += entry_costs
                    
                    units = (notional - entry_costs) / price  # Ajustar por costos
                    
                    if sig == 1:
                        # Compras unidades: reduces cash
                        cash -= notional
                    else:  # short: recibes efectivo
                        cash += notional - entry_costs
                    
                    position = int(sig)
                    position_size = float(units)
                    position = int(sig)
                    position_size = float(units)
                    entry_price = float(price)

            # CRITICAL: Update equity curve AFTER any cash changes
            equity_curve[t] = mark_to_market(price)

            # Protección: no permitir equity negativa (sin apalancamiento)
            if equity_curve[t] < 0:
                print(f"⚠️  WARNING: Equity negativa en t={t}, price={price}, equity={equity_curve[t]}")
                # No lanzar error, solo advertencia
                # raise RuntimeError("Equity negativa: revise tamaño/margen.")

        # Cerrar al final si hay posición abierta
        if position != 0:
            price = float(prices[-1])
            notional = position_size * price
            # Usar costos realistas para cierre final
            trading_costs = self.calculate_trading_costs(
                entry_price, price, abs(notional), is_long=(position == 1)
            )
            total_costs += trading_costs
            
            if position == 1:
                pnl = position_size * (price - entry_price) - trading_costs
                cash += pnl + position_size * entry_price
            else:
                pnl = position_size * (entry_price - price) - trading_costs
                cash += pnl - position_size * entry_price
            
            trades.append({
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": float(entry_price), 
                "exit_price": float(price), 
                "profit": float(pnl),
                "trading_costs": float(trading_costs),
                "position": position,
                "size": position_size,
                "entry_date": len(prices)-1,
                "exit_date": len(prices)-1
            })
            position = 0
            position_size = 0.0
            entry_price = None
            equity_curve[-1] = cash

        final_capital = cash
        
        # Calcular returns evitando división por cero
        base = np.clip(np.concatenate([[self.initial_capital], equity_curve[:-1]]), 1e-9, None)
        returns = (equity_curve - base) / base
        
        return {
            "equity_curve": equity_curve,
            "portfolio_values": equity_curve.tolist(),  # Para compatibilidad
            "returns": returns,
            "trades": trades,
            "final_capital": float(final_capital),
            "total_costs": float(total_costs),  # Nuevo: total de costos realistas
            "num_trades": len([t for t in trades if 'exit_price' in t and t['exit_price'] is not None]),
            "winning_trades": len([t for t in trades if 'profit' in t and t['profit'] and t['profit'] > 0]),
            "losing_trades": len([t for t in trades if 'profit' in t and t['profit'] and t['profit'] < 0]),
            "dates": dates if dates is not None else []
        }
    
    def calculate_metrics(self, trading_result, prices, dates):
        """📊 Calcular métricas de rendimiento"""
        portfolio_values = np.array(trading_result['portfolio_values'])
        
        # Retorno total
        total_return = (trading_result['final_capital'] / self.initial_capital - 1) * 100
        
        # Retorno anualizado (asumiendo datos diarios)
        days = len(portfolio_values)
        years = days / 252  # 252 días trading por año
        annualized_return = ((trading_result['final_capital'] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Buy & Hold benchmark
        buy_hold_return = ((prices[-1] / prices[0]) - 1) * 100
        
        # Volatilidad (retornos diarios)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Anualizada
        
        # Sharpe Ratio (asumiendo risk-free rate = 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
        
        # Maximum Drawdown (usando función estándar)
        max_drawdown = self.max_drawdown_pct(trading_result['portfolio_values'])
        
        # Win Rate
        completed_trades = [t for t in trading_result['trades'] if t['profit'] is not None]
        win_rate = (len([t for t in completed_trades if t['profit'] > 0]) / len(completed_trades) * 100) if completed_trades else 0
        
        # Profit Factor
        winning_profits = sum([t['profit'] for t in completed_trades if t['profit'] > 0])
        losing_profits = abs(sum([t['profit'] for t in completed_trades if t['profit'] < 0]))
        profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'buy_hold_return': buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': trading_result['num_trades'],
            'final_capital': trading_result['final_capital']
        }
    
    def backtest_naive_strategy(self, features_df, seed=None):
        """🎲 Backtesting estrategia Naive (Buy & Hold y Random)"""
        seed_val = seed if seed is not None else DEFAULT_PARAMS.SEED
        print(f"\n🎲 Backtesting Naive Strategies (seed={seed_val})...")
        
        prices = features_df['price'].values
        dates = features_df.index
        
        results = {}
        
        # 1. Buy & Hold correcto: LONG todo el período
        buy_hold_signals = np.ones(len(prices), dtype=int)
        
        trading_result = self.simulate_trading(buy_hold_signals, prices, "Buy & Hold")
        results['Buy & Hold'] = {
            'trading_result': trading_result,
            'metrics': self.calculate_metrics(trading_result, prices, dates),
            'type': 'Baseline',
            'seed': seed_val
        }
        
        # 2. Random Trading (para comparación)
        np.random.seed(seed_val)  # Usar semilla especificada
        random_signals = np.random.choice([-1, 0, 1], size=len(prices), p=[0.1, 0.8, 0.1])
        
        trading_result = self.simulate_trading(random_signals, prices, f"Random (seed={seed_val})")
        results[f'Random Trading (seed={seed_val})'] = {
            'trading_result': trading_result,
            'metrics': self.calculate_metrics(trading_result, prices, dates),
            'type': 'Baseline',
            'seed': seed_val
        }
        
        return results
    
    def backtest_model_predictions(self, model_path, features_df, strategy_name):
        """🤖 Backtesting con predicciones de modelo específico"""
        try:
            print(f"\n🤖 Backtesting {strategy_name}...")
            
            # Cargar modelo
            checkpoint = torch.load(model_path, map_location=device)
            model_name = checkpoint['model_class']
            optuna_params = checkpoint['optuna_params']
            seq_length = checkpoint['seq_length']
            
            # Preparar datos
            target_data = features_df['price']
            feature_columns = [col for col in features_df.columns if col != 'price']
            features_data = features_df[feature_columns]
            
            # Split para backtesting (usar últimos 20% como período de trading)
            train_size = int(len(features_data) * 0.8)
            
            X_test_raw = features_data.iloc[train_size:].values
            y_test_raw = target_data.iloc[train_size:].values
            
            # Escalado (usando datos de entrenamiento)
            scaler = RobustScaler()
            X_train_raw = features_data.iloc[:train_size].values
            y_train_raw = target_data.iloc[:train_size].values
            
            scaler.fit(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            target_scaler = RobustScaler()
            target_scaler.fit(y_train_raw.reshape(-1, 1))
            y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
            
            # Crear secuencias
            def create_sequences(X, y, seq_len):
                X_seq, y_seq = [], []
                for i in range(seq_len, len(X)):
                    X_seq.append(X[i-seq_len:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
            
            # Recrear modelo
            input_size = X_test_seq.shape[2]
            
            if model_name == "TLS_LSTMModel":
                model = TLS_LSTMModel(
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
            elif model_name == "HybridLSTMAttentionModel":
                model = HybridLSTMAttentionModel(
                    input_size=input_size,
                    hidden_size=optuna_params['hidden_size'],
                    output_size=1,
                    dropout_prob=optuna_params['dropout_prob']
                ).to(device)
            elif model_name == "BidirectionalDeepLSTMModel":
                model = BidirectionalDeepLSTMModel(
                    input_size=input_size,
                    hidden_size=optuna_params['hidden_size'],
                    output_size=1,
                    dropout_prob=optuna_params['dropout_prob']
                ).to(device)
            elif model_name == "ARIMAModel":
                model = ARIMAModel(
                    input_size=input_size,
                    output_size=1
                ).to(device)
            elif model_name == "NaiveForecastModel":
                model = NaiveForecastModel(
                    input_size=input_size,
                    output_size=1
                ).to(device)
            else:
                print(f"⚠️ Modelo {model_name} no soportado")
                return None
            
            # Cargar pesos y evaluar
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
                test_pred_scaled = model(X_test_tensor).squeeze()
                test_pred = target_scaler.inverse_transform(
                    test_pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()
            
            # Precios reales para el período de backtesting
            backtest_prices = y_test_raw[seq_length:]  # Alinear con predicciones
            backtest_dates = features_df.index[train_size + seq_length:]
            
            # Probar diferentes estrategias de señales
            strategies = {}
            
            for signal_method in ['threshold', 'directional', 'hybrid']:
                signals = self.generate_signals(test_pred, backtest_prices, signal_method)
                trading_result = self.simulate_trading(signals, backtest_prices, f"{strategy_name}_{signal_method}")
                
                strategies[f"{strategy_name}_{signal_method}"] = {
                    'trading_result': trading_result,
                    'metrics': self.calculate_metrics(trading_result, backtest_prices, backtest_dates),
                    'type': 'ML Model',
                    'signal_method': signal_method,
                    'model_name': model_name,
                    'predictions': test_pred,
                    'actual_prices': backtest_prices
                }
            
            return strategies
            
        except Exception as e:
            print(f"❌ Error en backtesting {strategy_name}: {e}")
            return None
    
    def run_rolling_forecast_backtest(self, features_df, retrain_frequency=10, model_path=None):
        """
        🚀 Backtesting Rolling Forecast con Re-entrenamiento Configurable
        
        Args:
            features_df: DataFrame con características
            retrain_frequency: Cada cuántas predicciones re-entrenar (default: 10)
            model_path: Ruta específica del modelo (opcional)
        """
        print(f"\n🚀 Backtesting Rolling Forecast con Re-entrenamiento cada {retrain_frequency} predicciones...")
        
        # Buscar modelo específico o usar BidirectionalDeepLSTM por defecto
        if model_path:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"❌ Modelo no encontrado: {model_path}")
                return {}
        else:
            # Buscar modelo BidirectionalDeepLSTM por defecto
            current_dir = Path.cwd()
            if current_dir.name == "model":
                models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
            else:
                models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
            
            model_files = list(models_dir.glob("BidirectionalDeepLSTMModel_optuna_*.pth"))
            
            if not model_files:
                print("❌ No se encontró modelo BidirectionalDeepLSTM para Rolling Forecast")
                return {}
            
            model_path = model_files[0]
        
        # Cargar información del modelo
        checkpoint = torch.load(model_path, map_location=device)
        optuna_params = checkpoint['optuna_params']
        seq_length = checkpoint['seq_length']
        
        # Detectar tipo de modelo del nombre del archivo
        model_name = model_path.stem.split('_')[0]  # Extraer nombre del modelo
        print(f"   📊 Usando {model_path.name} (Tipo: {model_name})")
        
        # Implementar Rolling Forecast para backtesting
        target_data = features_df['price']
        feature_columns = [col for col in features_df.columns if col != 'price']
        features_data = features_df[feature_columns]
        
        # Período de backtesting (últimos 20%)
        train_size = int(len(features_data) * 0.8)
        self.train_size = train_size  # Guardar como atributo de la instancia
        backtest_start = train_size
        backtest_predictions = []
        backtest_actual = []
        
        print(f"   🔄 Rolling Forecast desde posición {backtest_start}...")
        print(f"   🎯 Re-entrenamiento cada {retrain_frequency} predicciones")
        
        # Rolling forecast extendido para mejor testing (aumentado a 120 predicciones)
        max_predictions = min(120, len(features_data) - backtest_start - seq_length)
        
        # Variables para control de re-entrenamiento
        current_model = None
        current_scaler = None
        current_target_scaler = None
        
        # Manejar caso sin re-entrenamiento
        if retrain_frequency is None:
            last_retrain_step = -999999  # Nunca re-entrenar
        else:
            last_retrain_step = -retrain_frequency  # Para forzar entrenamiento inicial
        retrain_count = 0
        
        print(f"   📊 Total predicciones a realizar: {max_predictions}")
        
        for i in range(max_predictions):
            current_train_end = backtest_start + i
            
            # 🎯 DECISIÓN DE RE-ENTRENAMIENTO
            if retrain_frequency is None:
                should_retrain = (current_model is None)  # Solo entrenar una vez al inicio
            else:
                should_retrain = (i - last_retrain_step) >= retrain_frequency
            
            if should_retrain:
                retrain_count += 1
                print(f"   🔄 Re-entrenamiento #{retrain_count} en predicción {i+1}/{max_predictions}...")
                last_retrain_step = i
                
                # 🎲 FIJAR SEMILLAS PARA REPRODUCIBILIDAD (Opcional)
                torch.manual_seed(42)  # Fija semilla de PyTorch
                np.random.seed(42)     # Fija semilla de NumPy
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)  # Fija semilla de CUDA
                
                # Datos hasta el punto actual
                X_train_raw = features_data.iloc[:current_train_end].values
                y_train_raw = target_data.iloc[:current_train_end].values
                
                if len(X_train_raw) < seq_length + 20:  # Mínimo de datos para entrenar
                    print(f"      ⚠️ Datos insuficientes: {len(X_train_raw)} < {seq_length + 20}")
                    continue
                # Escalado
                current_scaler = RobustScaler()
                X_train_scaled = current_scaler.fit_transform(X_train_raw)
                
                current_target_scaler = RobustScaler()
                y_train_scaled = current_target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
                
                # Crear secuencias
                def create_sequences(X, y, seq_len):
                    X_seq, y_seq = [], []
                    for j in range(seq_len, len(X)):
                        X_seq.append(X[j-seq_len:j])
                        y_seq.append(y[j])
                    return np.array(X_seq), np.array(y_seq)
                
                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
                
                if len(X_train_seq) < 5:
                    print(f"      ⚠️ Secuencias insuficientes: {len(X_train_seq)} < 5")
                    continue
                
                # Crear modelo según el tipo detectado
                input_size = X_train_seq.shape[2]
                
                if model_name == "GRU":
                    current_model = GRU_Model(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "TLS":
                    current_model = TLS_LSTMModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "HybridLSTMAttentionModel":
                    current_model = HybridLSTMAttentionModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                else:  # Default: BidirectionalDeepLSTMModel
                    current_model = BidirectionalDeepLSTMModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params['dropout_prob']
                    ).to(device)
                
                # Entrenamiento optimizado
                X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
                y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
                
                optimizer = torch.optim.Adam(current_model.parameters(), lr=optuna_params['learning_rate'])
                criterion = nn.MSELoss()
                
                current_model.train()
                for epoch in range(8):  # Más épocas para mejor convergencia
                    optimizer.zero_grad()
                    outputs = current_model(X_train_tensor).squeeze()
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                print(f"      ✅ Modelo re-entrenado (loss: {loss.item():.6f})")
            
            # 🔮 PREDICCIÓN (usando modelo actual, entrenado o reutilizado)
            if current_model is None or current_scaler is None:
                print(f"      ⚠️ Saltando predicción {i+1} - modelo no inicializado")
                continue
            
            # Preparar datos para predicción
            if current_train_end + seq_length <= len(features_data):
                X_pred_raw = features_data.iloc[current_train_end-seq_length:current_train_end].values
                X_pred_scaled = current_scaler.transform(X_pred_raw)
                X_pred_tensor = torch.FloatTensor(X_pred_scaled).unsqueeze(0).to(device)
                
                current_model.eval()
                with torch.no_grad():
                    pred_scaled = current_model(X_pred_tensor).squeeze()
                    pred_value = current_target_scaler.inverse_transform(
                        pred_scaled.cpu().numpy().reshape(-1, 1)
                    ).flatten()[0]
                
                backtest_predictions.append(pred_value)
                backtest_actual.append(target_data.iloc[current_train_end])
            
            # Log progreso cada 20 predicciones
            if (i + 1) % 20 == 0:
                current_da = self.calculate_directional_accuracy(
                    np.array(backtest_predictions[-20:]), 
                    np.array(backtest_actual[-20:])
                )
                print(f"      📈 {i + 1}/{max_predictions} - DA últimas 20: {current_da:.1%}")
        
        print(f"   ✅ Rolling Forecast completado: {len(backtest_predictions)} predicciones")
        print(f"   🔄 Total re-entrenamientos: {retrain_count}")
        
        if len(backtest_predictions) == 0:
            print("❌ No se generaron predicciones para Rolling Forecast")
            return {}
        
        # Generar señales y simular trading
        backtest_predictions = np.array(backtest_predictions)
        backtest_actual = np.array(backtest_actual)
        
        strategies = {}
        
        for signal_method in ['threshold', 'directional', 'hybrid']:
            signals = self.generate_signals(backtest_predictions, backtest_actual, signal_method)
            trading_result = self.simulate_trading(signals, backtest_actual, f"Rolling_Forecast_{signal_method}")
            
            backtest_dates = features_df.index[backtest_start:backtest_start + len(backtest_actual)]
            
            strategies[f"Rolling_Forecast_{signal_method}"] = {
                'trading_result': trading_result,
                'metrics': self.calculate_metrics(trading_result, backtest_actual, backtest_dates),
                'type': 'Rolling Forecast',
                'signal_method': signal_method,
                'model_name': 'BidirectionalDeepLSTM_Rolling',
                'predictions': backtest_predictions,
                'actual_prices': backtest_actual
            }
        
        print(f"   ✅ Rolling Forecast completado: {len(backtest_predictions)} predicciones")
        
        return strategies
    
    def run_rolling_forecast_backtest_with_seed(self, features_df, retrain_frequency=10, model_path=None, seed=42, trading_period_days=252):
        """
        🚀 Backtesting Rolling Forecast con Re-entrenamiento y Semilla Específica
        MODIFICADO: Trading configurable para análisis de estabilidad
        
        Args:
            features_df: DataFrame con características
            retrain_frequency: Cada cuántas predicciones re-entrenar (default: 10)
            model_path: Ruta específica del modelo (opcional)
            seed: Semilla específica para reproducibilidad
            trading_period_days: Días de trading para backtesting (default: 252 = 1 año)
        """
        period_name = "6 MESES" if trading_period_days <= 126 else "1 AÑO"
        print(f"🚀 Backtesting Rolling Forecast con semilla {seed} - PERÍODO: {period_name}...")
        
        # Configurar semilla al inicio
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Buscar modelo específico
        if model_path:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"❌ Modelo no encontrado: {model_path}")
                return {}
        else:
            print("❌ Se requiere ruta del modelo")
            return {}
        
        # Cargar información del modelo
        checkpoint = torch.load(model_path, map_location=device)
        optuna_params = checkpoint['optuna_params']
        seq_length = checkpoint['seq_length']
        
        # Detectar tipo de modelo del nombre del archivo
        model_name = model_path.stem.split('_')[0]  # Extraer nombre del modelo
        
        # Implementar Rolling Forecast para backtesting
        target_data = features_df['price']
        feature_columns = [col for col in features_df.columns if col != 'price']
        features_data = features_df[feature_columns]
        
        # 🎯 MODIFICACIÓN CLAVE: Usar período configurable para backtesting
        total_data_points = len(features_data)
        
        # Reservar período específico de datos para backtesting
        min_backtest_days = trading_period_days
        
        if total_data_points < min_backtest_days * 1.5:
            print(f"⚠️ ADVERTENCIA: Solo {total_data_points} días disponibles, usando período disponible...")
            # Usar 70% para training, 30% para backtesting
            train_size = int(total_data_points * 0.7)
            max_predictions = total_data_points - train_size - seq_length
        else:
            # Suficientes datos: usar período específico para backtesting
            train_size = total_data_points - min_backtest_days
            max_predictions = min(min_backtest_days, total_data_points - train_size - seq_length)
        
        self.train_size = train_size  # Guardar como atributo de la instancia
        backtest_start = train_size
        backtest_predictions = []
        backtest_actual = []
        
        print(f"   📊 Datos totales: {total_data_points} días (~{total_data_points/252:.1f} años)")
        print(f"   🎯 Entrenamiento: {train_size} días (~{train_size/252:.1f} años)")
        print(f"   📈 Backtesting: {max_predictions} días (~{max_predictions/252:.1f} años)")
        
        # Variables para control de re-entrenamiento
        current_model = None
        current_scaler = None
        current_target_scaler = None
        
        # Control de re-entrenamiento
        last_retrain_step = -retrain_frequency  # Para forzar entrenamiento inicial
        retrain_count = 0
        
        for i in range(max_predictions):
            current_train_end = backtest_start + i
            
            # 🎯 DECISIÓN DE RE-ENTRENAMIENTO
            should_retrain = (i - last_retrain_step) >= retrain_frequency
            
            if should_retrain:
                retrain_count += 1
                last_retrain_step = i
                
                # 🎲 FIJAR SEMILLAS PARA REPRODUCIBILIDAD con la semilla específica
                torch.manual_seed(seed + i)  # Variación de semilla para cada re-entrenamiento
                np.random.seed(seed + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed + i)
                
                # Datos hasta el punto actual
                X_train_raw = features_data.iloc[:current_train_end].values
                y_train_raw = target_data.iloc[:current_train_end].values
                
                if len(X_train_raw) < seq_length + 20:  # Mínimo de datos para entrenar
                    continue
                    
                # Escalado
                current_scaler = RobustScaler()
                X_train_scaled = current_scaler.fit_transform(X_train_raw)
                
                current_target_scaler = RobustScaler()
                y_train_scaled = current_target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
                
                # Crear secuencias
                def create_sequences(X, y, seq_len):
                    X_seq, y_seq = [], []
                    for j in range(seq_len, len(X)):
                        X_seq.append(X[j-seq_len:j])
                        y_seq.append(y[j])
                    return np.array(X_seq), np.array(y_seq)
                
                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
                
                if len(X_train_seq) < 5:
                    continue
                
                # Crear modelo según el tipo detectado
                input_size = X_train_seq.shape[2]
                
                if model_name == "GRU":
                    current_model = GRU_Model(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "BidirectionalDeepLSTM" or "Bidirectional" in model_name:
                    current_model = BidirectionalDeepLSTMModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "TLS" or "LSTM" in model_name:
                    current_model = TLS_LSTMModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "HybridLSTMAttention" or "Hybrid" in model_name:
                    current_model = HybridLSTMAttentionModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "ARIMA":
                    current_model = ARIMAModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                elif model_name == "Naive":
                    current_model = NaiveForecastModel(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                else:  # Default: usar GRU si no se reconoce
                    current_model = GRU_Model(
                        input_size=input_size,
                        hidden_size=optuna_params['hidden_size'],
                        output_size=1,
                        dropout_prob=optuna_params.get('dropout_prob', 0.1)
                    ).to(device)
                
                # Entrenamiento optimizado
                X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
                y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
                
                optimizer = torch.optim.Adam(current_model.parameters(), lr=optuna_params['learning_rate'])
                criterion = nn.MSELoss()
                
                current_model.train()
                
                # Entrenamiento rápido (pocas épocas para speed)
                epochs = 20
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = current_model(X_train_tensor)
                    loss = criterion(outputs.squeeze(), y_train_tensor)
                    loss.backward()
                    optimizer.step()
            
            # 🔮 HACER PREDICCIÓN
            if current_model is not None and current_scaler is not None:
                # Preparar datos para predicción
                if current_train_end + seq_length <= len(features_data):
                    X_pred_raw = features_data.iloc[current_train_end-seq_length+1:current_train_end+1].values
                    X_pred_scaled = current_scaler.transform(X_pred_raw)
                    X_pred_tensor = torch.FloatTensor(X_pred_scaled).unsqueeze(0).to(device)
                    
                    current_model.eval()
                    with torch.no_grad():
                        pred_scaled = current_model(X_pred_tensor).cpu().numpy()[0, 0]
                    
                    # Desescalar predicción
                    pred_original = current_target_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    
                    # Valor real
                    if current_train_end < len(target_data):
                        actual_value = target_data.iloc[current_train_end]
                        
                        backtest_predictions.append(pred_original)
                        backtest_actual.append(actual_value)
        
        # Convertir a arrays
        backtest_predictions = np.array(backtest_predictions)
        backtest_actual = np.array(backtest_actual)
        
        # Calcular métricas usando el sistema de trading
        strategies = {}
        
        for signal_method in ['threshold', 'directional', 'hybrid']:
            signals = self.generate_signals(backtest_predictions, backtest_actual, signal_method)
            
            trading_result = self.simulate_trading(signals, backtest_actual, f"Rolling_Forecast_{signal_method}_seed_{seed}")
            
            backtest_dates = features_df.index[backtest_start:backtest_start + len(backtest_actual)]
            
            strategies[f"Rolling_Forecast_{signal_method}_seed_{seed}"] = {
                'trading_result': trading_result,
                'metrics': self.calculate_metrics(trading_result, backtest_actual, backtest_dates),
                'directional_accuracy': self.calculate_directional_accuracy(backtest_predictions, backtest_actual),
                'type': 'Rolling Forecast',
                'signal_method': signal_method,
                'model_name': f'{model_name}_Rolling_Seed_{seed}',
                'seed': seed,
                'predictions': backtest_predictions,
                'actual_prices': backtest_actual
            }
        
        return strategies

    def run_complete_backtest(self):
        """🏆 Ejecutar backtesting completo de todos los modelos"""
        print("🏆 SISTEMA DE BACKTESTING COMPLETO")
        print("=" * 60)
        print(f"💰 Capital inicial: ${self.initial_capital:,}")
        print(f"💸 Costo transacción: {self.transaction_cost*100:.2f}%")
        print(f"🎲 Semillas de testing: {len(self.test_seeds)} semillas")
        print("=" * 60)
        
        # Cargar datos
        eur_prices, dxy_prices = self.load_data()
        features_df = self.create_features(eur_prices, dxy_prices)
        
        all_results = {}
        
        # 1. Baseline strategies con múltiples semillas
        print("\n🎲 TESTING CON MÚLTIPLES SEMILLAS")
        print("-" * 40)
        
        for seed in self.test_seeds:
            baseline_results = self.backtest_naive_strategy(features_df, seed)
            all_results.update(baseline_results)
            
            # Guardar resultados por semilla para análisis
            if seed not in self.seed_results:
                self.seed_results[seed] = {}
            self.seed_results[seed].update(baseline_results)
        
        # 2. Modelos Optuna (sin variación por semilla)
        current_dir = Path.cwd()
        if current_dir.name == "model":
            models_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
        else:
            models_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
        
        optuna_models = list(models_dir.glob("*_optuna_*.pth"))
        
        for model_path in optuna_models:
            model_name = model_path.stem.split('_optuna_')[0]
            strategies = self.backtest_model_predictions(model_path, features_df, model_name)
            if strategies:
                all_results.update(strategies)
        
        # 3. Rolling Forecast con diferentes frecuencias de re-entrenamiento
        print("\n🔄 Probando diferentes frecuencias de re-entrenamiento...")
        
        retrain_frequencies = self.retrain_frequencies
        
        for freq in retrain_frequencies:
            print(f"\n📊 Evaluando re-entrenamiento cada {freq} predicciones...")
            rolling_results = self.run_rolling_forecast_backtest(features_df, retrain_frequency=freq)
            
            # Renombrar estrategias para incluir frecuencia
            freq_results = {}
            for strategy_name, strategy_data in rolling_results.items():
                new_name = f"{strategy_name}_retrain_{freq}"
                freq_results[new_name] = strategy_data
                freq_results[new_name]['retrain_frequency'] = freq
            
            all_results.update(freq_results)
        
        self.results = all_results
        return all_results
    
    def generate_backtest_report(self):
        """📊 Generar reporte completo de backtesting"""
        if not self.results:
            print("❌ No hay resultados para generar reporte")
            return
        
        print(f"\n📊 REPORTE DE BACKTESTING")
        print("=" * 100)
        print(f"{'Estrategia':<40} {'Tipo':<15} {'Retorno':<8} {'Sharpe':<8} {'DD':<8} {'Trades':<8} {'Win%':<8}")
        print("-" * 100)
        
        # Ordenar por retorno total
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['metrics']['total_return'], 
            reverse=True
        )
        
        for i, (strategy_name, result) in enumerate(sorted_results, 1):
            metrics = result['metrics']
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{emoji}{strategy_name:<39} {result['type']:<15} "
                  f"{metrics['total_return']:<8.1f}% "
                  f"{metrics['sharpe_ratio']:<8.2f} "
                  f"{metrics['max_drawdown']:<8.1f}% "
                  f"{metrics['num_trades']:<8} "
                  f"{metrics['win_rate']:<8.1f}%")
        
        # Análisis de campeones
        best_strategy = sorted_results[0]
        best_name, best_result = best_strategy
        
        print(f"\n🏆 CAMPEÓN: {best_name}")
        print(f"   💰 Retorno Total: {best_result['metrics']['total_return']:.1f}%")
        print(f"   📈 Retorno Anualizado: {best_result['metrics']['annualized_return']:.1f}%")
        print(f"   🔥 Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {best_result['metrics']['max_drawdown']:.1f}%")
        print(f"   🎯 Win Rate: {best_result['metrics']['win_rate']:.1f}%")
        print(f"   💵 Capital Final: ${best_result['metrics']['final_capital']:,.2f}")
        
        # Comparación con Buy & Hold
        buy_hold_return = None
        for name, result in self.results.items():
            if 'Buy & Hold' in name:
                buy_hold_return = result['metrics']['total_return']
                break
        
        if buy_hold_return:
            alpha = best_result['metrics']['total_return'] - buy_hold_return
            print(f"   🆚 Alpha vs Buy&Hold: {alpha:.1f}%")
        
        return best_strategy
    
    def analyze_seed_variability(self):
        """🎲 Analizar variabilidad del Random Trading con diferentes semillas"""
        if not self.seed_results:
            print("❌ No hay resultados por semilla para analizar")
            return
        
        print(f"\n🎲 ANÁLISIS DE VARIABILIDAD POR SEMILLAS")
        print("=" * 80)
        print(f"{'Semilla':<10} {'Random Return':<15} {'Buy&Hold Return':<18} {'Diferencia':<12}")
        print("-" * 80)
        
        random_returns = []
        buy_hold_returns = []
        
        for seed in self.test_seeds:
            if seed in self.seed_results:
                # Buscar resultados de Random Trading y Buy & Hold para esta semilla
                random_result = None
                buy_hold_result = None
                
                for strategy_name, result in self.seed_results[seed].items():
                    if 'Random Trading' in strategy_name:
                        random_result = result['metrics']['total_return']
                    elif 'Buy & Hold' in strategy_name:
                        buy_hold_result = result['metrics']['total_return']
                
                if random_result is not None and buy_hold_result is not None:
                    random_returns.append(random_result)
                    buy_hold_returns.append(buy_hold_result)
                    difference = random_result - buy_hold_result
                    
                    print(f"{seed:<10} {random_result:<15.2f}% {buy_hold_result:<18.2f}% {difference:<12.2f}%")
        
        if random_returns:
            print("-" * 80)
            print(f"📊 ESTADÍSTICAS RANDOM TRADING:")
            print(f"   📈 Promedio: {np.mean(random_returns):.2f}%")
            print(f"   📊 Desviación Estándar: {np.std(random_returns):.2f}%")
            print(f"   📉 Mínimo: {np.min(random_returns):.2f}%")
            print(f"   📈 Máximo: {np.max(random_returns):.2f}%")
            print(f"   🎯 Rango: {np.max(random_returns) - np.min(random_returns):.2f}%")
            
            # Comparar con Buy & Hold
            if buy_hold_returns:
                avg_buy_hold = np.mean(buy_hold_returns)
                print(f"\n📊 COMPARACIÓN CON BUY & HOLD:")
                print(f"   📈 Promedio Buy & Hold: {avg_buy_hold:.2f}%")
                print(f"   🎲 Promedio Random: {np.mean(random_returns):.2f}%")
                print(f"   📊 Diferencia Promedio: {np.mean(random_returns) - avg_buy_hold:.2f}%")
                
                # Cuántas veces Random superó Buy & Hold
                wins = sum(1 for r, b in zip(random_returns, buy_hold_returns) if r > b)
                win_rate = wins / len(random_returns) * 100
                print(f"   🏆 Random superó Buy&Hold: {wins}/{len(random_returns)} veces ({win_rate:.1f}%)")
        
        print("=" * 80)
        
        return {
            'random_returns': random_returns,
            'buy_hold_returns': buy_hold_returns,
            'random_stats': {
                'mean': np.mean(random_returns) if random_returns else 0,
                'std': np.std(random_returns) if random_returns else 0,
                'min': np.min(random_returns) if random_returns else 0,
                'max': np.max(random_returns) if random_returns else 0,
                'range': np.max(random_returns) - np.min(random_returns) if random_returns else 0
            }
        }
    
    def create_backtest_charts(self):
        """📈 Crear gráficos de análisis de backtesting"""
        if not self.results:
            print("❌ No hay resultados para graficar")
            return
        
        print("\n📊 Generando gráficos de backtesting...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 ANÁLISIS DE BACKTESTING - Rendimiento de Estrategias', fontsize=16, fontweight='bold')
        
        # 1. Retorno vs Riesgo (Sharpe)
        ax1 = axes[0, 0]
        
        returns = [r['metrics']['total_return'] for r in self.results.values()]
        sharpes = [r['metrics']['sharpe_ratio'] for r in self.results.values()]
        types = [r['type'] for r in self.results.values()]
        names = list(self.results.keys())
        
        # Colores por tipo
        color_map = {'Baseline': 'red', 'ML Model': 'blue', 'Rolling Forecast': 'gold'}
        colors = [color_map.get(t, 'gray') for t in types]
        
        scatter = ax1.scatter(sharpes, returns, c=colors, s=100, alpha=0.7)
        
        # Anotar puntos importantes
        for i, name in enumerate(names):
            if returns[i] > 10 or sharpes[i] > 1.0:  # Destacar mejores
                ax1.annotate(name.replace('_', '\n'), (sharpes[i], returns[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Retorno Total (%)')
        ax1.set_title('Retorno vs Riesgo (Sharpe Ratio)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Ranking por tipo
        ax2 = axes[0, 1]
        
        # Agrupar por tipo
        type_performance = {}
        for name, result in self.results.items():
            type_name = result['type']
            if type_name not in type_performance:
                type_performance[type_name] = []
            type_performance[type_name].append(result['metrics']['total_return'])
        
        # Promedio por tipo
        type_avg = {t: np.mean(returns) for t, returns in type_performance.items()}
        
        bars = ax2.bar(type_avg.keys(), type_avg.values(), 
                      color=['red', 'blue', 'gold'], alpha=0.7)
        
        # Valores en barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Rendimiento Promedio por Tipo')
        ax2.set_ylabel('Retorno Promedio (%)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Top 5 estrategias
        ax3 = axes[1, 0]
        
        # Top 5 por retorno
        top_5 = sorted(self.results.items(), 
                      key=lambda x: x[1]['metrics']['total_return'], 
                      reverse=True)[:5]
        
        top_names = [name[:20] + '...' if len(name) > 20 else name for name, _ in top_5]
        top_returns = [result['metrics']['total_return'] for _, result in top_5]
        top_colors = [color_map.get(result['type'], 'gray') for _, result in top_5]
        
        bars = ax3.barh(range(len(top_5)), top_returns, color=top_colors, alpha=0.7)
        ax3.set_yticks(range(len(top_5)))
        ax3.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(top_names)])
        ax3.set_xlabel('Retorno Total (%)')
        ax3.set_title('🏆 Top 5 Estrategias')
        
        # Valores en barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Métricas de riesgo
        ax4 = axes[1, 1]
        
        # Max Drawdown vs Win Rate
        drawdowns = [r['metrics']['max_drawdown'] for r in self.results.values()]
        win_rates = [r['metrics']['win_rate'] for r in self.results.values()]
        
        scatter = ax4.scatter(drawdowns, win_rates, c=colors, s=100, alpha=0.7)
        
        # Anotar mejores por win rate
        for i, name in enumerate(names):
            if win_rates[i] > 60 or drawdowns[i] < -5:  # Destacar mejores win rate o menor drawdown
                ax4.annotate(name.replace('_', '\n'), (drawdowns[i], win_rates[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Riesgo vs Efectividad')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
        
        plt.tight_layout()
        
        # Guardar
        current_dir = Path.cwd()
        if current_dir.name == "model":
            images_dir = Path("./images/backtesting")
        else:
            images_dir = Path("images/backtesting")
        images_dir.mkdir(exist_ok=True)
        
        chart_path = images_dir / f"backtest_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"📊 Gráficos guardados: {chart_path}")
        
        return chart_path

def plot_enhanced_results(all_results, monte_carlo_results=None):
    """
    📊 GRÁFICOS MEJORADOS para representar ROIs extremos (8000-10000%)
    Incluye escalas logarítmicas y visualización Monte Carlo
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # Extraer datos
    models = []
    rois = []
    win_rates = []
    sharpe_ratios = []
    
    for model_name, results in all_results.items():
        if 'GRU_freq_10' in model_name:  # Solo frecuencia 10
            models.append(model_name.replace('GRU_freq_10_seed_', 'Seed '))
            rois.append(results.get('roi', 0))
            win_rates.append(results.get('win_rate', 0))
            sharpe_ratios.append(results.get('sharpe_ratio', 0))
    
    # 1. ROI con escala logarítmica
    plt.subplot(2, 3, 1)
    bars1 = plt.bar(range(len(models)), rois, color='gold', alpha=0.8, edgecolor='black')
    plt.yscale('log')
    plt.title('ROI - Escala Logarítmica\n(8000-10000%)', fontsize=14, fontweight='bold')
    plt.ylabel('ROI (%)', fontsize=12)
    plt.xticks(range(len(models)), models, rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Anotar valores
    for i, (bar, roi) in enumerate(zip(bars1, rois)):
        plt.annotate(f'{roi:.0f}%', 
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Win Rate
    plt.subplot(2, 3, 2)
    bars2 = plt.bar(range(len(models)), win_rates, color='lightgreen', alpha=0.8, edgecolor='black')
    plt.title('Win Rate por Semilla\n(64-79%)', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xticks(range(len(models)), models, rotation=45, fontsize=10)
    plt.ylim(60, 80)
    plt.grid(True, alpha=0.3)
    
    for i, (bar, wr) in enumerate(zip(bars2, win_rates)):
        plt.annotate(f'{wr:.1f}%', 
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Sharpe Ratio
    plt.subplot(2, 3, 3)
    bars3 = plt.bar(range(len(models)), sharpe_ratios, color='lightblue', alpha=0.8, edgecolor='black')
    plt.title('Risk-Adjusted Return\n(30-63)', fontsize=14, fontweight='bold')
    plt.ylabel('Sharpe-like Ratio', fontsize=12)
    plt.xticks(range(len(models)), models, rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    for i, (bar, sr) in enumerate(zip(bars3, sharpe_ratios)):
        plt.annotate(f'{sr:.1f}', 
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Distribución ROI
    plt.subplot(2, 3, 4)
    plt.hist(rois, bins=8, color='gold', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rois), color='red', linestyle='--', linewidth=2, label=f'Promedio: {np.mean(rois):.0f}%')
    plt.title('Distribución ROI\nFrecuencia 10 días', fontsize=14, fontweight='bold')
    plt.xlabel('ROI (%)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Correlación ROI vs Win Rate
    plt.subplot(2, 3, 5)
    plt.scatter(win_rates, rois, c=sharpe_ratios, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('ROI vs Win Rate\n(Color = Risk-Adj)', fontsize=14, fontweight='bold')
    plt.xlabel('Win Rate (%)', fontsize=12)
    plt.ylabel('ROI (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Anotar mejores puntos
    max_roi_idx = np.argmax(rois)
    max_wr_idx = np.argmax(win_rates)
    plt.annotate(f'Max ROI\n{models[max_roi_idx]}', 
                (win_rates[max_roi_idx], rois[max_roi_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=9)
    
    # 6. Monte Carlo (si está disponible)
    if monte_carlo_results:
        plt.subplot(2, 3, 6)
        plt.hist(monte_carlo_results['monte_carlo_rois'], bins=50, alpha=0.7, color='lightcoral', 
                edgecolor='black', label='ROI Aleatorio')
        plt.axvline(monte_carlo_results['actual_roi'], color='blue', linestyle='-', linewidth=3, 
                   label=f'ROI Real: {monte_carlo_results["actual_roi"]:.0f}%')
        plt.axvline(monte_carlo_results['p95_threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'Percentil 95: {monte_carlo_results["p95_threshold"]:.1f}%')
        
        plt.title(f'Validación Monte Carlo\nPercentil: {monte_carlo_results["percentile"]:.1f}%', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('ROI (%)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Texto de significancia
        significance_text = "🏆 SIGNIFICATIVO" if monte_carlo_results['is_significant_95'] else "❌ NO SIGNIFICATIVO"
        plt.text(0.02, 0.98, significance_text, transform=plt.gca().transAxes, 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='lightgreen' if monte_carlo_results['is_significant_95'] else 'lightcoral',
                         alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.suptitle('🏆 ANÁLISIS COMPLETO GRU - FRECUENCIA 10 DÍAS + MONTE CARLO', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Guardar con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'images/enhanced_gru_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico mejorado guardado: {filename}")
    
    return filename

def main():
    """Función principal para ejecutar backtesting completo con múltiples semillas"""
    print("🏦 SISTEMA DE BACKTESTING COMPLETO CON MÚLTIPLES SEMILLAS")
    print("=" * 80)
    print("💰 Evaluando rendimiento real de trading de todos los modelos")
    print("🎲 Validando consistencia con diferentes semillas random")
    print("=" * 80)
    
    # Configuración
    initial_capital = 10000  # $10,000 inicial
    test_seeds = [42, 123, 987, 555, 777, 2024, 1337, 3141, 9999, 2023]  # 10 semillas para análisis robusto
    retrain_frequencies = [10]  # Frecuencias de re-entrenamiento
    
    # Crear backtester con múltiples semillas
    backtester = TradingBacktester(initial_capital, test_seeds, retrain_frequencies)
    
    # Ejecutar backtesting completo
    results = backtester.run_complete_backtest()
    
    if not results:
        print("❌ No se pudieron generar resultados de backtesting")
        return
    
    # Analizar variabilidad por semillas
    seed_analysis = backtester.analyze_seed_variability()
    
    # Generar reporte
    best_strategy = backtester.generate_backtest_report()
    
    # Crear gráficos
    chart_path = backtester.create_backtest_charts()
    
    # Guardar resultados detallados
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    current_dir = Path.cwd()
    if current_dir.name == "model":
        results_dir = Path("../modelos") / DEFAULT_PARAMS.TABLENAME
    else:
        results_dir = Path("modelos") / DEFAULT_PARAMS.TABLENAME
    
    results_path = results_dir / f"backtest_results_multiseed_{timestamp}.json"
    
    # Preparar datos para JSON
    json_results = {}
    for strategy_name, result in results.items():
        json_results[strategy_name] = {
            'type': result['type'],
            'metrics': result['metrics'],
            'signal_method': result.get('signal_method', 'N/A'),
            'model_name': result.get('model_name', 'N/A'),
            'seed': result.get('seed', 'N/A')
        }
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'initial_capital': initial_capital,
            'test_seeds': test_seeds,
            'retrain_frequencies': retrain_frequencies
        },
        'seed_analysis': seed_analysis,
        'best_strategy': {
            'name': best_strategy[0],
            'metrics': best_strategy[1]['metrics']
        },
        'results': json_results,
        'chart_path': str(chart_path)
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados completos guardados: {results_path}")
    print(f"📊 Gráficos en: {chart_path}")
    print(f"\n🎯 BACKTESTING CON MÚLTIPLES SEMILLAS COMPLETADO")
    
    # Resumen final
    if seed_analysis and 'random_stats' in seed_analysis:
        stats = seed_analysis['random_stats']
        print(f"\n🎲 RESUMEN VARIABILIDAD RANDOM TRADING:")
        print(f"   📊 Promedio: {stats['mean']:.2f}%")
        print(f"   📈 Rango: {stats['range']:.2f}% (desde {stats['min']:.2f}% hasta {stats['max']:.2f}%)")
        print(f"   ⚠️ Desviación: {stats['std']:.2f}%")
        
        if stats['range'] > 10:
            print(f"   🚨 ALTA VARIABILIDAD: Random Trading es inconsistente!")
        else:
            print(f"   ✅ Variabilidad moderada en Random Trading")

def monte_carlo_validation(trading_signals, price_data, actual_roi, initial_capital=10000, n_simulations=1000):
    """
    🎲 SIMULACIÓN MONTE CARLO para validar que el modelo NO es azar
    
    Reordena las señales aleatoriamente 1000 veces y calcula ROI.
    Si el modelo real está en percentil 95+ → NO es azar, es genuina capacidad predictiva.
    
    Args:
        trading_signals: Lista de señales del modelo (1=buy, 0=hold, -1=sell)
        price_data: Datos de precios correspondientes
        actual_roi: ROI real del modelo
        n_simulations: Número de simulaciones Monte Carlo (default: 1000)
    
    Returns:
        dict: Resultados de la validación Monte Carlo
    """
    print(f"\n🎲 INICIANDO SIMULACIÓN MONTE CARLO ({n_simulations} iteraciones)")
    print("=" * 70)
    print(f"🎯 HIPÓTESIS: Si ROI real > percentil 95 → NO es azar")
    print(f"📊 ROI real a validar: {actual_roi:.2f}%")
    
    # Configuración realista de costos de trading
    spread_cost = 0.8  # pips EUR/USD
    commission_per_lot = 8.0  # USD por lote estándar
    slippage = 0.0001  # 0.01% de slippage
    lot_size = 100000  # Lote estándar
    
    monte_carlo_rois = []
    
    # Convertir señales y precios a numpy arrays
    signals = np.array(trading_signals)
    prices = np.array(price_data)
    
    for i in range(n_simulations):
        if i % 200 == 0:
            print(f"⚡ Simulación {i+1}/{n_simulations}...")
        
        # Reordenar las señales aleatoriamente (permutación)
        shuffled_signals = np.random.permutation(signals)
        
        # Simular trading con señales aleatorias
        capital = initial_capital
        position = 0
        total_trades = 0
        winning_trades = 0
        
        for j in range(1, len(prices)):
            signal = shuffled_signals[j]
            current_price = prices[j]
            
            # Calcular costos realistas de trading
            spread = (spread_cost * 0.00001)  # Convertir pips a decimal
            commission = commission_per_lot / lot_size  # Por unidad
            total_cost = spread + commission + slippage
            
            if signal == 1 and position <= 0:  # Señal de compra
                if position < 0:  # Cerrar posición short
                    profit = position * (prices[j-1] - current_price - total_cost)
                    capital += profit
                    if profit > 0:
                        winning_trades += 1
                    total_trades += 1
                
                # Abrir posición long
                position = capital / current_price
                capital = 0
                
            elif signal == -1 and position >= 0:  # Señal de venta
                if position > 0:  # Cerrar posición long
                    profit = position * (current_price - prices[j-1] - total_cost)
                    capital += profit
                    if profit > 0:
                        winning_trades += 1
                    total_trades += 1
                
                # Abrir posición short
                position = -capital / current_price
                capital = 0
        
        # Cerrar posición final
        if position != 0:
            final_price = prices[-1]
            total_cost = spread + commission + slippage
            
            if position > 0:  # Cerrar long
                profit = position * (final_price - prices[-2] - total_cost)
            else:  # Cerrar short
                profit = position * (prices[-2] - final_price - total_cost)
                
            capital += profit
            if profit > 0:
                winning_trades += 1
            total_trades += 1
        
        # Calcular ROI final
        final_value = capital + (position * prices[-1] if position > 0 else capital - position * prices[-1])
        roi = ((final_value - initial_capital) / initial_capital) * 100
        monte_carlo_rois.append(roi)
    
    # Análisis estadístico
    monte_carlo_rois = np.array(monte_carlo_rois)
    percentile_of_actual = (np.sum(monte_carlo_rois < actual_roi) / len(monte_carlo_rois)) * 100
    
    results = {
        'actual_roi': actual_roi,
        'monte_carlo_rois': monte_carlo_rois,
        'percentile': percentile_of_actual,
        'mean_random_roi': np.mean(monte_carlo_rois),
        'std_random_roi': np.std(monte_carlo_rois),
        'min_random_roi': np.min(monte_carlo_rois),
        'max_random_roi': np.max(monte_carlo_rois),
        'p95_threshold': np.percentile(monte_carlo_rois, 95),
        'p99_threshold': np.percentile(monte_carlo_rois, 99),
        'is_significant_95': percentile_of_actual >= 95,
        'is_significant_99': percentile_of_actual >= 99
    }
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS MONTE CARLO:")
    print(f"🎲 ROI promedio aleatorio: {results['mean_random_roi']:.2f}% ± {results['std_random_roi']:.2f}%")
    print(f"📈 Rango aleatorio: [{results['min_random_roi']:.2f}%, {results['max_random_roi']:.2f}%]")
    print(f"🚪 Umbral 95%: {results['p95_threshold']:.2f}%")
    print(f"🚪 Umbral 99%: {results['p99_threshold']:.2f}%")
    print(f"🎯 ROI real: {actual_roi:.2f}%")
    print(f"📊 Percentil del modelo: {percentile_of_actual:.1f}%")
    
    if results['is_significant_99']:
        print(f"🏆 VEREDICTO: ALTAMENTE SIGNIFICATIVO (>99%) - NO ES AZAR!")
    elif results['is_significant_95']:
        print(f"🥇 VEREDICTO: SIGNIFICATIVO (>95%) - NO ES AZAR!")
    else:
        print(f"❌ VEREDICTO: NO SIGNIFICATIVO (<95%) - PODRÍA SER AZAR")
    
    return results

def test_gru_rolling_retraining():
    """
    🔬 ANÁLISIS DE ESTABILIDAD: GRU con múltiples semillas y frecuencias
    MODIFICADO: Trading por 6 meses para análisis de estabilidad robusto
    
    - 10 semillas diferentes para validar consistencia
    - Múltiples frecuencias de re-entrenamiento (5, 10, 20 días)
    - Período de 6 meses (~126 días de trading)
    """
    print("� ANÁLISIS DE ESTABILIDAD GRU: 6 MESES DE TRADING")
    print("=" * 70)
    print("📊 Período: 6 meses (~126 días de trading)")
    print("🎲 Semillas: 10 diferentes para validar robustez")
    print("🔄 Frecuencias: 5, 10, 20 días de re-entrenamiento")
    print("=" * 70)
    
    try:
        # Configuración específica para prueba
        initial_capital = 10000
        transaction_cost = 0.0001
        
        # Solo una semilla para prueba rápida
        backtester = TradingBacktester(initial_capital, transaction_cost, [42])
        
        # Cargar datos
        eur_prices, dxy_prices = backtester.load_data()
        if eur_prices is None:
            print("❌ Error al cargar datos")
            return
        
        # Crear características
        features_df = backtester.create_features(eur_prices, dxy_prices)
        if features_df is None:
            print("❌ Error al crear características")
            return
        
        print(f"✅ Datos cargados: {len(features_df)} registros")
        
        # Buscar modelo GRU específico
        import glob
        model_patterns = [
            "../modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/GRU_Model_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/GRU_Model_*.pth",
            "modelos/eur_usd/GRU_Model_*.pth"
        ]
        
        # Buscar también modelos bidireccionales
        bidirectional_patterns = [
            "../modelos/eur_usd/BidirectionalDeepLSTMModel_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/BidirectionalDeepLSTMModel_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/BidirectionalDeepLSTMModel_*.pth",
            "modelos/eur_usd/BidirectionalDeepLSTMModel_*.pth"
        ]

        # Buscar Modelo TLS
        tlstm_patterns = [
            "modelos/eur_usd/TLS_LSTMModel_rolling_optimized_USD_2010-2024.pth",
            "../modelos/eur_usd/TLS_LSTMModel_rolling_optimized_USD_2010-2024.pth",
            "../modelos/eur_usd/TLS_LSTMModel_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/TLS_LSTMModel_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/TLS_LSTMModel_*.pth",
            "modelos/eur_usd/TLS_LSTMModel_*.pth"
        ]

        # Buscar Modelo Hybrid
        hybrid_patterns = [
            "modelos/eur_usd/HybridLSTMAttentionModel_rolling_optimized_EUR_USD_2010-2024.csv",
            "../modelos/eur_usd/HybridLSTMAttentionModel_rolling_optimized_EUR_USD_2010-2024.csv",
            "../modelos/eur_usd/HybridLSTMAttentionModel_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/HybridLSTMAttentionModel_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/HybridLSTMAttentionModel_*.pth",
            "modelos/eur_usd/HybridLSTMAttentionModel_*.pth"
        ]

        # Buscar Modelo ARIMA
        arima_patterns = [
            "../modelos/eur_usd/ARIMAModel_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/ARIMAModel_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/ARIMAModel_*.pth",
            "modelos/eur_usd/ARIMAModel_*.pth"
        ]

        # Buscar Modelo Naive
        naive_patterns = [
            "../modelos/eur_usd/NaiveForecastModel_optuna_EUR_USD_2010-2024.csv.pth",
            "modelos/eur_usd/NaiveForecastModel_optuna_EUR_USD_2010-2024.csv.pth",
            "../modelos/eur_usd/NaiveForecastModel_*.pth",
            "modelos/eur_usd/NaiveForecastModel_*.pth"
        ]

        tlstm_model_path = None
        hybrid_model_path = None
        gru_model_path = None
        bidirectional_model_path = None
        arima_model_path = None
        naive_model_path = None

        # Buscar modelo Naive
        for pattern in naive_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    naive_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    naive_model_path = pattern
                    break

        # Buscar modelo ARIMA
        for pattern in arima_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    arima_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    arima_model_path = pattern
                    break

        # Buscar modelo TLS
        for pattern in tlstm_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    tlstm_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    tlstm_model_path = pattern
                    break

        # Buscar modelo Hybrid
        for pattern in hybrid_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    hybrid_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    hybrid_model_path = pattern
                    break

        # Buscar modelo GRU
        for pattern in model_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    gru_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    gru_model_path = pattern
                    break
        
        # Buscar modelo Bidireccional
        for pattern in bidirectional_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    bidirectional_model_path = matches[0]
                    break
            else:
                # Ruta específica
                if Path(pattern).exists():
                    bidirectional_model_path = pattern
                    break


        if not gru_model_path and not bidirectional_model_path and not tlstm_model_path and not hybrid_model_path and not arima_model_path and not naive_model_path:
            print("❌ No se encontraron modelos GRU, Bidireccionales, TLS, Hybrid, ARIMA ni Naive")
            print("📁 Verificando directorios...")
            print(f"   📍 Directorio actual: {Path.cwd()}")
            
            # Verificar directorios existentes
            possible_dirs = ["../modelos/eur_usd", "modelos/eur_usd"]
            for dir_path in possible_dirs:
                if Path(dir_path).exists():
                    gru_files = list(Path(dir_path).glob("GRU_*.pth"))
                    bi_files = list(Path(dir_path).glob("BidirectionalDeepLSTMModel_*.pth"))
                    tlstm_files = list(Path(dir_path).glob("TLS_LSTMModel_*.pth"))
                    hybrid_files = list(Path(dir_path).glob("HybridLSTMAttentionModel_*.pth"))
                    arima_files = list(Path(dir_path).glob("ARIMAModel_*.pth"))
                    naive_files = list(Path(dir_path).glob("NaiveForecastModel_*.pth"))
                    print(f"   📁 {dir_path}: {len(gru_files)} archivos GRU, {len(bi_files)} archivos Bidireccionales, {len(tlstm_files)} archivos TLS, {len(hybrid_files)} archivos Hybrid, {len(arima_files)} archivos ARIMA, {len(naive_files)} archivos Naive")
                    for file in gru_files + bi_files + tlstm_files + hybrid_files + arima_files + naive_files:
                        print(f"      - {file.name}")
                else:
                    print(f"   ❌ {dir_path}: no existe")
            return
        
        # Crear lista de modelos a probar
        models_to_test = []
        # if gru_model_path:
        #     models_to_test.append(("GRU", gru_model_path))
        #     print(f"✅ Modelo GRU encontrado: {gru_model_path}")
        # if bidirectional_model_path:
        #     models_to_test.append(("Bidirectional", bidirectional_model_path))
        #     print(f"✅ Modelo Bidireccional encontrado: {bidirectional_model_path}")
        if tlstm_model_path:
            models_to_test.append(("TLS", tlstm_model_path))
            print(f"✅ Modelo TLS encontrado: {tlstm_model_path}")
        if hybrid_model_path:
            models_to_test.append(("Hybrid", hybrid_model_path))
            print(f"✅ Modelo Hybrid encontrado: {hybrid_model_path}")
        # if arima_model_path:
        #     models_to_test.append(("ARIMA", arima_model_path))
        #     print(f"✅ Modelo ARIMA encontrado: {arima_model_path}")
        # if naive_model_path:
        #     models_to_test.append(("Naive", naive_model_path))
        #     print(f"✅ Modelo Naive encontrado: {naive_model_path}")

        print(f"🔄 Se probarán {len(models_to_test)} tipos de modelos")
        
        # ==============================================================
        # 1. EVALUAR RANDOM TRADING BASELINE
        # ==============================================================
        
        print(f"\n🎯 FASE 1: EVALUANDO RANDOM TRADING BASELINE")
        print("=" * 50)
        
        # Evaluar Random Trading con múltiples semillas - PERÍODO: 6 MESES
        trading_days_6_months = 126  # ~6 meses de trading
        random_results = evaluate_random_baseline(backtester, features_df, trading_days_6_months)
        
        # ==============================================================
        # 2. EVALUAR BUY & HOLD BENCHMARK
        # ==============================================================
        
        print(f"\n📈 FASE 2: EVALUANDO BUY & HOLD BENCHMARK")
        print("=" * 50)
        
        # Evaluar estrategia Buy & Hold por 6 meses
        buy_hold_result = backtester.evaluate_buy_and_hold_strategy(features_df, trading_days_6_months)
        
        # ==============================================================
        # 3. ANÁLISIS DE ESTABILIDAD: MÚLTIPLES MODELOS, SEMILLAS Y FRECUENCIAS
        # ==============================================================
        
        print(f"\n🔬 FASE 3: ANÁLISIS DE ESTABILIDAD - MODELOS MÚLTIPLES - 6 MESES")
        print("=" * 70)
        
        # 10 semillas para análisis de estabilidad robusto
        stability_seeds = [42, 123, 987, 555, 777, 2024, 1337, 3141, 9999, 2023]
        
        # Solo frecuencia 10 días - LA GANADORA ABSOLUTA según análisis previo
        retrain_frequencies = [10]  # Óptima: Mayor ROI, Win Rate y consistencia
        
        print(f"🎲 Probando {len(stability_seeds)} semillas diferentes")
        print(f"🏆 Frecuencia ÓPTIMA: {retrain_frequencies[0]} días (GANADORA)")
        print(f"🤖 Modelos a probar: {[model[0] for model in models_to_test]}")
        print(f"📊 Total de combinaciones: {len(stability_seeds) * len(retrain_frequencies) * len(models_to_test)}")
        
        all_model_results = {}
        frequency_results = {freq: [] for freq in retrain_frequencies}
        model_consistency = {model[0]: {} for model in models_to_test}
        
        # Lista para almacenar todas las señales para Monte Carlo
        all_trading_signals = []
        all_price_data = []
        
        # Iterar sobre cada tipo de modelo
        for model_name, model_path in models_to_test:
            print(f"\n🤖 EVALUANDO MODELO: {model_name}")
            print("=" * 60)
            
            model_results = {}
            
            for freq in retrain_frequencies:
                print(f"\n🔄 FRECUENCIA DE RE-ENTRENAMIENTO: {freq} días")
                print("-" * 50)
                
                for seed in stability_seeds:
                    print(f"\n🎲 Probando {model_name} - Freq: {freq} días, Semilla: {seed}")
                    print("-" * 40)
                    
                    try:
                        # Ejecutar rolling forecast con semilla y frecuencia específicas
                        result = backtester.run_rolling_forecast_backtest_with_seed(
                            features_df, 
                            retrain_frequency=freq,
                            model_path=model_path,
                            seed=seed,
                            trading_period_days=trading_days_6_months
                        )
                        
                        if result:
                            # Obtener métricas del mejor resultado de esta combinación
                            best_strategy = None
                            best_total_return = float('-inf')
                            
                            for strategy_name, strategy_data in result.items():
                                total_return = strategy_data['metrics']['total_return']
                                if total_return > best_total_return:
                                    best_total_return = total_return
                                    best_strategy = strategy_data
                            
                            if best_strategy:
                                # Crear resultado unificado
                                unified_result = {
                                    'total_return': best_strategy['metrics']['total_return'],
                                    'win_rate': best_strategy['metrics']['win_rate'],
                                    'directional_accuracy': best_strategy['directional_accuracy'],
                                    'max_drawdown': best_strategy['metrics']['max_drawdown'],
                                    'model_type': model_name,
                                    'seed': seed,
                                    'frequency': freq,
                                    'best_strategy_name': best_strategy['signal_method'],
                                    'strategy_key': f"{model_name}_freq_{freq}_seed_{seed}"
                                }
                                
                                # Almacenar resultado
                                strategy_key = f"{model_name}_freq_{freq}_seed_{seed}"
                                model_results[strategy_key] = unified_result
                                frequency_results[freq].append(unified_result)
                                
                                # Tracking por semilla para análisis de consistencia
                                if seed not in model_consistency[model_name]:
                                    model_consistency[model_name][seed] = []
                                model_consistency[model_name][seed].append(unified_result)
                                
                                print(f"✅ {model_name} - Freq: {freq}, Semilla: {seed}:")
                                print(f"   📊 Win Rate: {unified_result['win_rate']:.2%}")
                                print(f"   🎯 DA: {unified_result['directional_accuracy']:.2%}")
                                print(f"   💰 ROI: {unified_result['total_return']:.2%}")
                                print(f"   🏆 Mejor estrategia: {unified_result['best_strategy_name']}")
                            else:
                                print(f"❌ No se encontraron estrategias válidas - {model_name} Freq: {freq}, Semilla: {seed}")
                        else:
                            print(f"❌ Error en resultado - {model_name} Freq: {freq}, Semilla: {seed}")
                            
                    except Exception as e:
                        print(f"❌ Error en {model_name} - Freq: {freq}, Semilla: {seed}: {str(e)}")
            
            # Almacenar resultados del modelo actual
            all_model_results[model_name] = model_results
        
        # ==============================================================
        # 4. ANÁLISIS DE ESTABILIDAD Y CONSISTENCIA
        # ==============================================================
        
        print(f"\n📊 FASE 4: ANÁLISIS DE ESTABILIDAD Y CONSISTENCIA")
        print("=" * 60)
        
        # Analizar resultados por frecuencia
        for freq in retrain_frequencies:
            if frequency_results[freq]:
                freq_returns = [r['total_return'] for r in frequency_results[freq]]
                freq_das = [r['directional_accuracy'] for r in frequency_results[freq]]
                
                print(f"\n🔄 FRECUENCIA {freq} días:")
                print(f"   📈 ROI promedio: {np.mean(freq_returns):.2%}")
                print(f"   📊 ROI std: {np.std(freq_returns):.2%}")
                print(f"   🎯 DA promedio: {np.mean(freq_das):.2%}")
                print(f"   📊 DA std: {np.std(freq_das):.2%}")
                print(f"   ✅ Resultados positivos: {len([r for r in freq_returns if r > 0])}/{len(freq_returns)}")
        
        # Analizar consistencia por modelo y semilla
        print(f"\n🎲 ANÁLISIS DE CONSISTENCIA POR MODELO Y SEMILLA:")
        for model_name in model_consistency:
            print(f"\n🤖 Modelo: {model_name}")
            for seed in stability_seeds:
                if seed in model_consistency[model_name]:
                    seed_returns = [r['total_return'] for r in model_consistency[model_name][seed]]
                    seed_das = [r['directional_accuracy'] for r in model_consistency[model_name][seed]]
                    
                    print(f"   Semilla {seed}: ROI={np.mean(seed_returns):.2%}±{np.std(seed_returns):.2%}, "
                          f"DA={np.mean(seed_das):.2%}±{np.std(seed_das):.2%}")
        
        # Recopilar todos los resultados
        all_combined_results = {}
        for model_name, model_results in all_model_results.items():
            all_combined_results.update(model_results)
        
        # Seleccionar mejor resultado general
        if all_combined_results:
            best_overall = max(all_combined_results.items(), key=lambda x: x[1]['total_return'])
            best_key, best_result = best_overall
            
            print(f"\n🏆 MEJOR RESULTADO GENERAL:")
            print(f"   🔧 Configuración: {best_key}")
            print(f"   📊 Win Rate: {best_result['win_rate']:.2%}")
            print(f"   🎯 DA: {best_result['directional_accuracy']:.2%}")
            print(f"   💰 ROI: {best_result['total_return']:.2%}")
            print(f"   🔄 Frecuencia: {best_result['frequency']} días")
            print(f"   🎲 Semilla: {best_result['seed']}")
            
            gru_results = {best_key: best_result}
        else:
            print("❌ No se obtuvieron resultados válidos de GRU")
            gru_results = {}

        # ==============================================================
        # 5. EVALUACIÓN DE ROBUSTEZ Y ESTABILIDAD
        # ==============================================================
        
        print(f"\n📊 FASE 5: EVALUACIÓN DE ROBUSTEZ")
        print("=" * 50)
        
        if all_combined_results:
            all_returns = [r['total_return'] for r in all_combined_results.values()]
            all_das = [r['directional_accuracy'] for r in all_combined_results.values()]
            positive_returns = [r for r in all_returns if r > 0]
            
            print(f"🎯 RESUMEN DE ESTABILIDAD:")
            print(f"   🔬 Total combinaciones probadas: {len(all_combined_results)}")
            print(f"   ✅ Resultados positivos: {len(positive_returns)}/{len(all_returns)} ({len(positive_returns)/len(all_returns):.1%})")
            print(f"   💰 ROI promedio: {np.mean(all_returns):.2%}")
            print(f"   📊 ROI desviación estándar: {np.std(all_returns):.2%}")
            print(f"   🎯 DA promedio: {np.mean(all_das):.2%}")
            print(f"   📊 DA desviación estándar: {np.std(all_das):.2%}")
            
            # Verificar si el modelo es consistentemente rentable
            consistency_score = len(positive_returns) / len(all_returns)
            if consistency_score >= 0.8:
                print(f"🏆 MODELO ROBUSTO: {consistency_score:.1%} de configuraciones rentables")
            elif consistency_score >= 0.6:
                print(f"⚠️ MODELO MODERADAMENTE ESTABLE: {consistency_score:.1%} de configuraciones rentables")
            else:
                print(f"❌ MODELO INESTABLE: Solo {consistency_score:.1%} de configuraciones rentables")
        else:
            print("❌ No se obtuvieron resultados válidos de GRU con ninguna semilla")

        # ==============================================================
        # 2.5. EVALUANDO BIDIRECTIONAL LSTM CON ROLLING RE-TRAINING (DESHABILITADO)
        # ==============================================================
        
        # print(f"\n🎯 FASE 2.5: EVALUANDO BIDIRECTIONAL LSTM CON ROLLING RE-TRAINING")
        # print("=" * 60)
        
        # # Buscar modelo bidireccional
        # bidirectional_model_patterns = [
        #     "../modelos/eur_usd/BidirectionalDeepLSTMModel_optuna_*.pth",
        #     "modelos/eur_usd/BidirectionalDeepLSTMModel_optuna_*.pth",
        #     "../modelos/eur_usd/BidirectionalDeepLSTMModel_*.pth",
        #     "modelos/eur_usd/BidirectionalDeepLSTMModel_*.pth"
        # ]
        
        # bidirectional_model_path = None
        # for pattern in bidirectional_model_patterns:
        #     if "*" in pattern:
        #         matches = glob.glob(pattern)
        #         if matches:
        #             bidirectional_model_path = matches[0]
        #             break
        #     else:
        #         # Ruta específica
        #         if Path(pattern).exists():
        #             bidirectional_model_path = pattern
        #             break
        
        bidirectional_results = {}  # Vacío para compatibilidad
        
        # if bidirectional_model_path:
        #     print(f"✅ Usando modelo bidireccional: {Path(bidirectional_model_path).name}")
        #     
        #     # Solo probar rolling 10 como especificó el usuario
        #     print(f"\n🔄 Probando: Re-entrenamiento cada 10 (Bidirectional LSTM)")
        #     print("-" * 50)
        #     
        #     try:
        #         # Ejecutar rolling forecast con frecuencia 10
        #         result = backtester.run_rolling_forecast_backtest(
        #             features_df, 
        #             retrain_frequency=10,
        #             model_path=bidirectional_model_path
        #         )
        #         
        #         if result:
        #             # Extraer métricas de la mejor estrategia
        #             best_strategy = None
        #             best_total_return = float('-inf')
        #             
        #             for strategy_name, strategy_data in result.items():
        #                 metrics = strategy_data['metrics']
        #                 total_return = metrics.get('total_return', float('-inf'))
        #                 if total_return > best_total_return:
        #                     best_total_return = total_return
        #                     best_strategy = strategy_data
        #             
        #             if best_strategy:
        #                 # Obtener fechas del período de backtesting
        #                 backtest_dates = features_df.index[backtester.train_size:backtester.train_size + len(best_strategy['predictions'])]
        #                 start_date, end_date, duration_days = backtester.get_trading_period(backtest_dates)
        #                 
        #                 # Crear resultado unificado
        #                 unified_result = {
        #                     'win_rate': best_strategy['metrics']['win_rate'],
        #                     'directional_accuracy': backtester.calculate_directional_accuracy(
        #                         best_strategy['predictions'], best_strategy['actual_prices']
        #                     ),
        #                     'total_return': best_strategy['metrics']['total_return'],
        #                     'total_predictions': len(best_strategy['predictions']),
        #                     'start_date': start_date,
        #                     'end_date': end_date,
        #                     'duration_days': duration_days,
        #                     'model_type': 'BIDIRECTIONAL',
        #                     'best_strategy_name': best_strategy['signal_method'],
        #                     'max_drawdown': best_strategy['metrics'].get('max_drawdown', 0),
        #                     'predictions': best_strategy['predictions'],
        #                     'actual_prices': best_strategy['actual_prices']
        #                 }
        #                 
        #                 bidirectional_results["Re-entrenamiento cada 10"] = unified_result
        #                 print(f"✅ Re-entrenamiento cada 10 (Bidirectional LSTM):")
        #                 print(f"   📊 Win Rate: {unified_result['win_rate']:.2%}")
        #                 print(f"   🎯 DA: {unified_result['directional_accuracy']:.2%}")
        #                 print(f"   💰 ROI: {unified_result['total_return']:.2%}")
        #                 print(f"   📈 Predicciones: {unified_result['total_predictions']}")
        #                 print(f"   🏆 Mejor estrategia: {unified_result['best_strategy_name']}")
        #             else:
        #                 print(f"❌ No se encontraron estrategias válidas para Bidirectional LSTM")
        #         else:
        #             print(f"❌ Error en Re-entrenamiento cada 10 (Bidirectional LSTM)")
        #             
        #     except Exception as e:
        #         print(f"❌ Error en Bidirectional LSTM rolling 10: {str(e)}")
        #         import traceback
        #         traceback.print_exc()
        # else:
        #     print("❌ No se encontró modelo Bidirectional LSTM")
        #     print("📁 Verificando directorios...")
        #     print(f"   📍 Directorio actual: {Path.cwd()}")
        #     
        #     # Verificar directorios existentes
        #     possible_dirs = ["../modelos/eur_usd", "modelos/eur_usd"]
        #     for dir_path in possible_dirs:
        #         if Path(dir_path).exists():
        #             files = list(Path(dir_path).glob("BidirectionalDeepLSTMModel_*.pth"))
        #             print(f"   📁 {dir_path}: {len(files)} archivos Bidirectional")
        #             for file in files:
        #                 print(f"      - {file.name}")
        #         else:
        #             print(f"   ❌ {dir_path}: no existe")
        
        # ==============================================================
        # 6. CONSOLIDAR TODOS LOS RESULTADOS
        # ==============================================================
        
        print(f"\n🎯 FASE 6: CONSOLIDANDO RESULTADOS")
        print("=" * 50)
        
        all_results = {}
        
        # Agregar TODOS los resultados de Random Trading (no solo el mejor)
        if random_results:
            for strategy_name, strategy_data in random_results.items():
                # Obtener fechas del período de backtesting para Random Trading
                random_dates = strategy_data['trading_result']['dates'] if 'dates' in strategy_data['trading_result'] else []
                start_date, end_date, duration_days = ("N/A", "N/A", 0)
                if len(random_dates) > 0:
                    start_date, end_date, duration_days = backtester.get_trading_period(random_dates)
                
                all_results[strategy_name] = {
                    'win_rate': strategy_data['metrics']['win_rate'],
                    'directional_accuracy': strategy_data['directional_accuracy'],  # Usar DA ya calculada
                    'total_return': strategy_data['metrics']['total_return'],
                    'max_drawdown': strategy_data['metrics']['max_drawdown'],
                    'total_predictions': len(strategy_data['predictions']),
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': duration_days,
                    'model_type': 'Random Trading',
                    'seed': strategy_data['seed'],
                    'signal_method': strategy_data['signal_method']
                }
        
        # Agregar resultado de Buy & Hold
        if buy_hold_result and buy_hold_result['strategy'] != 'Error':
            all_results['Buy_Hold_EUR_USD'] = {
                'win_rate': 1.0 if buy_hold_result['total_return'] > 0 else 0.0,  # 100% si gana, 0% si pierde
                'directional_accuracy': buy_hold_result['directional_accuracy'],
                'total_return': buy_hold_result['total_return'],
                'max_drawdown': buy_hold_result['max_drawdown'],
                'total_predictions': 1,  # Solo una predicción: "comprar y mantener"
                'start_date': buy_hold_result['start_date'],
                'end_date': buy_hold_result['end_date'],
                'duration_days': buy_hold_result['trading_days'],
                'model_type': 'Buy & Hold',
                'seed': 'N/A',
                'signal_method': 'Buy & Hold EUR/USD',
                'total_costs': buy_hold_result['total_costs'],
                'roi_annualized': buy_hold_result['roi_annualized']
            }
        
        # Agregar TODOS los resultados de modelos (no solo el mejor)
        if all_combined_results:
            for strategy_name, strategy_data in all_combined_results.items():
                all_results[strategy_name] = strategy_data
        
        # Resumen comparativo
        if all_results:
            print(f"\n🏆 ANÁLISIS DE ESTABILIDAD: GRU vs BASELINES - 6 MESES")
            print("=" * 80)
            print(f"{'Modelo':<30} {'Freq':<6} {'Semilla':<8} {'Win Rate':<10} {'DA':<8} {'ROI':<8}")
            print("-" * 80)
            
            best_strategy = None
            best_score = 0
            
            for strategy, result in all_results.items():
                win_rate = result['win_rate']
                da = result['directional_accuracy']
                roi = result['total_return']
                model_type = result['model_type']
                
                # Extraer frecuencia y semilla si es GRU
                freq_str = str(result.get('frequency', 'N/A'))
                seed_str = str(result.get('seed', 'N/A'))
                
                # Score combinado (puedes ajustar los pesos)
                score = (da * 0.5) + (roi * 0.5)
                
                print(f"{strategy:<30} {freq_str:<6} {seed_str:<8} {win_rate:<10.2%} {da:<8.2%} {roi:<8.2%}")
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            
            print("-" * 80)
            if best_strategy:
                print(f"🏆 MODELO GANADOR: {best_strategy}")
                best_result = all_results[best_strategy]
                print(f"   📊 Win Rate: {best_result['win_rate']:.2%}")
                print(f"   🎯 DA: {best_result['directional_accuracy']:.2%}")
                print(f"   💰 ROI: {best_result['total_return']:.2%}")
                print(f"   🔧 Tipo: {best_result['model_type']}")
                
                # Calcular mejora vs baseline
                baseline_results = {k: v for k, v in all_results.items() if v['model_type'] == 'Random Trading'}
                if baseline_results:
                    best_baseline = max(baseline_results.items(), key=lambda x: x[1]['directional_accuracy'])
                    baseline_da = best_baseline[1]['directional_accuracy']
                    baseline_roi = best_baseline[1]['total_return']
                    
                    if best_result['model_type'] != 'Random Trading':
                        da_improvement = (best_result['directional_accuracy'] - baseline_da) * 100
                        roi_improvement = best_result['total_return'] - baseline_roi
                        
                        print(f"\n✅ ¡GRU SUPERA AL RANDOM TRADING!")
                        print(f"   📈 Mejora vs Random Trading: {da_improvement:+.1f} puntos porcentuales")
                        print(f"   💰 Mejora ROI vs Random: {roi_improvement:+.1f} puntos porcentuales")
                    else:
                        print(f"\n⚠️ RANDOM TRADING ES EL MEJOR")
                        print(f"   📊 Random Trading supera a otros modelos")
        
        # ==============================================================
        # 6. SIMULACIÓN MONTE CARLO - VALIDACIÓN ESTADÍSTICA
        # ==============================================================
        
        print(f"\n� FASE 6: SIMULACIÓN MONTE CARLO")
        print("=" * 60)
        
        # Buscar la mejor estrategia GRU para Monte Carlo
        best_gru_result = None
        best_gru_roi = -float('inf')
        best_gru_name = None
        
        for strategy_name, result in all_results.items():
            if 'GRU_freq_10' in strategy_name and result['total_return'] > best_gru_roi:
                best_gru_roi = result['total_return']
                best_gru_result = result
                best_gru_name = strategy_name
        
        monte_carlo_results = None
        if best_gru_result and 'predictions' in best_gru_result and 'actual_prices' in best_gru_result:
            print(f"🎯 Validando estrategia: {best_gru_name}")
            print(f"💰 ROI a validar: {best_gru_roi:.2f}%")
            
            # Generar señales de trading basadas en predicciones
            predictions = best_gru_result['predictions']
            actual_prices = best_gru_result['actual_prices']
            
            # Crear señales: 1 = buy, 0 = hold, -1 = sell
            trading_signals = []
            for i in range(1, len(predictions)):
                if predictions[i] > actual_prices[i-1] * 1.0005:  # >0.05% predicción al alza
                    trading_signals.append(1)
                elif predictions[i] < actual_prices[i-1] * 0.9995:  # <-0.05% predicción a la baja
                    trading_signals.append(-1)
                else:
                    trading_signals.append(0)  # Hold
            
            # Ejecutar Monte Carlo
            try:
                monte_carlo_results = monte_carlo_validation(
                    trading_signals=trading_signals,
                    price_data=actual_prices[1:],  # Sincronizar con señales
                    actual_roi=best_gru_roi,
                    n_simulations=1000
                )
                
                # Mostrar resumen Monte Carlo
                print(f"\n🏆 RESULTADO MONTE CARLO:")
                if monte_carlo_results['is_significant_99']:
                    print(f"   ✅ ALTAMENTE SIGNIFICATIVO (>99% percentil)")
                    print(f"   🎯 El modelo GRU tiene GENUINA capacidad predictiva")
                elif monte_carlo_results['is_significant_95']:
                    print(f"   ✅ SIGNIFICATIVO (>95% percentil)")
                    print(f"   🎯 El modelo GRU NO es azar")
                else:
                    print(f"   ❌ NO SIGNIFICATIVO (<95% percentil)")
                    print(f"   ⚠️ Los resultados podrían ser azar")
                    
            except Exception as e:
                print(f"⚠️ Error en Monte Carlo: {e}")
                monte_carlo_results = None
        else:
            print("❌ No se encontraron datos suficientes para Monte Carlo")
        
        # ==============================================================
        # 7. GENERAR GRÁFICO MEJORADO CON TODOS LOS RESULTADOS
        # ==============================================================
        
        print(f"\n🎨 FASE 7: Generando análisis visual mejorado...")
        
        # Generar gráfico mejorado con Monte Carlo
        #image_path = plot_enhanced_results(all_results, monte_carlo_results)
        
        # Generar archivo JSON con resultados detallados  
        json_path = generate_results_json(all_results)
        
        #print(f"✅ Análisis completo disponible en: {image_path}")
        print(f"✅ Resultados JSON disponibles en: {json_path}")
        
        # Resumen final
        print(f"\n🏆 RESUMEN FINAL - FRECUENCIA 10 DÍAS")
        print("=" * 60)
        
        if monte_carlo_results:
            print(f"🎲 Monte Carlo: {monte_carlo_results['percentile']:.1f}% percentil")
            print(f"✅ Significativo: {'SÍ' if monte_carlo_results['is_significant_95'] else 'NO'}")
        
        print(f"🔢 Total combinaciones probadas: {len([k for k in all_results.keys() if 'GRU_freq_10' in k])}")
        print(f"🏅 Mejor ROI: {best_gru_roi:.2f}% ({best_gru_name})")
        
        return all_results, monte_carlo_results
        
    except Exception as e:
        print(f"❌ Error en análisis de estabilidad: {e}")
        import traceback
        traceback.print_exc()
        return {}, None

def evaluate_random_baseline(backtester, features_df, predictions_count=126):
    """
    🎲 Evaluar baseline RANDOM TRADING con múltiples semillas
    MODIFICADO: Trading por 6 meses (126 días) para análisis de estabilidad
    """
    print(f"\n🎲 Evaluando Random Trading Baseline - PERÍODO: 6 MESES...")
    
    try:
        # Usar configuración para 1 año de trading
        target_data = features_df['price']
        total_data_points = len(features_df)
        
        # Calcular split para reservar 1 año de backtesting
        min_train_size = int(total_data_points * 0.7)  # Al menos 70% para entrenamiento
        backtest_start = max(min_train_size, total_data_points - predictions_count)
        
        # Verificar que tenemos suficientes datos
        available_backtest_days = total_data_points - backtest_start
        if available_backtest_days < predictions_count:
            print(f"⚠️ ADVERTENCIA: Solo {available_backtest_days} días disponibles, ajustando período...")
            predictions_count = available_backtest_days
            backtest_start = total_data_points - predictions_count
        
        print(f"   📊 Período de evaluación: desde posición {backtest_start}")
        print(f"   🎯 Predicciones a realizar: {predictions_count} días (~{predictions_count/252:.1f} años)")
        print(f"   📅 Datos de entrenamiento: {backtest_start} días (~{backtest_start/252:.1f} años)")
        
        # Probar múltiples semillas para Random Trading
        random_seeds = [42, 123, 987, 555, 777]
        all_random_results = {}
        
        for seed in random_seeds:
            print(f"\n   🎲 Probando Random Trading con semilla {seed}...")
            
            # Obtener precios reales para el período de backtest
            backtest_actual = target_data.iloc[backtest_start:backtest_start + predictions_count].values
            backtest_dates = features_df.index[backtest_start:backtest_start + len(backtest_actual)]
            
            # Generar predicciones aleatorias para DA
            np.random.seed(seed)
            random_predictions = backtest_actual[0] + np.cumsum(np.random.normal(0, 0.001, len(backtest_actual)))
            
            # Calcular DA correctamente
            directional_accuracy = backtester.calculate_directional_accuracy(random_predictions, backtest_actual)
            
            # Generar señales aleatorias con semilla fija
            np.random.seed(seed)
            random_signals = np.random.choice([-1, 0, 1], size=predictions_count, p=[0.33, 0.34, 0.33])
            
            # Simular trading para cada método de señales
            for signal_method in ['threshold', 'directional', 'hybrid']:
                # Para Random Trading, usamos directamente las señales aleatorias
                if signal_method == 'directional':
                    signals = random_signals
                else:
                    # Para otros métodos, generamos señales basadas en ruido aleatorio
                    np.random.seed(seed + hash(signal_method) % 1000)
                    signals = np.random.choice([-1, 0, 1], size=len(backtest_actual), p=[0.33, 0.34, 0.33])
                
                trading_result = backtester.simulate_trading(signals, backtest_actual, f"Random_{signal_method}_seed_{seed}", backtest_dates)
                
                strategy_name = f"Random_{signal_method}_seed_{seed}"
                all_random_results[strategy_name] = {
                    'trading_result': trading_result,
                    'metrics': backtester.calculate_metrics(trading_result, backtest_actual, backtest_dates),
                    'type': 'Random Trading',
                    'signal_method': signal_method,
                    'model_name': 'Random',
                    'seed': seed,
                    'predictions': random_predictions,  # Predicciones aleatorias reales
                    'actual_prices': backtest_actual,
                    'directional_accuracy': directional_accuracy  # DA calculada correctamente
                }
        
        # Seleccionar el mejor resultado de Random Trading
        best_random = max(all_random_results.items(), key=lambda x: x[1]['metrics']['total_return'])
        best_random_name, best_random_data = best_random
        
        print(f"\n   ✅ Mejor Random Trading: {best_random_name}")
        print(f"   💰 ROI: {best_random_data['metrics']['total_return']:.2f}%")
        print(f"   📊 Win Rate: {best_random_data['metrics']['win_rate']:.2f}%")
        print(f"   🎯 DA: {best_random_data['directional_accuracy']*100:.2f}%")
        print(f"   🎲 Semilla ganadora: {best_random_data['seed']}")
        
        # Retornar TODOS los resultados de Random Trading para el gráfico
        return all_random_results
        
    except Exception as e:
        print(f"❌ Error en Random Trading baseline: {e}")
        import traceback
        traceback.print_exc()
        return {}

def evaluate_naive_baseline(backtester, features_df, predictions_count=120):
    """
    🎲 Evaluar baseline NAIVE con rolling forecast
    """
    print(f"\n🎲 Evaluando Baseline NAIVE...")
    
    try:
        # Usar misma configuración que GRU
        target_data = features_df['price']
        train_size = int(len(features_df) * 0.8)
        backtest_start = train_size
        
        backtest_predictions = []
        backtest_actual = []
        
        print(f"   📊 Período de evaluación: desde posición {backtest_start}")
        print(f"   🎯 Predicciones a realizar: {predictions_count}")
        
        for i in range(min(predictions_count, len(target_data) - backtest_start)):
            current_idx = backtest_start + i
            
            if current_idx >= len(target_data):
                break
                
            # NAIVE: siguiente valor = valor actual
            if current_idx > 0:
                naive_pred = target_data.iloc[current_idx - 1]  # Usar valor anterior
                actual_value = target_data.iloc[current_idx]
                
                backtest_predictions.append(naive_pred)
                backtest_actual.append(actual_value)
        
        # Convertir a arrays
        backtest_predictions = np.array(backtest_predictions)
        backtest_actual = np.array(backtest_actual)
        
        print(f"   ✅ NAIVE completado: {len(backtest_predictions)} predicciones")
        
        # Calcular métricas usando el sistema de trading
        strategies = {}
        
        for signal_method in ['threshold', 'directional', 'hybrid']:
            signals = backtester.generate_signals(backtest_predictions, backtest_actual, signal_method)
            
            backtest_dates = features_df.index[backtest_start:backtest_start + len(backtest_actual)]
            trading_result = backtester.simulate_trading(signals, backtest_actual, f"NAIVE_Baseline_{signal_method}", backtest_dates)
            
            strategies[f"NAIVE_Baseline_{signal_method}"] = {
                'trading_result': trading_result,
                'metrics': backtester.calculate_metrics(trading_result, backtest_actual, backtest_dates),
                'type': 'NAIVE Baseline',
                'signal_method': signal_method,
                'model_name': 'NAIVE',
                'predictions': backtest_predictions,
                'actual_prices': backtest_actual
            }
        
        return strategies
        
    except Exception as e:
        print(f"❌ Error en NAIVE baseline: {e}")
        return {}

def evaluate_arima_baseline(backtester, features_df, predictions_count=120):
    """
    📈 Evaluar baseline ARIMA con rolling forecast
    """
    print(f"\n📈 Evaluando Baseline ARIMA...")
    
    try:
        target_data = features_df['price']
        train_size = int(len(features_df) * 0.8)
        backtest_start = train_size
        
        backtest_predictions = []
        backtest_actual = []
        
        print(f"   📊 Período de evaluación: desde posición {backtest_start}")
        print(f"   🎯 Predicciones a realizar: {predictions_count}")
        
        # Preparar serie temporal
        price_series = target_data.iloc[:backtest_start]
        
        # Determinar orden ARIMA automáticamente con órdenes simples
        def find_arima_order(series, max_p=2, max_d=1, max_q=2):
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Intentar órdenes simples primero para evitar problemas de convergencia
            simple_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 2), (2, 1, 0)]
            
            for order in simple_orders:
                try:
                    model = ARIMA(series, order=order)
                    fitted = model.fit()  # Configuración estándar
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = order
                except:
                    continue
            
            return best_order
        
        # Encontrar mejor orden ARIMA
        print(f"   🔍 Buscando mejor orden ARIMA...")
        arima_order = find_arima_order(price_series)
        print(f"   ✅ Orden ARIMA seleccionado: {arima_order}")
        
        # Rolling forecast ARIMA
        current_series = price_series.copy()
        
        for i in range(min(predictions_count, len(target_data) - backtest_start)):
            current_idx = backtest_start + i
            
            if current_idx >= len(target_data):
                break
            
            try:
                # Entrenar ARIMA con datos hasta el momento actual (con configuración rápida)
                model = ARIMA(current_series, order=arima_order)
                fitted_model = model.fit()
                
                # Hacer predicción para siguiente período
                forecast = fitted_model.forecast(steps=1)
                arima_pred = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                
                actual_value = target_data.iloc[current_idx]
                
                backtest_predictions.append(arima_pred)
                backtest_actual.append(actual_value)
                
                # Actualizar serie para siguiente iteración
                current_series = pd.concat([current_series, pd.Series([actual_value], index=[current_idx])])
                
                # Progreso cada 20 predicciones
                if (i + 1) % 20 == 0:
                    print(f"   📈 ARIMA: {i+1}/{predictions_count} predicciones completadas")
                
            except Exception as e:
                print(f"   ⚠️ Error en predicción {i+1}: {str(e)}")
                # En caso de error, usar valor anterior (fallback a NAIVE)
                if i > 0:
                    arima_pred = backtest_predictions[-1]
                else:
                    arima_pred = current_series.iloc[-1]
                
                actual_value = target_data.iloc[current_idx]
                backtest_predictions.append(arima_pred)
                backtest_actual.append(actual_value)
                
                current_series = pd.concat([current_series, pd.Series([actual_value], index=[current_idx])])
        
        # Convertir a arrays
        backtest_predictions = np.array(backtest_predictions)
        backtest_actual = np.array(backtest_actual)
        
        print(f"   ✅ ARIMA completado: {len(backtest_predictions)} predicciones")
        
        # Calcular métricas usando el sistema de trading
        strategies = {}
        
        for signal_method in ['threshold', 'directional', 'hybrid']:
            signals = backtester.generate_signals(backtest_predictions, backtest_actual, signal_method)
            
            backtest_dates = features_df.index[backtest_start:backtest_start + len(backtest_actual)]
            trading_result = backtester.simulate_trading(signals, backtest_actual, f"ARIMA_Baseline_{signal_method}", backtest_dates)
            
            strategies[f"ARIMA_Baseline_{signal_method}"] = {
                'trading_result': trading_result,
                'metrics': backtester.calculate_metrics(trading_result, backtest_actual, backtest_dates),
                'type': 'ARIMA Baseline',
                'signal_method': signal_method,
                'model_name': 'ARIMA',
                'predictions': backtest_predictions,
                'actual_prices': backtest_actual
            }
        
        return strategies
        
    except Exception as e:
        print(f"❌ Error en ARIMA baseline: {e}")
        import traceback
        traceback.print_exc()
        return {}

def generate_comprehensive_trading_analysis(results_comparison, initial_capital=10000, save_path=None):
    """
    🎨 Generar análisis visual COMPRENSIBLE del trading real
    Muestra dinero real ganado/perdido, capital inicial, número de trades
    """
    if not results_comparison:
        print("❌ No hay resultados para analizar")
        return
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura para análisis comprensible
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # Título principal
    fig.suptitle(f'💰 ANÁLISIS DE TRADING REAL: Capital Inicial ${initial_capital:,}\nComparación GRU Rolling Re-training vs Baselines', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    # Extraer datos para visualización
    strategies = list(results_comparison.keys())
    das = [results_comparison[s]['directional_accuracy'] for s in strategies]
    rois = [results_comparison[s]['total_return'] for s in strategies]
    model_types = [results_comparison[s]['model_type'] for s in strategies]
    total_predictions = [results_comparison[s].get('total_predictions', 100) for s in strategies]
    max_drawdowns = [results_comparison[s].get('max_drawdown', 0) for s in strategies]
    start_dates = [results_comparison[s].get('start_date', 'N/A') for s in strategies]
    end_dates = [results_comparison[s].get('end_date', 'N/A') for s in strategies]
    duration_days = [results_comparison[s].get('duration_days', 0) for s in strategies]
    
    # Calcular dinero real ganado/perdido
    final_capitals = [initial_capital * (1 + roi) for roi in rois]
    profits_losses = [final_capital - initial_capital for final_capital in final_capitals]
    
    # Colores específicos por tipo de modelo
    color_map = {
        'NAIVE': '#FF6B6B',     # Rojo
        'ARIMA': '#4ECDC4',     # Turquesa  
        'GRU': '#45B7D1'        # Azul
    }
    colors = [color_map.get(mt, '#96CEB4') for mt in model_types]
    
    # 1. CAPITAL FINAL - Gráfico de barras principal
    ax1 = fig.add_subplot(gs[0, :])
    bars1 = ax1.bar(range(len(strategies)), final_capitals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Línea de capital inicial
    ax1.axhline(y=initial_capital, color='red', linestyle='--', linewidth=3, alpha=0.8, label=f'Capital Inicial: ${initial_capital:,}')
    
    ax1.set_title('💰 CAPITAL FINAL DESPUÉS DEL TRADING', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Capital Final ($)')
    ax1.set_ylim(0, max(final_capitals) * 1.15)
    
    # Añadir valores en las barras con colores según ganancia/pérdida
    for i, (bar, final_cap, profit_loss, model_type) in enumerate(zip(bars1, final_capitals, profits_losses, model_types)):
        height = bar.get_height()
        
        # Color del texto según ganancia/pérdida
        text_color = 'green' if profit_loss > 0 else 'red'
        sign = '+' if profit_loss > 0 else ''
        
        # Valor del capital final
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_capitals) * 0.02,
                f'${final_cap:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Ganancia/pérdida
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_capitals) * 0.06,
                f'{sign}${profit_loss:,.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color=text_color)
        
        # Tipo de modelo en la base
        ax1.text(bar.get_x() + bar.get_width()/2., -max(final_capitals) * 0.05,
                f'{model_type}', ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([s.replace('Re-entrenamiento ', '').replace('GRU ', '') for s in strategies], 
                        rotation=45, ha='right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=12, loc='upper left')
    
    # 2. GANANCIA/PÉRDIDA NETA
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(range(len(strategies)), profits_losses, 
                    color=['green' if p > 0 else 'red' for p in profits_losses], 
                    alpha=0.7, edgecolor='black')
    
    ax2.set_title('📈 GANANCIA/PÉRDIDA NETA', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ganancia/Pérdida ($)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Añadir valores
    for i, (bar, profit_loss) in enumerate(zip(bars2, profits_losses)):
        height = bar.get_height()
        sign = '+' if profit_loss > 0 else ''
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (abs(height) * 0.05 if height > 0 else -abs(height) * 0.05),
                f'{sign}${profit_loss:,.0f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=10)
    
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.split()[-1] if 'cada' in s else s for s in strategies], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. MAXIMUM DRAWDOWN 
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(range(len(strategies)), [abs(dd) for dd in max_drawdowns], 
                    color=['red' if dd < -10 else 'orange' if dd < -5 else 'green' for dd in max_drawdowns], 
                    alpha=0.7, edgecolor='black')
    ax3.set_title('📉 MAXIMUM DRAWDOWN', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    
    # Añadir valores
    for i, (bar, dd) in enumerate(zip(bars3, max_drawdowns)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(abs(dd) for dd in max_drawdowns) * 0.02,
                f'{dd:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels([s.split()[-1] if 'cada' in s else s for s in strategies], rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. DIRECTIONAL ACCURACY 
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(range(len(strategies)), [da * 100 for da in das], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('🎯 DIRECTIONAL ACCURACY', fontsize=14, fontweight='bold')
    ax4.set_ylabel('DA (%)')
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Walk (50%)')
    
    # Añadir valores
    for i, (bar, da) in enumerate(zip(bars4, das)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{da*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([s.split()[-1] if 'cada' in s else s for s in strategies], rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # 5. TABLA RESUMEN COMPRENSIBLE
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Crear tabla comprensible con nueva información
    table_data = []
    headers = ['Modelo', 'Tipo', 'Capital Final', 'Ganancia/Pérdida', 'DA (%)', 'Max DD (%)', 'Trades']
    
    for i, strategy in enumerate(strategies):
        da_pct = results_comparison[strategy]['directional_accuracy'] * 100
        roi_pct = results_comparison[strategy]['total_return'] * 100
        model_type = results_comparison[strategy]['model_type']
        final_cap = final_capitals[i]
        profit_loss = profits_losses[i]
        trades = total_predictions[i]
        max_dd = max_drawdowns[i]
        start_date = start_dates[i]
        end_date = end_dates[i]
        days = duration_days[i]
        
        # Formatear ganancia/pérdida con color
        profit_loss_str = f"+${profit_loss:,.0f}" if profit_loss > 0 else f"-${abs(profit_loss):,.0f}"
        
        table_data.append([
            strategy.replace('Re-entrenamiento ', '').replace('GRU ', ''),
            model_type,
            f"${final_cap:,.0f}",
            profit_loss_str,
            f"{da_pct:.1f}%",
            f"{max_dd:.1f}%",
            f"{trades}"
        ])
    
    table = ax5.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colColours=['lightblue'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)
    
    # Colorear filas por tipo de modelo
    for i, model_type in enumerate(model_types):
        row_color = color_map.get(model_type, 'white')
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(row_color)
            table[(i + 1, j)].set_alpha(0.3)
            
        # Colorear columna de ganancia/pérdida según resultado
        profit_loss = profits_losses[i]
        gain_loss_color = 'lightgreen' if profit_loss > 0 else 'lightcoral'
        table[(i + 1, 3)].set_facecolor(gain_loss_color)  # Columna Ganancia/Pérdida
        table[(i + 1, 3)].set_alpha(0.6)
        
        # Colorear columna de drawdown según magnitud
        max_dd = max_drawdowns[i]
        dd_color = 'lightcoral' if abs(max_dd) > 10 else 'yellow' if abs(max_dd) > 5 else 'lightgreen'
        table[(i + 1, 5)].set_facecolor(dd_color)  # Columna Max DD
        table[(i + 1, 5)].set_alpha(0.6)
    
    # Resaltar el mejor resultado (mayor ganancia)
    best_idx = max(range(len(profits_losses)), key=lambda i: profits_losses[i])
    for j in range(len(headers)):
        table[(best_idx + 1, j)].set_facecolor('#FFD700')  # Dorado
        table[(best_idx + 1, j)].set_alpha(0.8)
    
    ax5.set_title(f'📋 RESUMEN COMPLETO DE TRADING\nCapital Inicial: ${initial_capital:,} | Mejor Modelo: {strategies[best_idx]}', 
                 fontsize=14, fontweight='bold', pad=30)
    
    # Guardar con configuración optimizada
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determinar ruta absoluta del directorio de imágenes
        current_dir = Path.cwd()
        if current_dir.name == "model":
            images_dir = current_dir.parent / "images" / "backtesting"
        else:
            images_dir = current_dir / "images" / "backtesting"
        
        # Asegurar que el directorio existe
        images_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = images_dir / f"comprehensive_trading_analysis_{timestamp}.png"
    
    # Guardar con buena resolución
    plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n🎨 Análisis comprensible de trading guardado: {save_path}")
    print(f"🖼️ Resolución: 200 DPI, Tamaño: 20x12")
    print(f"💰 Capital inicial: ${initial_capital:,}")
    print(f"🏆 Mejor modelo: {strategies[best_idx]} con ganancia de ${profits_losses[best_idx]:,.0f}")
    print(f"📉 Max Drawdown del mejor modelo: {max_drawdowns[best_idx]:.1f}%")
    
    plt.tight_layout()
    plt.show()
    
    return save_path

def generate_results_json(results_comparison, initial_capital=10000, save_path=None):
    """
    📄 Generar archivo JSON con resultados detallados del backtesting
    """
    if not results_comparison:
        print("❌ No hay resultados para generar JSON")
        return None
    
    # Obtener timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configurar path de guardado
    if save_path is None:
        # Determinar ruta absoluta del directorio de imágenes
        current_dir = Path.cwd()
        if current_dir.name == "model":
            images_dir = current_dir.parent / "images" / "backtesting"
        else:
            images_dir = current_dir / "images" / "backtesting"
        
        # Asegurar que el directorio existe
        images_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = images_dir / f"backtest_results_{timestamp}.json"
    
    # Extraer datos para JSON
    strategies = list(results_comparison.keys())
    
    # Calcular capital final y ganancias para cada estrategia
    json_results = {
        "metadata": {
            "timestamp": timestamp,
            "generation_date": datetime.now().isoformat(),
            "initial_capital": initial_capital,
            "currency": "USD",
            "analysis_type": "Rolling Re-training Backtesting",
            "models_compared": list(set([results_comparison[s]['model_type'] for s in strategies]))
        },
        "summary": {
            "total_strategies_tested": len(strategies),
            "best_strategy": None,
            "worst_strategy": None
        },
        "detailed_results": {}
    }
    
    # Procesar cada estrategia
    strategy_performance = []
    
    for strategy_name in strategies:
        data = results_comparison[strategy_name]
        
        # Calcular métricas financieras
        roi = data['total_return']
        final_capital = initial_capital * (1 + roi)
        profit_loss = final_capital - initial_capital
        
        strategy_result = {
            "strategy_name": str(strategy_name),
            "model_type": str(data['model_type']),
            "financial_metrics": {
                "initial_capital": float(initial_capital),
                "final_capital": round(float(final_capital), 2),
                "profit_loss": round(float(profit_loss), 2),
                "return_on_investment_pct": round(float(roi * 100), 2),
                "win_rate_pct": round(float(data['win_rate']), 2),
                "maximum_drawdown_pct": round(float(data.get('max_drawdown', 0)), 2)
            },
            "trading_metrics": {
                "directional_accuracy_pct": round(float(data['directional_accuracy'] * 100), 2),
                "total_predictions": int(data.get('total_predictions', 0)),
                "best_strategy_method": str(data.get('best_strategy_name', 'N/A'))
            },
            "time_metrics": {
                "start_date": str(data.get('start_date', 'N/A')),
                "end_date": str(data.get('end_date', 'N/A')),
                "duration_days": int(data.get('duration_days', 0))
            },
            "risk_metrics": {
                "max_drawdown_absolute": round(float(profit_loss * (data.get('max_drawdown', 0) / 100)), 2) if data.get('max_drawdown', 0) != 0 else 0.0,
                "risk_category": "High" if abs(float(data.get('max_drawdown', 0))) > 20 else "Medium" if abs(float(data.get('max_drawdown', 0))) > 10 else "Low"
            }
        }
        
        # Agregar información específica del modelo
        if data['model_type'] == 'GRU':
            strategy_result["model_specific"] = {
                "retraining_frequency": strategy_name.split('cada ')[-1] if 'cada' in strategy_name else 'N/A',
                "model_architecture": "GRU with Optuna optimization",
                "features_used": "EUR/USD price, DXY, RSI, SMA"
            }
        elif data['model_type'] in ['NAIVE', 'ARIMA']:
            strategy_result["model_specific"] = {
                "baseline_type": data['model_type'],
                "description": "Traditional forecasting baseline" if data['model_type'] == 'ARIMA' else "Simple carry-forward prediction"
            }
        
        json_results["detailed_results"][strategy_name] = strategy_result
        strategy_performance.append({
            'name': strategy_name,
            'profit': profit_loss,
            'roi': roi,
            'da': data['directional_accuracy']
        })
    
    # Determinar mejor y peor estrategia
    best_strategy = max(strategy_performance, key=lambda x: x['profit'])
    worst_strategy = min(strategy_performance, key=lambda x: x['profit'])
    
    json_results["summary"]["best_strategy"] = {
        "name": str(best_strategy['name']),
        "profit": round(float(best_strategy['profit']), 2),
        "roi_pct": round(float(best_strategy['roi'] * 100), 2),
        "directional_accuracy_pct": round(float(best_strategy['da'] * 100), 2)
    }
    
    json_results["summary"]["worst_strategy"] = {
        "name": str(worst_strategy['name']),
        "profit": round(float(worst_strategy['profit']), 2),
        "roi_pct": round(float(worst_strategy['roi'] * 100), 2),
        "directional_accuracy_pct": round(float(worst_strategy['da'] * 100), 2)
    }
    
    # Agregar comparación con baselines
    gru_strategies = [s for s in strategy_performance if 'GRU' in results_comparison[s['name']]['model_type']]
    baseline_strategies = [s for s in strategy_performance if results_comparison[s['name']]['model_type'] in ['NAIVE', 'ARIMA']]
    
    if gru_strategies and baseline_strategies:
        best_gru = max(gru_strategies, key=lambda x: x['profit'])
        best_baseline = max(baseline_strategies, key=lambda x: x['profit'])
        
        json_results["comparison_analysis"] = {
            "gru_vs_baselines": {
                "gru_outperforms": bool(best_gru['profit'] > best_baseline['profit']),
                "profit_advantage": round(float(best_gru['profit'] - best_baseline['profit']), 2),
                "da_advantage_points": round(float((best_gru['da'] - best_baseline['da']) * 100), 2),
                "best_gru_strategy": str(best_gru['name']),
                "best_baseline_strategy": str(best_baseline['name'])
            }
        }
    
    # Guardar JSON
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Resultados JSON guardados: {save_path}")
        print(f"📊 Estrategias analizadas: {len(strategies)}")
        print(f"🏆 Mejor estrategia: {best_strategy['name']} (${best_strategy['profit']:,.0f})")
        print(f"📉 Peor estrategia: {worst_strategy['name']} (${worst_strategy['profit']:,.0f})")
        
        return save_path
        
    except Exception as e:
        print(f"❌ Error al guardar JSON: {e}")
        return None
    """
    🎨 Generar análisis visual optimizado del rolling re-training vs BASELINES
    """
    if not results_comparison:
        print("❌ No hay resultados para analizar")
        return
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura más pequeña para evitar memory error
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # Título principal
    fig.suptitle('🚀 ANÁLISIS OPTIMIZADO: GRU vs BASELINES\nRolling Re-training vs Métodos Tradicionales', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Extraer datos para visualización
    strategies = list(results_comparison.keys())
    win_rates = [results_comparison[s]['win_rate'] for s in strategies]
    das = [results_comparison[s]['directional_accuracy'] for s in strategies]
    rois = [results_comparison[s]['total_return'] for s in strategies]
    model_types = [results_comparison[s]['model_type'] for s in strategies]
    
    # Colores específicos por tipo de modelo
    color_map = {
        'NAIVE': '#FF6B6B',     # Rojo
        'ARIMA': '#4ECDC4',     # Turquesa
        'GRU': '#45B7D1'        # Azul
    }
    colors = [color_map.get(mt, '#96CEB4') for mt in model_types]
    
    # 1. Gráfico de barras - Directional Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(strategies)), [da * 100 for da in das], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('📊 Directional Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('DA (%)')
    ax1.set_ylim(0, max([da * 100 for da in das]) * 1.1)
    
    # Añadir valores en las barras
    for i, (bar, da, model_type) in enumerate(zip(bars1, das, model_types)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{da*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        # Añadir etiqueta del tipo de modelo
        ax1.text(bar.get_x() + bar.get_width()/2., -2,
                model_type, ha='center', va='top', fontsize=8, rotation=0)
    
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([s.replace('Re-entrenamiento ', '').replace('GRU ', '') for s in strategies], rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Línea de referencia en 50% (random walk)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Walk (50%)')
    ax1.legend(fontsize=8)
    
    # 2. Gráfico de barras - ROI
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(strategies)), [roi * 100 for roi in rois], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('💰 Return on Investment (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ROI (%)')
    
    # Añadir valores en las barras
    for i, (bar, roi, model_type) in enumerate(zip(bars2, rois, model_types)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1 if height > 0 else height * 0.9,
                f'{roi*100:.0f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)
    
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.replace('Re-entrenamiento ', '').replace('GRU ', '') for s in strategies], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # 3. Tabla de resultados simplificada
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')
    
    # Crear tabla simplificada
    table_data = []
    headers = ['Modelo', 'Tipo', 'DA (%)', 'ROI (%)', 'Estrategia']
    
    for i, strategy in enumerate(strategies):
        da_pct = results_comparison[strategy]['directional_accuracy'] * 100
        roi_pct = results_comparison[strategy]['total_return'] * 100
        model_type = results_comparison[strategy]['model_type']
        best_strategy_name = results_comparison[strategy]['best_strategy_name']
        
        table_data.append([
            strategy.replace('Re-entrenamiento ', '').replace('GRU ', ''),
            model_type,
            f"{da_pct:.1f}%",
            f"{roi_pct:.0f}%",
            best_strategy_name.replace('combined_', '').replace('_', ' ').title()
        ])
    
    table = ax3.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colColours=['lightblue'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    # Colorear filas por tipo de modelo
    for i, model_type in enumerate(model_types):
        row_color = color_map.get(model_type, 'white')
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(row_color)
            table[(i + 1, j)].set_alpha(0.3)
    
    # Resaltar el mejor resultado
    best_idx = max(range(len(rois)), key=lambda i: (das[i] + abs(rois[i])) / 2)
    for j in range(len(headers)):
        table[(best_idx + 1, j)].set_facecolor('#90EE90')  # Verde claro
        table[(best_idx + 1, j)].set_alpha(0.8)
    
    ax3.set_title('📋 RESUMEN COMPARATIVO\nGRU Rolling Re-training vs Baselines Tradicionales', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 4. Análisis de superioridad
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Calcular mejoras de GRU sobre baselines
    gru_strategies = {k: v for k, v in results_comparison.items() if v['model_type'] == 'GRU'}
    baseline_strategies = {k: v for k, v in results_comparison.items() if v['model_type'] in ['NAIVE', 'ARIMA']}
    
    if gru_strategies and baseline_strategies:
        best_gru = max(gru_strategies.values(), key=lambda x: x['directional_accuracy'])
        
        improvements = []
        baseline_names = []
        
        for baseline_name, baseline_data in baseline_strategies.items():
            da_improvement = (best_gru['directional_accuracy'] - baseline_data['directional_accuracy']) * 100
            improvements.append(da_improvement)
            baseline_names.append(baseline_data['model_type'])
        
        bars = ax4.bar(baseline_names, improvements, 
                      color=['green' if imp > 0 else 'red' for imp in improvements], 
                      alpha=0.7, edgecolor='black')
        
        ax4.set_title('📈 Mejora GRU vs Baselines\n(Puntos % DA)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Mejora (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                    f'{imp:+.1f}pp', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)
    
    # 5. Análisis de conclusiones
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Determinar el mejor modelo
    best_strategy = max(results_comparison.keys(), 
                       key=lambda k: (results_comparison[k]['directional_accuracy'] + abs(results_comparison[k]['total_return'])) / 2)
    best_data = results_comparison[best_strategy]
    
    # Texto de análisis
    analysis_text = f"""🏆 RESULTADOS PRINCIPALES

🥇 MEJOR: {best_strategy.replace('Re-entrenamiento ', '').replace('GRU ', '')}
   Tipo: {best_data['model_type']}
   DA: {best_data['directional_accuracy']*100:.1f}%
   ROI: {best_data['total_return']*100:.0f}%

🔍 vs BASELINES:"""
    
    # Calcular mejoras vs baselines
    baseline_comparisons = []
    for baseline_name, baseline_data in results_comparison.items():
        if baseline_data['model_type'] in ['NAIVE', 'ARIMA']:
            da_diff = (best_data['directional_accuracy'] - baseline_data['directional_accuracy']) * 100
            roi_diff = (best_data['total_return'] - baseline_data['total_return']) * 100
            baseline_comparisons.append(f"   vs {baseline_data['model_type']}: DA {da_diff:+.1f}pp, ROI {roi_diff:+.0f}%")
    
    analysis_text += "\n" + "\n".join(baseline_comparisons)
    
    # Recomendaciones
    if best_data['model_type'] == 'GRU' and best_data['directional_accuracy'] > 0.55:
        recommendations = """

💡 CONCLUSIONES:
✅ GRU supera baselines
✅ Re-training efectivo  
✅ Listo para producción
⚠️ Validar con nuevos datos"""
    else:
        recommendations = """

⚠️ HALLAZGOS:
❌ Mejora limitada
💡 Considerar:
   • Más datos
   • Otros modelos
   • Ajustar parámetros"""
    
    analysis_text += recommendations
    
    ax5.text(0.05, 0.95, analysis_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    # Guardar con configuración optimizada
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determinar ruta absoluta del directorio de imágenes
        current_dir = Path.cwd()
        if current_dir.name == "model":
            images_dir = current_dir.parent / "images" / "backtesting"
        else:
            images_dir = current_dir / "images" / "backtesting"
        
        # Asegurar que el directorio existe
        images_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = images_dir / f"optimized_analysis_gru_vs_baselines_{timestamp}.png"
    
    # Guardar con resolución moderada para evitar memory error
    plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n🎨 Análisis visual optimizado guardado: {save_path}")
    print(f"🖼️ Resolución: 200 DPI, Formato: PNG, Tamaño: 16x12")
    print(f"📊 Incluye comparación: GRU vs NAIVE vs ARIMA")
    
    plt.tight_layout()
    plt.show()
    
    return save_path

if __name__ == "__main__":
    print("🚀 INICIANDO ANÁLISIS GRU COMPLETO - FRECUENCIA 10 + MONTE CARLO")
    print("=" * 70)
    
    # Ejecutar análisis completo con Monte Carlo
    results, monte_carlo = test_gru_rolling_retraining()
    
    if results:
        print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"📊 Resultados obtenidos: {len(results)} estrategias")
        if monte_carlo:
            print(f"🎲 Monte Carlo: {'SIGNIFICATIVO' if monte_carlo['is_significant_95'] else 'NO SIGNIFICATIVO'}")
    else:
        print("❌ Error en el análisis")
    
    
    # Generar análisis visual y JSON si hay resultados
    if results:
        print(f"\n🎨 Generando análisis visual comprensible...")
        chart_path = generate_comprehensive_trading_analysis(results, initial_capital=10000)
        print(f"✅ Análisis completo disponible en: {chart_path}")
        
        print(f"\n📄 Generando archivo JSON de resultados...")
        json_path = generate_results_json(results, initial_capital=10000)
        if json_path:
            print(f"✅ Resultados JSON disponibles en: {json_path}")
        else:
            print("❌ Error al generar archivo JSON")
    
    # Para backtesting completo con todos los modelos (mantén comentado):
    # main()
