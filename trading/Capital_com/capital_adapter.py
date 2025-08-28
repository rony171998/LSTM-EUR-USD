# capital_adapter.py
"""
Adaptador de Capital.com integrado con sistema de logging de transacciones
"""
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import capital_api as api
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import traceback

# Imports del sistema existente
from capital_api import (
    login, get_account_info, get_market_data, get_prices, 
    get_open_positions as get_api_open_positions, 
    place_order, close_position, modify_position
)

# Imports del sistema de logging mejorado
from save_data_trading import (
    init_trading_tables, log_transaction, close_transaction,
    get_open_positions as get_db_open_positions, get_trading_summary,
    save_daily_performance, log_message
)

class MLTradingBot:
    """Bot de trading con ML integrado con sistema de logging completo"""
    
    def __init__(self, model_path: str, initial_capital: float = 10000, 
                 risk_percentage: float = 2.0, max_positions: int = 3, 
                 retrain_frequency: int = 10):
        
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_percentage = risk_percentage
        self.max_positions = max_positions
        self.retrain_frequency = retrain_frequency
        
        # Estados del bot
        self.is_initialized = False
        self.model = None
        self.account_info = None
        self.last_retrain_date = None
        
        # Métricas de seguimiento
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        print(f"🤖 Bot ML inicializado - Capital: ${initial_capital}")

    def initialize(self) -> bool:
        """Inicializa el bot: login, tablas DB, modelo ML"""
        try:
            print("🚀 Inicializando bot de trading...")
            
            # 1. Inicializar base de datos
            print("📊 Configurando base de datos...")
            init_trading_tables()
            
            # 2. Login en Capital.com
            print("🔐 Iniciando sesión en Capital.com...")
            login()
            
            # 3. Obtener información de cuenta
            self.account_info = get_account_info()
            current_balance = self.account_info.get('balance', {}).get('available', 0)
            
            print(f"💰 Balance disponible: ${current_balance}")
            log_message(f"🤖 Bot inicializado - Balance: ${current_balance}")
            
            # 4. Cargar modelo ML (aquí cargarías tu modelo real)
            self.load_model()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            error_msg = f"❌ Error al inicializar bot: {str(e)}"
            print(error_msg)
            log_message(error_msg)
            print(traceback.format_exc())
            return False

    def load_model(self):
        """Carga el modelo ML (implementar según tu arquitectura)"""
        try:
            # Aquí cargarías tu modelo real
            # self.model = torch.load(self.model_path)
            print(f"🧠 Modelo ML cargado: {self.model_path}")
            log_message(f"🧠 Modelo ML cargado: {self.model_path}")
            # Placeholder para modelo
            self.model = "ML_MODEL_PLACEHOLDER"
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def get_ml_prediction(self, epic: str, market_data: Dict) -> Tuple[float, float, str]:
        """
        Obtiene predicción del modelo ML
        
        Returns:
            Tuple[prediction_value, confidence, direction]
        """
        try:
            # Aquí implementarías tu lógica de predicción real
            # Por ahora, simulamos una predicción
            
            # Obtener datos históricos
            prices = get_prices(epic, resolution="HOUR", max_values=50)
            
            if len(prices) < 20:
                return 0.0, 0.0, "HOLD"
            
            # Simular predicción (reemplazar con tu modelo real)
            recent_prices = [p['closePrice']['mid'] for p in prices[-10:]]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Lógica simple de ejemplo (reemplazar con ML real)
            if price_change > 0.001:  # 0.1% de cambio
                prediction = abs(price_change) * 100
                confidence = min(0.8, abs(price_change) * 1000)
                direction = "BUY"
            elif price_change < -0.001:
                prediction = abs(price_change) * 100
                confidence = min(0.8, abs(price_change) * 1000)
                direction = "SELL"
            else:
                prediction = 0.0
                confidence = 0.3
                direction = "HOLD"
            
            print(f"🧠 Predicción ML: {direction} (conf: {confidence:.2f})")
            return prediction, confidence, direction
            
        except Exception as e:
            print(f"❌ Error en predicción ML: {e}")
            return 0.0, 0.0, "HOLD"

    def calculate_position_size(self, epic: str, entry_price: float, 
                              stop_loss: float) -> float:
        """Calcula el tamaño de posición basado en gestión de riesgo"""
        try:
            # Obtener balance actual
            account = get_account_info()
            available_balance = account['balance']['available']
            
            # Calcular riesgo en dinero
            risk_amount = available_balance * (self.risk_percentage / 100)
            
            # Calcular distancia al stop loss
            price_distance = abs(entry_price - stop_loss)
            
            # Calcular tamaño de posición
            if price_distance > 0:
                position_size = risk_amount / price_distance
                # Ajustar según especificaciones del mercado
                position_size = round(position_size, 2)
                position_size = max(0.01, min(position_size, 10.0))  # Límites
            else:
                position_size = 0.01
                
            print(f"📏 Tamaño calculado: {position_size} (riesgo: ${risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculando tamaño: {e}")
            return 0.01

    def open_position(self, epic: str, direction: str, prediction: float, 
                     confidence: float) -> Optional[str]:
        """
        Abre una posición y la registra en la base de datos
        """
        try:
            # Obtener datos del mercado
            market_data = get_market_data(epic)
            
            if direction == "BUY":
                entry_price = market_data['snapshot']['offer']
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.04  # 4% take profit
            else:  # SELL
                entry_price = market_data['snapshot']['bid']
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profit = entry_price * 0.96  # 4% take profit
            
            # Calcular tamaño de posición
            position_size = self.calculate_position_size(epic, entry_price, stop_loss)
            
            # Calcular valores
            entry_value = position_size * entry_price
            risk_amount = abs(entry_value - (position_size * stop_loss))
            
            # Obtener capital actual
            account = get_account_info()
            current_balance = account['balance']['available']
            
            print(f"📈 Abriendo posición: {epic} {direction} {position_size} @ {entry_price}")
            
            # Colocar orden en Capital.com
            order_response = place_order(
                epic=epic,
                direction=direction,
                size=position_size,
                stop_price=stop_loss,
                profit_price=take_profit
            )
            
            deal_id = order_response.get('dealReference') or f"{epic}_{int(time.time())}"
            
            # Registrar transacción en base de datos
            success = log_transaction(
                deal_id=deal_id,
                epic=epic,
                action=direction,
                size=position_size,
                entry_price=entry_price,
                entry_value=entry_value,
                risk_amount=risk_amount,
                risk_percentage=self.risk_percentage,
                capital_before=current_balance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                ml_prediction=prediction,
                ml_confidence=confidence,
                strategy_used="ML_BIDIRECTIONAL_LSTM",
                notes=f"Predicción ML: {prediction:.4f}, Confianza: {confidence:.2f}"
            )
            
            if success:
                self.total_trades += 1
                print(f"✅ Posición abierta y registrada: {deal_id}")
                log_message(f"📈 Posición abierta: {epic} {direction} {position_size} @ {entry_price}")
                return deal_id
            else:
                print("❌ Error registrando transacción")
                return None
                
        except Exception as e:
            error_msg = f"❌ Error abriendo posición: {str(e)}"
            print(error_msg)
            log_message(error_msg)
            print(traceback.format_exc())
            return None

    def close_position_by_id(self, deal_id: str, reason: str = "Manual") -> bool:
        """
        Cierra una posición específica y actualiza la base de datos
        """
        try:
            print(f"🔄 Cerrando posición: {deal_id}")
            
            # Obtener información actual de la posición desde la API
            api_positions = get_api_open_positions()
            position_info = None
            
            for pos in api_positions:
                if pos.get('dealId') == deal_id:
                    position_info = pos
                    break
            
            if not position_info:
                print(f"⚠️ Posición no encontrada en API: {deal_id}")
                return False
            
            # Cerrar posición en Capital.com
            close_response = close_position(deal_id)
            
            # Obtener datos para el cierre
            exit_price = position_info.get('level', 0)
            position_size = position_info.get('size', 0)
            direction = position_info.get('direction', 'BUY')
            
            # Calcular valores de salida
            exit_value = position_size * exit_price
            
            # Estimar comisión (ajustar según Capital.com)
            commission = exit_value * 0.0001  # 0.01% estimado
            
            # Obtener balance actual
            account = get_account_info()
            current_balance = account['balance']['available']
            
            # Actualizar transacción en base de datos
            success = close_transaction(
                deal_id=deal_id,
                exit_price=exit_price,
                exit_value=exit_value,
                commission=commission,
                capital_after=current_balance,
                notes=f"Cerrada por: {reason}"
            )
            
            if success:
                # Actualizar métricas locales
                # (El P&L se calcula en close_transaction)
                print(f"✅ Posición cerrada y registrada: {deal_id}")
                log_message(f"📉 Posición cerrada: {deal_id} por {reason}")
                return True
            else:
                print(f"❌ Error actualizando cierre en DB: {deal_id}")
                return False
                
        except Exception as e:
            error_msg = f"❌ Error cerrando posición {deal_id}: {str(e)}"
            print(error_msg)
            log_message(error_msg)
            print(traceback.format_exc())
            return False

    def manage_open_positions(self):
        """Gestiona las posiciones abiertas (stop loss, take profit, etc.)"""
        try:
            # Obtener posiciones desde la base de datos
            db_positions = get_db_open_positions()
            
            if db_positions.empty:
                return
                
            print(f"📊 Gestionando {len(db_positions)} posiciones abiertas...")
            
            # Obtener posiciones actuales de la API
            api_positions = get_api_open_positions()
            api_deal_ids = {pos.get('dealId') for pos in api_positions}
            
            for _, position in db_positions.iterrows():
                deal_id = position['deal_id']
                
                # Verificar si la posición aún existe en la API
                if deal_id not in api_deal_ids:
                    print(f"🔍 Posición {deal_id} no encontrada en API, cerrando en DB...")
                    # La posición se cerró externamente, actualizar DB
                    self.close_position_by_id(deal_id, "Cerrada externamente")
                    continue
                
                # Verificar tiempo de la posición
                entry_time = pd.to_datetime(position['entry_time'])
                duration = (datetime.utcnow() - entry_time.to_pydatetime()).total_seconds() / 3600  # horas
                
                # Cerrar posiciones muy antiguas (ej: más de 24 horas)
                if duration > 24:
                    print(f"⏰ Cerrando posición antigua: {deal_id} ({duration:.1f}h)")
                    self.close_position_by_id(deal_id, "Timeout - 24h")
                    continue
                
                # Aquí puedes agregar más lógica de gestión
                # Por ejemplo, trailing stop, ajustar stop loss, etc.
                
        except Exception as e:
            error_msg = f"❌ Error gestionando posiciones: {str(e)}"
            print(error_msg)
            log_message(error_msg)

    def analyze_and_trade(self, epic: str):
        """Analiza un instrumento y ejecuta trading si es necesario"""
        try:
            print(f"\n🔍 Analizando {epic}...")
            
            # Verificar número de posiciones abiertas
            open_positions = get_db_open_positions()
            current_positions = len(open_positions[open_positions['epic'] == epic])
            
            if current_positions >= self.max_positions:
                print(f"⚠️ Máximo de posiciones alcanzado para {epic}: {current_positions}")
                return
            
            # Obtener datos del mercado
            market_data = get_market_data(epic)
            
            # Obtener predicción ML
            prediction, confidence, direction = self.get_ml_prediction(epic, market_data)
            
            # Condiciones para abrir posición
            min_confidence = 0.6  # Confianza mínima
            
            if confidence < min_confidence:
                print(f"📊 Confianza insuficiente: {confidence:.2f} < {min_confidence}")
                return
            
            if direction == "HOLD":
                print("📊 Señal HOLD - No se abre posición")
                return
            
            # Abrir posición
            deal_id = self.open_position(epic, direction, prediction, confidence)
            
            if deal_id:
                print(f"🎯 Nueva posición abierta: {deal_id}")
            else:
                print("❌ No se pudo abrir la posición")
                
        except Exception as e:
            error_msg = f"❌ Error analizando {epic}: {str(e)}"
            print(error_msg)
            log_message(error_msg)

    def print_performance_summary(self):
        """Muestra resumen de performance"""
        try:
            # Obtener resumen de los últimos 7 días
            summary = get_trading_summary(days=7)
            
            if "error" in summary:
                print("📊 No hay datos de trading recientes")
                return
            
            print("\n" + "="*50)
            print("📊 RESUMEN DE PERFORMANCE (7 días)")
            print("="*50)
            print(f"🎯 Total trades: {summary['total_trades']}")
            print(f"💚 Trades ganadores: {summary['winning_trades']}")
            print(f"❤️ Trades perdedores: {summary['losing_trades']}")
            print(f"🎯 Win Rate: {summary['win_rate']}%")
            print(f"💰 P&L Total: ${summary['total_pnl']:.2f}")
            print(f"📈 Retorno: {summary['return_percentage']:.2f}%")
            print(f"⏱️ Duración promedio: {summary['avg_duration_minutes']:.1f} min")
            
            if summary.get('profit_factor'):
                print(f"🔥 Profit Factor: {summary['profit_factor']:.2f}")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error mostrando resumen: {e}")

    def run_strategy(self, epics: List[str], interval_minutes: int = 60):
        """
        Ejecuta la estrategia de trading principal
        
        Args:
            epics: Lista de instrumentos a tradear
            interval_minutes: Intervalo de análisis en minutos
        """
        if not self.is_initialized:
            print("❌ Bot no inicializado")
            return
        
        print(f"\n🚀 Iniciando estrategia de trading...")
        print(f"📈 Instrumentos: {epics}")
        print(f"⏰ Intervalo: {interval_minutes} minutos")
        print("="*50)
        
        try:
            while True:
                loop_start = time.time()
                
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Gestionar posiciones abiertas
                self.manage_open_positions()
                
                # Analizar cada instrumento
                for epic in epics:
                    try:
                        self.analyze_and_trade(epic)
                        time.sleep(2)  # Pequeña pausa entre instrumentos
                    except Exception as e:
                        print(f"❌ Error con {epic}: {e}")
                        continue
                
                # Mostrar resumen cada hora
                if self.total_trades > 0 and self.total_trades % 5 == 0:
                    self.print_performance_summary()
                
                # Guardar performance diaria (cada 6 horas)
                current_hour = datetime.now().hour
                if current_hour in [0, 6, 12, 18] and datetime.now().minute < 5:
                    save_daily_performance()
                
                # Esperar hasta el siguiente ciclo
                elapsed = time.time() - loop_start
                wait_time = max(0, (interval_minutes * 60) - elapsed)
                
                if wait_time > 0:
                    print(f"😴 Esperando {wait_time/60:.1f} minutos hasta el siguiente análisis...")
                    time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n⏹️ Deteniendo bot...")
            self.shutdown()
        except Exception as e:
            error_msg = f"❌ Error crítico en estrategia: {str(e)}"
            print(error_msg)
            log_message(error_msg)
            print(traceback.format_exc())

    def shutdown(self):
        """Cierra el bot ordenadamente"""
        try:
            print("🔄 Cerrando bot...")
            
            # Cerrar todas las posiciones abiertas (opcional)
            # open_positions = get_db_open_positions()
            # for _, position in open_positions.iterrows():
            #     self.close_position_by_id(position['deal_id'], "Bot shutdown")
            
            # Guardar performance final
            save_daily_performance()
            
            # Mostrar resumen final
            self.print_performance_summary()
            
            log_message("🛑 Bot cerrado correctamente")
            print("✅ Bot cerrado correctamente")
            
        except Exception as e:
            print(f"❌ Error cerrando bot: {e}")


class TradingBot:
    def __init__(self, risk_percentage: float = 2.0, max_positions: int = 3):
        """
        Inicializa el bot de trading
        
        Args:
            risk_percentage: Porcentaje de riesgo por operación
            max_positions: Número máximo de posiciones abiertas
        """
        self.risk_percentage = risk_percentage
        self.max_positions = max_positions
        self.positions = []
        self.account_info = None
        
    def initialize(self):
        """Inicializa la conexión con Capital.com"""
        try:
            api.login()
            self.account_info = api.get_account_info()
            print(f"Bot inicializado. Balance: ${self.account_info['balance']:.2f}")
            return True
        except Exception as e:
            print(f"Error al inicializar: {e}")
            return False
    
    def calculate_position_size(self, stop_distance: float, price: float) -> float:
        """
        Calcula el tamaño de la posición basado en el riesgo
        
        Args:
            stop_distance: Distancia al stop loss en puntos
            price: Precio actual del activo
        """
        if not self.account_info:
            return 0
        
        risk_amount = self.account_info['balance'] * (self.risk_percentage / 100)
        position_size = risk_amount / stop_distance
        
        # Ajustar al tamaño mínimo/máximo permitido
        return round(position_size, 2)
    
    def analyze_market(self, epic: str) -> Optional[Dict]:
        """
        Analiza el mercado para un instrumento específico
        
        Args:
            epic: Identificador del instrumento
        """
        try:
            # Obtener datos del mercado
            market_data = api.get_market_data(epic)
            prices = api.get_prices(epic, resolution="HOUR", max_values=100)
            
            if not prices:
                return None
            
            # Calcular indicadores técnicos simples
            analysis = self.calculate_indicators(prices)
            analysis['current_price'] = market_data['snapshot']['bid']
            analysis['epic'] = epic
            
            return analysis
            
        except Exception as e:
            print(f"Error en análisis de mercado: {e}")
            return None
    
    def calculate_indicators(self, prices: List[Dict]) -> Dict:
        """
        Calcula indicadores técnicos básicos
        
        Args:
            prices: Lista de precios históricos
        """
        close_prices = [p['closePrice']['bid'] for p in prices]
        
        # Media móvil simple (20 períodos)
        sma20 = sum(close_prices[-20:]) / 20 if len(close_prices) >= 20 else None
        
        # Media móvil simple (50 períodos)
        sma50 = sum(close_prices[-50:]) / 50 if len(close_prices) >= 50 else None
        
        # RSI simplificado
        rsi = self.calculate_rsi(close_prices, 14)
        
        # Determinar tendencia
        trend = "NEUTRAL"
        if sma20 and sma50:
            if sma20 > sma50:
                trend = "BULLISH"
            elif sma20 < sma50:
                trend = "BEARISH"
        
        return {
            'sma20': sma20,
            'sma50': sma50,
            'rsi': rsi,
            'trend': trend,
            'last_close': close_prices[-1] if close_prices else None
        }
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calcula el RSI (Relative Strength Index)
        
        Args:
            prices: Lista de precios
            period: Período para el cálculo
        """
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def generate_signal(self, analysis: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Genera señales de trading basadas en el análisis
        
        Args:
            analysis: Resultados del análisis técnico
        """
        if not analysis or not all(k in analysis for k in ['rsi', 'trend', 'current_price']):
            return None, None
        
        signal = None
        params = {}
        
        # Estrategia simple: RSI + Tendencia
        if analysis['trend'] == 'BULLISH' and analysis['rsi'] < 70:
            if analysis['rsi'] < 40:  # Sobreventa en tendencia alcista
                signal = 'BUY'
                params = {
                    'stop_distance': analysis['current_price'] * 0.02,  # Stop 2% abajo
                    'profit_distance': analysis['current_price'] * 0.04  # TP 4% arriba
                }
        elif analysis['trend'] == 'BEARISH' and analysis['rsi'] > 30:
            if analysis['rsi'] > 60:  # Sobrecompra en tendencia bajista
                signal = 'SELL'
                params = {
                    'stop_distance': analysis['current_price'] * 0.02,  # Stop 2% arriba
                    'profit_distance': analysis['current_price'] * 0.04  # TP 4% abajo
                }
        
        return signal, params
    
    def execute_trade(self, epic: str, signal: str, params: Dict) -> bool:
        """
        Ejecuta una operación de trading
        
        Args:
            epic: Identificador del instrumento
            signal: Señal de trading (BUY/SELL)
            params: Parámetros de la operación
        """
        try:
            # Verificar número de posiciones abiertas
            open_positions = api.get_open_positions()
            if len(open_positions) >= self.max_positions:
                print(f"Máximo de posiciones alcanzado ({self.max_positions})")
                return False
            
            # Obtener precio actual
            market_data = api.get_market_data(epic)
            current_price = market_data['snapshot']['bid']
            
            # Calcular niveles
            if signal == 'BUY':
                stop_price = current_price - params['stop_distance']
                profit_price = current_price + params['profit_distance']
            else:
                stop_price = current_price + params['stop_distance']
                profit_price = current_price - params['profit_distance']
            
            # Calcular tamaño de posición
            size = self.calculate_position_size(params['stop_distance'], current_price)
            
            if size <= 0:
                print("Tamaño de posición inválido")
                return False
            
            # Ejecutar orden
            result = api.place_order(
                epic=epic,
                direction=signal,
                size=size,
                stop_price=stop_price,
                profit_price=profit_price
            )
            
            print(f"✅ Orden ejecutada: {signal} {size} unidades de {epic}")
            print(f"   Stop Loss: {stop_price:.5f}, Take Profit: {profit_price:.5f}")
            
            return True
            
        except Exception as e:
            print(f"Error al ejecutar trade: {e}")
            return False
    
    def monitor_positions(self):
        """Monitorea las posiciones abiertas"""
        try:
            positions = api.get_open_positions()
            
            if not positions:
                print("No hay posiciones abiertas")
                return
            
            print(f"\n📊 Posiciones abiertas: {len(positions)}")
            for pos in positions:
                pnl = pos.get('profit', 0)
                symbol = "🟢" if pnl >= 0 else "🔴"
                print(f"{symbol} {pos['market']['epic']}: PnL: ${pnl:.2f}")
                
        except Exception as e:
            print(f"Error al monitorear posiciones: {e}")
    
    def run_strategy(self, epics: List[str], interval_seconds: int = 300):
        """
        Ejecuta la estrategia de trading continuamente
        
        Args:
            epics: Lista de instrumentos a operar
            interval_seconds: Intervalo entre análisis en segundos
        """
        print(f"🤖 Bot de trading iniciado")
        print(f"Instrumentos: {', '.join(epics)}")
        print(f"Intervalo: {interval_seconds} segundos")
        
        while True:
            try:
                for epic in epics:
                    print(f"\n⏰ Analizando {epic}...")
                    
                    # Analizar mercado
                    analysis = self.analyze_market(epic)
                    if not analysis:
                        continue
                    
                    # Generar señal
                    signal, params = self.generate_signal(analysis)
                    
                    if signal:
                        print(f"📡 Señal detectada: {signal}")
                        self.execute_trade(epic, signal, params)
                    else:
                        print(f"No hay señales para {epic}")
                
                # Monitorear posiciones existentes
                self.monitor_positions()
                
                # Actualizar información de cuenta
                self.account_info = api.get_account_info()
                print(f"\n💰 Balance actual: ${self.account_info['balance']:.2f}")
                
                # Esperar antes del próximo ciclo
                print(f"\n⏳ Esperando {interval_seconds} segundos...")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot detenido por el usuario")
                break
            except Exception as e:
                print(f"Error en el ciclo principal: {e}")
                time.sleep(30)  # Esperar 30 segundos en caso de error

