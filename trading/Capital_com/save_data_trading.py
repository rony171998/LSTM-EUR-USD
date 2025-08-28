# save_data_trading.py
import pandas as pd
import os
import logging
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL" , "postgresql://default:3OJkChlXe7ag@ep-throbbing-tooth-a4nrxkkp-pooler.us-east-1.aws.neon.tech/lstmdb").replace("postgres://", "postgresql://")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Configuración de logging
logging.basicConfig(filename='trading_system.log', level=logging.INFO, 
                   format='%(asctime)s - %(message)s')

# ======================== MODELOS DE DATOS ========================

class TradingTransaction(Base):
    """Modelo para registrar todas las transacciones de trading"""
    __tablename__ = 'trading_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    deal_id = Column(String(100), unique=True)  # ID único de la operación
    epic = Column(String(20), nullable=False)  # Instrumento (EURUSD, etc.)
    
    # Datos de la operación
    action = Column(String(10), nullable=False)  # BUY/SELL
    size = Column(Float, nullable=False)  # Tamaño de la posición
    entry_price = Column(Float, nullable=False)  # Precio de entrada
    exit_price = Column(Float, nullable=True)  # Precio de salida
    
    # Estado de la operación
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    
    # Métricas financieras
    entry_value = Column(Float, nullable=False)  # Valor total de entrada
    exit_value = Column(Float, nullable=True)  # Valor total de salida
    gross_pnl = Column(Float, default=0.0)  # P&L bruto
    commission = Column(Float, default=0.0)  # Comisiones
    net_pnl = Column(Float, default=0.0)  # P&L neto (después de comisiones)
    
    # Gestión de riesgo
    stop_loss = Column(Float, nullable=True)  # Precio de stop loss
    take_profit = Column(Float, nullable=True)  # Precio de take profit
    risk_amount = Column(Float, nullable=False)  # Cantidad arriesgada
    risk_percentage = Column(Float, nullable=False)  # % de riesgo del capital
    
    # Duración
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    
    # Datos del modelo (si aplica)
    ml_prediction = Column(Float, nullable=True)  # Predicción del modelo ML
    ml_confidence = Column(Float, nullable=True)  # Confianza de la predicción
    strategy_used = Column(String(50), nullable=True)  # Estrategia utilizada
    
    # Capital en el momento
    capital_before = Column(Float, nullable=False)  # Capital antes de la operación
    capital_after = Column(Float, nullable=True)  # Capital después de la operación
    
    # Metadatos
    notes = Column(Text, nullable=True)  # Notas adicionales
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradingPerformance(Base):
    """Modelo para métricas de performance agregadas"""
    __tablename__ = 'trading_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)  # Fecha del resumen
    
    # Métricas básicas
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Métricas financieras
    total_pnl = Column(Float, default=0.0)
    total_commission = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    
    # Métricas de riesgo
    max_drawdown = Column(Float, default=0.0)
    max_profit = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    # Capital
    starting_capital = Column(Float, nullable=False)
    ending_capital = Column(Float, nullable=False)
    return_percentage = Column(Float, default=0.0)
    
    # Promedios
    avg_winning_trade = Column(Float, default=0.0)
    avg_losing_trade = Column(Float, default=0.0)
    avg_trade_duration = Column(Float, default=0.0)  # En minutos
    
    created_at = Column(DateTime, default=datetime.utcnow)

# ======================== FUNCIONES DE BASE DE DATOS ========================

def init_trading_tables():
    """Inicializa todas las tablas de trading"""
    try:
        Base.metadata.create_all(engine)
        print("✅ Tablas de trading inicializadas correctamente")
        log_message("✅ Tablas de trading inicializadas correctamente")
    except Exception as e:
        print(f"🚨 Error al inicializar tablas: {e}")
        log_message(f"🚨 Error al inicializar tablas: {e}")
        raise

def log_message(message: str):
    """Registra mensaje en la tabla de logs y archivo"""
    session = Session()
    try:
        # Crear tabla de logs si no existe
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT
        );
        """))
        
        # Insertar mensaje
        session.execute(
            text("INSERT INTO logs (message) VALUES (:message)"),
            {'message': message}
        )
        session.commit()
        logging.info(message)
    except Exception as e:
        session.rollback()
        print(f"🚨 Error al guardar log: {e}")
        logging.error(f"Error al guardar log: {e}")
    finally:
        session.close()

# ======================== FUNCIONES DE TRADING LOGS ========================

def log_transaction(
    deal_id: str,
    epic: str,
    action: str,  # BUY/SELL
    size: float,
    entry_price: float,
    entry_value: float,
    risk_amount: float,
    risk_percentage: float,
    capital_before: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    ml_prediction: Optional[float] = None,
    ml_confidence: Optional[float] = None,
    strategy_used: Optional[str] = None,
    notes: Optional[str] = None
) -> bool:
    """
    Registra una nueva transacción de trading
    
    Args:
        deal_id: ID único de la operación
        epic: Instrumento financiero
        action: BUY o SELL
        size: Tamaño de la posición
        entry_price: Precio de entrada
        entry_value: Valor total de la entrada
        risk_amount: Cantidad arriesgada
        risk_percentage: Porcentaje de riesgo
        capital_before: Capital antes de la operación
        stop_loss: Precio de stop loss (opcional)
        take_profit: Precio de take profit (opcional)
        ml_prediction: Predicción del modelo ML (opcional)
        ml_confidence: Confianza de la predicción (opcional)
        strategy_used: Estrategia utilizada (opcional)
        notes: Notas adicionales (opcional)
    
    Returns:
        bool: True si se registró correctamente
    """
    session = Session()
    try:
        transaction = TradingTransaction(
            deal_id=deal_id,
            epic=epic,
            action=action,
            size=size,
            entry_price=entry_price,
            entry_value=entry_value,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            capital_before=capital_before,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            strategy_used=strategy_used,
            entry_time=datetime.utcnow(),
            notes=notes,
            status='OPEN'
        )
        
        session.add(transaction)
        session.commit()
        
        log_message(f"📊 Nueva transacción registrada: {deal_id} - {epic} {action} {size} @ {entry_price}")
        print(f"✅ Transacción registrada: {deal_id}")
        return True
        
    except Exception as e:
        session.rollback()
        error_msg = f"🚨 Error al registrar transacción {deal_id}: {e}"
        print(error_msg)
        log_message(error_msg)
        return False
    finally:
        session.close()

def close_transaction(
    deal_id: str,
    exit_price: float,
    exit_value: float,
    commission: float,
    capital_after: float,
    notes: Optional[str] = None
) -> bool:
    """
    Cierra una transacción existente y calcula las métricas finales
    
    Args:
        deal_id: ID único de la operación
        exit_price: Precio de salida
        exit_value: Valor total de salida
        commission: Comisiones cobradas
        capital_after: Capital después de la operación
        notes: Notas adicionales
    
    Returns:
        bool: True si se actualizó correctamente
    """
    session = Session()
    try:
        # Buscar la transacción
        transaction = session.query(TradingTransaction).filter_by(deal_id=deal_id).first()
        
        if not transaction:
            print(f"❌ Transacción no encontrada: {deal_id}")
            return False
        
        # Calcular métricas
        gross_pnl = exit_value - transaction.entry_value
        if transaction.action == 'SELL':  # Para posiciones cortas
            gross_pnl = transaction.entry_value - exit_value
        
        net_pnl = gross_pnl - commission
        exit_time = datetime.utcnow()
        duration = (exit_time - transaction.entry_time).total_seconds() / 60  # minutos
        
        # Actualizar la transacción
        transaction.exit_price = exit_price
        transaction.exit_value = exit_value
        transaction.gross_pnl = gross_pnl
        transaction.commission = commission
        transaction.net_pnl = net_pnl
        transaction.capital_after = capital_after
        transaction.exit_time = exit_time
        transaction.duration_minutes = int(duration)
        transaction.status = 'CLOSED'
        transaction.updated_at = datetime.utcnow()
        
        if notes:
            transaction.notes = f"{transaction.notes or ''}\n{notes}".strip()
        
        session.commit()
        
        # Log del resultado
        result_emoji = "💚" if net_pnl >= 0 else "❤️"
        log_message(f"{result_emoji} Transacción cerrada: {deal_id} - P&L: ${net_pnl:.2f} ({duration:.1f}min)")
        print(f"✅ Transacción cerrada: {deal_id} - P&L: ${net_pnl:.2f}")
        
        return True
        
    except Exception as e:
        session.rollback()
        error_msg = f"🚨 Error al cerrar transacción {deal_id}: {e}"
        print(error_msg)
        log_message(error_msg)
        return False
    finally:
        session.close()

def get_open_positions() -> pd.DataFrame:
    """Obtiene todas las posiciones abiertas"""
    try:
        query = """
        SELECT * FROM trading_transactions 
        WHERE status = 'OPEN' 
        ORDER BY entry_time DESC
        """
        
        df = pd.read_sql(
            text(query),
            con=engine,
            parse_dates=['entry_time', 'exit_time', 'created_at', 'updated_at']
        )
        
        print(f"📊 Se encontraron {len(df)} posiciones abiertas")
        return df
        
    except Exception as e:
        print(f"🚨 Error al obtener posiciones abiertas: {e}")
        log_message(f"🚨 Error al obtener posiciones abiertas: {e}")
        return pd.DataFrame()

def get_trading_summary(days: int = 30) -> Dict[str, Any]:
    """
    Obtiene un resumen de trading para los últimos N días
    
    Args:
        days: Número de días hacia atrás para el resumen
        
    Returns:
        Dict con métricas de performance
    """
    session = Session()
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Consultar transacciones cerradas en el período
        transactions = session.query(TradingTransaction).filter(
            TradingTransaction.status == 'CLOSED',
            TradingTransaction.exit_time >= start_date
        ).all()
        
        if not transactions:
            return {"error": "No hay transacciones en el período especificado"}
        
        # Calcular métricas
        total_trades = len(transactions)
        winning_trades = sum(1 for t in transactions if t.net_pnl > 0)
        losing_trades = sum(1 for t in transactions if t.net_pnl < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.net_pnl for t in transactions)
        total_commission = sum(t.commission for t in transactions)
        
        winning_amounts = [t.net_pnl for t in transactions if t.net_pnl > 0]
        losing_amounts = [t.net_pnl for t in transactions if t.net_pnl < 0]
        
        avg_winning_trade = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0
        avg_losing_trade = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0
        
        avg_duration = sum(t.duration_minutes for t in transactions if t.duration_minutes) / total_trades
        
        # Capital inicial y final
        first_trade = min(transactions, key=lambda x: x.entry_time)
        last_trade = max(transactions, key=lambda x: x.exit_time)
        
        starting_capital = first_trade.capital_before
        ending_capital = last_trade.capital_after
        return_percentage = ((ending_capital - starting_capital) / starting_capital) * 100
        
        summary = {
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_commission": round(total_commission, 2),
            "avg_winning_trade": round(avg_winning_trade, 2),
            "avg_losing_trade": round(avg_losing_trade, 2),
            "avg_duration_minutes": round(avg_duration, 1),
            "starting_capital": round(starting_capital, 2),
            "ending_capital": round(ending_capital, 2),
            "return_percentage": round(return_percentage, 2),
            "profit_factor": round(abs(avg_winning_trade / avg_losing_trade), 2) if avg_losing_trade != 0 else float('inf')
        }
        
        return summary
        
    except Exception as e:
        error_msg = f"🚨 Error al generar resumen: {e}"
        print(error_msg)
        log_message(error_msg)
        return {"error": str(e)}
    finally:
        session.close()

def save_daily_performance():
    """Guarda las métricas de performance diarias"""
    session = Session()
    try:
        today = datetime.utcnow().date()
        
        # Verificar si ya existe entrada para hoy
        existing = session.query(TradingPerformance).filter(
            TradingPerformance.date == today
        ).first()
        
        if existing:
            print("📊 Performance diaria ya registrada para hoy")
            return
        
        # Obtener resumen del día
        summary = get_trading_summary(days=1)
        
        if "error" in summary:
            print(f"⚠️ No se pudo generar performance diaria: {summary['error']}")
            return
        
        # Crear registro de performance
        performance = TradingPerformance(
            date=today,
            total_trades=summary.get('total_trades', 0),
            winning_trades=summary.get('winning_trades', 0),
            losing_trades=summary.get('losing_trades', 0),
            win_rate=summary.get('win_rate', 0.0),
            total_pnl=summary.get('total_pnl', 0.0),
            total_commission=summary.get('total_commission', 0.0),
            net_pnl=summary.get('total_pnl', 0.0),
            starting_capital=summary.get('starting_capital', 0.0),
            ending_capital=summary.get('ending_capital', 0.0),
            return_percentage=summary.get('return_percentage', 0.0),
            avg_winning_trade=summary.get('avg_winning_trade', 0.0),
            avg_losing_trade=summary.get('avg_losing_trade', 0.0),
            avg_trade_duration=summary.get('avg_duration_minutes', 0.0)
        )
        
        session.add(performance)
        session.commit()
        
        log_message(f"📊 Performance diaria guardada: {summary['total_trades']} trades, P&L: ${summary['total_pnl']:.2f}")
        print(f"✅ Performance diaria guardada")
        
    except Exception as e:
        session.rollback()
        error_msg = f"🚨 Error al guardar performance diaria: {e}"
        print(error_msg)
        log_message(error_msg)
    finally:
        session.close()

# ======================== FUNCIONES ORIGINALES (MANTENIDAS) ========================

def create_table_if_not_exists(table_name: str):
    """Crea la tabla con nombres de columnas consistentes"""
    session = Session()
    try:
        sql = f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                fecha TIMESTAMP PRIMARY KEY,
                último DOUBLE PRECISION,
                apertura DOUBLE PRECISION,
                máximo DOUBLE PRECISION,
                mínimo DOUBLE PRECISION,
                vol DOUBLE PRECISION,
                var DOUBLE PRECISION
            );
        """
        session.execute(text(sql))
        session.commit()
        print(f"✅ Tabla '{table_name}' creada/verificada con éxito")
    except Exception as e:
        session.rollback()
        print(f"🚨 Error al crear/verificar tabla '{table_name}': {e}")
        raise
    finally:
        session.close()

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para coincidir con la BD"""
    column_mapping = {
        "Último": "último",
        "Apertura": "apertura",
        "Máximo": "máximo",
        "Mínimo": "mínimo",
        "Vol.": "vol",
        "% var.": "var"
    }
    return df.rename(columns=column_mapping)

def validate_data(df: pd.DataFrame) -> bool:
    """Valida la integridad del DataFrame antes de guardar."""
    print("✨ Validando datos...")
    log_message("🌸 Validación de datos iniciada...")

    if df['último'].isnull().any():
        print("🚨 Hay valores nulos en los datos.")
        log_message("🚨 Error: Hay valores nulos en los datos.")
        return False
    
    if df.index.isnull().any():
        print("🚨 La columna 'Fecha' tiene valores nulos.")
        log_message("🚨 Error: La columna 'Fecha' tiene valores nulos.")
        return False

    if not df.index.is_unique:
        print("🚨 Las fechas en el índice no son únicas.")
        log_message("🚨 Error: Las fechas en el índice no son únicas.")
        return False

    if (df[["último", "apertura", "máximo", "mínimo"]] < 0).any().any():
        print("🚨 Hay precios negativos, eso no debería pasar.")
        log_message("🚨 Error: Hay precios negativos en los datos.")
        return False

    print("✅ Datos validados correctamente~!")
    log_message("✅ Datos validados correctamente~!")
    return True

def check_existing_dates(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Filtra fechas que ya existen en la base de datos"""
    if df.empty:
        return df

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️ El índice del DataFrame no es de tipo datetime")
            return df
            
        fechas_unicas = df.index.unique()
        if len(fechas_unicas) == 0:
            return df
            
        query = text(f'''
            SELECT DISTINCT fecha 
            FROM "{table_name}" 
            WHERE fecha BETWEEN :min_date AND :max_date
        ''')
        
        existing_dates = pd.read_sql(
            query,
            con=engine,
            params={'min_date': fechas_unicas.min(), 'max_date': fechas_unicas.max()}
        )
        
        if existing_dates.empty:
            print("ℹ️ No se encontraron fechas existentes en el rango especificado")
            return df
            
        existing_dates['fecha'] = pd.to_datetime(existing_dates['fecha'])
        fechas_existentes_set = set(existing_dates['fecha'])
        
        mascara = ~df.index.isin(fechas_existentes_set)
        filas_a_mantener = mascara.sum()
        
        if filas_a_mantener == 0:
            print("ℹ️ Todas las fechas ya existen en la base de datos")
        else:
            print(f"📅 Se encontraron {len(df) - filas_a_mantener} fechas existentes de {len(df)} totales")
            print(f"📌 Se mantendrán {filas_a_mantener} filas para subir")
            
        return df[mascara]
        
    except Exception as e:
        print(f"⚠️ Error al verificar fechas: {str(e)}")
        return df

def save_data_to_db(df: pd.DataFrame, table_name: str):
    """Guarda el DataFrame a la base de datos, evitando duplicados."""
    
    if df is not None and not df.empty:
        print(f"🌸 Subiendo datos a la tabla '{table_name}'...")
        
        df = check_existing_dates(df, table_name)  
        if not df.empty:
            try:
                df.to_sql(
                    table_name,
                    con=engine,
                    if_exists='append',
                    index=True,
                    index_label='fecha',
                    method='multi'
                )
                print(f"✅ Datos subidos exitosamente con {len(df)} filas~!")
                log_message(f"✅ Subida exitosa de {len(df)} filas a la tabla '{table_name}'.")
            except Exception as e:
                print(f"🚨 Error al subir datos: {e}")
                log_message(f"🚨 Error al subir datos: {e}")
        else:
            print("🚫 No hay nuevos datos para subir, todo ya existe.")
            log_message("🚫 No hay nuevos datos para subir, todo ya existe.")
    else:
        print("🚫 DataFrame vacío, no se sube nada.")
        log_message("🚫 DataFrame vacío, no se sube nada.")

def desnormalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Desnormaliza nombres de columnas para coincidir con la BD"""
    column_mapping = {
        "último": "Último",
        "apertura": "Apertura",
        "máximo": "Máximo",
        "mínimo": "Mínimo",
        "vol": "Vol.",
        "var": "% var."
    }
    return df.rename(columns=column_mapping)

# ======================== MAIN FLOW ========================

if __name__ == "__main__":
    # Inicializar todas las tablas
    init_trading_tables()
    
    # Mostrar resumen de los últimos 7 días
    summary = get_trading_summary(days=7)
    if "error" not in summary:
        print(f"📊 Resumen últimos 7 días:")
        print(f"   • Total trades: {summary['total_trades']}")
        print(f"   • Win rate: {summary['win_rate']}%")
        print(f"   • P&L total: ${summary['total_pnl']}")
        print(f"   • Retorno: {summary['return_percentage']}%")
    
    # Mostrar posiciones abiertas
    open_positions = get_open_positions()
    if not open_positions.empty:
        print(f"📈 Hay {len(open_positions)} posiciones abiertas")
    else:
        print("💤 No hay posiciones abiertas actualmente")
    
    print("✅ Sistema listo para registrar transacciones de trading!")