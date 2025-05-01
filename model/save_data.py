# save_data.py
import pandas as pd
import os
import logging
from sqlalchemy import create_engine ,text
from sqlalchemy.orm import sessionmaker
from config import DEFAULT_PARAMS
from train_model import load_and_prepare_data

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Configuración de logging
logging.basicConfig(filename='data_upload.log', level=logging.INFO, 
                   format='%(asctime)s - %(message)s')

def log_message(message: str):
    session = Session()
    try:
        # Verifica si existe la tabla logs
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT
        );
        """))
        
        # Ahora inserta el mensaje
        session.execute(
            text("INSERT INTO logs (message) VALUES (:message)"),
            {'message': message}
        )
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"🚨 Error al guardar log en DB: {e}")
    finally:
        session.close()

# Modifica las consultas así:
def create_table_if_not_exists():
    """Crea la tabla con nombres de columnas consistentes"""
    session = Session()
    try:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS eur_usd (
                fecha TIMESTAMP PRIMARY KEY,
                último DOUBLE PRECISION,
                apertura DOUBLE PRECISION,
                máximo DOUBLE PRECISION,
                mínimo DOUBLE PRECISION,
                vol DOUBLE PRECISION,
                var DOUBLE PRECISION
            );
        """))
        session.commit()
        print("✅ Tabla creada con nombres estandarizados")
    except Exception as e:
        session.rollback()
        print(f"🚨 Error al crear tabla: {e}")
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
    """💎 Valida la integridad del DataFrame antes de guardar."""
    print("✨ Validando datos...")
    log_message("🌸 Validación de datos iniciada...")

    if df['último'].isnull().any():
        print("🚨 Hay valores nulos en los datos.")
        log_message("🚨 Error: Hay valores nulos en los datos.")
        return False
    
    # Validar si la columna 'Fecha' tiene valores nulos
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

def check_existing_dates(df: pd.DataFrame, table_name: str = "eur_usd") -> pd.DataFrame:
    """Filtra fechas que ya existen en la base de datos (insensible a mayúsculas)"""
    if df.empty:
        return df

    try:
        # Consulta con nombres exactos (en minúsculas)
        query = text("""
            SELECT fecha 
            FROM eur_usd 
            WHERE fecha BETWEEN :min_date AND :max_date
        """)
        
        existing_dates = pd.read_sql(
            query,
            con=engine,
            params={'min_date': df.index.min(), 'max_date': df.index.max()}
        )
        
        # Filtrar el DataFrame original
        if not existing_dates.empty:
            existing_dates['fecha'] = pd.to_datetime(existing_dates['fecha'])
            return df[~df.index.isin(existing_dates['fecha'])]
        
        return df
        
    except Exception as e:
        print(f"⚠️ Error al verificar fechas: {str(e)}")
        return df  # Si hay error, retorna el DataFrame sin filtrar
def save_data_to_db(df: pd.DataFrame, table_name: str):
    """Guarda el DataFrame a la base de datos, evitando duplicados."""
    if df is not None and not df.empty:
        print(f"🌸 Subiendo datos a la tabla '{table_name}'...")
        
        # Verificar si hay datos duplicados por fecha
        df = check_existing_dates(df, table_name)
        
        if not df.empty:
            try:
                df.to_sql(
                    table_name,
                    con=engine,
                    if_exists='append',  # Agregar datos a la tabla
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

def desnormalize_column_names (df: pd.DataFrame) -> pd.DataFrame:
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

def get_df(from_date: str = None, to_date: str = None , table_name: str = None) -> pd.DataFrame:
    """
    Obtiene todos los datos históricos de la tabla eur_usd
    con opción de filtrar por rango de fechas.
    
    Parámetros:
        from_date (str): Fecha inicial en formato 'YYYY-MM-DD' (opcional)
        to_date (str): Fecha final en formato 'YYYY-MM-DD' (opcional)
    
    Retorna:
        pd.DataFrame: DataFrame con los datos solicitados
    """
    try:
        query = f"SELECT * FROM {table_name}"
        params = {}
        
        # Construir consulta con filtros de fecha si se especifican
        if from_date or to_date:
            conditions = []
            if from_date:
                conditions.append("fecha >= :from_date")
                params['from_date'] = from_date
            if to_date:
                conditions.append("fecha <= :to_date")
                params['to_date'] = to_date
            
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY fecha ASC"
        
        # Ejecutar consulta
        df = pd.read_sql(
            text(query),
            con=engine,
            params=params,
            parse_dates=['fecha'],
            index_col='fecha'
        )
        
        # Log exitoso
        log_message(f"✅ Datos obtenidos exitosamente ({len(df)} filas)")
        print(f"✅ Se obtuvieron {len(df)} registros de la base de datos")

        df = desnormalize_column_names(df)  # Desnormalizar nombres de columnas
        return df
        
    except Exception as e:
        error_msg = f"🚨 Error al obtener datos: {str(e)}"
        print(error_msg)
        log_message(error_msg)
        return pd.DataFrame()  # Retorna DataFrame vacío en caso de error

# 🛠️ MAIN FLOW
if __name__ == "__main__":
    create_table_if_not_exists()  # Asegura que la tabla exista

    table_name = "eur_usd"  # Nombre de la tabla en la base de datos
    get_df(table_name=table_name)  # Obtener datos existentes
    df = None #load_and_prepare_data(DEFAULT_PARAMS.FILEPATH)

    if df is not None:
        df = normalize_column_names(df)

        df.index = pd.to_datetime(df.index)
        df.index.name = 'fecha'

        # 💖 Limpiar duplicados de fechas antes de guardar
        df = df[~df.index.duplicated(keep='first')]

        # Verificar fechas existentes en la base de datos
        df = check_existing_dates(df, table_name)

        # Validación de datos
        if validate_data(df):
            print("✨ Datos validados, procediendo a guardar...")
            log_message("✨ Datos validados, procediendo a guardar...")
        else:
            print("🚫 Datos no válidos, abortando subida.")
            log_message("🚫 Datos no válidos, abortando subida.")
            exit()

        # Guardar los datos en la base de datos
        save_data_to_db(df, table_name)
    else:
        print("🚫 No se cargaron datos, abortando subida.")
        log_message("🚫 No se cargaron datos, abortando subida.")
