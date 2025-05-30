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
def create_table_if_not_exists(table_name: str):
    """Crea la tabla con nombres de columnas consistentes"""
    session = Session()
    try:
        # Usar texto sin formato con parámetros de SQLAlchemy
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
        raise  # Relanzar la excepción para manejo posterior
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

def check_existing_dates(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Filtra fechas que ya existen en la base de datos, retornando solo las filas
    cuyas fechas no existen en la tabla especificada.
    
    Args:
        df: DataFrame con los datos a filtrar. Debe tener un índice de tipo datetime.
        table_name: Nombre de la tabla en la base de datos a verificar.
        
    Returns:
        DataFrame con solo las filas cuyas fechas no existen en la base de datos.
    """
    if df.empty:
        return df

    try:
        # 1. Verificar que el índice sea de tipo datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️ El índice del DataFrame no es de tipo datetime")
            return df
            
        # 2. Obtener fechas únicas del DataFrame
        fechas_unicas = df.index.unique()
        if len(fechas_unicas) == 0:
            return df
            
        # 3. Consultar fechas existentes en la base de datos
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
        
        # 4. Si no hay fechas existentes, retornar el DataFrame completo
        if existing_dates.empty:
            print("ℹ️ No se encontraron fechas existentes en el rango especificado")
            return df
            
        # 5. Convertir fechas existentes a datetime y crear un conjunto para búsqueda rápida
        existing_dates['fecha'] = pd.to_datetime(existing_dates['fecha'])
        fechas_existentes_set = set(existing_dates['fecha'])
        
        # 6. Filtrar el DataFrame original
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
    Obtiene todos los datos históricos de la tabla {table_name}
    con opción de filtrar por rango de fechas.
    
    Parámetros:
        from_date (str): Fecha inicial en formato 'YYYY-MM-DD' (opcional)
        to_date (str): Fecha final en formato 'YYYY-MM-DD' (opcional)
        table_name (str): Nombre de la tabla
    
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
    table_name = DEFAULT_PARAMS.TABLENAME  # Nombre de la tabla en la base de datos
    create_table_if_not_exists(table_name=table_name)  # Asegura que la tabla exista

    #df =get_df(table_name=table_name)  # Obtener datos existentes
    df = load_and_prepare_data(DEFAULT_PARAMS.FILEPATH)

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
            print(" Datos validados, procediendo a guardar...")
            log_message(" Datos validados, procediendo a guardar...")
            # Guardar los datos en la base de datos
            save_data_to_db(df, table_name)
        else:
            print(" Datos no válidos, abortando subida.")
            log_message(" Datos no válidos, abortando subida.")
            import sys
            sys.exit(0)
    else:
        print("🚫 No se cargaron datos, abortando subida.")
        log_message("🚫 No se cargaron datos, abortando subida.")
        import sys
        sys.exit(0)
