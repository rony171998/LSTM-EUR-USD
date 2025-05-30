import pandas as pd
from datetime import datetime
import sys
import os
from config import DEFAULT_PARAMS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predict import predict_future_prices, PredictRequest
from model.save_data import save_data_to_db, normalize_column_names
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Logging
logging.basicConfig(filename='predictions.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# DB Setup
DATABASE_URL = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def create_prediction_table():
    """Crea la tabla central de predicciones si no existe"""
    session = Session()
    try:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                fecha TIMESTAMP NOT NULL,
                último DOUBLE PRECISION NOT NULL,
                apertura DOUBLE PRECISION,
                máximo DOUBLE PRECISION,
                mínimo DOUBLE PRECISION,
                is_prediction BOOLEAN DEFAULT TRUE,
                model_name VARCHAR(100) NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE (fecha, model_name, ticker)
            );
        """))
        session.commit()
        logging.info("✅ Tabla de predicciones creada/verificada.")
    except Exception as e:
        session.rollback()
        logging.error(f"🚨 Error creando tabla de predicciones: {e}")
    finally:
        session.close()

def generate_predictions(days_to_predict: int, model_name: str, ticker: str) -> pd.DataFrame:
    """Genera y guarda predicciones en la tabla central"""
    create_prediction_table()

    try:
        request = PredictRequest(
            ticker=ticker,
            model_name=model_name,
            future_steps_to_predict=days_to_predict
        )
        predictions = predict_future_prices(request)

        if not predictions or 'date' not in predictions[0] or 'close' not in predictions[0]:
            raise ValueError(f"❌ Estructura inesperada en predicciones: {predictions}")

        df = pd.DataFrame(predictions)
        df['fecha'] = pd.to_datetime(df['date'])
        df.drop(columns=['date'], inplace=True)
        df.set_index('fecha', inplace=True)
        df.rename(columns={'close': 'último'}, inplace=True)

        df['apertura'] = df['último'] * 0.999
        df['máximo'] = df['último'] * 1.002
        df['mínimo'] = df['último'] * 0.998
        df['is_prediction'] = True
        df['model_name'] = model_name
        df['ticker'] = ticker

        df.reset_index(inplace=True)
        df = normalize_column_names(df)

        required_cols = ['fecha', 'último', 'apertura', 'máximo', 'mínimo', 'is_prediction', 'model_name', 'ticker']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Falta columna requerida: {col}")

        # Filtrar duplicados existentes en DB
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT fecha FROM predictions
                WHERE model_name = :model AND ticker = :ticker
            """), {'model': model_name, 'ticker': ticker})
            existing_dates = set([row[0] for row in result])
            df = df[~df['fecha'].isin(existing_dates)]

        if df.empty:
            logging.info(f"ℹ️ No hay nuevas predicciones para insertar ({model_name} - {ticker})")
            return pd.DataFrame()

        df.set_index('fecha', inplace=True)
        save_data_to_db(df, 'predictions')
        logging.info(f"✅ Predicciones guardadas: {model_name} - {ticker}")
        return df

    except Exception as e:
        logging.error(f"🚨 Error generando predicciones: {model_name} - {ticker}: {str(e)}")
        raise

def generate_all_models_predictions(days_to_predict: int = 30, tickers: list = [DEFAULT_PARAMS.TICKER]):
    models = ["GRU_Model", "TLS_LSTMModel", "BidirectionalDeepLSTM", "HybridLSTMAttention"]
    all_predictions = {}

    for ticker in tickers:
        for model in models:
            try:
                df = generate_predictions(days_to_predict, model, ticker)
                all_predictions[f"{model}_{ticker}"] = df
                print(f"✅ Predicciones generadas: {model} - {ticker}")
            except Exception as e:
                print(f"❌ Error en {model} - {ticker}: {str(e)}")

    return all_predictions

if __name__ == "__main__":
    predictions = generate_all_models_predictions(days_to_predict=30, tickers=[DEFAULT_PARAMS.TICKER])
    print(f"✅ Proceso completo. Total modelos procesados: {len(predictions)}")
