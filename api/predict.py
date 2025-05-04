from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import timedelta
from model.train_model import add_indicator, load_and_prepare_data, get_model, device
from model.config import DEFAULT_PARAMS, ModelParams
from pydantic import BaseModel
from typing import List, Literal
import os
from model.modelos import TLS_LSTMModel, GRU_Model, HybridLSTMAttentionModel, BidirectionalDeepLSTMModel

router = APIRouter()

# Modelo de entrada de la API
class PredictRequest(BaseModel):
    ticker: str  # No se usa aún, pero podrías adaptarlo para cambiar el CSV
    model_name: Literal["TLS_LSTMModel", "BidirectionalDeepLSTM", "HybridLSTMAttention", "GRU_Model"] = "GRU_Model"
    future_steps_to_predict: int = 30

# Mapeo de nombres de modelo a clases
MODEL_CLASSES = {
    "TLS_LSTMModel": TLS_LSTMModel,
    "BidirectionalDeepLSTM": BidirectionalDeepLSTMModel,
    "HybridLSTMAttention": HybridLSTMAttentionModel,
    "GRU_Model": GRU_Model
}

@router.post("/predict")
def predict_future_prices(request: PredictRequest):
    try:
        # Crear una nueva instancia de parámetros con el modelo seleccionado
        params = ModelParams()
        params.MODELNAME = request.model_name
        params.MODELPATH = f"{params.MODELNAME}_{params.FILEPATH}.pth"
        params.SCALER_PATH = f"modelos/{params.FILEPATH}_scaler.pkl"
        
        # Verificar que los archivos existan
        model_path = os.path.join("modelos", params.MODELPATH)
        scaler_path = params.SCALER_PATH
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Scaler file not found: {scaler_path}")
            
        scaler = joblib.load(scaler_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando scaler: {str(e)}")

    try:
        # Obtener la clase del modelo correcta
        model_class = MODEL_CLASSES[request.model_name]
        model = model_class(
            input_size=len(params.FEATURES),
            hidden_size=params.HIDDEN_SIZE,
            output_size=params.FORECAST_HORIZON,
            dropout_prob=params.DROPOUT_PROB
        ).to(device)
        
        # Cargar los pesos del modelo
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

    try:
        df = load_and_prepare_data(params.FILEPATH)
        indicators = add_indicator(df)
        for name, values in indicators.items():
            df[name] = values
        last_known = df[params.FEATURES].iloc[-params.SEQ_LENGTH:].values
        last_known_scaled = scaler.transform(last_known)
        current_sequence = torch.tensor(last_known_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        future_preds_scaled = []
        with torch.no_grad():
            for _ in range(request.future_steps_to_predict):
                pred = model(current_sequence)
                neutral_rsi = (50 - scaler.center_[1]) / scaler.scale_[1]
                last_sma = (df["SMA"].iloc[-1] - scaler.center_[2]) / scaler.scale_[2]
                next_step = torch.tensor([[pred.item(), neutral_rsi, last_sma]], dtype=torch.float32).unsqueeze(0).to(device)
                future_preds_scaled.append(pred.item())
                current_sequence = torch.cat((current_sequence[:, 1:, :], next_step), dim=1)

        preds_for_inverse = np.zeros((len(future_preds_scaled), len(params.FEATURES)))
        preds_for_inverse[:, 0] = future_preds_scaled
        preds_for_inverse[:, 1] = 50
        preds_for_inverse[:, 2] = df["SMA"].iloc[-1]
        future_predictions = scaler.inverse_transform(preds_for_inverse)[:, 0]

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=request.future_steps_to_predict, freq='B')

        predictions = []
        for date, pred in zip(future_dates, future_predictions):
            predictions.append({
                "date": date.strftime('%Y-%m-%d'),
                "close": float(pred)
            })

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
