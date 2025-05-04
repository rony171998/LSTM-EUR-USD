from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class ModelInfo(BaseModel):
    name: str
    description: str

@router.get("/models", response_model=List[ModelInfo])
def get_available_models():
    """
    Returns a list of available models for prediction
    """
    return [
        ModelInfo(
            name="TLS_LSTMModel",
            description="Two-Layer Stacked LSTM model with attention mechanism"
        ),
        ModelInfo(
            name="BidirectionalDeepLSTM",
            description="Bidirectional LSTM model with deep architecture"
        ),
        ModelInfo(
            name="HybridLSTMAttention",
            description="Hybrid model combining LSTM with attention mechanism"
        ),
        ModelInfo(
            name="GRU_Model",
            description="Gated Recurrent Unit model for time series prediction"
        )
    ] 