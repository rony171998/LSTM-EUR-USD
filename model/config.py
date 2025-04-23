from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelParams:
    FILEPATH: str = "EUR_USD_2010-2024.csv"  # Cambia esto según necesites
    TARGET_COLUMN: str = "Último"
    SEQ_LENGTH: int = 120
    FORECAST_HORIZON: int = 1
    TRAIN_SPLIT_RATIO: float = 0.90
    BATCH_SIZE: int = 16
    EPOCHS: int = 150
    PATIENCE: int = 10
    LEARNING_RATE: float = 0.0023883992351518826
    HIDDEN_SIZE: int = 512
    DROPOUT_PROB: float = 0.10976329984400868
    FEATURES: List[str] = field(default_factory=lambda: ["Último", "RSI", "SMA"])
    MODELNAME: str = "TLS_LSTMModel" 
    #TLS_LSTMModel 
    #BidirectionalDeepLSTM
    #HybridLSTMAttention
    #TemporalAutoencoderLSTM
    #GRU_Model
    MODELPATH: str = f"{MODELNAME}_{FILEPATH}.pth"
    SCALER_PATH: str = f"modelos/{FILEPATH}_scaler.pkl"

# Instancia única de parámetros
DEFAULT_PARAMS = ModelParams()