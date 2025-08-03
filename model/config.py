# model/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelParams:
    FILEPATH: str = "USD_COP_2010-2024.csv"  # EUR_USD_2010-2024.csv , USD_COP_2010-2024.csv
    TABLENAME: str = "usd_cop" # eur_usd , usd_cop
    TICKER: str = "USD/COP" # EUR/USD , USD/COP
    TARGET_COLUMN: str = "Último"
    SEQ_LENGTH: int = 120
    FORECAST_HORIZON: int = 1
    TRAIN_SPLIT_RATIO: float = 0.80
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
    #GRU_Model
    #ContextualLSTMTransformerFlexible
    #NaiveForecastModel
    #ARIMAModel
    MODELPATH: str = f"modelos/{TABLENAME}/{MODELNAME}_{FILEPATH}.pth"
    SCALER_PATH: str = f"modelos/{TABLENAME}/{FILEPATH}_scaler.pkl"
    WINDOW_SIZE: int = 10
    MAX_NEIGHBORS: int = 2
    LSTM_UNITS: int = 64
    NUM_HEADS: int = 8
    EMBED_DIM: int = 128

# Instancia única de parámetros
DEFAULT_PARAMS = ModelParams()