# GitHub Copilot Instructions for LSTM-EUR-USD

## 🎯 Project Overview
Financial time series prediction system for EUR/USD exchange rates using LSTM neural networks compared against statistical baselines. The project emphasizes **rigorous baseline comparison** - any ML model must significantly outperform Naive and ARIMA baselines to demonstrate real value.

## 🏗️ Core Architecture

### Configuration-Driven Design
- **Central config**: `model/config.py` - ModelParams dataclass controls all model parameters
- **Multi-model support**: Switch between TLS_LSTM, GRU, HybridLSTMAttention, etc. via `MODELNAME` parameter
- **Consistent paths**: Model files follow pattern `modelos/{TABLENAME}/{MODELNAME}_{FILEPATH}.pth`

### Data Pipeline Pattern
```python
# Standard workflow across all model training scripts
df = load_and_prepare_data(FILEPATH)  # Loads CSV with financial data converters
indicators = add_indicator(df)        # Adds RSI, SMA technical indicators  
scaler = RobustScaler(quantile_range=(5, 95))  # Always use RobustScaler
X, y = create_sequences(data, SEQ_LENGTH, FORECAST_HORIZON)  # Sliding windows
```

### Model Architecture Patterns
- **Adapter Pattern**: `evaluate_models.py` uses adapter classes (UltraIntelligentAdapter, FinalOptimizedAdapter) to make models compatible with evaluation system
- **Checkpoint Pattern**: Models save comprehensive metadata: `{'model_state_dict', 'test_rmse', 'naive_rmse', 'improvement_pct', 'direction_accuracy'}`
- **Device Management**: Always use `device = torch.device("cuda" if torch.cuda.is_available())` and verify GPU availability

## 🔬 Critical Financial ML Principles

### Baseline-First Philosophy
1. **Naive Baseline**: Last value carried forward - surprisingly effective in financial markets
2. **ARIMA Statistical**: Classical econometric approach with stationarity tests
3. **Beat Both**: ML models must achieve RMSE < 0.004 and DA > 52% to be considered successful

### Temporal Data Integrity
- **NO k-fold validation**: Use rolling windows for time series
- **Strict chronological splits**: `train_split_ratio=0.80` with no shuffling
- **Data leakage prevention**: Never scale train/test together, fit scaler only on training

### Financial Market Specifics
- **Random walk assumption**: EUR/USD follows near-random walk (Hurst exponent ~0.5)
- **Directional accuracy matters**: Predicting direction (up/down) more important than exact values
- **Sequence lengths**: Standard `SEQ_LENGTH=120` (trading days), `FORECAST_HORIZON=1`

## 🛠️ Key Development Workflows

### GPU-Only Training Policy
**CRITICAL**: All model training must be performed on GPU for optimal performance and convergence. CPU training is inefficient and produces inferior results.

**GPU Setup Requirements**:
- If CUDA GPU not detected or PyTorch GPU support fails, reinstall with:
```bash
pip uninstall torch torchvision torchaudio
```
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- Verify GPU availability: `torch.cuda.is_available()` must return `True`
- Monitor GPU memory usage during training to prevent OOM errors

### Training New Models
```bash
# Standard training with configurable parameters (GPU REQUIRED)
python model/train_model.py --model TLS_LSTMModel --data EUR_USD_2010-2024.csv

# Advanced models with GPU optimization (GPU MANDATORY)
python train_ultra_intelligent_model_fixed.py  # 1.5M parameters, requires CUDA
```

### Model Evaluation & Comparison
```bash
# Compare all models including baselines
python model/evaluate_models.py  # Generates comparison tables and charts

# Individual baseline training
python model/train_naive_baseline.py      # Simple but effective baseline
python model/train_arima_professional.py  # Rigorous statistical baseline
```

### API Deployment
```bash
# Local development
python main.py  # FastAPI server on localhost:8000

# Production deployment patterns in railway.json, render.yaml, Procfile
```

## 📁 File System Conventions

### Directory Structure Logic
- `model/`: Core ML training scripts and model definitions
- `modelos/`: Trained model artifacts (.pth files) and scalers (.pkl)
- `data/`: Historical price data and optimization parameters
- `images/`: Generated analysis charts organized by type (correlaciones/, estadisticas/, prediccion/)
- `api/`: FastAPI endpoints for model serving

### Import Patterns
```python
# Standard imports across training scripts
from config import DEFAULT_PARAMS
from modelos import TLS_LSTMModel, GRU_Model, HybridLSTMAttentionModel
from train_model import load_and_prepare_data, create_sequences, add_indicator
```

### Model Registry Pattern
Models defined in `modelos.py` follow consistent interfaces:
- `__init__(input_size, hidden_size, output_size, dropout_prob)`
- `forward(x)` returns predictions with proper tensor shapes
- Special models (Naive, ARIMA) inherit from nn.Module but implement statistical methods

## 🔧 Technical Implementation Details

### Data Preprocessing Standards
- **CSV parsing**: Uses custom converters for European number formats (comma decimals, dot thousands)
- **Feature engineering**: RSI(14), SMA(20) technical indicators standard
- **Scaling**: RobustScaler with quantile_range=(5,95) to handle outliers

### Model Compatibility System
- **Evaluation adapter**: `evaluate_models.py` handles sequence length mismatches (120 vs 30)
- **Checkpoint loading**: Always check for both `state_dict` and `model_state_dict` keys
- **Path resolution**: Models searched in multiple locations with glob patterns

### Performance Benchmarks
- **Naive Baseline**: RMSE: 0.005025, R²: 0.976684, DA: 50.0%
- **Target Performance**: RMSE < 0.004, R² > 0.98, DA > 52%
- **UltraIntelligent**: Current best with RMSE: 0.005024, R²: 0.976696

## 🚨 Common Pitfalls & Solutions

### Tensor Shape Mismatches
- Always use `.unsqueeze(-1)` when adapting between model expectations
- Check sequence lengths: evaluation system uses 120, some models expect 30
- Handle both 1D and 2D output tensors in evaluation functions

### Model Loading Issues
- Use `map_location=device` for cross-platform compatibility
- Check both direct state_dict and nested model.state_dict patterns
- Implement glob-based path searching for checkpoint files

### Financial Data Specifics
- Verify stationarity with ADF tests before modeling
- Calculate Hurst exponent to understand market efficiency
- Always include directional accuracy alongside RMSE metrics

When working on this codebase, prioritize beating baseline performance over architectural complexity. The financial markets are incredibly efficient, making significant improvements over simple baselines genuinely challenging and meaningful.
