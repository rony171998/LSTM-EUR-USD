# 📊 RESUMEN COMPLETO DE MODELOS ENTRENADOS - LSTM EUR/USD

## 🎯 Información General del Dataset
- **Dataset**: EUR/USD + DXY (2010-2024)
- **Registros totales**: 3,882 muestras
- **Features utilizadas**: 4 ['returns', 'rsi', 'sma20', 'dxy_returns']
- **Split temporal**: 80% train (3,105) / 20% test (777)
- **Secuencias**: Train (3,075, 30, 4) | Test (747, 30, 4)
- **Escalado**: RobustScaler (sin quantile_range personalizado)
- **Criterio de pérdida**: MSE Loss
- **Optimizador**: Adam (lr=0.001)
- **Batch size**: 32
- **GPU**: NVIDIA GeForce RTX 4060

---

## 📋 TABLA COMPARATIVA COMPLETA

| **Modelo** | **Capas** | **Hidden Size** | **Output Size** | **Seq Length** | **Parámetros** | **Épocas** | **Dropout** | **Características Especiales** |
|------------|-----------|-----------------|-----------------|----------------|----------------|------------|-------------|--------------------------------|
| **TLS_LSTMModel** | 2 LSTM | 50 | 1 | 30 | 31,651 | 100 | 0.2 | 2 capas LSTM secuenciales con dropout |
| **GRU_Model** | 2 GRU | 50 | 1 | 30 | 54,351 | 100 | 0.2 | 2 capas GRU con num_layers=2 |
| **HybridLSTMAttentionModel** | 2 LSTM + Attention | 50 | 1 | 30 | 34,202 | 100 | 0.2 | LSTM + Mecanismo de Atención + FC |
| **BidirectionalDeepLSTMModel** | 2 LSTM Bidireccional | 50 | 1 | 30 | 88,301 | 100 | 0.2 | LSTM Bidireccional + Capas Profundas |
| **TLS_LSTMModel Optimizado** | 2 LSTM | 128 | 1 | 30 | 200,833 | 150 | 0.1 | Versión optimizada con mayor capacidad |
| **ContextualLSTMTransformerFlexible** | LSTM + Transformer | 32 (LSTM) | 1 | 30 | 134,977 | 150 | 0.2 | LSTM + Self-Attention + Cross-Attention |

---

## 🏆 RESULTADOS DE RENDIMIENTO

| **Modelo** | **Test RMSE** | **Test R²** | **Train DA** | **Test DA** | **Tiempo** | **Ranking** |
|------------|---------------|-------------|--------------|-------------|------------|-------------|
| **HybridLSTMAttentionModel** | **0.006669** | **0.958940** | 49.1% | 50.4% | 1:36 | 🥇 **1°** |
| **TLS_LSTMModel Optimizado** | **0.006269** | **0.963723** | 49.5% | 50.3% | 1:44 | 🥈 **2°** |
| **BidirectionalDeepLSTMModel** | 0.012893 | 0.846564 | 48.9% | **51.7%** | 1:20 | 🥉 **3°** |
| **GRU_Model** | 0.014184 | 0.814289 | 48.6% | 49.6% | 1:19 | **4°** |
| **TLS_LSTMModel** | 0.014611 | 0.802931 | 48.6% | 50.7% | 1:12 | **5°** |
| **ContextualLSTMTransformerFlexible** | 0.021419 | 0.576492 | 51.2% | **51.3%** | 10:03 | **6°** |

---

## 🔧 DETALLES ARQUITECTÓNICOS ESPECÍFICOS

### 1. **TLS_LSTMModel (Básico)**
```python
- LSTM1: input_size=4 → hidden_size=50
- Dropout1: 0.2
- LSTM2: hidden_size=50 → hidden_size=50  
- Dropout2: 0.2
- FC: hidden_size=50 → output_size=1
```

### 2. **GRU_Model**
```python
- GRU1: input_size=4 → hidden_size=50, num_layers=2
- Dropout1: 0.2
- GRU2: hidden_size=50 → hidden_size=50, num_layers=2
- Dropout2: 0.2
- FC: hidden_size=50 → output_size=1
```

### 3. **HybridLSTMAttentionModel** ⭐
```python
- LSTM1: input_size=4 → hidden_size=50
- Dropout1: 0.2
- LSTM2: hidden_size=50 → hidden_size=50
- Attention: Linear(50→25) + Tanh + Linear(25→1) + Softmax
- FC: Linear(50→25) + ReLU + Dropout(0.2) + Linear(25→1)
```

### 4. **BidirectionalDeepLSTMModel**
```python
- LSTM: input_size=4 → hidden_size=50, bidirectional=True, num_layers=2
- Dropout: 0.2
- FC: Linear(100→50) + ReLU + Dropout(0.2) + Linear(50→1)
```

### 5. **TLS_LSTMModel Optimizado** ⭐
```python
- LSTM1: input_size=4 → hidden_size=128
- Dropout1: 0.1 (reducido)
- LSTM2: hidden_size=128 → hidden_size=128
- Dropout2: 0.1 (reducido)
- FC: hidden_size=128 → output_size=1
- Épocas: 150 (aumentadas)
```

### 6. **ContextualLSTMTransformerFlexible**
```python
- Window Size: 6 (divide seq_len=30 en 5 ventanas)
- Max Neighbors: 1
- LSTM Units: 32 (bidireccional → 64)
- Num Heads: 2
- Embed Dim: 64
- Componentes:
  * ReshapeToWindows
  * LSTMWithSelfAttention (LSTM + MultiheadAttention)
  * CrossAttentionBlock (para ventanas contextuales)
  * Final Dense: 64 → 1
```

---

## 📈 ANÁLISIS COMPARATIVO

### 🏅 **CAMPEONES POR CATEGORÍA:**

1. **🎯 Mejor RMSE**: TLS_LSTMModel Optimizado (0.006269)
2. **📊 Mejor R²**: TLS_LSTMModel Optimizado (0.963723)
3. **🎲 Mejor Directional Accuracy**: ContextualLSTMTransformerFlexible (51.3%)
4. **⚡ Más Eficiente**: HybridLSTMAttentionModel (34K parámetros, 2° lugar)
5. **🚀 Más Rápido**: TLS_LSTMModel (1:12 min)

### 📊 **PATRONES IDENTIFICADOS:**

1. **Tamaño vs Rendimiento**: Más parámetros no siempre = mejor rendimiento
2. **Atención Funciona**: Los 2 mejores modelos usan atención o mayor capacidad
3. **Dropout Optimizado**: Reducir dropout (0.1 vs 0.2) mejoró el rendimiento
4. **Épocas Importan**: 150 épocas vs 100 mejoró la convergencia
5. **Transformer Complejo**: Alta capacidad direccional pero overfitting en RMSE

### ⚠️ **OBSERVACIONES TÉCNICAS:**

- **Overfitting**: ContextualLSTMTransformer muestra Train R² (0.996) vs Test R² (0.576)
- **Sweet Spot**: 34K-200K parámetros parecen óptimos para este dataset
- **Directional vs RMSE**: Existe trade-off entre precisión numérica y direccional
- **GPU Necesaria**: Todos los modelos requieren CUDA para entrenamiento eficiente

---

## 🎯 CONCLUSIONES FINALES

1. **Top Tier**: HybridLSTMAttentionModel y TLS_LSTM Optimizado están en una liga propia
2. **Atención > Complejidad**: Mecanismos de atención superan a arquitecturas más complejas
3. **Configuración Importa**: Ajustar dropout, épocas y hidden_size es crucial
4. **Especialización**: Algunos modelos mejor para RMSE, otros para direccionalidad

---

*Generado el 13 de agosto de 2025 - Experimento LSTM EUR/USD*
