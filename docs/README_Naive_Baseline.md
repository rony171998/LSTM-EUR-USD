# Modelo Baseline Naive Forecast

## Descripción

El modelo **Naive Forecast** es un modelo baseline simple que predice que el valor de mañana será igual al de hoy. Este tipo de modelo es fundamental en el análisis de series temporales porque:

1. **Referencia de comparación**: Cualquier modelo más complejo debe superar significativamente al Naive para justificar su complejidad
2. **Benchmark mínimo**: Si un modelo LSTM no puede superar al Naive, probablemente hay problemas en los datos o la implementación
3. **Interpretabilidad**: Es fácil de entender y explicar

## Características del Modelo

- **Tipo**: Baseline/Benchmark
- **Complejidad**: Mínima (sin parámetros entrenables)
- **Predicción**: `y(t+1) = y(t)`
- **Tiempo de entrenamiento**: Instantáneo
- **Memoria requerida**: Mínima

## Uso

### 1. Entrenar el modelo Naive

```bash
cd model
python train_naive_baseline.py
```

### 2. Comparar todos los modelos

```bash
python evaluate_models.py
```

## Interpretación de Resultados

### Métricas a considerar:

- **RMSE/MAE**: Menor es mejor
- **R²**: Mayor es mejor (cercano a 1)
- **MAPE**: Menor es mejor
- **Directional Accuracy**: Mayor es mejor (>50% indica capacidad predictiva direccional)

### Cómo interpretar:

1. **Si tu modelo LSTM tiene RMSE similar al Naive**: Tu modelo no está aprendiendo patrones útiles
2. **Si tu modelo LSTM supera al Naive en un 20-30%**: Tu modelo está capturando patrones reales
3. **Si tu modelo LSTM es mucho peor que Naive**: Hay overfitting o problemas en los datos

## Ejemplo de Resultados Esperados

```
TABLA COMPARATIVA DE MODELOS
================================================================================
                        Model       MSE      RMSE       MAE        R2      MAPE        DA
0           NaiveForecastModel  0.000012  0.003500  0.002800  0.456789    0.25     51.2
1                TLS_LSTMModel  0.000008  0.002800  0.002200  0.634521    0.19     56.8
2                   GRU_Model  0.000007  0.002650  0.002100  0.678912    0.18     58.4
```

En este ejemplo:
- Los modelos LSTM/GRU superan al Naive en todas las métricas
- La mejora en RMSE es de ~25-30%, indicando aprendizaje real
- La Directional Accuracy superior al 50% indica capacidad predictiva direccional

## Ventajas del Modelo Naive

1. **Simplicidad**: Fácil de implementar y entender
2. **Rapidez**: Predicciones instantáneas
3. **Robustez**: No sufre overfitting
4. **Baseline sólido**: En mercados eficientes, puede ser difícil de superar

## Limitaciones

1. **Sin aprendizaje**: No captura patrones o tendencias
2. **Sin adaptación**: No se ajusta a cambios en el mercado
3. **Predicción única**: Solo predice un horizonte temporal
4. **Sin información contextual**: Ignora indicadores técnicos

## Código del Modelo

```python
class NaiveForecastModel(nn.Module):
    def forward(self, x):
        # Tomar el último valor observado
        last_value = x[:, -1, 0]  # (batch_size,)
        return last_value.unsqueeze(-1)  # (batch_size, 1)
```

## Próximos Pasos

1. **Analiza los resultados**: ¿Tus modelos superan consistentemente al Naive?
2. **Investiga**: Si no lo superan, revisa la preparación de datos y arquitectura
3. **Optimiza**: Usa técnicas como grid search para mejorar hiperparámetros
4. **Valida**: Usa validación cruzada temporal para confirmar resultados
