# 📚 Documentación del Proyecto LSTM-EUR-USD

Bienvenido a la documentación completa del proyecto de predicción EUR/USD con modelos de Machine Learning y análisis estadístico.

## 📋 Índice de Documentación

### 🏠 [README Principal](../readme.md)
Documentación principal del proyecto con guía de instalación, uso y comparación de modelos.

### 📊 Modelos Baseline

#### 🎯 [Naive Baseline](README_Naive_Baseline.md)
- **Descripción**: Modelo baseline más simple (último valor conocido)
- **Características**: Benchmark mínimo que cualquier modelo debe superar
- **Resultados**: RMSE 0.005025, R² 0.9767, DA 50.0%
- **Importancia**: Verdadero desafío en mercados eficientes

#### 📈 [ARIMA Professional](README_ARIMA_Professional.md)
- **Descripción**: Implementación rigurosa del modelo ARIMA con `statsmodels`
- **Características**: Grid search automático, validación rolling window, tests estadísticos
- **Resultados**: RMSE 0.01436, R² 98.36%, DA 50.75%
- **Análisis**: Comparación detallada vs Naive, interpretación financiera

#### 🎯 [Guía Completa de Baselines](README_Baseline_Models.md)
- **Descripción**: Comparación exhaustiva de todos los modelos baseline
- **Modelos**: Naive, ARIMA, comparación con LSTM
- **Benchmarks**: Objetivos para futuros modelos ML
- **Metodología**: Validación temporal y métricas estadísticas

## 🔬 Análisis Técnico

### 📊 Métricas Comparativas

| Modelo | RMSE | R² | DA (%) | Tiempo | Archivo |
|--------|------|----|----|--------|---------|
| **Naive** | 0.005025 | 0.9767 | 50.0 | < 1s | - |
| **ARIMA** | 0.070406 | -3.58 | 50.7 | ~7min | [🔗](README_ARIMA_Professional.md) |
| **LSTM** | 0.0052* | 0.9749* | TBD | ~30min | [🔗](../readme.md) |

*Resultados según imagen del usuario

### 🎯 Benchmarks para Nuevos Modelos

**Objetivos mínimos:**
- **RMSE**: < 0.004 (superar significativamente a Naive)
- **DA**: > 52% (utilidad comercial)
- **R²**: > 0.98 (mejor explicación de varianza)

## 🛠 Metodología

### 📅 Validación Temporal
- **Rolling Window**: 3,834 ventanas de validación
- **Ventana de entrenamiento**: 100 observaciones
- **Horizonte de predicción**: 10 pasos

### 🧪 Tests Estadísticos
- **Diebold-Mariano**: Comparación de precisión
- **ADF/KPSS**: Pruebas de estacionariedad
- **Ljung-Box**: Autocorrelación de residuos
- **Jarque-Bera**: Normalidad de errores

## 🎓 Insights Clave

### 💡 Hallazgos Principales

1. **Naive domina en FX**: EUR/USD sigue random walk muy cercano
2. **ARIMA limitado**: Mejor interpretabilidad pero menor precisión
3. **ML debe justificar complejidad**: Mejoras marginales no valen el costo
4. **Direccionalidad importa**: DA > 52% crítico para trading

### 📈 Implicaciones Financieras

- **Mercado eficiente**: Información incorporada rápidamente
- **Microestructura**: Spreads pequeños, alta liquidez
- **Predicción**: Desafío técnico considerable
- **Benchmark realista**: Naive es el verdadero adversario

## 🚀 Uso de la Documentación

### 📖 Para Desarrolladores
1. Leer [README principal](../readme.md) para configuración
2. Revisar [Guía de Baselines](README_Baseline_Models.md) para contexto
3. Consultar [ARIMA Professional](README_ARIMA_Professional.md) para detalles técnicos

### 🔬 Para Investigadores
1. Estudiar metodología de validación temporal
2. Analizar tests estadísticos implementados
3. Comprender benchmarks para modelos ML

### 💼 Para Analistas Financieros
1. Entender por qué Naive supera a modelos complejos
2. Interpretar métricas en contexto de trading
3. Evaluar viabilidad comercial de modelos

## 📁 Estructura de Archivos

```
docs/
├── README.md                     # Este índice
├── README_Naive_Baseline.md      # Modelo Naive simple
├── README_ARIMA_Professional.md  # ARIMA técnico completo
├── README_ARIMA_Baseline.md      # ARIMA básico
└── README_Baseline_Models.md     # Comparación completa
```

## 🔄 Actualizaciones

- **Última actualización**: Agosto 2, 2025
- **Versión de modelos**: ARIMA Professional v1.0
- **Estado**: Documentación completa y organizada

---

**📚 Documentación completa para análisis riguroso de series temporales financieras**
