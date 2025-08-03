# 📌 Predicción del EUR/USD con LSTM y Modelos Baseline

Este proyecto utiliza datos históricos del tipo de cambio **EUR/USD** para entrenar y comparar diferentes modelos de predicción:
- 🧠 **Redes neuronales LSTM** (Deep Learning)
- 📊 **Modelos baseline estadísticos** (Naive, ARIMA)
- 📈 **Análisis comparativo** con métricas estadísticas rigurosas

## ⚙️ Instalación y Configuración

### 1️⃣ Clonar el repositorio
```sh
git clone https://github.com/rony171998/LSTM-EUR-USD.git
cd tu_repositorio
```

### 2️⃣ Crear un entorno virtual
#### 🔹 En Windows (cmd/powershell):
```sh
python -m venv venv
venv\Scripts\activate
```

#### 🔹 En macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar dependencias
```sh
pip install -r requirements.txt
```

## 🧠 Entrenar el modelo LSTM
Para entrenar la red neuronal LSTM, ejecuta:
```sh
python model/train_model.py
```
El modelo entrenado se guardará como `TLS_LSTMModel_EUR_USD_2010-2024.csv.pth`.

## 📊 Modelos Baseline Estadísticos

### 🎯 Naive Baseline
El modelo Naive utiliza el último valor conocido como predicción. Es el benchmark mínimo que cualquier modelo debe superar:
```sh
python model/train_naive_baseline.py
```
**Resultados:** RMSE: 0.005025, R²: 0.976684, DA: 50.00%

### 📈 ARIMA Profesional
Modelo estadístico clásico con selección automática de parámetros y validación rigurosa:
```sh
python model/train_arima_professional.py
```

**🔧 Características del ARIMA:**
- ✅ Selección automática de parámetros (p,d,q) usando grid search
- ✅ Pruebas de estacionariedad (ADF, KPSS)
- ✅ Validación cruzada temporal con rolling windows
- ✅ Test de Diebold-Mariano para comparación estadística
- ✅ Análisis completo de residuos (normalidad, autocorrelación)
- ✅ 6 métricas de evaluación (RMSE, MAE, R², MAPE, DA, MSE)

**📊 Resultados ARIMA vs Naive:**
```
                Model      MSE     RMSE      MAE        R²     MAPE        DA
    NaiveForecastModel 0.000025 0.005025 0.003790  0.976684 0.357320 50.000000
            ARIMAModel 0.004957 0.070406 0.062241 -3.577081 5.933923 50.748503
```

**💡 Interpretación financiera:**
- El mercado EUR/USD sigue un patrón muy cercano a random walk
- El modelo Naive es superior (RMSE 14x mejor)
- ARIMA muestra ligera ventaja direccional (50.75% vs 50%)
- Los modelos ML deben superar significativamente a Naive para ser útiles

### 📋 Benchmark para modelos ML:
- **RMSE objetivo:** < 0.004 (mejor que Naive)
- **Directional Accuracy:** > 52% (superar el azar)
- **R² objetivo:** > 0.98 (explicar más varianza)

## Ejecuta el modelo:
```sh
python model/execute_model.py
```

## 🔬 Análisis y Comparación de Modelos

### 📊 Evaluación Estadística Completa
Cada modelo incluye:
- **Métricas de precisión:** RMSE, MAE, MSE
- **Métricas de ajuste:** R², MAPE
- **Precisión direccional:** Porcentaje de direcciones correctas
- **Tests estadísticos:** Diebold-Mariano, normalidad de residuos
- **Validación temporal:** Rolling windows para series temporales

### 🏆 Comparación de Rendimiento
| Modelo | RMSE | R² | DA (%) | Tiempo | Complejidad |
|--------|------|----|----|--------|-------------|
| **Naive** | 0.005025 | 0.9767 | 50.0 | < 1s | Muy baja |
| **ARIMA** | 0.070406 | -3.58 | 50.7 | ~7min | Media |
| **LSTM** | 0.0052* | 0.9749* | TBD | ~30min | Alta |

*Resultados del modelo LSTM pueden variar según configuración

## 🔍 Realizar predicciones
Una vez entrenado el modelo, puedes generar predicciones con:
```sh
python model/predecir_future.py
```
Esto graficará los valores reales vs. predichos.

## 📁 Estructura del Proyecto

```
LSTM-EUR-USD/
├── 📂 data/                    # Datos históricos EUR/USD
│   ├── EUR_USD_2010-2024.csv  # Dataset principal
│   └── best_params_*.json     # Parámetros optimizados
├── 📂 model/                   # Scripts de modelos
│   ├── train_model.py         # Entrenamiento LSTM
│   ├── train_naive_baseline.py    # Modelo Naive
│   ├── train_arima_professional.py # ARIMA estadístico
│   ├── execute_model.py       # Ejecución de modelos
│   └── predecir_future.py     # Predicciones futuras
├── 📂 docs/                    # 📚 Documentación completa
│   ├── README.md              # Índice de documentación
│   ├── README_ARIMA_Professional.md # ARIMA técnico
│   ├── README_ARIMA_Baseline.md     # ARIMA básico
│   └── README_Baseline_Models.md    # Comparación completa
├── 📂 modelos/                 # Modelos entrenados (.pth)
├── 📂 images/                  # Gráficas y análisis
│   ├── correlaciones/         # Análisis de correlaciones
│   ├── estadisticas/          # Estadísticas descriptivas
│   └── prediccion/            # Resultados de predicción
├── 📂 api/                     # API REST para predicciones
└── requirements.txt           # Dependencias del proyecto
```

## 📚 Documentación Completa

Para documentación técnica detallada, consulta la carpeta [`docs/`](docs/README.md):

- **[🎯 Naive Baseline](docs/README_Naive_Baseline.md)**: Modelo baseline más simple y efectivo
- **[📊 ARIMA Professional](docs/README_ARIMA_Professional.md)**: Implementación rigurosa con statsmodels
- **[📈 ARIMA Baseline](docs/README_ARIMA_Baseline.md)**: Modelo estadístico básico  
- **[🎯 Guía de Baselines](docs/README_Baseline_Models.md)**: Comparación completa de modelos
- **[📚 Índice General](docs/README.md)**: Navegación completa de documentación

## 🚀 ¡Listo!
Tu modelo LSTM ya está funcionando para predecir el EUR/USD. 🎯

---
### 🛠 Requisitos
- Python 3.8+
- `pip install -r requirements.txt`

### 📌 Dependencias principales
- `torch` → Framework PyTorch para redes neuronales LSTM
- `statsmodels` → Modelos estadísticos profesionales (ARIMA)
- `scipy` → Tests estadísticos avanzados
- `pandas` → Manejo y limpieza de datos financieros
- `numpy` → Manipulación de datos numéricos
- `scikit-learn` → Métricas de evaluación y normalización
- `matplotlib` → Visualización de predicciones y análisis
- `seaborn` → Gráficas estadísticas avanzadas

## 🔬 Metodología Científica

### 📊 Validación Temporal
- **Rolling Window Validation:** Simulación realista de trading
- **3,834 ventanas** de validación para robustez estadística
- **Prevención de data leakage** en series temporales

### 🧪 Tests Estadísticos
- **Diebold-Mariano:** Comparación significativa entre modelos
- **ADF/KPSS:** Pruebas de estacionariedad
- **Ljung-Box:** Autocorrelación de residuos
- **Jarque-Bera:** Normalidad de errores

### 📈 Insights Financieros
- **Mercado eficiente:** EUR/USD sigue random walk (H=0.5299)
- **Desafío realista:** Superar a Naive es más difícil que a ARIMA
- **Benchmark práctico:** DA > 52% para rentabilidad en trading

## 🎯 Conclusiones del Análisis

1. **🏆 Naive Baseline domina:** En mercados eficientes como EUR/USD
2. **📊 ARIMA limitado:** Mejor interpretabilidad pero menor precisión  
3. **🧠 Deep Learning:** Debe demostrar ventaja significativa (RMSE < 0.004)
4. **⚖️ Trade-off importante:** Complejidad vs mejora marginal

⚡ **¡A entrenar y comparar modelos!** 🚀

---

## 📚 Referencias Académicas

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy*
- Makridakis, S., et al. (1998). *Forecasting methods and applications*
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## � Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

