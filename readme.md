# 📌 Predicción del EUR/USD con LSTM

Este proyecto utiliza datos históricos del tipo de cambio **EUR/USD** para entrenar una red neuronal **LSTM** que predice sus variaciones.

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
python train_model.py
```
El modelo entrenado se guardará como `TLS_LSTMModel_EUR_USD_2010-2024.csv.pth`.

## Ejecuta el modelo:
```sh
Una Ejecuta el script execute_model.py para ver el modelo LSTM:
```sh
python execute_model.py
```

## 🔍 Realizar predicciones
Una vez entrenado el modelo, puedes generar predicciones con:
```sh
python predecir_future.py
```
Esto graficará los valores reales vs. predichos.

## 🚀 ¡Listo!
Tu modelo LSTM ya está funcionando para predecir el EUR/USD. 🎯

---
### 🛠 Requisitos
- Python 3.8+
- `pip install -r requirements.txt`

### 📌 Dependencias principales
- `yfinance` → Para descargar datos históricos.
- `pandas` → Manejo y limpieza de datos.
- `numpy` → Manipulación de datos numéricos.
- `tensorflow` → Construcción y entrenamiento de la red LSTM.
- `scikit-learn` → Normalización de datos.
- `matplotlib` → Visualización de predicciones.

⚡ **¡A entrenar y predecir!** 🚀

