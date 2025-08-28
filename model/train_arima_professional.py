#!/usr/bin/env python3
#train_arima_professional.py
"""
ARIMA Professional Implementation using statsmodels
Entrenamiento de modelo ARIMA con statsmodels para predicción EUR/USD
Este script implementa un modelo ARIMA profesional con:
- Selección automática de parámetros con grid search
- Pruebas de estacionariedad (ADF, KPSS)
- Validación cruzada temporal (rolling window)
- Test de Diebold-Mariano para comparación con Naive
- Análisis completo de residuos
- Métricas estadísticas completas
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Configuración para suprimir warnings molestos
warnings.filterwarnings('ignore')

# Imports estadísticos
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

# Imports de métricas y utilidades
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import itertools
from config import DEFAULT_PARAMS
# Añadir directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuraciones por defecto
SEQUENCE_LENGTH = DEFAULT_PARAMS.SEQ_LENGTH
BATCH_SIZE = DEFAULT_PARAMS.BATCH_SIZE
LEARNING_RATE = DEFAULT_PARAMS.LEARNING_RATE
EPOCHS = DEFAULT_PARAMS.EPOCHS
FILE_PATH = f"data/{DEFAULT_PARAMS.FILEPATH}"

def print_section(title):
    """Imprime una sección con formato"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print('='*80)

def load_data():
    """Carga los datos EUR/USD"""
    print(f"📊 Cargando datos {DEFAULT_PARAMS.TICKER}...")
    
    # Obtener el directorio del proyecto (un nivel arriba de model/)
    project_dir = Path(__file__).parent.parent
    data_path = project_dir / FILE_PATH
    
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    
    # Cargar CSV con configuración robusta para diferentes formatos
    df = pd.read_csv(
        data_path,
        index_col="Fecha",
        parse_dates=True,
        dayfirst=True,
        decimal=",",
        thousands=".",
        converters={
            "Último": lambda x: convert_to_float(x) if x else np.nan,
            "Apertura": lambda x: convert_to_float(x) if x else np.nan,
            "Máximo": lambda x: convert_to_float(x) if x else np.nan,
            "Mínimo": lambda x: convert_to_float(x) if x else np.nan,
            "% var.": lambda x: convert_to_float(str(x).replace("%", "")) if x else np.nan
        }
    )
    
    # Verificar columnas necesarias
    required_columns = ['Último']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el dataset")
    
    # Ordenar por fecha ascendente
    df = df.sort_index()
    
    print(f"✅ Datos cargados: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
    print(f"📈 Precio {DEFAULT_PARAMS.TICKER} - Min: {df['Último'].min():.6f}, Max: {df['Último'].max():.6f}")
    
    return df

def convert_to_float(value_str):
    """
    Convierte string a float manejando diferentes formatos numéricos:
    - '1.234,56' (formato europeo: punto miles, coma decimal)
    - '4.130.50' (formato con doble punto)
    - '1,234.56' (formato americano: coma miles, punto decimal)
    - '1234.56' (formato simple)
    """
    if pd.isna(value_str) or value_str == '' or value_str is None:
        return np.nan
    
    # Convertir a string por si acaso
    value_str = str(value_str).strip()
    
    try:
        # Caso 1: Formato simple sin separadores especiales
        if ',' not in value_str and value_str.count('.') <= 1:
            return float(value_str)
        
        # Caso 2: Formato europeo estándar (1.234,56)
        if ',' in value_str and value_str.rfind(',') > value_str.rfind('.'):
            # La coma está después del último punto = formato europeo
            # Remover puntos (miles) y cambiar coma por punto (decimal)
            cleaned = value_str.replace('.', '').replace(',', '.')
            return float(cleaned)
        
        # Caso 3: Formato americano estándar (1,234.56)
        elif '.' in value_str and value_str.rfind('.') > value_str.rfind(','):
            # El punto está después de la última coma = formato americano
            # Remover comas (miles)
            cleaned = value_str.replace(',', '')
            return float(cleaned)
        
        # Caso 4: Formato problemático como '4.130.50'
        elif value_str.count('.') > 1:
            # Múltiples puntos - asumir que el último es decimal
            parts = value_str.split('.')
            if len(parts) >= 2:
                # Unir todas las partes excepto la última (miles)
                # La última parte es decimal
                integer_part = ''.join(parts[:-1])
                decimal_part = parts[-1]
                cleaned = f"{integer_part}.{decimal_part}"
                return float(cleaned)
        
        # Caso 5: Solo comas (formato '1,234')
        elif ',' in value_str and '.' not in value_str:
            # Si hay 3 dígitos después de la última coma, probablemente es miles
            # Si hay 1-2 dígitos, probablemente es decimal
            parts = value_str.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Formato decimal con coma
                cleaned = value_str.replace(',', '.')
                return float(cleaned)
            else:
                # Formato miles con coma
                cleaned = value_str.replace(',', '')
                return float(cleaned)
        
        # Último recurso: intentar conversión directa
        return float(value_str)
        
    except (ValueError, AttributeError) as e:
        print(f"⚠️ No se pudo convertir '{value_str}' a float: {e}")
        return np.nan

def test_stationarity(series, series_name="Serie"):
    """Pruebas de estacionariedad (ADF y KPSS)"""
    print(f"\n🔍 Pruebas de Estacionariedad - {series_name}")
    print("-" * 60)
    
    # Test ADF (Augmented Dickey-Fuller)
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        print(f"Test ADF:")
        print(f"  • Estadístico ADF: {adf_result[0]:.6f}")
        print(f"  • p-value: {adf_result[1]:.6f}")
        print(f"  • Valores críticos: {dict(adf_result[4])}")
        adf_stationary = adf_result[1] <= 0.05
        print(f"  • Resultado: {'✅ ESTACIONARIA' if adf_stationary else '❌ NO ESTACIONARIA'} (p-value ≤ 0.05)")
    except Exception as e:
        print(f"  • Error en ADF: {e}")
        adf_stationary = False
    
    # Test KPSS
    try:
        kpss_result = kpss(series.dropna(), regression='c')
        print(f"\nTest KPSS:")
        print(f"  • Estadístico KPSS: {kpss_result[0]:.6f}")
        print(f"  • p-value: {kpss_result[1]:.6f}")
        print(f"  • Valores críticos: {dict(kpss_result[3])}")
        kpss_stationary = kpss_result[1] > 0.05
        print(f"  • Resultado: {'✅ ESTACIONARIA' if kpss_stationary else '❌ NO ESTACIONARIA'} (p-value > 0.05)")
    except Exception as e:
        print(f"  • Error en KPSS: {e}")
        kpss_stationary = False
    
    # Conclusión
    both_agree = adf_stationary and kpss_stationary
    print(f"\n📊 Conclusión: {'✅ SERIE ESTACIONARIA' if both_agree else '⚠️ SERIE REQUIERE DIFERENCIACIÓN'}")
    
    return adf_stationary, kpss_stationary

def find_best_arima_params(series, max_p=3, max_d=2, max_q=3):
    """Encuentra los mejores parámetros ARIMA usando grid search"""
    print(f"\n🔍 Búsqueda de mejores parámetros ARIMA (p≤{max_p}, d≤{max_d}, q≤{max_q})...")
    
    # Generar todas las combinaciones de parámetros
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    
    best_aic = np.inf
    best_params = None
    best_model = None
    results = []
    
    total_combinations = len(p_values) * len(d_values) * len(q_values)
    current = 0
    
    print(f"🔄 Evaluando {total_combinations} combinaciones...")
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        current += 1
        try:
            model = ARIMA(series, order=(p, d, q))
            fitted_model = model.fit()
            aic = fitted_model.aic
            
            results.append({
                'p': p, 'd': d, 'q': q,
                'AIC': aic,
                'BIC': fitted_model.bic,
                'HQIC': fitted_model.hqic,
                'status': '✅'
            })
            
            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)
                best_model = fitted_model
                print(f"  [{current:2d}/{total_combinations}] Nuevo mejor: ARIMA({p},{d},{q}) - AIC: {aic:.4f}")
            
        except Exception as e:
            results.append({
                'p': p, 'd': d, 'q': q,
                'AIC': np.inf,
                'BIC': np.inf,
                'HQIC': np.inf,
                'status': f'❌ {str(e)[:30]}'
            })
    
    # Mostrar top 5 modelos
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['AIC'] != np.inf].sort_values('AIC').head(5)
    
    print(f"\n🏆 Top 5 modelos ARIMA:")
    print("-" * 60)
    for idx, row in valid_results.iterrows():
        print(f"  {row['status']} ARIMA({row['p']},{row['d']},{row['q']}) - AIC: {row['AIC']:.4f}, BIC: {row['BIC']:.4f}")
    
    print(f"\n✅ Mejor modelo: ARIMA{best_params} con AIC: {best_aic:.4f}")
    
    return best_params, best_model, results_df

def analyze_residuals(residuals, model_name="ARIMA"):
    """Análisis completo de residuos"""
    print(f"\n🔬 Análisis de Residuos - {model_name}")
    print("-" * 60)
    
    residuals_clean = residuals.dropna()
    
    # 1. Estadísticas básicas
    print(f"📊 Estadísticas de Residuos:")
    print(f"  • Media: {residuals_clean.mean():.8f}")
    print(f"  • Desviación estándar: {residuals_clean.std():.8f}")
    print(f"  • Mínimo: {residuals_clean.min():.8f}")
    print(f"  • Máximo: {residuals_clean.max():.8f}")
    print(f"  • Asimetría: {residuals_clean.skew():.4f}")
    print(f"  • Curtosis: {residuals_clean.kurtosis():.4f}")
    
    # 2. Test de normalidad (Jarque-Bera)
    try:
        jb_stat, jb_pvalue = jarque_bera(residuals_clean)
        print(f"\n🔍 Test de Normalidad (Jarque-Bera):")
        print(f"  • Estadístico: {jb_stat:.4f}")
        print(f"  • p-value: {jb_pvalue:.6f}")
        print(f"  • Resultado: {'✅ NORMAL' if jb_pvalue > 0.05 else '❌ NO NORMAL'} (p-value > 0.05)")
    except Exception as e:
        print(f"  • Error en Jarque-Bera: {e}")
    
    # 3. Test de autocorrelación (Ljung-Box)
    try:
        lb_result = acorr_ljungbox(residuals_clean, lags=10, return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[-1]  # Último lag
        print(f"\n🔍 Test de Autocorrelación (Ljung-Box):")
        print(f"  • p-value (lag 10): {lb_pvalue:.6f}")
        print(f"  • Resultado: {'✅ NO AUTOCORRELACIÓN' if lb_pvalue > 0.05 else '❌ AUTOCORRELACIÓN PRESENTE'} (p-value > 0.05)")
    except Exception as e:
        print(f"  • Error en Ljung-Box: {e}")

def calculate_comprehensive_metrics(y_true, y_pred, model_name="Model"):
    """Calcula métricas completas de evaluación"""
    print(f"\n📈 Métricas de Evaluación - {model_name}")
    print("-" * 60)
    
    # Métricas básicas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'MSE': mse
    }
    
    print(f"  • RMSE: {rmse:.8f}")
    print(f"  • MAE: {mae:.8f}")
    print(f"  • R² Score: {r2:.6f}")
    print(f"  • MAPE: {mape:.4f}%")
    print(f"  • Precisión Direccional: {directional_accuracy:.2f}%")
    
    return metrics

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Test de Diebold-Mariano para comparar la precisión de dos modelos
    H0: Los modelos tienen la misma precisión de predicción
    H1: Los modelos tienen diferente precisión de predicción
    """
    print(f"\n🔬 Test de Diebold-Mariano")
    print("-" * 60)
    
    # Calcular diferencias de errores cuadráticos
    d = errors1**2 - errors2**2
    
    # Estadísticas básicas
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)
    
    # Estadístico DM
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # p-value (distribución t con n-1 grados de libertad)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))
    
    print(f"  • Estadístico DM: {dm_stat:.6f}")
    print(f"  • p-value: {p_value:.6f}")
    print(f"  • Diferencia media: {d_mean:.8f}")
    
    if p_value < 0.05:
        if dm_stat > 0:
            conclusion = "❌ Modelo 1 es significativamente PEOR que Modelo 2"
        else:
            conclusion = "✅ Modelo 1 es significativamente MEJOR que Modelo 2"
    else:
        conclusion = "⚖️ No hay diferencia significativa entre los modelos"
    
    print(f"  • Conclusión: {conclusion}")
    
    return dm_stat, p_value

def rolling_window_validation(data, model_params, window_size=100, forecast_horizon=10):
    """Validación con ventana deslizante"""
    print(f"\n🔄 Validación Rolling Window")
    print(f"  • Tamaño de ventana: {window_size}")
    print(f"  • Horizonte de predicción: {forecast_horizon}")
    print("-" * 60)
    
    series = data['Último'].values
    predictions = []
    actuals = []
    start_time = time.time()
    
    n_windows = len(series) - window_size - forecast_horizon + 1
    print(f"🔄 Ejecutando {n_windows} ventanas de validación...")
    
    for i in range(n_windows):
        if i % 20 == 0:
            elapsed = time.time() - start_time
            progress = (i / n_windows) * 100
            print(f"  Progreso: {progress:.1f}% ({i}/{n_windows}) - Tiempo: {elapsed:.1f}s")
        
        # Ventana de entrenamiento
        train_data = series[i:i + window_size]
        
        # Datos de prueba
        test_data = series[i + window_size:i + window_size + forecast_horizon]
        
        try:
            # Entrenar modelo ARIMA
            model = ARIMA(train_data, order=model_params)
            fitted_model = model.fit()
            
            # Predicir
            forecast = fitted_model.forecast(steps=forecast_horizon)
            
            predictions.extend(forecast)
            actuals.extend(test_data)
            
        except Exception as e:
            # En caso de error, usar última observación (naive)
            last_value = train_data[-1]
            forecast = [last_value] * forecast_horizon
            predictions.extend(forecast)
            actuals.extend(test_data)
    
    total_time = time.time() - start_time
    print(f"✅ Validación completada en {total_time:.2f} segundos")
    
    return np.array(actuals), np.array(predictions)

def create_visualizations(data, train_size, predictions, actuals, model_name="ARIMA"):
    """Crear visualizaciones del modelo"""
    print(f"\n📊 Creando visualizaciones...")
    
    # Configurar el estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Análisis del Modelo {model_name} - {DEFAULT_PARAMS.TICKER}', fontsize=16, fontweight='bold')
    
    # 1. Predicción vs Realidad
    ax1 = axes[0, 0]
    test_indices = range(len(predictions))
    ax1.plot(test_indices, actuals, label='Valores Reales', color='blue', alpha=0.7)
    ax1.plot(test_indices, predictions, label='Predicciones', color='red', alpha=0.7)
    ax1.set_title('Predicciones vs Valores Reales')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel(f'Precio {DEFAULT_PARAMS.TICKER}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(actuals, predictions, alpha=0.6, color='green')
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax2.set_xlabel('Valores Reales')
    ax2.set_ylabel('Predicciones')
    ax2.set_title('Dispersión: Predicciones vs Reales')
    ax2.grid(True, alpha=0.3)
    
    # 3. Errores
    ax3 = axes[1, 0]
    errors = predictions - actuals
    ax3.plot(errors, color='purple', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Errores de Predicción')
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('Error')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución de errores
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_title('Distribución de Errores')
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frecuencia')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfica
    project_dir = Path(__file__).parent.parent
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)
    filename = images_dir / f"{DEFAULT_PARAMS.TABLENAME}/arima_professional_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfica guardada: {filename}")
    
    plt.show()

def main():
    """Función principal"""
    print_section("MODELO ARIMA PROFESIONAL CON STATSMODELS")
    print(f"🎯 Implementación profesional de ARIMA para {DEFAULT_PARAMS.TICKER}")
    print("📊 Incluye: Grid search, validación rolling, test Diebold-Mariano")
    
    start_time = time.time()
    
    try:
        # 1. Cargar datos
        print_section("1. CARGA Y PREPARACIÓN DE DATOS")
        data = load_data()
        
        # 2. Análisis de estacionariedad
        print_section("2. ANÁLISIS DE ESTACIONARIEDAD")
        series = data['Último']
        adf_stationary, kpss_stationary = test_stationarity(series, f"{DEFAULT_PARAMS.TICKER} Último")
        
        # 3. Búsqueda de mejores parámetros
        print_section("3. SELECCIÓN DE PARÁMETROS ARIMA")
        best_params, best_model, search_results = find_best_arima_params(series)
        
        # 4. Análisis de residuos del mejor modelo
        print_section("4. ANÁLISIS DE RESIDUOS")
        residuals = best_model.resid
        analyze_residuals(residuals, f"ARIMA{best_params}")
        
        # 5. Validación rolling window
        print_section("5. VALIDACIÓN ROLLING WINDOW")
        actuals, predictions = rolling_window_validation(data, best_params)
        
        # 6. Métricas completas
        print_section("6. EVALUACIÓN DE RENDIMIENTO")
        arima_metrics = calculate_comprehensive_metrics(actuals, predictions, f"ARIMA{best_params}")
        
        # 7. Comparación con Naive (cargar resultados del Naive)
        print_section("7. COMPARACIÓN CON MODELO NAIVE")
        try:
            # Implementar naive simple para comparación
            naive_predictions = []
            for i in range(len(actuals)):
                if i == 0:
                    naive_predictions.append(actuals[0])  # Primera predicción
                else:
                    naive_predictions.append(actuals[i-1])  # Usar valor anterior
            
            naive_predictions = np.array(naive_predictions)
            naive_metrics = calculate_comprehensive_metrics(actuals, naive_predictions, "Naive Baseline")
            
            # Test de Diebold-Mariano
            arima_errors = predictions - actuals
            naive_errors = naive_predictions - actuals
            dm_stat, dm_pvalue = diebold_mariano_test(arima_errors, naive_errors)
            
        except Exception as e:
            print(f"⚠️ No se pudo comparar con Naive: {e}")
        
        # 8. Visualizaciones
        print_section("8. VISUALIZACIONES")
        train_size = len(data) - len(predictions)
        create_visualizations(data, train_size, predictions, actuals, f"ARIMA{best_params}")
        
        # 9. Resumen final
        print_section("9. RESUMEN FINAL")
        total_time = time.time() - start_time
        
        print(f"🏆 Mejor modelo ARIMA: {best_params}")
        print(f"📊 RMSE: {arima_metrics['RMSE']:.8f}")
        print(f"📊 R² Score: {arima_metrics['R²']:.6f}")
        print(f"📊 MAPE: {arima_metrics['MAPE']:.4f}%")
        print(f"📊 Precisión Direccional: {arima_metrics['Directional_Accuracy']:.2f}%")
        print(f"⏱️ Tiempo total de ejecución: {total_time:.2f} segundos")
        
        print(f"\n✅ Entrenamiento ARIMA profesional completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
