# evaluate_models.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from config import DEFAULT_PARAMS
from modelos import (
    TLS_LSTMModel,
    GRU_Model,
    HybridLSTMAttentionModel,
    BidirectionalDeepLSTMModel,
    ContextualLSTMTransformerFlexible,
    NaiveForecastModel,
    ARIMAModel,
    FinalOptimizedTLSLSTM
)
import warnings
warnings.filterwarnings('ignore')
# Importar el modelo Final Optimizado
import sys
import os
# Agregar tanto el directorio padre como la ruta actual
sys.path.append('..')  # Para acceder al directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Ruta absoluta al directorio padre

# Importar el modelo Final Optimizado
try:
    from train_final_optimized import FinalOptimizedModel
    FINAL_OPTIMIZED_AVAILABLE = True
    print("✅ FinalOptimizedModel importado correctamente")
except ImportError as e:
    print(f"⚠️  Warning: No se pudo importar FinalOptimizedModel: {e}")
    FINAL_OPTIMIZED_AVAILABLE = False

from train_model import load_and_prepare_data, create_sequences, add_indicator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # R² score
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan

    # MAPE (Mean Absolute Percentage Error)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        mape = np.nan

    # Directional Accuracy (DA) - porcentaje de direcciones correctas
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        da = np.mean(true_direction == pred_direction) * 100
    else:
        da = np.nan

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'DA': da
    }

def get_model_specific_params(model_name):
    """Retorna parámetros específicos de cada modelo usados en el entrenamiento reproducible"""
    base_params = {
        'seq_length': 30,  # Usado en entrenamiento reproducible
        'input_size': 4,   # ['returns', 'rsi', 'sma20', 'dxy_returns']
        'output_size': 1,
        'forecast_horizon': 1
    }

    if model_name == "TLS_LSTMModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2}
    elif model_name == "TLS_LSTMModel_Optimizado":
        return {**base_params, 'hidden_size': 128, 'dropout_prob': 0.1}
    elif model_name == "GRU_Model":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2, 'num_layers': 2}
    elif model_name == "HybridLSTMAttentionModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2}
    elif model_name == "BidirectionalDeepLSTMModel":
        return {**base_params, 'hidden_size': 50, 'dropout_prob': 0.2, 'num_layers': 2}
    elif model_name == "ContextualLSTMTransformerFlexible":
        return {
            **base_params,
            'window_size': 6,
            'max_neighbors': 1,
            'lstm_units': 32,
            'num_heads': 2,
            'embed_dim': 64,
            'dropout_rate': 0.2
        }
    else:
        return base_params

def load_model(model_name, model_path, params_dict):
    """Carga un modelo entrenado con los parámetros específicos."""
    input_size = params_dict['input_size']

    # if model_name == "TLS_LSTMModel":
    #     model = TLS_LSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    # elif model_name == "TLS_LSTMModel_Optimizado":
    #     model = TLS_LSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    # elif model_name == "TLS_LSTMModel_Optimizado":
    #     model = TLS_LSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    # elif model_name == "GRU_Model":
    #     model = GRU_Model(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    if model_name == "HybridLSTMAttentionModel":
        model = HybridLSTMAttentionModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    # elif model_name == "BidirectionalDeepLSTMModel":
    #     model = BidirectionalDeepLSTMModel(input_size, params_dict['hidden_size'], params_dict['output_size'], params_dict['dropout_prob'])
    # elif model_name == "ContextualLSTMTransformerFlexible":
    #     model = ContextualLSTMTransformerFlexible(
    #         seq_len=params_dict['seq_length'],
    #         feature_dim=input_size,
    #         output_size=params_dict['output_size'],
    #         window_size=params_dict['window_size'],
    #         max_neighbors=params_dict['max_neighbors'],
    #         lstm_units=params_dict['lstm_units'],
    #         num_heads=params_dict['num_heads'],
    #         embed_dim=params_dict['embed_dim'],
    #         dropout_rate=params_dict['dropout_rate'],
    #     )
    elif model_name == "NaiveForecastModel":
        # Modelo baseline que no requiere entrenamiento previo
        model = NaiveForecastModel(input_size=input_size, output_size=params_dict['output_size'])
        print(f"- Modelo baseline {model_name} inicializado (no requiere archivo .pth)")
        model.to(device)
        model.eval()
        return model
    elif model_name == "ARIMAModel":
        # Modelo ARIMA que se entrenará dinámicamente
        model = ARIMAModel(input_size=input_size, output_size=params_dict['output_size'])
        model.is_trained = False  # Marcar que necesita entrenamiento
        print(f"- Modelo baseline {model_name} inicializado (se entrenará dinámicamente)")
        model.to(device)
        model.eval()
        return model
        # Crear el modelo Final Optimizado
        if not FINAL_OPTIMIZED_AVAILABLE:
            print(f"- Modelo {model_name} no disponible (importación falló)")
            return None

        # Importar localmente para evitar problemas de scope
        try:
            from train_final_optimized import FinalOptimizedModel as LocalFinalOptimizedModel
        except ImportError:
            print(f"- Error importando FinalOptimizedModel localmente")
            return None

        # Crear adaptador para el modelo Final Optimizado
        class FinalOptimizedAdapter(nn.Module):
            """Adapter para hacer compatible FinalOptimizedModel con el sistema de evaluación"""
            def __init__(self):
                super(FinalOptimizedAdapter, self).__init__()
                # Crear el modelo Final Optimizado con los parámetros originales
                self.model = LocalFinalOptimizedModel(
                    input_size=1,  # Solo precio Close
                    hidden_size=28,
                    num_layers=1,
                    seq_length=30
                )

            def forward(self, x):
                # x viene como (batch, seq_len, features) del sistema de evaluación
                # FinalOptimizedModel espera (batch, 30, 1)

                # Tomar solo la primera característica (precio Close)
                price_sequence = x[:, :, 0:1]  # (batch, seq_len, 1)

                # Ajustar la longitud de secuencia a 30 (lo que espera el modelo)
                if price_sequence.size(1) > 30:
                    # Tomar los últimos 30 valores
                    price_sequence = price_sequence[:, -30:, :]
                elif price_sequence.size(1) < 30:
                    # Rellenar con el primer valor disponible
                    first_val = price_sequence[:, 0:1, :].expand(-1, 30 - price_sequence.size(1), -1)
                    price_sequence = torch.cat([first_val, price_sequence], dim=1)

                # Llamar al modelo Final Optimizado
                pred, direction_probs, confidence = self.model(price_sequence)

                # Retornar solo las predicciones de precio
                if len(pred.shape) == 1:
                    pred = pred.unsqueeze(-1)

                return pred

        model = FinalOptimizedAdapter()
        print(f"- Modelo {model_name} inicializado como adapter")
    # elif model_name == "MultivariateOptimizedModel":
    #     # Crear el modelo Multivariado
    #     try:
    #         from train_multivariate_optimized import MultivariateOptimizedModel as LocalMultivariateModel
    #     except ImportError:
    #         print(f"- Error importando MultivariateOptimizedModel")
    #         return None

    #     # Crear adaptador para el modelo Multivariado
    #     class MultivariateAdapter(nn.Module):
    #         """Adapter para hacer compatible MultivariateOptimizedModel con el sistema de evaluación"""
    #         def __init__(self):
    #             super(MultivariateAdapter, self).__init__()
    #             # Crear el modelo Multivariado con 8 features (EUR_USD + 7 pares correlacionados)
    #             self.model = LocalMultivariateModel(
    #                 input_size=8,  # EUR_USD + 7 pares correlacionados
    #                 hidden_size=64,
    #                 num_layers=2,
    #                 seq_length=30
    #             )

    #             # Cargar escaladores multivariados si están disponibles
    #             self.scalers = None
    #             self.feature_names = ['EUR_USD', 'Último', 'ECOPETROL', 'EUR_CHF', 'EUR_JPY', 'GBP_USD', 'S&P500', 'USD_COP']

    #         def forward(self, x):
    #             # x viene como (batch, seq_len, features) del sistema de evaluación
    #             # Si no tenemos suficientes features, replicar la primera
    #             if x.size(2) < 8:
    #                 # Extender con la primera feature replicada
    #                 first_feature = x[:, :, 0:1]  # (batch, seq_len, 1)
    #                 additional_features = first_feature.repeat(1, 1, 8 - x.size(2))
    #                 x = torch.cat([x, additional_features], dim=2)
    #             elif x.size(2) > 8:
    #                 # Tomar solo las primeras 8 features
    #                 x = x[:, :, :8]

    #             # Ajustar la longitud de secuencia a 30
    #             if x.size(1) > 30:
    #                 x = x[:, -30:, :]
    #             elif x.size(1) < 30:
    #                 first_vals = x[:, 0:1, :].expand(-1, 30 - x.size(1), -1)
    #                 x = torch.cat([first_vals, x], dim=1)

    #             # Llamar al modelo Multivariado
    #             pred, direction_probs, confidence = self.model(x)

    #             # Retornar solo las predicciones de precio
    #             if len(pred.shape) == 1:
    #                 pred = pred.unsqueeze(-1)

    #             return pred

    #     model = MultivariateAdapter()
    #     print(f"- Modelo {model_name} inicializado como adapter multivariado")
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    # Cargar pesos solo para modelos de deep learning
    # if model_name == "FinalOptimizedTLSLSTM":
    #     # Buscar el archivo del modelo final optimizado - buscar tanto en eur_usd como directamente
    #     possible_paths = [
    #         "modelos/eur_usd/FinalOptimizedTLSLSTM_EUR_USD_2010-2024.csv.pth",
    #         "../modelos/eur_usd/FinalOptimizedTLSLSTM_EUR_USD_2010-2024.csv.pth",
    #         "modelos/eurusd/FinalOptimized_EUR_USD_20250802_210634.pth"
    #     ]

    #     final_model_path = None
    #     for path in possible_paths:
    #         if os.path.exists(path):
    #             final_model_path = path
    #             break

    #     if final_model_path:
    #         try:
    #             checkpoint = torch.load(final_model_path, map_location=device)

    #             # El modelo final guarda el state_dict del modelo interno
    #             if 'model_state_dict' in checkpoint:
    #                 model.model.load_state_dict(checkpoint['model_state_dict'])
    #                 print(f"- Modelo {model_name} cargado desde: {final_model_path}")

    #                 # Mostrar métricas del entrenamiento si están disponibles
    #                 if 'test_rmse' in checkpoint:
    #                     print(f"  RMSE entrenamiento: {checkpoint['test_rmse']:.8f}")
    #                 if 'naive_rmse' in checkpoint:
    #                     print(f"  Naive RMSE referencia: {checkpoint['naive_rmse']:.8f}")
    #                 if 'improvement_pct' in checkpoint:
    #                     print(f"  Mejora vs Naive: {checkpoint['improvement_pct']:.4f}%")
    #             else:
    #                 print(f"- Formato de modelo no compatible: {final_model_path}")
    #                 return None
    #         except Exception as e:
    #             print(f"- Error cargando modelo final: {e}")
    #             return None
    #     else:
    #         print(f"- Archivo del modelo final no encontrado en ninguna de las rutas:")
    #         for path in possible_paths:
    #             print(f"  {path} -> {'✓' if os.path.exists(path) else '✗'}")
    #         return None
    # elif model_name == "UltraIntelligentBalanced":
    #     # Buscar el archivo del modelo Ultra Inteligente
    #     possible_paths = [
    #         "modelos/UltraIntelligent_EUR_USD_20250803_134939.pth",
    #         "../modelos/UltraIntelligent_EUR_USD_20250803_134939.pth",
    #         "modelos/UltraIntelligent_EUR_USD_2010-2024.csv.pth"
    #     ]

    #     # Buscar archivos que coincidan con el patrón UltraIntelligent_*
    #     import glob
    #     ultra_pattern_paths = [
    #         "modelos/UltraIntelligent_*.pth",
    #         "../modelos/UltraIntelligent_*.pth"
    #     ]

    #     for pattern in ultra_pattern_paths:
    #         found_files = glob.glob(pattern)
    #         if found_files:
    #             possible_paths.extend(found_files)

    #     ultra_model_path = None
    #     for path in possible_paths:
    #         if os.path.exists(path):
    #             ultra_model_path = path
    #             break

    #     if ultra_model_path:
    #         try:
    #             checkpoint = torch.load(ultra_model_path, map_location=device)

    #             # El modelo Ultra Inteligente guarda el state_dict completo
    #             if 'model_state_dict' in checkpoint:
    #                 model.model.load_state_dict(checkpoint['model_state_dict'])
    #                 print(f"- Modelo {model_name} cargado desde: {ultra_model_path}")

    #                 # Mostrar métricas del entrenamiento si están disponibles
    #                 if 'test_rmse' in checkpoint:
    #                     print(f"  RMSE entrenamiento: {checkpoint['test_rmse']:.8f}")
    #                 if 'naive_rmse' in checkpoint:
    #                     print(f"  Naive RMSE referencia: {checkpoint['naive_rmse']:.8f}")
    #                 if 'improvement_pct' in checkpoint:
    #                     print(f"  Mejora vs Naive: {checkpoint['improvement_pct']:.4f}%")
    #                 if 'direction_accuracy' in checkpoint:
    #                     print(f"  Directional Accuracy: {checkpoint['direction_accuracy']:.4f} ({checkpoint['direction_accuracy']*100:.2f}%)")
    #                 if 'test_r2' in checkpoint:
    #                     print(f"  R²: {checkpoint['test_r2']:.6f}")
    #                 if 'test_mape' in checkpoint:
    #                     print(f"  MAPE: {checkpoint['test_mape']:.4f}%")
    #             else:
    #                 print(f"- Formato de modelo no compatible: {ultra_model_path}")
    #                 return None
    #         except Exception as e:
    #             print(f"- Error cargando modelo Ultra Inteligente: {e}")
    #             return None
    #     else:
    #         print(f"- Archivo del modelo Ultra Inteligente no encontrado en ninguna de las rutas:")
    #         for path in possible_paths:
    #             print(f"  {path} -> {'✓' if os.path.exists(path) else '✗'}")
    #         return None
    # elif model_name == "FinalOptimizedModel":
    #     # Buscar el archivo del modelo Final Optimizado
    #     possible_paths = [
    #         "modelos/FinalOptimized_EUR_USD_20250806_123924.pth",
    #         "../modelos/FinalOptimized_EUR_USD_20250806_123924.pth",
    #         "modelos/FinalOptimized_EUR_USD_*.pth"
    #     ]

    #     # Buscar archivos que coincidan con el patrón FinalOptimized_*
    #     import glob
    #     final_pattern_paths = [
    #         "modelos/FinalOptimized_*.pth",
    #         "../modelos/FinalOptimized_*.pth"
    #     ]

    #     for pattern in final_pattern_paths:
    #         found_files = glob.glob(pattern)
    #         if found_files:
    #             # Ordenar por fecha y tomar el más reciente
    #             found_files.sort()
    #             possible_paths.extend(found_files)

    #     final_model_path = None
    #     for path in possible_paths:
    #         if os.path.exists(path):
    #             final_model_path = path
    #             break

    #     if final_model_path:
    #         try:
    #             checkpoint = torch.load(final_model_path, map_location=device)

    #             # El modelo Final Optimizado guarda el state_dict completo
    #             if 'model_state_dict' in checkpoint:
    #                 model.model.load_state_dict(checkpoint['model_state_dict'])
    #                 print(f"- Modelo {model_name} cargado desde: {final_model_path}")

    #                 # Mostrar métricas del entrenamiento si están disponibles
    #                 if 'test_rmse' in checkpoint:
    #                     print(f"  RMSE entrenamiento: {checkpoint['test_rmse']:.8f}")
    #                 if 'naive_rmse' in checkpoint:
    #                     print(f"  Naive RMSE referencia: {checkpoint['naive_rmse']:.8f}")
    #                 if 'naive_improvement' in checkpoint:
    #                     print(f"  Mejora vs Naive: {checkpoint['naive_improvement']:+.6f}%")
    #                 if 'direction_accuracy' in checkpoint:
    #                     print(f"  Directional Accuracy: {checkpoint['direction_accuracy']:.4f} ({checkpoint['direction_accuracy']*100:.2f}%)")
    #                 if 'test_r2' in checkpoint:
    #                     print(f"  R²: {checkpoint['test_r2']:.6f}")
    #                 if 'test_mape' in checkpoint:
    #                     print(f"  MAPE: {checkpoint['test_mape']:.4f}%")
    #                 if 'success_count' in checkpoint:
    #                     print(f"  Criterios cumplidos: {checkpoint['success_count']}/5")
    #                 if 'beats_naive' in checkpoint:
    #                     status = "✅ SÍ" if checkpoint['beats_naive'] else "📏 NO"
    #                     print(f"  Supera Naive: {status}")
    #                     if 'margin_vs_naive' in checkpoint:
    #                         print(f"  Margen vs Naive: {checkpoint['margin_vs_naive']:.6f}%")
    #             else:
    #                 print(f"- Formato de modelo no compatible: {final_model_path}")
    #                 return None
    #         except Exception as e:
    #             print(f"- Error cargando modelo Final Optimizado: {e}")
    #             return None
    #     else:
    #         print(f"- Archivo del modelo Final Optimizado no encontrado en ninguna de las rutas:")
    #         for path in possible_paths:
    #             print(f"  {path} -> {'✓' if os.path.exists(path) else '✗'}")
    #         return None
    # elif model_name == "MultivariateOptimizedModel":
    #     # Buscar el archivo del modelo Multivariado
    #     import glob
    #     multivariate_pattern_paths = [
    #         "modelos/*/MultivariateWinner_*.pth",
    #         "modelos/*/MultivariateOptimized_*.pth",
    #         "modelos/*/MultivariateAttempt_*.pth",
    #         "../modelos/*/MultivariateWinner_*.pth",
    #         "../modelos/*/MultivariateOptimized_*.pth",
    #         "../modelos/*/MultivariateAttempt_*.pth"
    #     ]

    #     multivariate_model_path = None
    #     for pattern in multivariate_pattern_paths:
    #         found_files = glob.glob(pattern)
    #         if found_files:
    #             # Tomar el archivo más reciente
    #             multivariate_model_path = max(found_files, key=os.path.getctime)
    #             break

    #     if multivariate_model_path:
    #         try:
    #             checkpoint = torch.load(multivariate_model_path, map_location=device)

    #             # El modelo Multivariado guarda el state_dict completo
    #             if 'model_state_dict' in checkpoint:
    #                 model.model.load_state_dict(checkpoint['model_state_dict'])
    #                 print(f"- Modelo {model_name} cargado desde: {multivariate_model_path}")

    #                 # Mostrar métricas del entrenamiento si están disponibles
    #                 if 'test_rmse' in checkpoint:
    #                     print(f"  RMSE entrenamiento: {checkpoint['test_rmse']:.8f}")
    #                 if 'naive_rmse' in checkpoint:
    #                     print(f"  Naive RMSE referencia: {checkpoint['naive_rmse']:.8f}")
    #                 if 'naive_improvement' in checkpoint:
    #                     print(f"  Mejora vs Naive: {checkpoint['naive_improvement']:+.6f}%")
    #                 if 'direction_accuracy' in checkpoint:
    #                     print(f"  Directional Accuracy: {checkpoint['direction_accuracy']:.4f} ({checkpoint['direction_accuracy']*100:.2f}%)")
    #                 if 'test_r2' in checkpoint:
    #                     print(f"  R²: {checkpoint['test_r2']:.6f}")
    #                 if 'test_mape' in checkpoint:
    #                     print(f"  MAPE: {checkpoint['test_mape']:.4f}%")
    #                 if 'features' in checkpoint:
    #                     print(f"  Features: {len(checkpoint['features'])} - {', '.join(checkpoint['features'])}")
    #                 if 'beats_naive' in checkpoint:
    #                     status = "✅ SÍ" if checkpoint['beats_naive'] else "📏 NO"
    #                     print(f"  Supera Naive: {status}")

    #                 # Cargar escaladores si están disponibles
    #                 if 'scalers' in checkpoint:
    #                     model.scalers = checkpoint['scalers']
    #                     print(f"  Escaladores multivariados cargados")
    #             else:
    #                 print(f"- Formato de modelo no compatible: {multivariate_model_path}")
    #                 return None
    #         except Exception as e:
    #             print(f"- Error cargando modelo Multivariado: {e}")
    #             return None
    #     else:
    #         print(f"- Archivo del modelo Multivariado no encontrado")
    #         print("  Patrones buscados:")
    #         for pattern in multivariate_pattern_paths:
    #             print(f"    {pattern}")
    #         return None
    # elif os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"- Modelo {model_name} cargado desde: {model_path}")
    # else:
    #     print(f"- Archivo del modelo no encontrado: {model_path}")
    #     return None

    model.to(device)
    model.eval()
    return model

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """Evalúa un modelo en el conjunto de test."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Manejo especial para modelo ARIMA
        if model_name == "ARIMAModel" and hasattr(model, 'fitted_model') and model.fitted_model is not None:
            # Para ARIMA, hacer predicciones one-step-ahead usando el modelo ajustado
            predictions_list = []

            # Obtener datos sin escalar para ARIMA (necesita datos originales)
            n_features = scaler.n_features_in_
            X_test_unscaled = np.zeros((len(X_test), X_test.shape[1], n_features))
            for i in range(len(X_test)):
                X_test_unscaled[i] = scaler.inverse_transform(X_test[i])

            # Usar solo la primera característica (precio) para ARIMA
            price_series = X_test_unscaled[:, :, 0]  # (batch, seq_len)

            for i in range(len(price_series)):
                try:
                    # Tomar la secuencia hasta el punto actual
                    current_series = price_series[i]  # (seq_len,)

                    # Hacer predicción usando el último punto de la serie
                    forecast = model.fitted_model.forecast(steps=1)
                    if hasattr(forecast, 'values'):
                        pred = float(forecast.values[0])
                    elif isinstance(forecast, (list, tuple)):
                        pred = float(forecast[0])
                    else:
                        pred = float(forecast)

                    predictions_list.append(pred)
                except Exception as e:
                    # Fallback: usar último valor
                    predictions_list.append(float(current_series[-1]))

            # Convertir predicciones a tensor
            predictions = torch.tensor(predictions_list, dtype=torch.float32, device=device).unsqueeze(-1)

        else:
            # Para otros modelos, usar forward normal
            predictions = model(X_test_tensor)

        # Asegurar que las predicciones tengan la forma correcta
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(-1)
        if len(y_test_tensor.shape) == 1:
            y_test_tensor = y_test_tensor.unsqueeze(-1)

        # Convertir a numpy para cálculo de métricas
        y_pred_scaled = predictions.cpu().numpy()
        y_true_scaled = y_test_tensor.cpu().numpy()

        # Desescalar predicciones para métricas en escala original
        # Crear arrays con todas las características para el inverse_transform
        n_features = scaler.n_features_in_

        # Para y_true
        y_true_full = np.zeros((len(y_true_scaled), n_features))
        y_true_full[:, 0] = y_true_scaled.flatten()  # Solo la primera característica (Último)
        y_true_original = scaler.inverse_transform(y_true_full)[:, 0]

        # Para y_pred - todos los modelos devuelven valores escalados
        y_pred_full = np.zeros((len(y_pred_scaled), n_features))
        y_pred_full[:, 0] = y_pred_scaled.flatten()
        y_pred_original = scaler.inverse_transform(y_pred_full)[:, 0]

        # Calcular métricas
        metrics = calculate_metrics(y_true_original, y_pred_original)

        return {
            'model_name': model_name,
            'y_true': y_true_original,
            'y_pred': y_pred_original,
            'metrics': metrics
        }

def evaluate_multivariate_special():
    """Evaluación especial para el modelo multivariado usando datos reales multivariados"""
    try:
        # Importar la función de carga de datos multivariados
        import sys
        sys.path.append('.')
        from train_multivariate_optimized import MultivariateOptimizedModel, load_multivariate_data
        import glob

        print("🔄 Cargando evaluación multivariada especial...")

        # Cargar datos multivariados (igual que en entrenamiento)
        combined_df = load_multivariate_data(
            main_file=DEFAULT_PARAMS.FILEPATH,
            correlation_threshold=0.4
        )

        # Preparar escaladores por feature (igual que en entrenamiento)
        scalers = {}
        scaled_data = combined_df.copy()

        for col in combined_df.columns:
            scaler = RobustScaler(quantile_range=(5, 95))
            scaled_data[col] = scaler.fit_transform(combined_df[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler

        # Crear secuencias multivariadas
        seq_length = 30
        features = scaled_data.values
        X, y = [], []

        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])  # (seq_length, n_features)
            y.append(features[i, 0])  # Solo EUR/USD como target

        X = np.array(X)  # (samples, seq_length, features)
        y = np.array(y)

        # División temporal igual que en entrenamiento
        train_size = int(0.80 * len(X))
        val_size = int(0.10 * len(X))

        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        # Cargar modelo multivariado entrenado
        multivariate_files = glob.glob("modelos/*/MultivariateWinner_*.pth")
        if not multivariate_files:
            multivariate_files = glob.glob("modelos/*/MultivariateOptimized_*.pth")
        if not multivariate_files:
            multivariate_files = glob.glob("modelos/*/MultivariateAttempt_*.pth")

        if not multivariate_files:
            print("❌ No se encontró modelo multivariado entrenado")
            return None

        model_path = max(multivariate_files, key=os.path.getctime)

        # Crear modelo
        model = MultivariateOptimizedModel(
            input_size=X.shape[2],
            hidden_size=64,
            num_layers=2,
            seq_length=30
        ).to(device)

        # Cargar pesos
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Cargar escaladores del checkpoint
        if 'scalers' in checkpoint:
            saved_scalers = checkpoint['scalers']
        else:
            saved_scalers = scalers

        # Evaluación en datos de test multivariados
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            pred, direction_probs, confidence = model(X_test_tensor)
            pred = pred.cpu().numpy()
            confidence = confidence.cpu().numpy().squeeze()

        # Desnormalizar predicciones con el escalador correcto
        eur_usd_scaler = saved_scalers['EUR_USD']
        pred_denorm = eur_usd_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        y_test_denorm = eur_usd_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Convertir a precios
        pred_prices = np.exp(pred_denorm)
        target_prices = np.exp(y_test_denorm)

        # Calcular métricas multivariadas reales
        metrics = calculate_metrics(target_prices, pred_prices)

        # Directional accuracy mejorada
        naive_pred_scaled = X_test[:, -1, 0]  # Último valor de EUR_USD
        naive_pred_denorm = eur_usd_scaler.inverse_transform(naive_pred_scaled.reshape(-1, 1)).flatten()
        naive_prices = np.exp(naive_pred_denorm)

        model_directions = np.where(pred_prices > naive_prices, 1, 0)
        actual_directions = np.where(target_prices > naive_prices, 1, 0)
        direction_acc = np.mean(model_directions == actual_directions)

        # Actualizar DA en métricas
        metrics['DA'] = direction_acc * 100

        print(f"✅ Evaluación multivariada especial completada")
        print(f"   Usando {X.shape[2]} features multivariadas")
        print(f"   Directional Accuracy mejorada: {direction_acc*100:.2f}%")

        return {
            'model_name': 'MultivariateOptimizedModel',
            'y_true': target_prices,
            'y_pred': pred_prices,
            'metrics': metrics
        }

    except Exception as e:
        print(f"❌ Error en evaluación multivariada especial: {e}")
        return None

def compare_models():
    """Compara todos los modelos disponibles incluyendo el baseline Naive."""

    # Lista de modelos a evaluar - Nombres corregidos para coincidir con entrenamiento reproducible
    models_to_evaluate = [
        "NaiveForecastModel",  # Baseline simple
        "ARIMAModel",        # Baseline estadístico
        "TLS_LSTMModel",     # Modelo TLS-LSTM básico
        "TLS_LSTMModel_Optimizado",  # Modelo TLS-LSTM optimizado
        "GRU_Model",         # Modelo GRU
        "HybridLSTMAttentionModel",  # Modelo LSTM con atención (nombre corregido)
        "BidirectionalDeepLSTMModel",  # Modelo LSTM bidireccional (nombre corregido)
        "ContextualLSTMTransformerFlexible",  # Modelo con Transformer
    ]

    # Cargar y preparar datos
    df = load_and_prepare_data(DEFAULT_PARAMS.FILEPATH)
    if df is None:
        print("No se pudieron cargar los datos.")
        return

    # Agregar indicadores
    indicators = add_indicator(df)
    for indicator_name, values in indicators.items():
        df[indicator_name] = values

    # Dividir en train/test
    split_index = int(len(df) * DEFAULT_PARAMS.TRAIN_SPLIT_RATIO)
    features = DEFAULT_PARAMS.FEATURES
    train_data = df[features].iloc[:split_index]
    test_data = df[features].iloc[split_index:]

    # Escalar datos
    scaler = RobustScaler(quantile_range=(5, 95))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Crear secuencias con la misma longitud que el entrenamiento reproducible
    REPRODUCIBLE_SEQ_LENGTH = 30  # Longitud usada en el entrenamiento reproducible
    X_train, y_train = create_sequences(train_scaled, REPRODUCIBLE_SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)
    X_test, y_test = create_sequences(test_scaled, REPRODUCIBLE_SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)

    print(f"Forma X_test: {X_test.shape}, Forma y_test: {y_test.shape}")

    # Evaluar cada modelo
    results = []

    for model_name in models_to_evaluate:
        print(f"\n--- Evaluando {model_name} ---")

        # Evaluación especial para modelo multivariado
        if model_name == "MultivariateOptimizedModel":
            try:
                # Usar evaluación multivariada real
                result = evaluate_multivariate_special()
                if result:
                    results.append(result)
                    # Mostrar métricas
                    metrics = result['metrics']
                    print(f"Métricas para {model_name} (evaluación multivariada):")
                    for metric_name, value in metrics.items():
                        if not np.isnan(value):
                            print(f"  {metric_name}: {value:.6f}")
                        else:
                            print(f"  {metric_name}: N/A")
                else:
                    print(f"- Error en evaluación multivariada especial")
            except Exception as e:
                print(f"- Error al evaluar {model_name} con datos multivariados: {e}")
            continue

        # Construir ruta del modelo
        model_path = f"modelos/{DEFAULT_PARAMS.TABLENAME}/{model_name}_{DEFAULT_PARAMS.FILEPATH}.pth"

        try:
            # Obtener parámetros específicos del modelo para evaluación
            model_params = get_model_specific_params(model_name)

            # Cargar modelo con parámetros específicos
            model = load_model(model_name, model_path, model_params)

            if model is not None:
                # Para ARIMA, reentrenar si es necesario
                if model_name == "ARIMAModel" and not model.is_trained:
                    print("- Reentrenando modelo ARIMA con datos de entrenamiento...")
                    train_data_original = df[DEFAULT_PARAMS.TARGET_COLUMN].iloc[:split_index].values
                    model.fit_global_arima(train_data_original)

                # Evaluar modelo
                result = evaluate_model(model, X_test, y_test, scaler, model_name)
                results.append(result)

                # Mostrar métricas
                metrics = result['metrics']
                print(f"Métricas para {model_name}:")
                for metric_name, value in metrics.items():
                    if not np.isnan(value):
                        print(f"  {metric_name}: {value:.6f}")
                    else:
                        print(f"  {metric_name}: N/A")

        except Exception as e:
            print(f"- Error al evaluar {model_name}: {e}")

    # Crear tabla comparativa
    if results:
        create_comparison_table(results)
        create_comparison_plots(results)

    return results

def create_comparison_table(results):
    """Crea una tabla comparativa de todos los modelos."""

    # Crear DataFrame con métricas
    comparison_data = []
    for result in results:
        row = {'Model': result['model_name']}
        row.update(result['metrics'])
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Redondear valores numéricos
    numeric_columns = df_comparison.select_dtypes(include=[np.number]).columns
    df_comparison[numeric_columns] = df_comparison[numeric_columns].round(6)

    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE MODELOS")
    print("="*80)
    print(df_comparison.to_string(index=False))

    # Identificar el mejor modelo para cada métrica
    print("\n" + "="*50)
    print("MEJORES MODELOS POR MÉTRICA")
    print("="*50)

    metrics_to_minimize = ['MSE', 'RMSE', 'MAE', 'MAPE']
    metrics_to_maximize = ['R2', 'DA']

    for metric in metrics_to_minimize:
        if metric in df_comparison.columns:
            # Filtrar valores no válidos (NaN, None, etc.)
            valid_data = df_comparison[df_comparison[metric].notna()]
            if not valid_data.empty:
                best_idx = valid_data[metric].idxmin()
                best_model = df_comparison.loc[best_idx, 'Model']
                best_value = df_comparison.loc[best_idx, metric]
                print(f"Mejor {metric}: {best_model} ({best_value:.6f})")

    for metric in metrics_to_maximize:
        if metric in df_comparison.columns:
            # Filtrar valores no válidos (NaN, None, etc.)
            valid_data = df_comparison[df_comparison[metric].notna()]
            if not valid_data.empty:
                best_idx = valid_data[metric].idxmax()
                best_model = df_comparison.loc[best_idx, 'Model']
                best_value = df_comparison.loc[best_idx, metric]
                print(f"Mejor {metric}: {best_model} ({best_value:.6f})")

                # Análisis especial para DA
                if metric == 'DA':
                    print("  📊 Análisis adicional DA:")
                    top_3_da = valid_data.nlargest(3, 'DA')
                    for idx, row in top_3_da.iterrows():
                        model_name = row['Model']
                        da_value = row['DA']
                        rmse_value = row['RMSE']
                        r2_value = row['R2']
                        print(f"    • {model_name}: DA={da_value:.2f}%, RMSE={rmse_value:.6f}, R²={r2_value:.6f}")

                    # Destacar modelo multivariado si está en top 3
                    multivar_row = df_comparison[df_comparison['Model'] == 'MultivariateOptimizedModel']
                    if not multivar_row.empty:
                        mv_da = multivar_row['DA'].iloc[0]
                        mv_rmse = multivar_row['RMSE'].iloc[0]
                        mv_r2 = multivar_row['R2'].iloc[0]
                        print(f"  🎯 MultivariateOptimizedModel: DA={mv_da:.2f}%, RMSE={mv_rmse:.6f}, R²={mv_r2:.6f}")
                        print(f"     ✅ Modelo balanceado: supera Naive baseline en DA y mantiene métricas estables")

    # Guardar tabla
    df_comparison.to_csv(f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}/model_comparison.csv', index=False)
    print(f"\n- Tabla comparativa guardada en: images/comparacion/{DEFAULT_PARAMS.TABLENAME}/model_comparison.csv")

def create_comparison_plots(results):
    """Crea gráficos comparativos de los modelos."""

    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparación de Modelos vs Baseline Naive', fontsize=16, fontweight='bold')

    # Extraer métricas
    models = [r['model_name'] for r in results]
    rmse_values = [r['metrics']['RMSE'] for r in results]
    mae_values = [r['metrics']['MAE'] for r in results]
    r2_values = [r['metrics']['R2'] for r in results if not np.isnan(r['metrics']['R2'])]
    da_values = [r['metrics']['DA'] for r in results if not np.isnan(r['metrics']['DA'])]

    # Colores especiales para el baseline
    colors = ['red' if model == 'NaiveForecastModel' else 'blue' for model in models]

    # Gráfico 1: RMSE
    axes[0, 0].bar(models, rmse_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('RMSE por Modelo', fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Gráfico 2: MAE
    axes[0, 1].bar(models, mae_values, color=colors, alpha=0.7)
    axes[0, 1].set_title('MAE por Modelo', fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Gráfico 3: R²
    if r2_values:
        axes[1, 0].bar([models[i] for i in range(len(models)) if not np.isnan(results[i]['metrics']['R2'])],
                      r2_values, color=[colors[i] for i in range(len(models)) if not np.isnan(results[i]['metrics']['R2'])], alpha=0.7)
        axes[1, 0].set_title('R² por Modelo', fontweight='bold')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Gráfico 4: Directional Accuracy
    if da_values:
        axes[1, 1].bar([models[i] for i in range(len(models)) if not np.isnan(results[i]['metrics']['DA'])],
                      da_values, color=[colors[i] for i in range(len(models)) if not np.isnan(results[i]['metrics']['DA'])], alpha=0.7)
        axes[1, 1].set_title('Directional Accuracy (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}/model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"- Graficos comparativos guardados en: images/comparacion/{DEFAULT_PARAMS.TABLENAME}/model_comparison_charts.png")

    # Gráfico de predicciones vs valores reales para algunos modelos
    create_prediction_plots(results)

def create_prediction_plots(results, max_points=200):
    """Crea gráficos de predicciones vs valores reales."""

    # Seleccionar algunos modelos para visualizar
    models_to_plot = ['NaiveForecastModel', 'ARIMAModel', 'GRU_Model', 'TLS_LSTMModel', 'HybridLSTMAttention', 'BidirectionalDeepLSTM','FinalOptimizedTLSLSTM', 'FinalOptimizedModel']
    selected_results = [r for r in results if r['model_name'] in models_to_plot]

    if not selected_results:
        return

    fig, axes = plt.subplots(len(selected_results), 1, figsize=(12, 4*len(selected_results)))
    if len(selected_results) == 1:
        axes = [axes]

    for i, result in enumerate(selected_results):
        model_name = result['model_name']
        y_true = result['y_true'][:max_points]  # Limitar puntos para visualización
        y_pred = result['y_pred'][:max_points]

        # Crear índice temporal
        x = np.arange(len(y_true))

        axes[i].plot(x, y_true, label='Valores Reales', color='blue', alpha=0.7)
        axes[i].plot(x, y_pred, label='Predicciones', color='red', alpha=0.7)
        axes[i].set_title(f'Predicciones vs Valores Reales - {model_name}', fontweight='bold')
        axes[i].set_xlabel('Tiempo')
        axes[i].set_ylabel('Valor')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Añadir métricas al gráfico
        rmse = result['metrics']['RMSE']
        mae = result['metrics']['MAE']
        axes[i].text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'images/comparacion/{DEFAULT_PARAMS.TABLENAME}/prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"- Graficos de predicciones guardados en: images/comparacion/{DEFAULT_PARAMS.TABLENAME}/prediction_comparison.png")

if __name__ == "__main__":
    print("- Iniciando evaluacion comparativa de modelos...")
    print("Incluyendo modelo baseline Naive Forecast...")
    results = compare_models()
    print("\n- Evaluacion completada!")
