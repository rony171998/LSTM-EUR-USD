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
)
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

def load_model(model_name, model_path, input_size, params):
    """Carga un modelo entrenado."""
    if model_name == "TLS_LSTMModel":
        model = TLS_LSTMModel(input_size, params.HIDDEN_SIZE, params.FORECAST_HORIZON, params.DROPOUT_PROB)
    elif model_name == "GRU_Model":
        model = GRU_Model(input_size, params.HIDDEN_SIZE, params.FORECAST_HORIZON, params.DROPOUT_PROB)
    elif model_name == "HybridLSTMAttention":
        model = HybridLSTMAttentionModel(input_size, params.HIDDEN_SIZE, params.FORECAST_HORIZON, params.DROPOUT_PROB)
    elif model_name == "BidirectionalDeepLSTM":
        model = BidirectionalDeepLSTMModel(input_size, params.HIDDEN_SIZE, params.FORECAST_HORIZON, params.DROPOUT_PROB)
    elif model_name == "ContextualLSTMTransformerFlexible":
        model = ContextualLSTMTransformerFlexible(
            seq_len=params.SEQ_LENGTH,
            feature_dim=input_size,
            output_size=params.FORECAST_HORIZON,
            window_size=params.WINDOW_SIZE,
            max_neighbors=params.MAX_NEIGHBORS,
            lstm_units=params.LSTM_UNITS,
            num_heads=params.NUM_HEADS,
            embed_dim=params.EMBED_DIM,
            dropout_rate=params.DROPOUT_PROB,
        )
    elif model_name == "NaiveForecastModel":
        # Modelo baseline que no requiere entrenamiento previo
        model = NaiveForecastModel(input_size=input_size, output_size=params.FORECAST_HORIZON)
        print(f"- Modelo baseline {model_name} inicializado (no requiere archivo .pth)")
        model.to(device)
        model.eval()
        return model
    elif model_name == "ARIMAModel":
        # Modelo ARIMA que se entrenará dinámicamente
        model = ARIMAModel(input_size=input_size, output_size=params.FORECAST_HORIZON)
        model.is_trained = False  # Marcar que necesita entrenamiento
        print(f"- Modelo baseline {model_name} inicializado (se entrenará dinámicamente)")
        model.to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")
    
    # Cargar pesos solo para modelos de deep learning
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"- Modelo {model_name} cargado desde: {model_path}")
    else:
        print(f"- Archivo del modelo no encontrado: {model_path}")
        return None
    
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

def compare_models():
    """Compara todos los modelos disponibles incluyendo el baseline Naive."""
    
    # Lista de modelos a evaluar
    models_to_evaluate = [
        "NaiveForecastModel",  # Baseline simple
        "ARIMAModel",          # Baseline estadístico
        "TLS_LSTMModel",
        "GRU_Model", 
        "HybridLSTMAttention",
        "BidirectionalDeepLSTM",
        "ContextualLSTMTransformerFlexible"
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
    
    # Crear secuencias
    X_train, y_train = create_sequences(train_scaled, DEFAULT_PARAMS.SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)
    X_test, y_test = create_sequences(test_scaled, DEFAULT_PARAMS.SEQ_LENGTH, DEFAULT_PARAMS.FORECAST_HORIZON)
    
    print(f"Forma X_test: {X_test.shape}, Forma y_test: {y_test.shape}")
    
    # Evaluar cada modelo
    results = []
    
    for model_name in models_to_evaluate:
        print(f"\n--- Evaluando {model_name} ---")
        
        # Construir ruta del modelo
        model_path = f"modelos/{DEFAULT_PARAMS.TABLENAME}/{model_name}_{DEFAULT_PARAMS.FILEPATH}.pth"
        
        try:
            # Cargar modelo
            model = load_model(model_name, model_path, len(features), DEFAULT_PARAMS)
            
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
            best_idx = df_comparison[metric].idxmin()
            best_model = df_comparison.loc[best_idx, 'Model']
            best_value = df_comparison.loc[best_idx, metric]
            print(f"Mejor {metric}: {best_model} ({best_value:.6f})")
    
    for metric in metrics_to_maximize:
        if metric in df_comparison.columns:
            best_idx = df_comparison[metric].idxmax()
            best_model = df_comparison.loc[best_idx, 'Model']
            best_value = df_comparison.loc[best_idx, metric]
            print(f"Mejor {metric}: {best_model} ({best_value:.6f})")
    
    # Guardar tabla
    df_comparison.to_csv('model_comparison.csv', index=False)
    print(f"\n- Tabla comparativa guardada en: model_comparison.csv")

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
    plt.savefig('image/test/model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"- Graficos comparativos guardados en: image/test/model_comparison_charts.png")
    
    # Gráfico de predicciones vs valores reales para algunos modelos
    create_prediction_plots(results)

def create_prediction_plots(results, max_points=200):
    """Crea gráficos de predicciones vs valores reales."""
    
    # Seleccionar algunos modelos para visualizar
    models_to_plot = ['NaiveForecastModel', 'ARIMAModel', 'GRU_Model', 'TLS_LSTMModel', 'HybridLSTMAttention', 'BidirectionalDeepLSTM']
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
    plt.savefig('image/test/prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"- Graficos de predicciones guardados en: image/test/prediction_comparison.png")

if __name__ == "__main__":
    print("- Iniciando evaluacion comparativa de modelos...")
    print("Incluyendo modelo baseline Naive Forecast...")
    results = compare_models()
    print("\n- Evaluacion completada!")
