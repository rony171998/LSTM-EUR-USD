# estadisticas.py
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_model import (
    add_indicator,
)
from save_data import get_df
import seaborn as sns
from config import DEFAULT_PARAMS
import os
from statsmodels.tsa.stattools import adfuller

# Crear la carpeta si no existe
os.makedirs("images/estadisticas", exist_ok=True)

def plot_historical_data(data, hurst, adf_result, title="Comportamiento histórico EUR/USD"):
    """
    Grafica el comportamiento histórico del precio con información estadística a la derecha
    """
    # Crear figura con dos subplots: uno para el gráfico y otro para los stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                                  gridspec_kw={'width_ratios': [3, 1]})
    
    # Gráfico principal en el primer subplot
    ax1.plot(data.index, data[DEFAULT_PARAMS.TARGET_COLUMN], 
             label='Precio EUR/USD', color='blue', alpha=0.7)
    
    # Configuración del gráfico principal
    ax1.set_title(title, fontsize=14, pad=20)
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Precio', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Crear texto con la información estadística
    stats_text = (
        f"📊 Estadísticas clave:\n"
        f"----------------------------\n"
        f"📅 Período: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}\n"
        f"📈 Datos: {len(data)} observaciones\n\n"
        
        f"📌 Exponente de Hurst:\n"
        f"H = {hurst:.4f}\n"
        f"{'🔮 Paseo aleatorio (H≈0.5)' if 0.4 < hurst < 0.6 else '📈 Tendencia persistente (H>0.6)' if hurst > 0.6 else '📉 Tendencia anti-persistente (H<0.4)'}\n\n"
        
        f"🧪 Test ADF:\n"
        f"Estadístico = {adf_result[0]:.4f}\n"
        f"p-valor = {adf_result[1]:.4f}\n"
        f"{'✅ Estacionaria (p<0.05)' if adf_result[1] < 0.05 else '❌ No estacionaria (p>0.05)'}\n\n"
        
        f"📊 Estadísticas descriptivas:\n"
        f"----------------------------\n"
        f"├─ Media: {data[DEFAULT_PARAMS.TARGET_COLUMN].mean():.6f}\n"
        f"├─ Std: {data[DEFAULT_PARAMS.TARGET_COLUMN].std():.6f}\n"
        f"├─ Min: {data[DEFAULT_PARAMS.TARGET_COLUMN].min():.6f}\n"
        f"├─ 25%: {data[DEFAULT_PARAMS.TARGET_COLUMN].quantile(0.25):.6f}\n"
        f"├─ 50%: {data[DEFAULT_PARAMS.TARGET_COLUMN].median():.6f}\n"
        f"├─ 75%: {data[DEFAULT_PARAMS.TARGET_COLUMN].quantile(0.75):.6f}\n"
        f"└─ Max: {data[DEFAULT_PARAMS.TARGET_COLUMN].max():.6f}"
    )
    
    # Mostrar texto en el segundo subplot
    ax2.text(0.1, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='whitesmoke', 
             alpha=0.8, edgecolor='lightgray'))
    
    # Configurar el segundo subplot (ocultar ejes)
    ax2.axis('off')
    ax2.set_title('Análisis Estadístico', fontsize=12, pad=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig("images/estadisticas/historical.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_historical_sma(data, hurst, adf_result, title="Comportamiento histórico EUR/USD con SMA"):
    """
    Grafica el comportamiento histórico del precio con SMA e información estadística a la derecha
    """
    # Crear figura con dos subplots: uno para el gráfico y otro para los stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                  gridspec_kw={'width_ratios': [3, 1]})
    
    # Gráfico principal en el primer subplot
    ax1.plot(data.index, data[DEFAULT_PARAMS.TARGET_COLUMN], 
             label='Precio EUR/USD', color='blue', alpha=0.5, linewidth=1)
    
    # Calcular y graficar SMA de 50 y 200 períodos
    sma_50 = data[DEFAULT_PARAMS.TARGET_COLUMN].rolling(window=50).mean()
    sma_200 = data[DEFAULT_PARAMS.TARGET_COLUMN].rolling(window=200).mean()
    
    ax1.plot(data.index, sma_50, label='SMA 50', color='green', linewidth=1.5)
    ax1.plot(data.index, sma_200, label='SMA 200', color='red', linewidth=1.5)
    
    # Resaltar cruces entre SMA 50 y SMA 200
    cross_above = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
    cross_below = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))
    
    ax1.scatter(data.index[cross_above], sma_50[cross_above], 
                color='lime', marker='^', s=100, label='Cruce alcista (SMA50 > SMA200)')
    ax1.scatter(data.index[cross_below], sma_50[cross_below], 
                color='darkred', marker='v', s=100, label='Cruce bajista (SMA50 < SMA200)')
    
    # Configuración del gráfico principal
    ax1.set_title(title, fontsize=14, pad=20)
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Precio', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='upper left')
    
    # Crear texto con la información estadística
    stats_text = (
        f"📊 Estadísticas clave:\n"
        f"----------------------------\n"
        f"📅 Período: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}\n"
        f"📈 Datos: {len(data)} observaciones\n\n"
        
        f"📌 Exponente de Hurst:\n"
        f"H = {hurst:.4f}\n"
        f"{'🔮 Paseo aleatorio (H≈0.5)' if 0.4 < hurst < 0.6 else '📈 Tendencia persistente (H>0.6)' if hurst > 0.6 else '📉 Tendencia anti-persistente (H<0.4)'}\n\n"
        
        f"🧪 Test ADF:\n"
        f"Estadístico = {adf_result[0]:.4f}\n"
        f"p-valor = {adf_result[1]:.4f}\n"
        f"{'✅ Estacionaria (p<0.05)' if adf_result[1] < 0.05 else '❌ No estacionaria (p>0.05)'}\n\n"
        
        f"📊 Estadísticas descriptivas:\n"
        f"----------------------------\n"
        f"├─ Media: {data[DEFAULT_PARAMS.TARGET_COLUMN].mean():.6f}\n"
        f"├─ Std: {data[DEFAULT_PARAMS.TARGET_COLUMN].std():.6f}\n"
        f"├─ Min: {data[DEFAULT_PARAMS.TARGET_COLUMN].min():.6f}\n"
        f"├─ 25%: {data[DEFAULT_PARAMS.TARGET_COLUMN].quantile(0.25):.6f}\n"
        f"├─ 50%: {data[DEFAULT_PARAMS.TARGET_COLUMN].median():.6f}\n"
        f"├─ 75%: {data[DEFAULT_PARAMS.TARGET_COLUMN].quantile(0.75):.6f}\n"
        f"└─ Max: {data[DEFAULT_PARAMS.TARGET_COLUMN].max():.6f}\n\n"
        
        f"📈 Medias Móviles:\n"
        f"----------------------------\n"
        f"├─ SMA 50 actual: {sma_50.iloc[-1]:.6f}\n"
        f"└─ SMA 200 actual: {sma_200.iloc[-1]:.6f}\n"
        f"Relación: {'SMA50 > SMA200' if sma_50.iloc[-1] > sma_200.iloc[-1] else 'SMA50 < SMA200'}"
    )
    
    # Mostrar texto en el segundo subplot
    ax2.text(0.1, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='whitesmoke', 
             alpha=0.8, edgecolor='lightgray'))
    
    # Configurar el segundo subplot (ocultar ejes)
    ax2.axis('off')
    ax2.set_title('Análisis Estadístico', fontsize=12, pad=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig("images/estadisticas/historical_sma.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplot(data, column_name):
    """Grafica un boxplot para visualizar mediana y distribución"""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column_name], color='lightblue')
    
    mediana = data[column_name].median()
    q1 = data[column_name].quantile(0.25)
    q3 = data[column_name].quantile(0.75)
    
    plt.title(f'Boxplot de {column_name}')
    plt.xlabel(column_name)
    
    stats_text = f'Mediana: {mediana:.2f}\nQ1: {q1:.2f}\nQ3: {q3:.2f}'
    plt.text(0.75, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"images/estadisticas/boxplot_{column_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_financial_tendency(data, rsi_col='RSI'):
    """Visualización especializada para datos financieros"""
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    sns.histplot(data[rsi_col], kde=True, ax=ax2, color='salmon')
    ax2.axvline(30, color='darkgreen', linestyle=':', label='Sobreventa')
    ax2.axvline(70, color='darkred', linestyle=':', label='Sobrecompra')
    ax2.axvline(data[rsi_col].median(), color='black', label='Mediana RSI')
    ax2.set_title(f'Distribución de RSI ({rsi_col})')
    ax2.legend()

    plt.tight_layout()
    fig.savefig("images/estadisticas/financial_tendency.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_central_tendency(data, column_name):
    """Grafica la distribución con media, mediana y moda"""
    media = data[column_name].mean()
    mediana = data[column_name].median()
    moda = data[column_name].mode()[0]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data[column_name], kde=True, color='skyblue', bins=30)
    
    plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    plt.axvline(moda, color='purple', linestyle='--', linewidth=2, label=f'Moda: {moda:.2f}')
    
    plt.title(f'Distribución de {column_name} con Medidas de Tendencia Central')
    plt.xlabel(column_name)
    plt.ylabel('Frecuencia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"images/estadisticas/central_tendency_{column_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def computehurst(ts):
    """Calcula el exponente de Hurst para una serie temporal"""
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

if __name__ == "__main__":
    # 1. Cargar los datos
    df = get_df(table_name='eur_usd')
    
    # 2. Calcular métricas estadísticas
    hurst = computehurst(df[DEFAULT_PARAMS.TARGET_COLUMN].values)
    adf_result = adfuller(df[DEFAULT_PARAMS.TARGET_COLUMN].values)
    
    # 3. Mostrar gráfico histórico con información estadística incorporada
    plot_historical_data(df, hurst, adf_result)

    plot_historical_sma(df, hurst, adf_result)
    
    # 4. Calcular y mostrar indicadores técnicos
    indicators = add_indicator(df)
    for indicator_name, values in indicators.items():
        df[indicator_name] = values
    
    # 5. Graficar análisis estadísticos
    columna_analisis = DEFAULT_PARAMS.TARGET_COLUMN
    plot_central_tendency(df, columna_analisis)
    plot_boxplot(df, columna_analisis)
    plot_financial_tendency(df)