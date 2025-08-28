#!/usr/bin/env python3
"""
Generador de gráficos simplificado para modelos reproducibles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

def create_simple_plots():
    """Crear gráficos simplificados de los resultados."""
    
    # Datos de los resultados (LSTM + Baselines)
    data = {
        'Modelo': ['NaiveForecast', 'ARIMA', 'BidirectionalDeepLSTMModel', 'TLS_LSTMModel_Optimizado', 
                  'HybridLSTMAttentionModel', 'GRU_Model', 'TLS_LSTMModel', 'ContextualLSTMTransformerFlexible'],
        'RMSE_Eval': [0.005025, 0.005063, 0.006302, 0.006326, 0.007059, 0.011394, 0.012781, 0.016287],
        'R2_Eval': [0.976684, 0.976333, 0.963337, 0.963065, 0.954000, 0.880169, 0.849215, 0.755137],
        'DA_Eval': [50.0, 50.449102, 51.072386, 50.402145, 50.402145, 50.268097, 49.865952, 52.412869],
        'Parametros': [0, 0, 88301, 200833, 34202, 54351, 31651, 134977]
    }
    
    df = pd.DataFrame(data)
    
    # Crear directorio si no existe (ruta desde la raíz del proyecto)
    save_dir = '../images/comparacion/eur_usd'
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RMSE por modelo
    ax1 = axes[0, 0]
    # Colores específicos: Baseline en grises, LSTM en colores
    colors = ['lightgray', 'gray', 'gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral']
    bars1 = ax1.bar(range(len(df)), df['RMSE_Eval'], color=colors)
    ax1.set_title('RMSE por Modelo (Menor es Mejor)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks(range(len(df)))
    # Nombres más cortos para mejor visualización
    short_names = ['Naive', 'ARIMA', 'BiLSTM', 'TLS_Opt', 'Hybrid', 'GRU', 'TLS', 'Contextual']
    ax1.set_xticklabels(short_names, rotation=45, ha='right')
    
    # Agregar valores
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. R² por modelo
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df)), df['R2_Eval']*100, color=colors)
    ax2.set_title('R² Score por Modelo (Mayor es Mejor)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² (%)', fontsize=12)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(short_names, rotation=45, ha='right')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Directional Accuracy
    ax3 = axes[1, 0]
    # Ordenar por DA
    df_da_sorted = df.sort_values('DA_Eval', ascending=False)
    # Mapear colores según el orden
    color_map = dict(zip(df['Modelo'], colors))
    da_colors = [color_map[model] for model in df_da_sorted['Modelo']]
    short_names_sorted = ['Contextual', 'BiLSTM', 'ARIMA', 'TLS_Opt', 'Hybrid', 'GRU', 'Naive', 'TLS']
    
    bars3 = ax3.bar(range(len(df_da_sorted)), df_da_sorted['DA_Eval'], color=da_colors)
    ax3.set_title('Directional Accuracy por Modelo (Mayor es Mejor)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('DA (%)', fontsize=12)
    ax3.set_xticks(range(len(df_da_sorted)))
    ax3.set_xticklabels(short_names_sorted, rotation=45, ha='right')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Scatter RMSE vs R²
    ax4 = axes[1, 1]
    scatter_colors = ['gray', 'darkgray', 'red', 'orange', 'yellow', 'lightgreen', 'blue', 'purple']
    
    # Separar baselines de LSTM
    baseline_mask = df['Parametros'] == 0
    lstm_mask = df['Parametros'] > 0
    
    # Plot baselines
    ax4.scatter(df[baseline_mask]['RMSE_Eval'], df[baseline_mask]['R2_Eval']*100, 
               c=['gray', 'darkgray'], s=200, alpha=0.8, marker='s', 
               edgecolors='black', linewidth=2, label='Baselines')
    
    # Plot LSTM models
    ax4.scatter(df[lstm_mask]['RMSE_Eval'], df[lstm_mask]['R2_Eval']*100, 
               c=['red', 'orange', 'yellow', 'lightgreen', 'blue', 'purple'], s=200, alpha=0.7, 
               edgecolors='black', linewidth=2, label='LSTM Models')
    
    # Añadir etiquetas
    for i, (model, short_name) in enumerate(zip(df['Modelo'], short_names)):
        ax4.annotate(f"{short_name}", 
                    (df['RMSE_Eval'].iloc[i], df['R2_Eval'].iloc[i]*100),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=scatter_colors[i], alpha=0.7))
    
    ax4.set_title('RMSE vs R² Score\n(Esquina Superior Izquierda = Mejor)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('RMSE (menor mejor)', fontsize=12)
    ax4.set_ylabel('R² Score (%) (mayor mejor)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    plot_path = Path(save_dir) / 'modelos_vs_baselines_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Grafico comparativo guardado en: {plot_path}")
    
    # Mostrar el gráfico
    plt.show()
    
    return plot_path

if __name__ == "__main__":
    print("Generando graficos de comparacion: LSTM vs Baselines...")
    print()
    
    # Crear gráficos
    plot_path = create_simple_plots()
    
    print()
    print("Graficos generados exitosamente!")
    print(f"- Graficos comparativos: {plot_path}")
    
    # Mostrar resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print("BASELINES:")
    print("  Naive:     RMSE=0.005025, R²=97.67%, DA=50.00%")
    print("  ARIMA:     RMSE=0.005063, R²=97.63%, DA=50.45%")
    print("\nMEJORES LSTM:")
    print("  BiLSTM:    RMSE=0.006302, R²=96.33%, DA=51.07%")
    print("  TLS_Opt:   RMSE=0.006326, R²=96.31%, DA=50.40%")
    print("  Hybrid:    RMSE=0.007059, R²=95.40%, DA=50.40%")
    print("\n⚠️  CONCLUSIÓN: Los baselines superan a los modelos LSTM")
    print("   en RMSE y R², confirmando la eficiencia del mercado EUR/USD")
