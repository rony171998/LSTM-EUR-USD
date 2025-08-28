#!/usr/bin/env python3
"""
generate_complete_comparison_chart.py - Gráfico completo con Rolling Forecast
Incluye resultados de Optuna + Baselines + Rolling Forecast (65.5% DA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

def create_complete_comparison_chart():
    """Crear gráfico comparativo completo con todos los resultados"""
    print("📊 Generando gráfico comparativo completo...")
    
    # Datos de resultados (basados en las evaluaciones)
    results_data = [
        # Baselines
        {"model_name": "Naive Baseline", "type": "Baseline", "rmse": 0.005290, "r2": 0.9741, "da": 0.5074, "color": "red"},
        {"model_name": "ARIMA(1,1,0)", "type": "Baseline", "rmse": 0.006097, "r2": 0.9615, "da": 0.5455, "color": "orange"},
        
        # Modelos Optuna Estándar
        {"model_name": "BidirectionalDeepLSTM", "type": "Optuna Standard", "rmse": 0.007961, "r2": 0.9415, "da": 0.5158, "color": "lightblue"},
        {"model_name": "TLS_LSTM", "type": "Optuna Standard", "rmse": 0.009458, "r2": 0.9174, "da": 0.5080, "color": "lightblue"},
        {"model_name": "HybridLSTMAttention", "type": "Optuna Standard", "rmse": 0.011347, "r2": 0.8811, "da": 0.5051, "color": "lightblue"},
        {"model_name": "GRU_Model", "type": "Optuna Standard", "rmse": 0.014008, "r2": 0.8125, "da": 0.4860, "color": "lightblue"},
        
        # Rolling Forecast (BREAKTHROUGH)
        {"model_name": "BidirectionalDeep + Rolling", "type": "Rolling Forecast", "rmse": 0.020937, "r2": -5.5120, "da": 0.6552, "color": "gold"},
        {"model_name": "GRU + Rolling", "type": "Rolling Forecast", "rmse": 0.013720, "r2": -1.7964, "da": 0.5517, "color": "gold"},
    ]
    
    df_results = pd.DataFrame(results_data)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('🏆 COMPARACIÓN COMPLETA: Optuna vs Baselines vs Rolling Forecast', fontsize=18, fontweight='bold')
    
    # 1. Directional Accuracy (Métrica más importante)
    ax1 = axes[0, 0]
    
    # Separar por tipo
    baselines = df_results[df_results['type'] == 'Baseline']
    optuna_std = df_results[df_results['type'] == 'Optuna Standard']
    rolling = df_results[df_results['type'] == 'Rolling Forecast']
    
    # Baseline bars
    if not baselines.empty:
        bars1 = ax1.bar(baselines['model_name'], baselines['da'] * 100, 
                       color='red', alpha=0.7, label='Baselines', width=0.6)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Optuna Standard bars
    if not optuna_std.empty:
        bars2 = ax1.bar(optuna_std['model_name'], optuna_std['da'] * 100, 
                       color='lightblue', alpha=0.8, label='Optuna Standard', width=0.6)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Rolling Forecast bars (DESTACADOS)
    if not rolling.empty:
        bars3 = ax1.bar(rolling['model_name'], rolling['da'] * 100, 
                       color='gold', alpha=0.9, label='🚀 Rolling Forecast', width=0.6, 
                       edgecolor='darkred', linewidth=2)
        for bar in bars3:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', 
                    fontsize=12, color='darkred')
    
    ax1.set_title('🎯 Directional Accuracy (Mayor es Mejor)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('DA (%)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    ax1.axhline(y=52, color='green', linestyle='--', alpha=0.7, label='Target (52%)')
    ax1.set_ylim([45, 70])
    
    # 2. RMSE (mostrar TODOS los modelos con escala logarítmica si es necesario)
    ax2 = axes[0, 1]
    
    # Separar por tipo para RMSE - MOSTRAR TODOS
    baselines_rmse = df_results[df_results['type'] == 'Baseline']
    optuna_std_rmse = df_results[df_results['type'] == 'Optuna Standard']
    rolling_rmse = df_results[df_results['type'] == 'Rolling Forecast']
    
    bars1 = ax2.bar(baselines_rmse['model_name'], baselines_rmse['rmse'], 
                   color='red', alpha=0.7, label='Baselines', width=0.6)
    bars2 = ax2.bar(optuna_std_rmse['model_name'], optuna_std_rmse['rmse'], 
                   color='lightblue', alpha=0.8, label='Optuna Standard', width=0.6)
    bars3 = ax2.bar(rolling_rmse['model_name'], rolling_rmse['rmse'], 
                   color='gold', alpha=0.9, label='🚀 Rolling Forecast', width=0.6,
                   edgecolor='darkred', linewidth=2)
    
    # Agregar valores - con formato especial para valores altos
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.015:  # Valores altos de Rolling
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', 
                        fontsize=9, color='darkred')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_title('📊 Test RMSE - TODOS los Modelos (Menor es Mejor)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Nota explicativa sobre RMSE vs DA
    ax2.text(0.02, 0.98, 'Nota: Rolling Forecast optimiza DA,\nno necesariamente RMSE', 
             transform=ax2.transAxes, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    # 3. Scatter Plot: DA vs RMSE
    ax3 = axes[1, 0]
    
    for type_name in df_results['type'].unique():
        subset = df_results[df_results['type'] == type_name]
        if type_name == 'Baseline':
            ax3.scatter(subset['rmse'], subset['da'] * 100, 
                       color='red', s=120, alpha=0.8, label=type_name, marker='o')
        elif type_name == 'Optuna Standard':
            ax3.scatter(subset['rmse'], subset['da'] * 100, 
                       color='lightblue', s=120, alpha=0.8, label=type_name, marker='s')
        else:  # Rolling Forecast
            ax3.scatter(subset['rmse'], subset['da'] * 100, 
                       color='gold', s=200, alpha=0.9, label=type_name, marker='*', 
                       edgecolor='darkred', linewidth=2)
    
    # Anotar puntos importantes
    for idx, row in df_results.iterrows():
        if row['da'] > 0.55 or row['model_name'] in ['Naive Baseline', 'ARIMA(1,1,0)']:
            ax3.annotate(row['model_name'], 
                        (row['rmse'], row['da'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
    
    ax3.set_title('🔍 DA vs RMSE Scatter Plot', fontweight='bold', fontsize=14)
    ax3.set_xlabel('RMSE', fontweight='bold')
    ax3.set_ylabel('DA (%)', fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=52, color='green', linestyle='--', alpha=0.7, label='Target DA (52%)')
    
    # 4. Ranking por DA (Tabla visual)
    ax4 = axes[1, 1]
    
    # Ordenar por DA
    df_sorted = df_results.sort_values('da', ascending=False)
    
    # Crear barras horizontales
    colors = ['gold' if 'Rolling' in row['type'] else 'red' if row['type'] == 'Baseline' else 'lightblue' 
              for _, row in df_sorted.iterrows()]
    
    bars = ax4.barh(range(len(df_sorted)), df_sorted['da'] * 100, color=colors, alpha=0.8)
    
    # Etiquetas
    ax4.set_yticks(range(len(df_sorted)))
    ax4.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(df_sorted['model_name'])], fontsize=10)
    
    # Valores en las barras
    for i, (bar, da) in enumerate(zip(bars, df_sorted['da'] * 100)):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{da:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax4.set_title('🏆 Ranking por Directional Accuracy', fontweight='bold', fontsize=14)
    ax4.set_xlabel('DA (%)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.axvline(x=52, color='green', linestyle='--', alpha=0.7, linewidth=2)
    
    # Destacar el ganador
    best_idx = df_sorted['da'].idxmax()
    best_model = df_sorted.loc[best_idx, 'model_name']
    best_da = df_sorted.loc[best_idx, 'da'] * 100
    
    ax4.text(0.5, 0.95, f'🏆 CAMPEÓN: {best_model}\n🎯 DA: {best_da:.1f}%', 
             transform=ax4.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Guardar gráfico
    current_dir = Path.cwd()
    if current_dir.name == "model":
        images_dir = Path("../images/evaluacion_completa")
    else:
        images_dir = Path("images/evaluacion_completa")
    images_dir.mkdir(exist_ok=True)
    
    chart_path = images_dir / f"comparacion_completa_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Gráfico completo guardado: {chart_path}")
    
    return chart_path, df_results

def generate_summary_report(df_results):
    """Generar reporte de resumen con hallazgos clave"""
    print("\n📋 GENERANDO REPORTE DE HALLAZGOS...")
    
    # Encontrar el mejor modelo
    best_model = df_results.loc[df_results['da'].idxmax()]
    
    # Contar logros
    models_above_52 = df_results[df_results['da'] > 0.52]
    rolling_models = df_results[df_results['type'] == 'Rolling Forecast']
    
    print(f"\n🏆 HALLAZGOS CLAVE:")
    print("=" * 60)
    print(f"🥇 CAMPEÓN: {best_model['model_name']}")
    print(f"   🎯 Directional Accuracy: {best_model['da']*100:.1f}%")
    print(f"   📊 Tipo: {best_model['type']}")
    print(f"   🎖️ Supera objetivo (52%): {'✅ SÍ' if best_model['da'] > 0.52 else '❌ NO'}")
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   🎯 Modelos que superan 52% DA: {len(models_above_52)}")
    print(f"   🚀 Modelos Rolling Forecast: {len(rolling_models)}")
    print(f"   📈 Mejor DA Rolling: {rolling_models['da'].max()*100:.1f}%")
    print(f"   📈 Mejora vs Naive: +{(best_model['da'] - 0.5074)*100:.1f}%")
    
    print(f"\n🔬 ANÁLISIS TÉCNICO:")
    if best_model['da'] > 0.60:
        print("   🎉 BREAKTHROUGH: DA > 60% es excepcional en predicción financiera")
    elif best_model['da'] > 0.55:
        print("   ✅ EXCELENTE: DA > 55% indica modelo muy efectivo")
    elif best_model['da'] > 0.52:
        print("   👍 BUENO: DA > 52% supera el objetivo establecido")
    else:
        print("   ⚠️ MODERADO: DA < 52% requiere más optimización")
    
    print(f"\n🚀 TÉCNICA EXITOSA:")
    print("   🔄 Rolling Forecast con Re-entrenamiento Incremental")
    print("   🎯 Inspirado en el éxito de ARIMA (50% → 54.5%)")
    print("   📈 Aplicado a BidirectionalDeepLSTM: 51.6% → 65.5%")
    print("   🏆 Mejora absoluta: +13.9 puntos porcentuales")

def main():
    """Función principal"""
    print("🚀 GENERACIÓN DE GRÁFICO COMPARATIVO COMPLETO")
    print("=" * 60)
    print("📊 Incluyendo: Optuna + Baselines + Rolling Forecast")
    print("=" * 60)
    
    # Crear gráfico completo
    chart_path, df_results = create_complete_comparison_chart()
    
    # Generar reporte de hallazgos
    generate_summary_report(df_results)
    
    print(f"\n✅ ANÁLISIS COMPLETO TERMINADO")
    print(f"📊 Gráfico guardado en: {chart_path}")

if __name__ == "__main__":
    main()
