#!/usr/bin/env python3
"""
advanced_backtest_analysis.py - Análisis Avanzado de Backtesting
Análisis profundo de las mejores estrategias identificadas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_backtest_results():
    """Cargar resultados del backtesting"""
    results_dir = Path("modelos/eur_usd")
    
    # Buscar el archivo más reciente
    result_files = list(results_dir.glob("backtest_results_*.json"))
    if not result_files:
        print("❌ No se encontraron resultados de backtesting")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Cargando resultados desde: {latest_file.name}")
    return data

def analyze_top_strategies(results_data):
    """Análisis detallado de las mejores estrategias"""
    print("\n🔍 ANÁLISIS PROFUNDO DE ESTRATEGIAS TOP")
    print("=" * 60)
    
    results = results_data['results']
    
    # Filtrar solo las mejores (retorno > 0%)
    profitable_strategies = {
        name: data for name, data in results.items() 
        if data['metrics']['total_return'] > 0
    }
    
    print(f"📈 Estrategias rentables: {len(profitable_strategies)}/{len(results)}")
    
    # Top 5 por retorno
    top_5 = sorted(profitable_strategies.items(), 
                   key=lambda x: x[1]['metrics']['total_return'], 
                   reverse=True)[:5]
    
    print(f"\n🏆 TOP 5 ESTRATEGIAS RENTABLES:")
    print("-" * 60)
    
    for i, (name, data) in enumerate(top_5, 1):
        metrics = data['metrics']
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        
        print(f"{emoji} {name}")
        print(f"   💰 Retorno Total: {metrics['total_return']:.1f}%")
        print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"   🎯 Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   🔄 Trades: {metrics['num_trades']}")
        print(f"   💵 Capital Final: ${metrics['final_capital']:,.2f}")
        print()
    
    return top_5, profitable_strategies

def risk_adjusted_analysis(profitable_strategies):
    """Análisis ajustado por riesgo"""
    print("⚖️ ANÁLISIS AJUSTADO POR RIESGO")
    print("=" * 60)
    
    # Crear DataFrame para análisis
    analysis_data = []
    for name, data in profitable_strategies.items():
        metrics = data['metrics']
        analysis_data.append({
            'Strategy': name,
            'Type': data['type'],
            'Return': metrics['total_return'],
            'Sharpe': metrics['sharpe_ratio'],
            'Max_DD': abs(metrics['max_drawdown']),
            'Win_Rate': metrics['win_rate'],
            'Profit_Factor': metrics['profit_factor'],
            'Trades': metrics['num_trades']
        })
    
    df = pd.DataFrame(analysis_data)
    
    # Calcular Score Compuesto (Risk-Adjusted Performance Score)
    # Fórmula: (Return * Win_Rate * Sharpe) / Max_DD
    df['Risk_Score'] = (df['Return'] * df['Win_Rate'] * np.maximum(df['Sharpe'], 0.01)) / np.maximum(df['Max_DD'], 1)
    
    # Ranking por Risk Score
    df_sorted = df.sort_values('Risk_Score', ascending=False)
    
    print("🎯 RANKING POR SCORE AJUSTADO POR RIESGO:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Estrategia':<35} {'Score':<8} {'Ret%':<6} {'Sharpe':<7} {'DD%':<6}")
    print("-" * 80)
    
    for i, row in df_sorted.head(10).iterrows():
        print(f"{df_sorted.index.get_loc(i)+1:<4} {row['Strategy'][:34]:<35} "
              f"{row['Risk_Score']:<8.2f} {row['Return']:<6.1f} "
              f"{row['Sharpe']:<7.3f} {row['Max_DD']:<6.1f}")
    
    # Mejores por categoría
    print(f"\n📊 MEJORES POR CATEGORÍA:")
    print("-" * 40)
    
    best_return = df.loc[df['Return'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    best_win_rate = df.loc[df['Win_Rate'].idxmax()]
    lowest_dd = df.loc[df['Max_DD'].idxmin()]
    
    print(f"🚀 Mayor Retorno: {best_return['Strategy'][:30]} ({best_return['Return']:.1f}%)")
    print(f"⚡ Mejor Sharpe: {best_sharpe['Strategy'][:30]} ({best_sharpe['Sharpe']:.3f})")
    print(f"🎯 Mayor Win Rate: {best_win_rate['Strategy'][:30]} ({best_win_rate['Win_Rate']:.1f}%)")
    print(f"🛡️ Menor Drawdown: {lowest_dd['Strategy'][:30]} ({lowest_dd['Max_DD']:.1f}%)")
    
    return df_sorted

def model_type_comparison(results_data):
    """Comparación por tipo de modelo"""
    print(f"\n🔬 COMPARACIÓN POR TIPO DE MODELO")
    print("=" * 60)
    
    results = results_data['results']
    
    # Agrupar por tipo
    type_stats = {}
    for name, data in results.items():
        model_type = data['type']
        if model_type not in type_stats:
            type_stats[model_type] = {
                'strategies': [],
                'returns': [],
                'sharpes': [],
                'win_rates': [],
                'drawdowns': []
            }
        
        metrics = data['metrics']
        type_stats[model_type]['strategies'].append(name)
        type_stats[model_type]['returns'].append(metrics['total_return'])
        type_stats[model_type]['sharpes'].append(metrics['sharpe_ratio'])
        type_stats[model_type]['win_rates'].append(metrics['win_rate'])
        type_stats[model_type]['drawdowns'].append(abs(metrics['max_drawdown']))
    
    # Estadísticas por tipo
    print(f"{'Tipo':<20} {'Count':<6} {'Avg_Ret%':<10} {'Best_Ret%':<10} {'Avg_Sharpe':<10} {'Avg_WinRate%':<12}")
    print("-" * 78)
    
    for model_type, stats in type_stats.items():
        avg_return = np.mean(stats['returns'])
        best_return = np.max(stats['returns'])
        avg_sharpe = np.mean(stats['sharpes'])
        avg_win_rate = np.mean(stats['win_rates'])
        count = len(stats['strategies'])
        
        print(f"{model_type:<20} {count:<6} {avg_return:<10.1f} {best_return:<10.1f} "
              f"{avg_sharpe:<10.3f} {avg_win_rate:<12.1f}")
    
    # Mejor estrategia por tipo
    print(f"\n🏆 CAMPEÓN POR TIPO:")
    print("-" * 50)
    
    for model_type, stats in type_stats.items():
        best_idx = np.argmax(stats['returns'])
        best_strategy = stats['strategies'][best_idx]
        best_return = stats['returns'][best_idx]
        
        print(f"{model_type}: {best_strategy[:35]} ({best_return:.1f}%)")
    
    return type_stats

def signal_method_analysis(results_data):
    """Análisis por método de señales"""
    print(f"\n📡 ANÁLISIS POR MÉTODO DE SEÑALES")
    print("=" * 60)
    
    results = results_data['results']
    
    # Agrupar por método de señal
    signal_stats = {}
    for name, data in results.items():
        if 'signal_method' in data and data['signal_method'] != 'N/A':
            signal_method = data['signal_method']
            if signal_method not in signal_stats:
                signal_stats[signal_method] = {
                    'returns': [],
                    'sharpes': [],
                    'win_rates': [],
                    'strategies': []
                }
            
            metrics = data['metrics']
            signal_stats[signal_method]['returns'].append(metrics['total_return'])
            signal_stats[signal_method]['sharpes'].append(metrics['sharpe_ratio'])
            signal_stats[signal_method]['win_rates'].append(metrics['win_rate'])
            signal_stats[signal_method]['strategies'].append(name)
    
    print(f"{'Método':<15} {'Count':<6} {'Avg_Ret%':<10} {'Best_Ret%':<10} {'Success_Rate%':<13}")
    print("-" * 64)
    
    for method, stats in signal_stats.items():
        avg_return = np.mean(stats['returns'])
        best_return = np.max(stats['returns'])
        profitable_count = len([r for r in stats['returns'] if r > 0])
        success_rate = (profitable_count / len(stats['returns'])) * 100
        
        print(f"{method:<15} {len(stats['strategies']):<6} {avg_return:<10.1f} "
              f"{best_return:<10.1f} {success_rate:<13.1f}")
    
    # Recomendación
    print(f"\n💡 RECOMENDACIONES POR MÉTODO:")
    print("-" * 40)
    
    best_method = max(signal_stats.keys(), 
                     key=lambda x: np.mean(signal_stats[x]['returns']))
    
    most_consistent = max(signal_stats.keys(),
                         key=lambda x: len([r for r in signal_stats[x]['returns'] if r > 0]) / len(signal_stats[x]['returns']))
    
    print(f"🚀 Mejor Rendimiento: {best_method}")
    print(f"🎯 Más Consistente: {most_consistent}")
    
    return signal_stats

def create_comprehensive_charts(results_data, df_analysis):
    """Crear gráficos comprensivos del análisis"""
    print(f"\n📊 Generando gráficos de análisis comprensivo...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('📊 ANÁLISIS COMPRENSIVO DE BACKTESTING', fontsize=16, fontweight='bold')
    
    # 1. Risk Score vs Return
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_analysis['Risk_Score'], df_analysis['Return'], 
                         c=df_analysis['Sharpe'], cmap='viridis', s=100, alpha=0.7)
    
    # Top 3 anotados
    top_3 = df_analysis.head(3)
    for _, row in top_3.iterrows():
        ax1.annotate(row['Strategy'][:15], 
                    (row['Risk_Score'], row['Return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Risk-Adjusted Score')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Risk Score vs Return (Color = Sharpe)')
    plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Rate vs Max Drawdown
    ax2 = axes[0, 1]
    colors = ['red' if t == 'Baseline' else 'blue' if t == 'ML Model' else 'gold' 
             for t in df_analysis['Type']]
    
    ax2.scatter(df_analysis['Max_DD'], df_analysis['Win_Rate'], 
               c=colors, s=100, alpha=0.7)
    
    ax2.set_xlabel('Max Drawdown (%)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Risk vs Accuracy')
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Performance por Tipo
    ax3 = axes[0, 2]
    type_performance = df_analysis.groupby('Type')['Return'].agg(['mean', 'max', 'count'])
    
    bars = ax3.bar(type_performance.index, type_performance['mean'], 
                  color=['red', 'blue', 'gold'], alpha=0.7, label='Promedio')
    
    # Máximos como puntos
    ax3.scatter(range(len(type_performance)), type_performance['max'], 
               color='darkred', s=100, label='Mejor', zorder=5)
    
    ax3.set_title('Rendimiento por Tipo de Modelo')
    ax3.set_ylabel('Retorno Total (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Rotar labels
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Top 10 Estrategias
    ax4 = axes[1, 0]
    top_10 = df_analysis.head(10)
    
    bars = ax4.barh(range(len(top_10)), top_10['Return'], 
                   color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10))))
    
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels([f"{i+1}. {name[:20]}" for i, name in enumerate(top_10['Strategy'])])
    ax4.set_xlabel('Retorno Total (%)')
    ax4.set_title('🏆 Top 10 Estrategias')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Análisis de Trades
    ax5 = axes[1, 1]
    
    # Número de trades vs Win Rate
    ax5.scatter(df_analysis['Trades'], df_analysis['Win_Rate'], 
               c=df_analysis['Return'], cmap='RdYlGn', s=100, alpha=0.7)
    
    ax5.set_xlabel('Número de Trades')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_title('Actividad vs Precisión')
    ax5.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    
    # 6. Rolling Forecast vs Others
    ax6 = axes[1, 2]
    
    rolling_strategies = df_analysis[df_analysis['Type'] == 'Rolling Forecast']
    other_strategies = df_analysis[df_analysis['Type'] != 'Rolling Forecast']
    
    ax6.hist([other_strategies['Return'], rolling_strategies['Return']], 
            bins=15, alpha=0.7, label=['Otros Modelos', 'Rolling Forecast'],
            color=['lightblue', 'gold'])
    
    ax6.set_xlabel('Retorno Total (%)')
    ax6.set_ylabel('Frecuencia')
    ax6.set_title('Distribución de Retornos')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    images_dir = Path("images/backtesting")
    images_dir.mkdir(exist_ok=True)
    
    comprehensive_path = images_dir / "comprehensive_backtest_analysis.png"
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Análisis comprensivo guardado: {comprehensive_path}")
    
    return comprehensive_path

def generate_final_recommendations(df_analysis, type_stats, signal_stats):
    """Generar recomendaciones finales"""
    print(f"\n🎯 RECOMENDACIONES FINALES")
    print("=" * 60)
    
    # Mejor estrategia general
    best_overall = df_analysis.iloc[0]
    
    print(f"🏆 ESTRATEGIA RECOMENDADA PRINCIPAL:")
    print(f"   📊 {best_overall['Strategy']}")
    print(f"   💰 Retorno: {best_overall['Return']:.1f}%")
    print(f"   ⚖️ Risk Score: {best_overall['Risk_Score']:.2f}")
    print(f"   🎯 Win Rate: {best_overall['Win_Rate']:.1f}%")
    print(f"   📉 Max DD: {best_overall['Max_DD']:.1f}%")
    
    # Estrategia conservadora
    conservative = df_analysis.loc[df_analysis['Max_DD'].idxmin()]
    
    print(f"\n🛡️ ESTRATEGIA CONSERVADORA:")
    print(f"   📊 {conservative['Strategy']}")
    print(f"   💰 Retorno: {conservative['Return']:.1f}%")
    print(f"   📉 Max DD: {conservative['Max_DD']:.1f}% (Menor Riesgo)")
    print(f"   🎯 Win Rate: {conservative['Win_Rate']:.1f}%")
    
    # Mejor modelo ML
    ml_models = df_analysis[df_analysis['Type'] == 'ML Model']
    if not ml_models.empty:
        best_ml = ml_models.iloc[0]
        print(f"\n🤖 MEJOR MODELO ML:")
        print(f"   📊 {best_ml['Strategy']}")
        print(f"   💰 Retorno: {best_ml['Return']:.1f}%")
        print(f"   🎯 Win Rate: {best_ml['Win_Rate']:.1f}%")
    
    # Rolling Forecast
    rolling_models = df_analysis[df_analysis['Type'] == 'Rolling Forecast']
    if not rolling_models.empty:
        best_rolling = rolling_models.iloc[0]
        print(f"\n🚀 MEJOR ROLLING FORECAST:")
        print(f"   📊 {best_rolling['Strategy']}")
        print(f"   💰 Retorno: {best_rolling['Return']:.1f}%")
        print(f"   🎯 Win Rate: {best_rolling['Win_Rate']:.1f}%")
        print(f"   📉 Max DD: {best_rolling['Max_DD']:.1f}%")
    
    # Insights clave
    print(f"\n💡 INSIGHTS CLAVE:")
    print(f"   🎲 Random Trading fue sorprendentemente el mejor (¡cuidado con overfitting!)")
    print(f"   🚀 Rolling Forecast mostró buen balance riesgo/retorno")
    print(f"   📊 Método 'directional' fue más efectivo que 'threshold'")
    print(f"   ⚖️ Mayor número de trades no garantiza mejor rendimiento")
    
    # Advertencias
    print(f"\n⚠️ ADVERTENCIAS IMPORTANTES:")
    print(f"   🎯 Random Trading no es una estrategia real (sesgo de selección)")
    print(f"   💰 Costos de transacción pueden impactar significativamente")
    print(f"   📈 Backtesting no garantiza rendimiento futuro")
    print(f"   🔄 Considerar walk-forward analysis para validación")

def main():
    """Función principal del análisis avanzado"""
    print("🔬 ANÁLISIS AVANZADO DE BACKTESTING")
    print("=" * 60)
    
    # Cargar resultados
    results_data = load_backtest_results()
    if not results_data:
        return
    
    # Análisis de estrategias top
    top_5, profitable_strategies = analyze_top_strategies(results_data)
    
    # Análisis ajustado por riesgo
    df_analysis = risk_adjusted_analysis(profitable_strategies)
    
    # Comparación por tipo de modelo
    type_stats = model_type_comparison(results_data)
    
    # Análisis por método de señales
    signal_stats = signal_method_analysis(results_data)
    
    # Crear gráficos comprensivos
    chart_path = create_comprehensive_charts(results_data, df_analysis)
    
    # Recomendaciones finales
    generate_final_recommendations(df_analysis, type_stats, signal_stats)
    
    print(f"\n✅ ANÁLISIS AVANZADO COMPLETADO")
    print(f"📊 Gráficos en: {chart_path}")

if __name__ == "__main__":
    main()
