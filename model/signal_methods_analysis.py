#!/usr/bin/env python3
"""
signal_methods_analysis.py - Análisis Comparativo de Métodos de Señales
Determina qué método funciona mejor: threshold, directional o hybrid
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_latest_backtest_results():
    """Cargar los resultados más recientes de backtesting"""
    results_dir = Path("modelos/eur_usd")
    result_files = list(results_dir.glob("backtest_results_*.json"))
    
    if not result_files:
        print("❌ No se encontraron resultados de backtesting")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Cargando desde: {latest_file.name}")
    return data

def analyze_signal_methods():
    """Análisis comprensivo de métodos de señales por modelo"""
    print("🔍 ANÁLISIS COMPRENSIVO DE MÉTODOS DE SEÑALES")
    print("=" * 80)
    
    # Cargar datos
    data = load_latest_backtest_results()
    if not data:
        return
    
    results = data['results']
    
    # Filtrar solo modelos ML (excluir Baselines y Rolling Forecast)
    ml_results = {}
    for name, result in results.items():
        if result['type'] == 'ML Model' and 'signal_method' in result:
            ml_results[name] = result
    
    # Organizar por modelo y método
    models_analysis = {}
    
    for strategy_name, result in ml_results.items():
        # Extraer modelo base y método
        parts = strategy_name.split('_')
        if len(parts) >= 2:
            model_name = '_'.join(parts[:-1])  # Todo excepto el último elemento
            signal_method = parts[-1]
            
            if model_name not in models_analysis:
                models_analysis[model_name] = {}
            
            models_analysis[model_name][signal_method] = result['metrics']
    
    print(f"\n📊 MODELOS ANALIZADOS: {len(models_analysis)}")
    for model in models_analysis.keys():
        print(f"   🤖 {model}")
    
    # Análisis por modelo
    print(f"\n🏆 RANKING POR MODELO Y MÉTODO")
    print("=" * 100)
    print(f"{'Modelo':<35} {'Método':<12} {'Retorno':<8} {'Sharpe':<8} {'DD':<8} {'WinRate':<8} {'Trades':<8}")
    print("-" * 100)
    
    all_model_results = []
    
    for model_name, methods in models_analysis.items():
        model_results = []
        
        for method, metrics in methods.items():
            model_results.append({
                'model': model_name,
                'method': method,
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'trades': metrics['num_trades']
            })
            all_model_results.append({
                'model': model_name,
                'method': method,
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'trades': metrics['num_trades']
            })
        
        # Ordenar por retorno dentro del modelo
        model_results.sort(key=lambda x: x['return'], reverse=True)
        
        for i, result in enumerate(model_results):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"{emoji}{result['model']:<34} {result['method']:<12} "
                  f"{result['return']:<8.1f}% {result['sharpe']:<8.3f} "
                  f"{result['max_dd']:<8.1f}% {result['win_rate']:<8.1f}% "
                  f"{result['trades']:<8}")
        print()
    
    # Estadísticas por método
    print(f"\n📊 ESTADÍSTICAS GENERALES POR MÉTODO")
    print("=" * 80)
    
    method_stats = {}
    for result in all_model_results:
        method = result['method']
        if method not in method_stats:
            method_stats[method] = {
                'returns': [],
                'sharpes': [],
                'win_rates': [],
                'max_dds': [],
                'trades': []
            }
        
        method_stats[method]['returns'].append(result['return'])
        method_stats[method]['sharpes'].append(result['sharpe'])
        method_stats[method]['win_rates'].append(result['win_rate'])
        method_stats[method]['max_dds'].append(abs(result['max_dd']))
        method_stats[method]['trades'].append(result['trades'])
    
    print(f"{'Método':<12} {'Count':<6} {'Avg_Ret':<9} {'Best_Ret':<9} {'Avg_Sharpe':<11} {'Avg_WinRate':<11} {'Winners':<8}")
    print("-" * 80)
    
    method_summary = {}
    for method, stats in method_stats.items():
        avg_return = np.mean(stats['returns'])
        best_return = np.max(stats['returns'])
        avg_sharpe = np.mean(stats['sharpes'])
        avg_win_rate = np.mean(stats['win_rates'])
        winners_count = len([r for r in stats['returns'] if r > 0])
        total_count = len(stats['returns'])
        
        method_summary[method] = {
            'avg_return': avg_return,
            'best_return': best_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'winners_count': winners_count,
            'total_count': total_count,
            'success_rate': (winners_count / total_count) * 100
        }
        
        print(f"{method:<12} {total_count:<6} {avg_return:<9.1f}% {best_return:<9.1f}% "
              f"{avg_sharpe:<11.3f} {avg_win_rate:<11.1f}% {winners_count}/{total_count:<6}")
    
    # Ranking final de métodos
    print(f"\n🏆 RANKING FINAL DE MÉTODOS DE SEÑALES")
    print("=" * 60)
    
    # Ordenar por retorno promedio
    sorted_methods = sorted(method_summary.items(), 
                           key=lambda x: x[1]['avg_return'], 
                           reverse=True)
    
    for i, (method, stats) in enumerate(sorted_methods, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{emoji} {method.upper()}")
        print(f"   💰 Retorno Promedio: {stats['avg_return']:.1f}%")
        print(f"   🚀 Mejor Resultado: {stats['best_return']:.1f}%")
        print(f"   📊 Sharpe Promedio: {stats['avg_sharpe']:.3f}")
        print(f"   🎯 Win Rate Promedio: {stats['avg_win_rate']:.1f}%")
        print(f"   ✅ Tasa de Éxito: {stats['success_rate']:.0f}% ({stats['winners_count']}/{stats['total_count']})")
        print()
    
    # Análisis específico por modelo
    print(f"\n🎯 GANADOR POR CADA MODELO")
    print("=" * 50)
    
    model_winners = {}
    for model_name, methods in models_analysis.items():
        best_method = max(methods.items(), key=lambda x: x[1]['total_return'])
        model_winners[model_name] = {
            'method': best_method[0],
            'return': best_method[1]['total_return'],
            'sharpe': best_method[1]['sharpe_ratio'],
            'win_rate': best_method[1]['win_rate']
        }
        
        print(f"🤖 {model_name}")
        print(f"   🥇 Mejor método: {best_method[0].upper()}")
        print(f"   💰 Retorno: {best_method[1]['total_return']:.1f}%")
        print(f"   📊 Sharpe: {best_method[1]['sharpe_ratio']:.3f}")
        print(f"   🎯 Win Rate: {best_method[1]['win_rate']:.1f}%")
        print()
    
    # Recomendaciones finales
    print(f"\n💡 RECOMENDACIONES FINALES")
    print("=" * 50)
    
    winner_method = sorted_methods[0][0]
    winner_stats = sorted_methods[0][1]
    
    print(f"🏆 MÉTODO GANADOR GENERAL: {winner_method.upper()}")
    print(f"   📈 Mejor rendimiento promedio: {winner_stats['avg_return']:.1f}%")
    print(f"   🎯 Mejor tasa de éxito: {winner_stats['success_rate']:.0f}%")
    print()
    
    # Contar votos por método
    method_votes = {}
    for model, winner in model_winners.items():
        method = winner['method']
        if method not in method_votes:
            method_votes[method] = 0
        method_votes[method] += 1
    
    print(f"📊 VOTACIÓN POR MODELOS:")
    for method, votes in sorted(method_votes.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method.upper()}: {votes}/{len(model_winners)} modelos lo prefieren")
    
    # Crear gráfico
    create_signal_methods_chart(all_model_results, method_summary)
    
    return method_summary, model_winners

def create_signal_methods_chart(all_results, method_summary):
    """Crear gráfico comparativo de métodos"""
    print(f"\n📊 Generando gráfico comparativo...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 ANÁLISIS COMPARATIVO DE MÉTODOS DE SEÑALES', fontsize=16, fontweight='bold')
    
    # Preparar datos
    df = pd.DataFrame(all_results)
    
    # 1. Boxplot de retornos por método
    ax1 = axes[0, 0]
    methods = ['threshold', 'directional', 'hybrid']
    returns_by_method = [df[df['method'] == method]['return'].values for method in methods]
    
    box_plot = ax1.boxplot(returns_by_method, labels=[m.capitalize() for m in methods],
                          patch_artist=True)
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Distribución de Retornos por Método')
    ax1.set_ylabel('Retorno Total (%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    ax1.legend()
    
    # 2. Promedio de retornos
    ax2 = axes[0, 1]
    method_names = list(method_summary.keys())
    avg_returns = [method_summary[method]['avg_return'] for method in method_names]
    
    bars = ax2.bar(method_names, avg_returns, color=colors)
    ax2.set_title('Retorno Promedio por Método')
    ax2.set_ylabel('Retorno Promedio (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en barras
    for bar, value in zip(bars, avg_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Tasa de éxito
    ax3 = axes[1, 0]
    success_rates = [method_summary[method]['success_rate'] for method in method_names]
    
    bars = ax3.bar(method_names, success_rates, color=colors)
    ax3.set_title('Tasa de Éxito por Método')
    ax3.set_ylabel('Porcentaje de Estrategias Rentables (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Scatter: Win Rate vs Retorno
    ax4 = axes[1, 1]
    
    method_colors = {'threshold': 'red', 'directional': 'blue', 'hybrid': 'green'}
    
    for method in methods:
        method_data = df[df['method'] == method]
        ax4.scatter(method_data['win_rate'], method_data['return'], 
                   c=method_colors[method], label=method.capitalize(), 
                   s=100, alpha=0.7)
    
    ax4.set_xlabel('Win Rate (%)')
    ax4.set_ylabel('Retorno Total (%)')
    ax4.set_title('Win Rate vs Retorno por Método')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    
    plt.tight_layout()
    
    # Guardar
    images_dir = Path("images/backtesting")
    images_dir.mkdir(exist_ok=True)
    
    chart_path = images_dir / "signal_methods_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Gráfico guardado: {chart_path}")
    
    return chart_path

def main():
    """Función principal"""
    print("🔍 ANÁLISIS COMPRENSIVO DE MÉTODOS DE SEÑALES")
    print("=" * 60)
    print("🎯 Determinando el mejor método: threshold vs directional vs hybrid")
    print("=" * 60)
    
    method_summary, model_winners = analyze_signal_methods()
    
    print(f"\n✅ ANÁLISIS COMPLETADO")
    print(f"📊 Revisar gráficos para análisis visual")

if __name__ == "__main__":
    main()
