#!/usr/bin/env python3
"""
🧪 Test simplificado: Solo NAIVE, ARIMA limitado, y GRU
"""
import sys
import os
sys.path.append('.')

from backtest_trading_models import TradingBacktester, evaluate_naive_baseline, evaluate_arima_baseline
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def test_simplified_models():
    """
    🎯 Test simplificado de modelos básicos
    """
    print("🧪 TEST SIMPLIFICADO: NAIVE + ARIMA + GRU")
    print("=" * 50)
    
    try:
        # Configuración básica
        initial_capital = 10000
        transaction_cost = 0.0001
        backtester = TradingBacktester(initial_capital, transaction_cost, [42])
        
        # Cargar datos
        eur_prices, dxy_prices = backtester.load_data()
        features_df = backtester.create_features(eur_prices, dxy_prices)
        print(f"✅ Datos cargados: {len(features_df)} registros")
        
        # Resultados
        all_results = {}
        
        # 1. NAIVE BASELINE (rápido)
        print("\n1️⃣ EVALUANDO NAIVE BASELINE")
        try:
            naive_results = evaluate_naive_baseline(backtester, features_df, 20)  # Solo 20 predicciones
            all_results.update(naive_results)
            print(f"   ✅ NAIVE completado: {len(naive_results)} estrategias")
        except Exception as e:
            print(f"   ❌ Error en NAIVE: {e}")
        
        # 2. ARIMA BASELINE (limitado)
        print("\n2️⃣ EVALUANDO ARIMA BASELINE")
        try:
            arima_results = evaluate_arima_baseline(backtester, features_df, 20)  # Solo 20 predicciones
            all_results.update(arima_results)
            print(f"   ✅ ARIMA completado: {len(arima_results)} estrategias")
        except Exception as e:
            print(f"   ❌ Error en ARIMA: {e}")
        
        # 3. Verificar que tenemos resultados
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   💼 Total estrategias evaluadas: {len(all_results)}")
        
        for strategy_name, result in all_results.items():
            metrics = result.get('metrics', {})
            final_capital = metrics.get('final_capital', 0)
            total_return = metrics.get('total_return', 0)
            print(f"   📈 {strategy_name}: ${final_capital:,.2f} ({total_return:+.2f}%)")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    results = test_simplified_models()
