# trader.py
"""
Script principal para ejecutar el bot de trading con ML integrado con sistema de logs
"""
from capital_adapter import MLTradingBot, TradingBot
from save_data_trading import get_trading_summary, get_open_positions, log_message
import sys
from pathlib import Path

def main():
    # Selección del modo de trading
    print("🤖 SISTEMA DE TRADING CAPITAL.COM")
    print("=" * 40)

    run_ml_trading()

def run_ml_trading():
    """Ejecutar bot con modelo ML y Rolling Forecast"""
    
    print("\n🧠 CONFIGURACIÓN DE TRADING ML")
    print("=" * 35)
    
    # Configuración ML
    MODEL_PATH = "modelos/eur_usd/BidirectionalDeepLSTMModel_optuna_EUR_USD_2010-2024.csv.pth"
    
    RISK_PERCENTAGE = 1.0
    MAX_POSITIONS = 1
    INTERVAL_MINUTES = 60
    INITIAL_CAPITAL = 11000
    
    RETRAIN_FREQUENCY = 10  # Re-entrenar cada 10 días
    
    # Verificar modelo
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {MODEL_PATH}")
        print("Buscando modelos disponibles...")
        
        models_dir = Path("modelos")
        if models_dir.exists():
            pth_files = list(models_dir.glob("**/*.pth"))
            if pth_files:
                print("\n📁 Modelos disponibles:")
                for i, file in enumerate(pth_files, 1):
                    print(f"{i}. {file}")
                
                try:
                    choice = input("\nSeleccione número de modelo (o Enter para salir): ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(pth_files):
                        model_path = pth_files[int(choice) - 1]
                    else:
                        print("👋 Saliendo...")
                        return
                except (ValueError, IndexError):
                    print("❌ Selección inválida")
                    return
            else:
                print("❌ No se encontraron modelos .pth")
                return
        else:
            print("❌ Directorio 'modelos' no encontrado")
            return
    
    # Lista de instrumentos
    print("\n📈 Selección de instrumentos:")
    print("1. Solo EUR/USD (Recomendado para inicio)")
    print("2. EUR/USD + GBP/USD")
    print("3. Majors (EUR/USD, GBP/USD, USD/JPY)")
    
    EPICS = ["EURUSD"]
    
    # Resumen de configuración
    print(f"\n🚀 CONFIGURACIÓN FINAL:")
    print(f"   • Modelo: {model_path.name}")
    print(f"   • Capital inicial: ${INITIAL_CAPITAL:,.2f}")
    print(f"   • Riesgo por operación: {RISK_PERCENTAGE}%")
    print(f"   • Máximo posiciones: {MAX_POSITIONS}")
    print(f"   • Intervalo análisis: {INTERVAL_MINUTES} min")
    print(f"   • Instrumentos: {', '.join(EPICS)}")
    
    confirm = input("\n¿Continuar con esta configuración? (s/N): ").strip().lower()
    if confirm != 's':
        print("❌ Operación cancelada")
        return
    
    # Crear instancia del bot ML
    print(f"\n🤖 Iniciando bot ML...")
    bot = MLTradingBot(
        model_path=str(model_path),
        initial_capital=INITIAL_CAPITAL,
        risk_percentage=RISK_PERCENTAGE,
        max_positions=MAX_POSITIONS,
        retrain_frequency=RETRAIN_FREQUENCY
    )
    
    # Inicializar
    print("🔄 Inicializando sistemas...")
    if not bot.initialize():
        print("❌ Error al inicializar el bot ML")
        input("Presione Enter para continuar...")
        return
    
    # Mostrar resumen actual antes de empezar
    print("\n📊 Estado actual del trading:")
    summary = get_trading_summary(days=7)
    if "error" not in summary and summary.get('total_trades', 0) > 0:
        print(f"   • Trades últimos 7 días: {summary['total_trades']}")
        print(f"   • Win rate: {summary['win_rate']}%")
        print(f"   • P&L: ${summary['total_pnl']:.2f}")
    else:
        print("   • No hay historial de trading reciente")
    
    try:
        print(f"\n🎯 Bot iniciado exitosamente!")
        print("🔄 Comenzando estrategia de trading...")
        print("📝 Presione Ctrl+C para detener el bot de forma segura")
        print("=" * 50)
        
        log_message(f"🚀 Bot ML iniciado - Instrumentos: {EPICS}, Riesgo: {RISK_PERCENTAGE}%")
        
        # Ejecutar estrategia
        bot.run_strategy(
            epics=EPICS,
            interval_minutes=INTERVAL_MINUTES
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo bot por solicitud del usuario...")
        bot.shutdown()
        print("👋 Bot ML detenido. ¡Hasta luego!")
        
    except Exception as e:
        error_msg = f"❌ Error crítico: {str(e)}"
        print(error_msg)
        log_message(error_msg)
        input("Presione Enter para continuar...")

if __name__ == "__main__":
    try:
        print("🔄 Iniciando script principal...")
        main()
    except Exception as e:
        print(f"❌ Error crítico en el script: {e}")
        sys.exit(1)