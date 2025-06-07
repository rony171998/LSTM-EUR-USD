# analisis_correlacion.py
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import DEFAULT_PARAMS
from train_model import load_and_prepare_data

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_correlations(df, main_pair="USD/COP", other_pairs=["USD/AUD", "EUR/USD", "USD/MXN", "USD/BRL"]):
    """
    Analiza correlaciones entre USD/COP y otros pares de divisas
    
    Args:
        df (DataFrame): DataFrame con los datos históricos
        main_pair (str): Par principal a analizar (default: "USD/COP")
        other_pairs (list): Lista de pares para comparar
    """
    print("\n=== Análisis de Correlación ===")
    print(f"Par principal: {main_pair}")
    
    # Verificar que el par principal existe en los datos
    if "Último" not in df.columns:
        raise ValueError("La columna 'Último' no existe en el DataFrame")
    
    # Calcular correlaciones
    correlations = {}
    for pair in other_pairs:
        try:
            # Cargar datos del par comparativo
            pair_df = pd.read_csv(
                f"data/{pair.replace('/', '_')}_2010-2024.csv",
                index_col="Fecha",
                parse_dates=True,
                dayfirst=True,
                decimal=",",
                thousands=".",
                converters={
                    "Último": lambda x: float(x.replace(".", "").replace(",", "."))
                }
            )
            
            # Alinear fechas
            aligned_df = df[["Último"]].join(pair_df[["Último"]], how='inner', lsuffix='_main', rsuffix='_pair')
            
            # Calcular correlación
            corr = aligned_df["Último_main"].corr(aligned_df["Último_pair"])
            correlations[pair] = corr
            
            # Interpretación
            print(f"\n{main_pair} vs {pair}:")
            print(f"Correlación: {corr:.4f}")
            
            if abs(corr) > 0.7:
                print("🔵 CORRELACIÓN FUERTE " + ("POSITIVA" if corr > 0 else "NEGATIVA"))
            elif abs(corr) > 0.4:
                print("🟢 Correlación moderada")
            else:
                print("🟡 Correlación débil o nula")
                
            print(f"Datos alineados: {len(aligned_df)} puntos temporales")
            
        except FileNotFoundError:
            print(f"\n⚠️ Datos no encontrados para {pair}")
            correlations[pair] = None
    
    # Visualización
    if len(correlations) > 0:
        plot_correlations(main_pair, correlations)

def plot_correlations(main_pair, correlations):
    """Visualiza las correlaciones encontradas"""
    # Filtrar pares con datos disponibles
    valid_pairs = {k: v for k, v in correlations.items() if v is not None}
    
    if not valid_pairs:
        print("\nNo hay datos suficientes para visualizar correlaciones")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in valid_pairs.values()]
    bars = plt.bar(valid_pairs.keys(), valid_pairs.values(), color=colors)
    
    plt.title(f"Correlación de {main_pair} con otros pares", fontsize=14)
    plt.ylabel("Coeficiente de correlación", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(0.7, color='blue', linestyle='--', linewidth=0.5)
    plt.axhline(-0.7, color='blue', linestyle='--', linewidth=0.5)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/correlaciones/{main_pair.replace("/", "_")}.png', dpi=300)
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    df = load_and_prepare_data(DEFAULT_PARAMS.FILEPATH)

    analyze_correlations(
        df,
        main_pair=DEFAULT_PARAMS.TICKER,
        other_pairs=["USD/AUD", "USD/COP", "GBP/USD" , "USD/CHF","EUR/USD","VIX","S&P500"]
    )
