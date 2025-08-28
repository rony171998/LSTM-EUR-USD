# analisis_correlacion.py
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import glob

from config import DEFAULT_PARAMS
from train_model import load_and_prepare_data

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_available_pairs(data_folder="data", exclude_file=None):
    """
    Obtiene automáticamente todos los pares disponibles en la carpeta data
    
    Args:
        data_folder (str): Ruta a la carpeta con los archivos CSV
        exclude_file (str): Archivo a excluir (generalmente el par principal)
    
    Returns:
        list: Lista de nombres de pares disponibles
    """
    # Buscar todos los archivos CSV en la carpeta data
    csv_files = glob.glob(os.path.join(data_folder, "*_2010-2024.csv"))
    
    pairs = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Excluir el archivo principal si se especifica
        if exclude_file and filename == exclude_file:
            continue
        
        # Extraer el nombre del par del archivo
        pair_name = filename.replace("_2010-2024.csv", "").replace("_", "/")
        
        # Convertir nombres especiales
        if pair_name == "S&P500":
            pair_name = "S&P500"
        elif pair_name == "VIX":
            pair_name = "VIX"
        elif pair_name == "ECOPETROL":
            pair_name = "ECOPETROL"
        elif pair_name == "DAX":
            pair_name = "DAX"
        
        pairs.append(pair_name)
    
    return sorted(pairs)

def analyze_correlations(df, main_pair="USD/COP", other_pairs=None):
    """
    Analiza correlaciones entre el par principal y otros pares de divisas
    
    Args:
        df (DataFrame): DataFrame con los datos históricos del par principal
        main_pair (str): Par principal a analizar
        other_pairs (list): Lista de pares para comparar. Si es None, se detectan automáticamente
    """
    print("\n=== Análisis de Correlación ===")
    print(f"Par principal: {main_pair}")
    
    # Si no se especifican pares, detectarlos automáticamente
    if other_pairs is None:
        print("🔍 Detectando pares disponibles automáticamente...")
        # Obtener el nombre del archivo principal basado en el par
        main_file = DEFAULT_PARAMS.FILEPATH
        other_pairs = get_available_pairs(data_folder="data", exclude_file=main_file)
        print(f"📊 Pares encontrados: {', '.join(other_pairs)}")
    
    # Verificar que el par principal existe en los datos
    if "Último" not in df.columns:
        raise ValueError("La columna 'Último' no existe en el DataFrame")
    
    # Calcular correlaciones
    correlations = {}
    for pair in other_pairs:
        try:
            # Construir el nombre del archivo basado en el par
            if pair in ["S&P500", "VIX", "ECOPETROL", "DAX"]:
                filename = f"{pair}_2010-2024.csv"
            else:
                filename = f"{pair.replace('/', '_')}_2010-2024.csv"
            
            # Cargar datos del par comparativo
            pair_df = pd.read_csv(
                f"data/{filename}",
                index_col="Fecha",
                parse_dates=True,
                dayfirst=True,
                decimal=",",
                thousands=".",
                converters={
                    "Último": lambda x: float(x.replace(".", "").replace(",", ".")) if isinstance(x, str) else x
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
        except Exception as e:
            print(f"\n❌ Error procesando {pair}: {str(e)}")
            correlations[pair] = None
    
    # Visualización
    if len(correlations) > 0:
        plot_correlations(main_pair, correlations)
    
    return correlations

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

    # Analizar correlaciones automáticamente detectando todos los pares disponibles
    correlations = analyze_correlations(
        df,
        main_pair=DEFAULT_PARAMS.TICKER,
        other_pairs=None  # Detectar automáticamente
    )
    
    print(f"\n📈 RESUMEN DE CORRELACIONES:")
    print("="*50)
    for pair, corr in correlations.items():
        if corr is not None:
            strength = ""
            if abs(corr) > 0.7:
                strength = "🔵 FUERTE"
            elif abs(corr) > 0.4:
                strength = "🟢 MODERADA"
            else:
                strength = "🟡 DÉBIL"
            
            direction = "+" if corr > 0 else "-"
            print(f"{DEFAULT_PARAMS.TICKER} vs {pair}: {corr:+.3f} {strength} {direction}")
        else:
            print(f"{DEFAULT_PARAMS.TICKER} vs {pair}: ❌ ERROR")
