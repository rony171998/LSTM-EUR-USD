"""
Script para verificar la disponibilidad y capacidades de GPU con PyTorch
Autor: Proyecto LSTM-EUR-USD
Fecha: 2025-02-08
"""

import torch
import platform
import sys
import subprocess
import time
import numpy as np

def check_system_info():
    """Información básica del sistema"""
    print("=" * 60)
    print("🖥️  INFORMACIÓN DEL SISTEMA")
    print("=" * 60)
    print(f"Sistema Operativo: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print()

def check_cuda_availability():
    """Verificar disponibilidad de CUDA"""
    print("=" * 60)
    print("🚀 VERIFICACIÓN DE CUDA")
    print("=" * 60)
    
    # Verificar si CUDA está disponible
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {'✅ SÍ' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        print(f"Versión de CUDA: {torch.version.cuda}")
        print(f"cuDNN disponible: {'✅ SÍ' if torch.backends.cudnn.enabled else '❌ NO'}")
        if torch.backends.cudnn.enabled:
            print(f"Versión de cuDNN: {torch.backends.cudnn.version()}")
        
        # Número de dispositivos GPU
        num_gpus = torch.cuda.device_count()
        print(f"Número de GPUs: {num_gpus}")
        
        # Información de cada GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # GPU actual
        current_device = torch.cuda.current_device()
        print(f"GPU actual: {current_device}")
        print(f"Nombre GPU actual: {torch.cuda.get_device_name(current_device)}")
        
    else:
        print("💡 Posibles soluciones:")
        print("   1. Instalar PyTorch con soporte CUDA")
        print("   2. Verificar drivers de NVIDIA")
        print("   3. Verificar compatibilidad GPU-CUDA")
    
    print()

def check_device_properties():
    """Propiedades detalladas de los dispositivos"""
    if not torch.cuda.is_available():
        return
    
    print("=" * 60)
    print("🔧 PROPIEDADES DETALLADAS DE GPU")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n📱 GPU {i}: {props.name}")
        print("-" * 40)
        print(f"Memoria total: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multiprocesadores: {props.multi_processor_count}")
        print(f"Memoria compartida por bloque: {props.shared_memory_per_block / 1024:.1f} KB")
        print(f"Memoria compartida por multiprocesador: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
        print(f"Registros por bloque: {props.regs_per_block:,}")
        print(f"Registros por multiprocesador: {props.regs_per_multiprocessor:,}")
        print(f"Máximo hilos por bloque: {props.max_threads_per_block:,}")
        print(f"Máximo hilos por multiprocesador: {props.max_threads_per_multi_processor:,}")
        print(f"Capacidad computacional: {props.major}.{props.minor}")
        print(f"Frecuencia de reloj: {props.clock_rate / 1000:.0f} MHz")
        print(f"Frecuencia de memoria: {props.memory_clock_rate / 1000:.0f} MHz")
        print(f"Ancho de bus de memoria: {props.memory_bus_width} bits")

def check_memory_usage():
    """Verificar uso de memoria de GPU"""
    if not torch.cuda.is_available():
        return
    
    print("\n" + "=" * 60)
    print("💾 USO DE MEMORIA GPU")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # Limpiar cache
        torch.cuda.empty_cache()
        
        # Obtener información de memoria
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
        free_memory = total_memory - cached_memory
        
        print(f"\n📱 GPU {i}:")
        print(f"  Memoria total: {total_memory / 1024**3:.2f} GB")
        print(f"  Memoria asignada: {allocated_memory / 1024**3:.3f} GB")
        print(f"  Memoria en caché: {cached_memory / 1024**3:.3f} GB")
        print(f"  Memoria libre: {free_memory / 1024**3:.2f} GB")
        print(f"  Utilización: {(cached_memory / total_memory) * 100:.1f}%")

def performance_test():
    """Test de rendimiento básico GPU vs CPU"""
    print("\n" + "=" * 60)
    print("⚡ TEST DE RENDIMIENTO")
    print("=" * 60)
    
    # Crear datos de prueba
    size = 5000
    device_cpu = torch.device('cpu')
    
    print(f"Creando matrices {size}x{size} para prueba...")
    
    # Test CPU
    print("\n🖥️  Test CPU:")
    a_cpu = torch.randn(size, size, device=device_cpu)
    b_cpu = torch.randn(size, size, device=device_cpu)
    
    start_time = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"  Multiplicación de matrices: {cpu_time:.3f} segundos")
    
    # Test GPU (si disponible)
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda:0')
        
        print("\n🚀 Test GPU:")
        
        # Transferir datos a GPU
        transfer_start = time.time()
        a_gpu = a_cpu.to(device_gpu)
        b_gpu = b_cpu.to(device_gpu)
        transfer_time = time.time() - transfer_start
        print(f"  Transferencia CPU→GPU: {transfer_time:.3f} segundos")
        
        # Operación en GPU
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Esperar a que termine
        gpu_time = time.time() - start_time
        print(f"  Multiplicación de matrices: {gpu_time:.3f} segundos")
        
        # Transferir resultado de vuelta
        transfer_start = time.time()
        c_gpu_cpu = c_gpu.to(device_cpu)
        transfer_back_time = time.time() - transfer_start
        print(f"  Transferencia GPU→CPU: {transfer_back_time:.3f} segundos")
        
        # Comparar resultados
        difference = torch.norm(c_cpu - c_gpu_cpu).item()
        print(f"  Diferencia entre resultados: {difference:.2e}")
        
        # Speedup
        total_gpu_time = transfer_time + gpu_time + transfer_back_time
        speedup_compute = cpu_time / gpu_time
        speedup_total = cpu_time / total_gpu_time
        
        print(f"\n📊 Análisis de rendimiento:")
        print(f"  Speedup solo computación: {speedup_compute:.2f}x")
        print(f"  Speedup total (con transferencias): {speedup_total:.2f}x")
        
        if speedup_compute > 1:
            print(f"  ✅ GPU es {speedup_compute:.1f}x más rápida para computación")
        else:
            print(f"  ❌ CPU es más rápida para este tamaño de problema")
            
        if speedup_total > 1:
            print(f"  ✅ Beneficio neto: {speedup_total:.1f}x incluyendo transferencias")
        else:
            print(f"  ⚠️  Overhead de transferencias reduce beneficio")

def lstm_gpu_test():
    """Test específico con LSTM para deep learning"""
    if not torch.cuda.is_available():
        return
    
    print("\n" + "=" * 60)
    print("🧠 TEST LSTM EN GPU")
    print("=" * 60)
    
    # Parámetros del test
    batch_size = 32
    seq_length = 120
    input_size = 3
    hidden_size = 512
    
    print(f"Configuración del test:")
    print(f"  Batch size: {batch_size}")
    print(f"  Longitud secuencia: {seq_length}")
    print(f"  Dimensión entrada: {input_size}")
    print(f"  Dimensión oculta: {hidden_size}")
    
    # Crear modelo y datos
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda:0')
    
    # Test CPU
    print(f"\n🖥️  Test LSTM en CPU:")
    lstm_cpu = torch.nn.LSTM(input_size, hidden_size, batch_first=True).to(device_cpu)
    data_cpu = torch.randn(batch_size, seq_length, input_size, device=device_cpu)
    
    start_time = time.time()
    output_cpu, _ = lstm_cpu(data_cpu)
    cpu_time = time.time() - start_time
    print(f"  Forward pass: {cpu_time:.3f} segundos")
    
    # Test GPU
    print(f"\n🚀 Test LSTM en GPU:")
    lstm_gpu = torch.nn.LSTM(input_size, hidden_size, batch_first=True).to(device_gpu)
    data_gpu = data_cpu.to(device_gpu)
    
    start_time = time.time()
    output_gpu, _ = lstm_gpu(data_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"  Forward pass: {gpu_time:.3f} segundos")
    
    # Comparar
    speedup = cpu_time / gpu_time
    print(f"\n📊 Speedup LSTM: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"  ✅ GPU es {speedup:.1f}x más rápida para LSTM")
        print(f"  💡 Recomendado usar GPU para entrenamiento")
    else:
        print(f"  ⚠️  CPU es más rápida para este tamaño de LSTM")
        print(f"  💡 Considera usar CPU para esta configuración")

def check_nvidia_smi():
    """Verificar nvidia-smi si está disponible"""
    print("\n" + "=" * 60)
    print("🔧 NVIDIA SYSTEM MANAGEMENT")
    print("=" * 60)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible:")
            print(result.stdout)
        else:
            print("❌ Error ejecutando nvidia-smi:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout ejecutando nvidia-smi")
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
        print("💡 Instala drivers de NVIDIA para más información")
    except Exception as e:
        print(f"❌ Error: {e}")

def recommendations():
    """Recomendaciones basadas en la configuración"""
    print("\n" + "=" * 60)
    print("💡 RECOMENDACIONES")
    print("=" * 60)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print("✅ Configuración GPU detectada:")
        print(f"  • {num_gpus} GPU(s) disponible(s)")
        print(f"  • {gpu_memory:.1f} GB de memoria GPU")
        
        print("\n🚀 Para optimizar entrenamiento:")
        if gpu_memory >= 8:
            print("  • Usa batch_size = 32-64 para LSTM")
            print("  • Habilita mixed precision (fp16) para mayor velocidad")
            print("  • Considera modelos más grandes (hidden_size = 512-1024)")
        elif gpu_memory >= 4:
            print("  • Usa batch_size = 16-32 para LSTM")
            print("  • Considera mixed precision (fp16)")
            print("  • Mantén hidden_size = 256-512")
        else:
            print("  • Usa batch_size = 8-16 para LSTM")
            print("  • Considera entrenar en CPU para modelos pequeños")
            print("  • Limita hidden_size = 128-256")
        
        print("\n📋 Configuración recomendada para config.py:")
        print("  BATCH_SIZE = 32 if gpu_memory >= 8 else 16")
        print("  HIDDEN_SIZE = 512 if gpu_memory >= 8 else 256")
        print("  device = torch.device('cuda')")
        
    else:
        print("❌ GPU no disponible:")
        print("\n🔧 Para habilitar GPU:")
        print("  1. Instala drivers NVIDIA actualizados")
        print("  2. Reinstala PyTorch con CUDA:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  3. Verifica compatibilidad GPU-CUDA")
        
        print("\n⚙️  Configuración para CPU:")
        print("  • Usa batch_size = 8-16")
        print("  • Limita hidden_size = 128-256")
        print("  • Aumenta num_workers para DataLoader")
        print("  • Considera usar modelos más simples")

def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN COMPLETA DE GPU PARA PYTORCH")
    print("=" * 60)
    print("Este script verifica la disponibilidad y rendimiento de GPU")
    print("para entrenamiento de modelos de deep learning con PyTorch.")
    print()
    
    # Ejecutar todos los tests
    check_system_info()
    check_cuda_availability()
    check_device_properties()
    check_memory_usage()
    performance_test()
    lstm_gpu_test()
    check_nvidia_smi()
    recommendations()
    
    print("\n" + "=" * 60)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("=" * 60)
    print("Usa esta información para optimizar la configuración")
    print("de entrenamiento en tu proyecto LSTM-EUR-USD.")

if __name__ == "__main__":
    main()