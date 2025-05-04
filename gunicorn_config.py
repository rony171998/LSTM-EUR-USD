import multiprocessing
import os

# Obtener el puerto del entorno o usar 10000 por defecto
port = int(os.getenv("PORT", 10000))

# Número de workers basado en el número de CPUs
workers = multiprocessing.cpu_count() * 2 + 1

# Configuración del worker
worker_class = "uvicorn.workers.UvicornWorker"
bind = f"0.0.0.0:{port}"

# Timeouts
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Configuración de seguridad
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Configuración adicional para producción
preload_app = True
max_requests = 1000
max_requests_jitter = 50 