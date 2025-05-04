import multiprocessing

# Número de workers basado en el número de CPUs
workers = multiprocessing.cpu_count() * 2 + 1

# Configuración del worker
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"

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