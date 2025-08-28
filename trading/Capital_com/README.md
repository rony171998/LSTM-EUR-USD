# README.md
"""
# Bot de Trading - Capital.com

## Descripción
Bot automatizado de trading que utiliza la API de Capital.com para ejecutar operaciones basadas en análisis técnico.

## Características
- ✅ Conexión segura con Capital.com API
- 📊 Análisis técnico (SMA, RSI)
- 🎯 Gestión de riesgo automática
- 🔄 Monitoreo continuo de posiciones
- 📈 Múltiples instrumentos simultáneos
- 🛡️ Stop Loss y Take Profit automáticos

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno:
- Copiar `.env.example` a `.env`
- Agregar tus credenciales de Capital.com

3. Ejecutar el bot:
```bash
python trader.py
```

## Configuración

Edita las variables en `trader.py`:
- `RISK_PERCENTAGE`: Porcentaje de riesgo por operación
- `MAX_POSITIONS`: Número máximo de posiciones abiertas
- `INTERVAL`: Intervalo entre análisis (segundos)
- `EPICS`: Lista de instrumentos a operar

## Estrategia

El bot utiliza una estrategia simple basada en:
- **Tendencia**: Cruces de medias móviles (SMA 20/50)
- **Momentum**: RSI para identificar sobrecompra/sobreventa
- **Gestión de riesgo**: Stop loss al 2%, Take profit al 4%

## Advertencias

⚠️ **IMPORTANTE**: 
- Este es un bot educativo. Úsalo bajo tu propio riesgo.
- Prueba primero en cuenta demo.
- El trading conlleva riesgos de pérdida de capital.
- Monitorea siempre las operaciones del bot.

## Mejoras Sugeridas

1. Implementar más indicadores técnicos
2. Añadir backtesting
3. Incluir análisis de volatilidad
4. Implementar trailing stop
5. Añadir notificaciones (email/telegram)
6. Crear dashboard web para monitoreo
"""