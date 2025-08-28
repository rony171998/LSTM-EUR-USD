# live_runner.py
import time, argparse, os
from collections import deque
from datetime import datetime
from model_wrapper import LiveModel
from oanda_adapter import get_price, place_bracket_safe, get_account_equity
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configuración de la base de datos (Postgres)
DATABASE_URL = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Crear tabla de fills/trades
def create_table_if_not_exists():
    session = Session()
    try:
        sql = """
            CREATE TABLE IF NOT EXISTS fills (
                fecha TIMESTAMP PRIMARY KEY,
                accion VARCHAR(10),
                units BIGINT,
                precio DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                take_profit DOUBLE PRECISION,
                estado VARCHAR(20),
                source VARCHAR(20)
            );
        """
        session.execute(text(sql))
        session.commit()
        print("✅ Tabla 'fills' creada/verificada con éxito en Postgres")
    except Exception as e:
        session.rollback()
        print(f"🚨 Error creando/verificando tabla 'fills': {e}")
        raise
    finally:
        session.close()

def pip_to_price(pips):
    # EUR/USD pip = 0.0001
    return pips * 0.0001

def calculate_position_size(equity, risk_pct, entry_price, stop_pips, pip_value=10):
    """
    risk_pct: ejemplo 0.01 => 1% del equity
    stop_pips: pips del stop (ej 20)
    pip_value: USD por pip por lote estándar (100k) ~ 10
    Resultado: units (aprox lot units)
    """
    risk_usd = equity * risk_pct
    units = risk_usd / (stop_pips * pip_value)
    units_contract = max(0, int(units * 100000))  # convertir a unidades de contrato
    return units_contract

def save_fill(fecha, accion, units, precio, sl, tp, estado, source):
    """Guardar trade en Postgres"""
    session = Session()
    try:
        sql = text("""
            INSERT INTO fills (fecha, accion, units, precio, stop_loss, take_profit, estado, source)
            VALUES (:fecha, :accion, :units, :precio, :sl, :tp, :estado, :source)
        """)
        session.execute(sql, {
            "fecha": fecha,
            "accion": accion,
            "units": units,
            "precio": precio,
            "sl": sl,
            "tp": tp,
            "estado": estado,
            "source": source
        })
        session.commit()
    except Exception as e:
        session.rollback()
        print("🚨 Error guardando fill:", e)
    finally:
        session.close()

def main(model_path=None, dry_run=False, risk_pct=0.01, seq_len=120):
    create_table_if_not_exists()
    model = LiveModel(model_path=model_path)
    buffer = deque(maxlen=seq_len)
    print("INICIANDO LIVE RUNNER (paper/practice recomendado). Dry run:", dry_run)

    equity = get_account_equity() if not dry_run else 10000.0
    print("Equity inicial:", equity)

    while True:
        try:
            bid, ask = get_price()
            mid = (bid + ask) / 2.0
            buffer.append(mid)

            if len(buffer) < seq_len:
                print(f"warmup {len(buffer)}/{seq_len}")
                time.sleep(1 if dry_run else 30)
                continue

            signal, stop_pips, tp_pips, conf = model.predict_live(list(buffer))
            print(f"Signal={signal}, stop={stop_pips}, tp={tp_pips}, conf={conf:.2f}, price={mid:.5f}")

            if signal == 0:
                print("No trade signal -> mantener posición")
            else:
                sl = mid - pip_to_price(stop_pips) if signal == 1 else mid + pip_to_price(stop_pips)
                tp = mid + pip_to_price(tp_pips) if signal == 1 else mid - pip_to_price(tp_pips)

                equity = get_account_equity() if not dry_run else equity
                units = calculate_position_size(equity, risk_pct, mid, stop_pips)

                if units <= 0:
                    print("Tamaño 0 -> saltando")
                else:
                    accion = "BUY" if signal == 1 else "SELL"
                    print(f"Enviando orden {accion} units={units}, sl={sl:.5f}, tp={tp:.5f}")

                    if not dry_run:
                        resp = place_bracket_safe("buy" if signal==1 else "sell", units, sl=sl, tp=tp)
                        print("Order resp:", resp)
                        save_fill(datetime.utcnow(), accion, units, mid, sl, tp, "FILLED", "live")
                    else:
                        save_fill(datetime.utcnow(), accion, units, mid, sl, tp, "FAKE_FILLED", "dry_run")
                        equity -= 6 + 0.0001 * equity  # comisión + slippage simulado

            time.sleep(60 if not dry_run else 1)
        except KeyboardInterrupt:
            print("Stopping by user...")
            break
        except Exception as e:
            print("Error en loop:", e)
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="ruta al modelo .pth", default=None)
    parser.add_argument("--dry-run", action="store_true", help="no enviar órdenes, solo simular")
    parser.add_argument("--risk", type=float, default=0.01, help="riesgo por trade en fracción (0.01=1%)")
    args = parser.parse_args()
    main(model_path=args.model, dry_run=args.dry_run, risk_pct=args.risk)
