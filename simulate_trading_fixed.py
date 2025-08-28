import numpy as np

def simulate_trading(prices, signals, initial_capital=10000, fee_rate=0.0001, risk_fraction=0.05):
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=int)
    assert prices.ndim == 1 and signals.ndim == 1 and len(prices) == len(signals)

    cash = float(initial_capital)
    position = 0              # -1 short, 0 flat, 1 long
    position_size = 0.0       # unidades del activo
    entry_price = None

    equity_curve = np.zeros_like(prices, dtype=float)
    trades = []

    def mark_to_market(price):
        if position == 1:
            return cash + position_size * price
        elif position == -1:
            return cash - position_size * price
        else:
            return cash

    for t, (price, sig) in enumerate(zip(prices, signals)):
        if sig != position:
            if position != 0:
                notional = position_size * price
                fee = abs(notional) * fee_rate
                if position == 1:
                    pnl = position_size * (price - entry_price) - fee
                    cash += pnl + position_size * entry_price
                else:
                    pnl = position_size * (entry_price - price) - fee
                    cash += pnl - position_size * entry_price
                trades.append({
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry": float(entry_price),
                    "exit": float(price),
                    "pnl": float(pnl)
                })
                position = 0
                position_size = 0.0
                entry_price = None

            if sig != 0:
                notional = cash * float(risk_fraction)
                if notional > 0:
                    units = notional / price
                    fee = notional * fee_rate
                    if sig == 1:
                        cash -= notional + fee
                    else:
                        cash += notional - fee
                    position = int(sig)
                    position_size = float(units)
                    entry_price = float(price)

        equity_curve[t] = mark_to_market(price)
        if equity_curve[t] < 0:
            raise RuntimeError("Equity negativa: revise tamaño/margen.")

    if position != 0:
        price = float(prices[-1])
        notional = position_size * price
        fee = abs(notional) * fee_rate
        if position == 1:
            pnl = position_size * (price - entry_price) - fee
            cash += pnl + position_size * entry_price
        else:
            pnl = position_size * (entry_price - price) - fee
            cash += pnl - position_size * entry_price
        trades.append({"side": "LONG" if position == 1 else "SHORT",
                       "entry": float(entry_price), "exit": float(price), "pnl": float(pnl)})
        position = 0
        position_size = 0.0
        entry_price = None
        equity_curve[-1] = cash

    final_capital = cash
    # retorno simple por paso (evita división por cero)
    base = np.clip(np.concatenate([[initial_capital], equity_curve[:-1]]), 1e-9, None)
    returns = (equity_curve - base) / base
    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "final_capital": float(final_capital)
    }

def buy_and_hold_signals(n):
    return np.ones(n, dtype=int)
