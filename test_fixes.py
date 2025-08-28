import numpy as np
from simulate_trading_fixed import simulate_trading, buy_and_hold_signals

def check(name, equity, want_min):
    ret = (equity[-1] / equity[0]) - 1 if equity[0] != 0 else float('nan')
    print(f"{name}: final={equity[-1]:.2f}  retorno={ret*100:.2f}%")
    assert ret >= want_min, f"{name} retorno {ret:.4f} < {want_min:.4f}"

# Serie de precios simple
prices = np.array([100, 110, 120, 130, 140], dtype=float)

# 1) Buy & Hold: señales 1 todo el período
signals = buy_and_hold_signals(len(prices))
res = simulate_trading(prices, signals, initial_capital=10000, risk_fraction=0.1)
print("Trades B&H:", res["trades"])
check("Buy&Hold LONG fraccional (riesgo 10%)", res["equity_curve"], 0.0)

# 2) LONG puro de punta a punta (idéntico a B&H en señales)
res2 = simulate_trading(prices, np.ones_like(prices), initial_capital=10000, risk_fraction=0.1)
check("LONG continuo", res2["equity_curve"], 0.0)

# 3) SHORT en tendencia alcista (debería perder)
short_signals = -np.ones_like(prices)
res3 = simulate_trading(prices, short_signals, initial_capital=10000, risk_fraction=0.1)
print("Trades SHORT:", res3["trades"])
ret3 = (res3["equity_curve"][-1] / res3["equity_curve"][0]) - 1
print(f"SHORT continuo retorno={ret3*100:.2f}% (debe ser negativo)")

# 4) Switch: LONG 2 pasos, FLAT 2, LONG 1
sig4 = np.array([1,1,0,0,1])
res4 = simulate_trading(prices, sig4, initial_capital=10000, risk_fraction=0.1)
print("Trades SWITCH:", res4["trades"])
print("OK: pruebas básicas pasaron.")
