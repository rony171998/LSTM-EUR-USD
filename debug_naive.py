import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simulate a simple NAIVE strategy that should lose money
def debug_naive_simulation():
    """Debug why NAIVE shows small drawdown despite huge losses"""
    
    print("🔍 DEBUGGING NAIVE STRATEGY SIMULATION")
    print("=" * 60)
    
    # Create a scenario where price trends down (NAIVE should lose)
    initial_capital = 10000
    prices = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])  # 50% price decline
    
    # NAIVE signals: always predict last value (buy and hold)
    signals = np.ones(len(prices), dtype=int)  # All LONG
    
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Price Series: {prices}")
    print(f"Signals: {signals} (all LONG)")
    print()
    
    # Simulate trading (corrected version)
    cash = float(initial_capital)
    position = 0
    position_size = 0.0
    entry_price = None
    risk_fraction = 0.05
    fee_rate = 0.0001
    
    equity_curve = np.zeros_like(prices, dtype=float)
    trades = []
    
    def mark_to_market(price):
        if position == 1:
            return cash + position_size * price
        elif position == -1:
            return cash - position_size * price
        else:
            return cash
    
    print("STEP-BY-STEP SIMULATION:")
    print("-" * 40)
    
    for t, (price, sig) in enumerate(zip(prices, signals)):
        print(f"\nStep {t}: Price={price:.3f}, Signal={sig}")
        
        # Position change logic
        if sig != position:
            # Close existing position
            if position != 0:
                notional = position_size * price
                fee = abs(notional) * fee_rate
                if position == 1:
                    pnl = position_size * (price - entry_price) - fee
                    cash += pnl + position_size * entry_price
                else:
                    pnl = position_size * (entry_price - price) - fee
                    cash += pnl - position_size * entry_price
                
                print(f"  Closed position: PnL=${pnl:.2f}")
                position = 0
                position_size = 0.0
                entry_price = None
            
            # Open new position
            if sig != 0:
                notional = cash * risk_fraction
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
                    print(f"  Opened LONG: {units:.2f} units at ${price:.3f}")
                    print(f"  Cash remaining: ${cash:.2f}")
        
        # Mark to market
        portfolio_value = mark_to_market(price)
        equity_curve[t] = portfolio_value
        
        print(f"  Portfolio Value: ${portfolio_value:.2f}")
    
    # Close final position
    if position != 0:
        price = float(prices[-1])
        notional = position_size * price
        fee = abs(notional) * fee_rate
        if position == 1:
            pnl = position_size * (price - entry_price) - fee
            cash += pnl + position_size * entry_price
        equity_curve[-1] = cash
        print(f"\nFinal position closed: PnL=${pnl:.2f}")
    
    final_capital = cash
    total_return = (final_capital / initial_capital - 1) * 100
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Equity Curve: {equity_curve}")
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak * 100
    max_dd = dd.min()
    
    print(f"\n📉 DRAWDOWN ANALYSIS:")
    print(f"Running Peaks: {peak}")
    print(f"Drawdowns: {dd}")
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    
    # Manual verification
    max_equity = equity_curve.max()
    min_equity = equity_curve.min()
    manual_max_dd = (min_equity - max_equity) / max_equity * 100
    
    print(f"\n✅ VERIFICATION:")
    print(f"Max Equity: ${max_equity:.2f}")
    print(f"Min Equity: ${min_equity:.2f}")
    print(f"Manual Max DD: {manual_max_dd:.2f}%")
    
    # Check if drawdown matches the total loss
    expected_min_dd = total_return  # Should be similar to total return if monotonic decline
    print(f"Expected DD (≈ total return for monotonic): {expected_min_dd:.2f}%")
    
    return equity_curve, max_dd

if __name__ == "__main__":
    debug_naive_simulation()
