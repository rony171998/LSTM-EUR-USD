import numpy as np

def debug_portfolio_simulation():
    """Debug the portfolio values calculation to see if drawdown is correct"""
    
    # Simulate a simple scenario
    initial_capital = 10000
    prices = np.array([100, 90, 80, 70, 85, 95])  # Significant price drop
    signals = np.array([1, 1, 1, 1, 1, 1])  # Stay LONG throughout
    
    print("🔍 DEBUGGING PORTFOLIO SIMULATION")
    print("=" * 50)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Price Movement: {prices}")
    print(f"Signals: {signals} (1 = LONG)")
    print()
    
    # Simulate with 5% position sizing
    cash = float(initial_capital)
    position = 0
    position_size = 0.0
    entry_price = None
    risk_fraction = 0.05
    fee_rate = 0.0001
    
    portfolio_values = []
    
    def mark_to_market(price):
        if position == 1:
            return cash + position_size * price
        elif position == -1:
            return cash - position_size * price
        else:
            return cash
    
    for t, (price, sig) in enumerate(zip(prices, signals)):
        print(f"\nStep {t}: Price=${price}, Signal={sig}")
        
        # Change position logic
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
                print(f"  Closed position: PnL=${pnl:.2f}, Cash=${cash:.2f}")
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
                    print(f"  Opened {['SHORT', 'FLAT', 'LONG'][sig+1]} position:")
                    print(f"    Units: {units:.4f}")
                    print(f"    Notional: ${notional:.2f}")
                    print(f"    Cash after: ${cash:.2f}")
        
        # Mark to market
        portfolio_value = mark_to_market(price)
        portfolio_values.append(portfolio_value)
        print(f"  Portfolio Value: ${portfolio_value:.2f}")
    
    # Calculate drawdown
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - peak) / peak
    max_dd = dd.min() * 100
    
    print(f"\n📊 RESULTS:")
    print(f"Portfolio Values: {portfolio_values}")
    print(f"Running Peaks: {peak}")
    print(f"Drawdowns: {dd * 100}")
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    
    # Calculate what we would expect manually
    max_portfolio = portfolio_values.max()
    min_after_peak = portfolio_values[np.argmax(portfolio_values):].min()
    expected_dd = (min_after_peak - max_portfolio) / max_portfolio * 100
    print(f"Expected DD (manual): {expected_dd:.2f}%")
    
    return max_dd

if __name__ == "__main__":
    debug_portfolio_simulation()
