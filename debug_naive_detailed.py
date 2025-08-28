import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from config import DEFAULT_PARAMS

def debug_naive_backtesting():
    """Debug the actual NAIVE baseline backtesting to see what's happening"""
    
    print("🔍 DEBUGGING ACTUAL NAIVE BASELINE")
    print("=" * 60)
    
    # Load the same data as the backtesting system
    data_path = Path("data/EUR_USD_2010-2024.csv")
    
    def european_float(x):
        if isinstance(x, str):
            return float(x.replace(',', '.').replace(' ', ''))
        return float(x)
    
    df = pd.read_csv(data_path, 
                     converters={'Último': european_float, 'Apertura': european_float,
                                'Máximo': european_float, 'Mínimo': european_float})
    df['Date'] = pd.to_datetime(df['Fecha'], format='%d.%m.%Y')
    df = df.set_index('Date').sort_index()
    df['price'] = df['Último']
    
    # Use the same train/test split as the backtesting
    train_split_ratio = 0.80
    split_idx = int(len(df) * train_split_ratio)
    test_idx = split_idx + DEFAULT_PARAMS.SEQ_LENGTH
    
    backtest_df = df.iloc[test_idx:test_idx + 120]  # Same 120 predictions
    
    print(f"Backtest period: {backtest_df.index[0]} to {backtest_df.index[-1]}")
    print(f"Price range: {backtest_df['price'].min():.5f} to {backtest_df['price'].max():.5f}")
    print(f"Number of periods: {len(backtest_df)}")
    
    # Get actual prices and create NAIVE predictions
    actual_prices = backtest_df['price'].values
    
    # NAIVE prediction: use last known value
    naive_predictions = np.zeros_like(actual_prices)
    naive_predictions[0] = actual_prices[0]  # First prediction
    for i in range(1, len(actual_prices)):
        naive_predictions[i] = actual_prices[i-1]  # Use previous actual value
    
    print(f"\nActual prices (first 10): {actual_prices[:10]}")
    print(f"NAIVE predictions (first 10): {naive_predictions[:10]}")
    
    # Generate signals using threshold method (same as backtesting)
    def generate_threshold_signals(predictions, actual_prices, threshold=0.0001):
        signals = np.zeros(len(predictions), dtype=int)
        for i in range(1, len(predictions)):
            price_change = (predictions[i] - actual_prices[i-1]) / actual_prices[i-1]
            if price_change > threshold:
                signals[i] = 1  # LONG
            elif price_change < -threshold:
                signals[i] = -1  # SHORT
            else:
                signals[i] = 0  # NEUTRAL
        return signals
    
    signals = generate_threshold_signals(naive_predictions, actual_prices)
    print(f"Signals (first 20): {signals[:20]}")
    print(f"Signal distribution: LONG={np.sum(signals==1)}, SHORT={np.sum(signals==-1)}, NEUTRAL={np.sum(signals==0)}")
    
    # Now simulate trading with detailed logging
    initial_capital = 10000
    cash = float(initial_capital)
    position = 0
    position_size = 0.0
    entry_price = None
    risk_fraction = 0.05
    fee_rate = 0.0001
    
    equity_curve = []
    trades = []
    
    def mark_to_market(price):
        if position == 1:
            return cash + position_size * price
        elif position == -1:
            return cash - position_size * price
        else:
            return cash
    
    print(f"\n🎯 DETAILED TRADING SIMULATION:")
    print("-" * 50)
    
    for t, (price, sig) in enumerate(zip(actual_prices, signals)):
        old_cash = cash
        old_position = position
        
        # Position change logic (same as corrected simulation)
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
                
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "profit": pnl,
                    "side": "LONG" if position == 1 else "SHORT"
                })
                
                if t < 10 or len(trades) % 10 == 0:  # Log first 10 and every 10th trade
                    print(f"  T{t:3d}: Closed {['SHORT','FLAT','LONG'][position+1]} | "
                          f"PnL: ${pnl:>7.2f} | Cash: ${cash:>8.2f}")
                
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
                    
                    if t < 10:  # Log first 10 trades
                        print(f"  T{t:3d}: Opened {['SHORT','FLAT','LONG'][sig+1]} | "
                              f"Size: {units:>6.2f} | Cash: ${cash:>8.2f}")
        
        # Mark to market
        portfolio_value = mark_to_market(price)
        equity_curve.append(portfolio_value)
    
    # Final results
    equity_curve = np.array(equity_curve)
    final_capital = cash if position == 0 else equity_curve[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak * 100
    max_dd = dd.min()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print(f"Equity Min: ${equity_curve.min():.2f}")
    print(f"Equity Max: ${equity_curve.max():.2f}")
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    
    print(f"\n🔍 EQUITY CURVE ANALYSIS:")
    print(f"First 10 values: {equity_curve[:10]}")
    print(f"Last 10 values: {equity_curve[-10:]}")
    
    # Check if equity curve shows the expected loss pattern
    if abs(total_return - max_dd) > 1.0:  # If they differ by more than 1%
        print(f"\n❌ PROBLEM DETECTED:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"These should be similar for a mostly-losing strategy!")
        
        # Show where the discrepancy comes from
        print(f"\nEquity curve range: ${equity_curve.min():.2f} to ${equity_curve.max():.2f}")
        peak_idx = np.argmax(equity_curve)
        min_idx = np.argmin(equity_curve)
        print(f"Peak at index {peak_idx}: ${equity_curve[peak_idx]:.2f}")
        print(f"Minimum at index {min_idx}: ${equity_curve[min_idx]:.2f}")

if __name__ == "__main__":
    debug_naive_backtesting()
