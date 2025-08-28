import numpy as np

def test_cumulative_losses():
    """Test scenario with many small losses to see if drawdown tracks correctly"""
    
    print("🔍 TESTING CUMULATIVE LOSSES SCENARIO")
    print("=" * 50)
    
    # Simulate losing 5% per trade over 20 trades (should lose ~64% total)
    initial_capital = 10000
    equity_values = [initial_capital]
    
    current_equity = initial_capital
    for i in range(20):
        # Lose 5% each trade
        loss = current_equity * 0.05
        current_equity -= loss
        equity_values.append(current_equity)
        print(f"Trade {i+1:2d}: Lost ${loss:>6.2f}, Equity: ${current_equity:>8.2f}")
    
    equity_curve = np.array(equity_values)
    total_loss_pct = (equity_curve[-1] / equity_curve[0] - 1) * 100
    
    print(f"\nTotal Loss: {total_loss_pct:.1f}%")
    print(f"Final Equity: ${equity_curve[-1]:.2f}")
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak * 100
    max_dd = dd.min()
    
    print(f"\nEquity Curve: {equity_curve[:5]}...{equity_curve[-3:]}")
    print(f"Running Peaks: {peak[:5]}...{peak[-3:]}")
    print(f"Drawdowns: {dd[:5]}...{dd[-3:]}")
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    
    # This should show that max drawdown ≈ total loss for monotonic decline
    print(f"\n✅ For monotonic losses, max DD should ≈ total loss")
    print(f"Total Loss: {total_loss_pct:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Difference: {abs(total_loss_pct - max_dd):.2f}% (should be ~0)")

if __name__ == "__main__":
    test_cumulative_losses()
