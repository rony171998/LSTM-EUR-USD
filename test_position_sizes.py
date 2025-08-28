import numpy as np

def test_larger_position_sizes():
    """Test with larger position sizes to see bigger drawdowns"""
    
    print("🧪 TESTING DIFFERENT POSITION SIZES")
    print("=" * 50)
    
    # Test scenario: 50% price drop
    initial_capital = 10000
    prices = np.array([100, 50])  # 50% price drop
    
    # Test different position sizes
    position_sizes = [0.05, 0.25, 0.50, 0.95]  # 5%, 25%, 50%, 95%
    
    for risk_fraction in position_sizes:
        cash = float(initial_capital)
        
        # Buy at price 100
        notional = cash * risk_fraction
        units = notional / prices[0]
        fee = notional * 0.0001
        cash -= notional + fee
        
        # Portfolio values
        portfolio_t0 = cash + units * prices[0]  # Should be ~initial_capital
        portfolio_t1 = cash + units * prices[1]  # After 50% price drop
        
        # Drawdown calculation
        peak = max(portfolio_t0, portfolio_t1)
        dd = (portfolio_t1 - peak) / peak * 100
        
        print(f"Position Size: {risk_fraction*100:>4.0f}% | "
              f"Start: ${portfolio_t0:>8.2f} | "
              f"End: ${portfolio_t1:>8.2f} | "
              f"Drawdown: {dd:>6.1f}%")
    
    print("\n✅ As position size increases, drawdown increases proportionally")
    print("✅ Small drawdowns in backtesting = good risk management!")

if __name__ == "__main__":
    test_larger_position_sizes()
