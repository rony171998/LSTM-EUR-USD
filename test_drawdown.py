import numpy as np

def max_drawdown(equity_curve: np.ndarray) -> float:
    """Test the corrected drawdown formula"""
    # equity_curve: array con el valor de la cuenta en cada paso
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min() * 100  # en %

# Test cases
print("🧪 TESTING MAXIMUM DRAWDOWN CALCULATION")
print("=" * 50)

# Test 1: Simple case - equity goes from 10000 to 5000
equity1 = np.array([10000, 8000, 5000, 6000, 7000])
dd1 = max_drawdown(equity1)
print(f"Test 1: {equity1}")
print(f"Expected: -50% (10000 → 5000)")
print(f"Calculated: {dd1:.2f}%")
print()

# Test 2: Multiple peaks and valleys
equity2 = np.array([10000, 12000, 8000, 15000, 6000, 9000])
dd2 = max_drawdown(equity2)
peak_at = np.maximum.accumulate(equity2)
print(f"Test 2: {equity2}")
print(f"Peaks: {peak_at}")
print(f"Expected: -60% (15000 → 6000)")
print(f"Calculated: {dd2:.2f}%")
print()

# Test 3: No drawdown (only gains)
equity3 = np.array([10000, 11000, 12000, 13000, 14000])
dd3 = max_drawdown(equity3)
print(f"Test 3: {equity3}")
print(f"Expected: 0% (no drawdown)")
print(f"Calculated: {dd3:.2f}%")
print()

# Test 4: Real example - what was happening before
equity4 = np.array([10000, 9500, 9000, 8500, 8000, 7500, 6000, 6500, 7000])
dd4 = max_drawdown(equity4)
print(f"Test 4: {equity4}")
print(f"Expected: -40% (10000 → 6000)")
print(f"Calculated: {dd4:.2f}%")

print("\n✅ If all tests show reasonable values, the formula is working correctly!")
