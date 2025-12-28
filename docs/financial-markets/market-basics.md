---
sidebar_position: 1
---

# Financial Markets Basics

Understanding financial markets is essential for quantitative trading. This guide covers the fundamentals of market structure and trading.

## Market Structure

### Types of Markets

1. **Stock Markets**
   - Primary market: IPOs and new issues
   - Secondary market: Trading existing securities
   - Examples: NYSE, NASDAQ, LSE

2. **Bond Markets**
   - Government bonds
   - Corporate bonds
   - Municipal bonds

3. **Derivatives Markets**
   - Futures
   - Options
   - Swaps

4. **Forex Markets**
   - Currency pairs
   - 24/7 trading
   - High liquidity

### Market Participants

- **Retail Investors**: Individual traders
- **Institutional Investors**: Mutual funds, pension funds
- **Market Makers**: Provide liquidity
- **Hedge Funds**: Alternative investment strategies
- **High-Frequency Traders**: Algorithmic trading at microsecond speeds

## Trading Mechanisms

### Order Types

```python
# Market Order
order = {
    'type': 'market',
    'side': 'buy',
    'quantity': 100,
    'symbol': 'AAPL'
}

# Limit Order
order = {
    'type': 'limit',
    'side': 'sell',
    'quantity': 100,
    'symbol': 'AAPL',
    'price': 150.00
}

# Stop-Loss Order
order = {
    'type': 'stop',
    'side': 'sell',
    'quantity': 100,
    'symbol': 'AAPL',
    'stop_price': 145.00
}
```

### Bid-Ask Spread

The difference between buying and selling prices:

```python
bid_price = 100.00  # Highest price buyers willing to pay
ask_price = 100.05  # Lowest price sellers willing to accept
spread = ask_price - bid_price  # 0.05

# Spread as percentage
spread_pct = (spread / ask_price) * 100
```

## Key Financial Instruments

### 1. Stocks (Equities)

Represent ownership in a company.

**Key Metrics:**
```python
# Price to Earnings Ratio
pe_ratio = stock_price / earnings_per_share

# Dividend Yield
dividend_yield = annual_dividend / stock_price

# Market Capitalization
market_cap = stock_price * shares_outstanding
```

### 2. Bonds (Fixed Income)

Debt instruments with fixed interest payments.

**Bond Pricing:**
```python
def bond_price(face_value, coupon_rate, years, discount_rate):
    """Calculate bond price"""
    coupon = face_value * coupon_rate
    present_value_coupons = sum(
        [coupon / (1 + discount_rate)**t for t in range(1, years + 1)]
    )
    present_value_face = face_value / (1 + discount_rate)**years
    return present_value_coupons + present_value_face

# Example
price = bond_price(1000, 0.05, 10, 0.06)
print(f"Bond Price: ${price:.2f}")
```

### 3. Options

Derivatives giving the right (not obligation) to buy/sell.

**Call Option Payoff:**
```python
def call_payoff(spot_price, strike_price, premium):
    """Calculate call option payoff"""
    intrinsic_value = max(spot_price - strike_price, 0)
    profit = intrinsic_value - premium
    return profit

# Example
profit = call_payoff(spot_price=110, strike_price=100, premium=5)
print(f"Profit: ${profit}")
```

**Put Option Payoff:**
```python
def put_payoff(spot_price, strike_price, premium):
    """Calculate put option payoff"""
    intrinsic_value = max(strike_price - spot_price, 0)
    profit = intrinsic_value - premium
    return profit
```

### 4. Futures

Contracts to buy/sell at predetermined price and date.

```python
# Futures profit calculation
def futures_profit(entry_price, exit_price, contracts, contract_size):
    """Calculate futures trading profit"""
    price_diff = exit_price - entry_price
    profit = price_diff * contracts * contract_size
    return profit

# Example: Crude Oil futures
profit = futures_profit(
    entry_price=75.00,
    exit_price=78.00,
    contracts=10,
    contract_size=1000  # barrels
)
print(f"Profit: ${profit:,.2f}")
```

## Market Indicators

### Price-Based Indicators

```python
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).rolling(period).mean()
    avg_loss = pd.Series(losses).rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = pd.Series(prices).ewm(span=fast).mean()
    ema_slow = pd.Series(prices).ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line
```

### Volume Indicators

```python
def on_balance_volume(prices, volumes):
    """On-Balance Volume"""
    obv = np.zeros(len(prices))
    obv[0] = volumes[0]
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    
    return obv
```

## Market Data

### OHLCV Data

Open, High, Low, Close, Volume - standard format for market data.

```python
import pandas as pd

# Sample OHLCV data structure
data = pd.DataFrame({
    'Open': [100, 102, 101, 105],
    'High': [103, 104, 103, 107],
    'Low': [99, 101, 100, 104],
    'Close': [102, 101, 105, 107],
    'Volume': [1000000, 1200000, 950000, 1100000]
}, index=pd.date_range('2024-01-01', periods=4))
```

## Trading Sessions

### Market Hours

```python
from datetime import time

market_hours = {
    'NYSE': {
        'pre_market': (time(4, 0), time(9, 30)),
        'regular': (time(9, 30), time(16, 0)),
        'after_hours': (time(16, 0), time(20, 0))
    },
    'LSE': {
        'regular': (time(8, 0), time(16, 30))
    }
}
```

## Market Efficiency

### Efficient Market Hypothesis (EMH)

1. **Weak Form**: Prices reflect all past market data
2. **Semi-Strong Form**: Prices reflect all public information
3. **Strong Form**: Prices reflect all information (public and private)

### Market Anomalies

- **January Effect**: Stocks tend to rise in January
- **Monday Effect**: Returns tend to be lower on Mondays
- **Size Effect**: Small-cap stocks outperform large-cap
- **Value Effect**: Value stocks outperform growth stocks

## Practice Exercises

1. **Calculate the profit/loss** for:
   - Buying 100 shares at $50, selling at $55
   - Selling short 50 shares at $80, buying back at $75

2. **Implement order book matching** algorithm:
   - Match buy and sell orders
   - Calculate execution prices

3. **Analyze market data**:
   - Calculate daily returns
   - Compute volatility
   - Identify support and resistance levels

## Key Takeaways

- Markets connect buyers and sellers
- Different instruments have different risk/return profiles
- Understanding order types is crucial for execution
- Market structure affects trading strategies
- Liquidity and volatility are key considerations

## Next Steps

- [Trading Strategies](/docs/financial-markets/trading-strategies)
- [Risk Management](/docs/financial-markets/risk-management)
- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
