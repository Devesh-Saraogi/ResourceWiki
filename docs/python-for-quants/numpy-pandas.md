---
sidebar_position: 2
---

# NumPy and Pandas for Finance

NumPy and Pandas are the backbone of quantitative finance in Python. This guide covers essential operations for financial data analysis.

## NumPy: Numerical Computing

NumPy provides efficient array operations crucial for quantitative analysis.

### Creating Arrays

```python
import numpy as np

# Stock prices
prices = np.array([100, 102, 101, 105, 107, 106, 110])

# Multiple stocks
portfolio = np.array([
    [100, 50, 75],  # Day 1
    [102, 52, 74],  # Day 2
    [101, 51, 76],  # Day 3
])
```

### Financial Calculations

```python
# Calculate returns
returns = np.diff(prices) / prices[:-1]

# Portfolio value
weights = np.array([0.5, 0.3, 0.2])
portfolio_value = np.dot(portfolio, weights)

# Volatility (standard deviation)
volatility = np.std(returns)
```

### Statistical Functions

```python
# Key statistics
mean_return = np.mean(returns)
median_return = np.median(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / std_return

print(f"Mean Return: {mean_return:.4f}")
print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
```

## Pandas: Data Manipulation

Pandas excels at handling time series financial data.

### Creating DataFrames

```python
import pandas as pd

# Stock data
data = {
    'Date': pd.date_range('2024-01-01', periods=5),
    'Open': [100, 102, 101, 105, 107],
    'High': [103, 104, 103, 107, 109],
    'Low': [99, 101, 100, 104, 106],
    'Close': [102, 101, 105, 107, 106],
    'Volume': [1000000, 1200000, 950000, 1100000, 1050000]
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)
```

### Time Series Operations

```python
# Calculate returns
df['Returns'] = df['Close'].pct_change()

# Moving averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Cumulative returns
df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
```

### Resampling

```python
# Convert daily to weekly data
weekly_data = df.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
```

## Practical Example: Multi-Asset Analysis

```python
import pandas as pd
import numpy as np

# Multiple stocks
stocks_data = {
    'AAPL': [150, 152, 151, 155, 157],
    'GOOGL': [2800, 2820, 2810, 2850, 2870],
    'MSFT': [300, 305, 303, 310, 315]
}

df = pd.DataFrame(stocks_data)

# Calculate returns for each stock
returns = df.pct_change()

# Correlation matrix
correlation = returns.corr()

# Portfolio statistics
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("Mean Returns:")
print(mean_returns)
print("\nCorrelation Matrix:")
print(correlation)
```

## Common Patterns in Quant Finance

### 1. Rolling Window Calculations

```python
# 20-day rolling volatility
df['Volatility'] = df['Returns'].rolling(window=20).std()
```

### 2. Group Operations

```python
# Group by month and calculate statistics
monthly_stats = df.groupby(pd.Grouper(freq='M')).agg({
    'Returns': ['mean', 'std', 'min', 'max']
})
```

### 3. Handling Missing Data

```python
# Forward fill missing values
df.fillna(method='ffill', inplace=True)

# Or drop missing values
df.dropna(inplace=True)
```

## Performance Tips

1. **Vectorization**: Use NumPy operations instead of loops
2. **Avoid Chaining**: Use single operations when possible
3. **Categorical Data**: Convert repeated strings to categories
4. **Efficient Storage**: Use appropriate data types

## Practice Exercises

1. Load historical stock data and calculate:
   - Daily returns
   - 30-day moving average
   - Rolling 20-day volatility

2. Create a correlation matrix for a portfolio of 5 stocks

3. Implement a function to calculate the Sharpe ratio for multiple assets

## Next Steps

- [Data Visualization](/docs/python-for-quants/data-visualization)
- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
