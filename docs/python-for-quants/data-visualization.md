---
sidebar_position: 3
---

# Data Visualization for Quant Finance

Effective visualization is crucial for understanding market data, identifying patterns, and communicating insights.

## Matplotlib Basics

Matplotlib is the fundamental plotting library in Python.

### Price Charts

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
dates = pd.date_range('2024-01-01', periods=100)
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(dates, prices, linewidth=2)
plt.title('Stock Price Over Time', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Returns Distribution

```python
returns = np.diff(prices) / prices[:-1]

plt.figure(figsize=(10, 6))
plt.hist(returns, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(returns.mean(), color='red', linestyle='--', label='Mean')
plt.title('Distribution of Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

## Candlestick Charts

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_candlestick(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, row in data.iterrows():
        color = 'green' if row['Close'] > row['Open'] else 'red'
        
        # Draw candle body
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        ax.add_patch(Rectangle((i, bottom), 0.8, height, 
                               facecolor=color, edgecolor='black'))
        
        # Draw wicks
        ax.plot([i+0.4, i+0.4], [row['Low'], row['High']], 
               color='black', linewidth=1)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.set_title('Candlestick Chart')
    plt.tight_layout()
    plt.show()
```

## Moving Averages

```python
# Calculate moving averages
sma_20 = prices.rolling(window=20).mean()
sma_50 = prices.rolling(window=50).mean()

plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='Price', linewidth=2)
plt.plot(dates[19:], sma_20[19:], label='20-day SMA', linewidth=2)
plt.plot(dates[49:], sma_50[49:], label='50-day SMA', linewidth=2)
plt.title('Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Portfolio Visualization

### Correlation Heatmap

```python
import seaborn as sns

# Sample correlation matrix
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
correlation = np.random.rand(5, 5)
correlation = (correlation + correlation.T) / 2
np.fill_diagonal(correlation, 1)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', 
            xticklabels=stocks, yticklabels=stocks,
            vmin=-1, vmax=1, center=0)
plt.title('Stock Correlation Matrix')
plt.tight_layout()
plt.show()
```

### Portfolio Allocation

```python
# Portfolio weights
assets = ['Stocks', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
weights = [0.4, 0.25, 0.15, 0.1, 0.1]

plt.figure(figsize=(10, 8))
plt.pie(weights, labels=assets, autopct='%1.1f%%', startangle=90)
plt.title('Portfolio Allocation')
plt.axis('equal')
plt.show()
```

## Interactive Plots with Plotly

```python
import plotly.graph_objects as go

# Create interactive candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig.update_layout(
    title='Interactive Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

fig.show()
```

## Backtesting Results

```python
def plot_backtest_results(strategy_returns, benchmark_returns):
    # Cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Cumulative returns
    ax1.plot(strategy_cum, label='Strategy', linewidth=2)
    ax1.plot(benchmark_cum, label='Benchmark', linewidth=2)
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    strategy_dd = (strategy_cum / strategy_cum.cummax() - 1)
    ax2.fill_between(range(len(strategy_dd)), strategy_dd, 0, 
                     alpha=0.3, color='red')
    ax2.set_title('Strategy Drawdown')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Risk-Return Scatter

```python
def plot_risk_return(returns_dict):
    """
    Plot risk vs return for multiple strategies
    """
    means = [np.mean(ret) * 252 for ret in returns_dict.values()]  # Annualized
    stds = [np.std(ret) * np.sqrt(252) for ret in returns_dict.values()]  # Annualized
    
    plt.figure(figsize=(10, 6))
    plt.scatter(stds, means, s=100)
    
    for i, name in enumerate(returns_dict.keys()):
        plt.annotate(name, (stds[i], means[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Risk-Return Profile')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## Best Practices

1. **Clear Labels**: Always label axes and include titles
2. **Color Choice**: Use colorblind-friendly palettes
3. **Grid Lines**: Help readers interpret values
4. **Legend Placement**: Don't obscure data
5. **Figure Size**: Make plots readable
6. **Consistency**: Use consistent styling across charts

## Common Visualization Tasks

### 1. Time Series Analysis
- Line charts for price movements
- Volume bars
- Technical indicators overlay

### 2. Distribution Analysis
- Histograms for returns
- Q-Q plots for normality
- Box plots for outliers

### 3. Comparison
- Multiple line charts
- Bar charts for performance
- Scatter plots for correlation

### 4. Composition
- Pie charts for allocation
- Stacked area charts
- Tree maps for sectors

## Practice Exercises

1. Create a dashboard with:
   - Price chart with volume
   - Returns histogram
   - Moving averages

2. Visualize portfolio performance:
   - Cumulative returns vs benchmark
   - Drawdown chart
   - Monthly returns heatmap

3. Build an interactive chart with Plotly showing:
   - Candlestick chart
   - Volume bars
   - Moving averages

## Next Steps

- [Financial Markets Basics](/docs/financial-markets/market-basics)
- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
