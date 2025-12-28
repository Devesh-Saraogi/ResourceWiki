---
sidebar_position: 2
---

# Backtesting

Backtesting is crucial for evaluating trading strategies. Learn how to properly test strategies on historical data.

## Backtesting Framework

### Basic Backtest Engine

```python
import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, data, strategy, initial_capital=100000, 
                 commission=0.001):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        
    def run(self):
        """Execute the backtest"""
        capital = self.initial_capital
        position = 0  # Current position size
        
        for i in range(len(self.data)):
            current_data = self.data.iloc[:i+1]
            signal = self.strategy.generate_signal(current_data)
            price = self.data['Close'].iloc[i]
            
            # Execute trades based on signal
            if signal == 1 and position == 0:  # Buy signal
                shares = int(capital / price)
                cost = shares * price * (1 + self.commission)
                
                if cost <= capital:
                    position = shares
                    capital -= cost
                    self.trades.append({
                        'date': self.data.index[i],
                        'type': 'BUY',
                        'shares': shares,
                        'price': price,
                        'cost': cost
                    })
            
            elif signal == -1 and position > 0:  # Sell signal
                proceeds = position * price * (1 - self.commission)
                capital += proceeds
                
                self.trades.append({
                    'date': self.data.index[i],
                    'type': 'SELL',
                    'shares': position,
                    'price': price,
                    'proceeds': proceeds
                })
                
                position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (position * price)
            self.portfolio_values.append(portfolio_value)
        
        return self.calculate_performance()
    
    def calculate_performance(self):
        """Calculate performance metrics"""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        total_return = (self.portfolio_values[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = len(self.portfolio_values)
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Sharpe ratio
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        # Maximum drawdown
        cumulative = pd.Series(self.portfolio_values)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'final_value': self.portfolio_values[-1]
        }
```

## Common Backtesting Pitfalls

### 1. Look-Ahead Bias

```python
# WRONG: Using future information
def wrong_strategy(data):
    # This uses the entire dataset, including future prices!
    future_return = data['Close'].pct_change().shift(-1)
    signal = np.where(future_return > 0, 1, -1)
    return signal

# CORRECT: Using only past information
def correct_strategy(data):
    # Use only historical data up to current point
    sma_short = data['Close'].rolling(window=20).mean()
    sma_long = data['Close'].rolling(window=50).mean()
    signal = np.where(sma_short > sma_long, 1, -1)
    return signal
```

### 2. Survivorship Bias

```python
def handle_survivorship_bias(data, delisted_stocks):
    """
    Include delisted stocks in backtest
    
    Parameters:
    - data: Current stock universe
    - delisted_stocks: Stocks that were delisted during period
    """
    # Combine current and delisted stocks
    full_universe = pd.concat([data, delisted_stocks])
    
    # This ensures we don't only test on survivors
    return full_universe
```

### 3. Overfitting

```python
def walk_forward_analysis(data, optimize_window=252, test_window=63):
    """
    Walk-forward analysis to detect overfitting
    
    Parameters:
    - optimize_window: Days for optimization
    - test_window: Days for out-of-sample testing
    """
    results = []
    
    for i in range(0, len(data) - optimize_window - test_window, test_window):
        # Optimization period
        train_data = data.iloc[i:i+optimize_window]
        
        # Test period
        test_data = data.iloc[i+optimize_window:i+optimize_window+test_window]
        
        # Optimize strategy on train data
        optimal_params = optimize_strategy(train_data)
        
        # Test on out-of-sample data
        test_result = backtest_with_params(test_data, optimal_params)
        
        results.append({
            'train_period': train_data.index[[0, -1]],
            'test_period': test_data.index[[0, -1]],
            'params': optimal_params,
            'test_return': test_result['return']
        })
    
    return pd.DataFrame(results)
```

## Transaction Costs

### Realistic Cost Model

```python
class TransactionCosts:
    def __init__(self, commission_pct=0.001, slippage_pct=0.001, 
                 min_commission=1.0):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
    
    def calculate_cost(self, price, quantity, side='buy'):
        """
        Calculate total transaction cost
        
        Parameters:
        - price: Execution price
        - quantity: Number of shares
        - side: 'buy' or 'sell'
        """
        # Commission
        commission = max(price * quantity * self.commission_pct, 
                        self.min_commission)
        
        # Slippage (adverse price movement)
        if side == 'buy':
            slippage = price * quantity * self.slippage_pct
        else:
            slippage = price * quantity * self.slippage_pct
        
        # Market impact (for large orders)
        market_impact = self._calculate_market_impact(quantity)
        
        total_cost = commission + slippage + market_impact
        
        return total_cost
    
    def _calculate_market_impact(self, quantity):
        """Estimate market impact for large orders"""
        # Simple square-root model
        impact = 0.1 * np.sqrt(quantity / 1000)
        return impact
```

## Performance Metrics

### Comprehensive Metrics

```python
def calculate_metrics(returns, benchmark_returns=None):
    """
    Calculate comprehensive performance metrics
    """
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe = annualized_return / annualized_vol
    
    # Downside deviation (for Sortino)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_std if len(downside_returns) > 0 else 0
    
    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean()
    
    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns)
    
    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Volatility': f"{annualized_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Avg Drawdown': f"{avg_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': f"{profit_factor:.2f}"
    }
    
    # Benchmark comparison
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (annualized_return - 
                           ((1 + benchmark_returns).prod() - 1)) / tracking_error
        
        metrics['Information Ratio'] = f"{information_ratio:.2f}"
        metrics['Tracking Error'] = f"{tracking_error:.2%}"
    
    return metrics
```

## Visualization

### Equity Curve

```python
import matplotlib.pyplot as plt

def plot_equity_curve(portfolio_values, benchmark_values=None):
    """Plot equity curve and drawdown"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Normalize to start at 100
    portfolio_norm = (portfolio_values / portfolio_values[0]) * 100
    
    # Plot equity curve
    ax1.plot(portfolio_norm, label='Strategy', linewidth=2)
    
    if benchmark_values is not None:
        benchmark_norm = (benchmark_values / benchmark_values[0]) * 100
        ax1.plot(benchmark_norm, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax1.set_title('Equity Curve', fontsize=14)
    ax1.set_ylabel('Portfolio Value (Base=100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot drawdown
    cumulative = pd.Series(portfolio_norm)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    
    ax2.fill_between(range(len(drawdown)), drawdown, 0, 
                     color='red', alpha=0.3)
    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Monthly Returns Heatmap

```python
import seaborn as sns

def plot_monthly_returns(returns):
    """Create monthly returns heatmap"""
    # Convert to DataFrame with date index
    returns_df = pd.DataFrame({'returns': returns})
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month
    
    # Calculate monthly returns
    monthly = returns_df.groupby(['year', 'month'])['returns'].apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Pivot for heatmap
    monthly_pivot = monthly.unstack(level='month')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', 
                cmap='RdYlGn', center=0, cbar_kws={'label': 'Return (%)'})
    plt.title('Monthly Returns Heatmap')
    plt.ylabel('Year')
    plt.xlabel('Month')
    plt.show()
```

## Cross-Validation

### Time Series Cross-Validation

```python
def time_series_cv(data, n_splits=5, test_size=252):
    """
    Time series cross-validation
    Maintains temporal order
    """
    results = []
    
    for i in range(n_splits):
        # Calculate split points
        test_end = len(data) - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start
        
        # Split data
        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]
        
        results.append({
            'fold': i + 1,
            'train': train,
            'test': test,
            'train_period': (train.index[0], train.index[-1]),
            'test_period': (test.index[0], test.index[-1])
        })
    
    return results
```

## Advanced Backtesting

### Event-Driven Backtesting

```python
class EventDrivenBacktest:
    """
    Event-driven backtesting engine
    More realistic simulation of trading
    """
    def __init__(self, data, strategy, initial_capital=100000):
        self.data = data
        self.strategy = strategy
        self.capital = initial_capital
        self.positions = {}
        self.orders = []
        
    def process_market_event(self, timestamp, market_data):
        """Process new market data"""
        # Generate signals
        signals = self.strategy.calculate_signals(market_data)
        
        # Generate orders
        orders = self.strategy.generate_orders(signals, self.positions)
        
        # Execute orders
        for order in orders:
            self.execute_order(order, market_data[order['symbol']])
    
    def execute_order(self, order, price):
        """Execute trading order"""
        if order['type'] == 'market':
            # Immediate execution at current price
            self.fill_order(order, price)
        elif order['type'] == 'limit':
            # Check if limit price is reached
            if (order['side'] == 'buy' and price <= order['limit_price']) or \
               (order['side'] == 'sell' and price >= order['limit_price']):
                self.fill_order(order, order['limit_price'])
    
    def fill_order(self, order, price):
        """Fill a trading order"""
        symbol = order['symbol']
        quantity = order['quantity']
        side = order['side']
        
        if side == 'buy':
            cost = quantity * price
            if cost <= self.capital:
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.capital -= cost
        else:  # sell
            if self.positions.get(symbol, 0) >= quantity:
                proceeds = quantity * price
                self.positions[symbol] -= quantity
                self.capital += proceeds
```

## Best Practices

1. **Use Out-of-Sample Testing**: Always test on data not used in optimization
2. **Include Transaction Costs**: Don't ignore commissions and slippage
3. **Avoid Survivorship Bias**: Include delisted stocks
4. **Check for Look-Ahead Bias**: Only use historical information
5. **Walk-Forward Analysis**: Validate strategy robustness
6. **Multiple Time Periods**: Test across different market conditions
7. **Document Assumptions**: Keep track of all assumptions made

## Practice Exercises

1. **Build a Complete Backtest**:
   - Implement moving average crossover strategy
   - Include transaction costs
   - Calculate performance metrics
   - Plot equity curve

2. **Detect Biases**:
   - Create examples of look-ahead bias
   - Fix the bias
   - Compare results

3. **Walk-Forward Analysis**:
   - Implement rolling optimization
   - Test on out-of-sample data
   - Analyze stability of results

## Next Steps

- [Portfolio Optimization](/docs/quantitative-analysis/portfolio-optimization)
- [Backtesting Frameworks](/docs/tools-libraries/backtesting-frameworks)
- [Risk Management](/docs/financial-markets/risk-management)
