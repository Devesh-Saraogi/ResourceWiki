---
sidebar_position: 2
---

# Trading Strategies

Learn about different trading strategies used in quantitative finance, from momentum to mean reversion.

## Strategy Categories

### 1. Trend Following

Capitalize on sustained price movements.

```python
import pandas as pd
import numpy as np

def moving_average_crossover(prices, short_window=50, long_window=200):
    """
    Simple Moving Average Crossover Strategy
    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices
    
    # Calculate moving averages
    signals['short_ma'] = prices.rolling(window=short_window).mean()
    signals['long_ma'] = prices.rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:],
        1, 0
    )
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    
    return signals
```

### 2. Mean Reversion

Profit from price returning to average.

```python
def bollinger_bands_strategy(prices, window=20, num_std=2):
    """
    Bollinger Bands Mean Reversion Strategy
    Buy when price touches lower band
    Sell when price touches upper band
    """
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices
    
    # Calculate Bollinger Bands
    signals['sma'] = prices.rolling(window=window).mean()
    signals['std'] = prices.rolling(window=window).std()
    signals['upper_band'] = signals['sma'] + (signals['std'] * num_std)
    signals['lower_band'] = signals['sma'] - (signals['std'] * num_std)
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[prices < signals['lower_band'], 'signal'] = 1  # Buy
    signals.loc[prices > signals['upper_band'], 'signal'] = -1  # Sell
    
    return signals
```

### 3. Momentum

Trade based on strength of price movement.

```python
def momentum_strategy(prices, lookback=20, holding_period=5):
    """
    Momentum Strategy
    Buy top performers, sell bottom performers
    """
    returns = prices.pct_change(lookback)
    
    signals = pd.DataFrame(index=prices.index)
    signals['returns'] = returns
    signals['signal'] = 0
    
    # Buy if momentum is positive
    signals.loc[returns > 0, 'signal'] = 1
    # Sell if momentum is negative
    signals.loc[returns < 0, 'signal'] = -1
    
    return signals
```

### 4. Statistical Arbitrage

Exploit statistical relationships between assets.

```python
def pairs_trading(price1, price2, window=20, entry_z=2, exit_z=0):
    """
    Pairs Trading Strategy
    Trade based on spread between two correlated assets
    """
    # Calculate spread
    spread = price1 - price2
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # Calculate z-score
    z_score = (spread - spread_mean) / spread_std
    
    signals = pd.DataFrame(index=price1.index)
    signals['spread'] = spread
    signals['z_score'] = z_score
    signals['signal'] = 0
    
    # Long spread (buy asset1, sell asset2) when z-score < -entry_z
    signals.loc[z_score < -entry_z, 'signal'] = 1
    # Short spread (sell asset1, buy asset2) when z-score > entry_z
    signals.loc[z_score > entry_z, 'signal'] = -1
    # Exit when z-score returns to zero
    signals.loc[abs(z_score) < exit_z, 'signal'] = 0
    
    return signals
```

## Advanced Strategies

### 5. Market Making

Provide liquidity and profit from bid-ask spread.

```python
class SimpleMarketMaker:
    def __init__(self, spread=0.01, inventory_limit=100):
        self.spread = spread
        self.inventory_limit = inventory_limit
        self.inventory = 0
    
    def quote(self, mid_price):
        """Generate bid and ask quotes"""
        half_spread = mid_price * self.spread / 2
        
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        
        # Adjust for inventory
        if self.inventory > self.inventory_limit * 0.8:
            # Too long, lower both sides
            bid -= half_spread * 0.5
            ask -= half_spread * 0.5
        elif self.inventory < -self.inventory_limit * 0.8:
            # Too short, raise both sides
            bid += half_spread * 0.5
            ask += half_spread * 0.5
        
        return {'bid': bid, 'ask': ask}
    
    def on_trade(self, side, quantity):
        """Update inventory after trade"""
        if side == 'buy':
            self.inventory += quantity
        else:
            self.inventory -= quantity
```

### 6. Volatility Trading

Trade options based on implied vs realized volatility.

```python
def volatility_strategy(returns, implied_vol, window=20):
    """
    Compare realized volatility with implied volatility
    Buy volatility when realized < implied
    Sell volatility when realized > implied
    """
    # Calculate realized volatility
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    signals = pd.DataFrame(index=returns.index)
    signals['realized_vol'] = realized_vol
    signals['implied_vol'] = implied_vol
    signals['vol_diff'] = implied_vol - realized_vol
    signals['signal'] = 0
    
    # Buy volatility (long straddle)
    signals.loc[signals['vol_diff'] > 0.05, 'signal'] = 1
    # Sell volatility (short straddle)
    signals.loc[signals['vol_diff'] < -0.05, 'signal'] = -1
    
    return signals
```

## Strategy Implementation

### Complete Strategy Template

```python
class TradingStrategy:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
    
    def generate_signals(self, data):
        """Override this method with strategy logic"""
        raise NotImplementedError
    
    def execute_trade(self, symbol, quantity, price, timestamp):
        """Execute a trade"""
        cost = quantity * price
        
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        self.positions[symbol] += quantity
        self.capital -= cost
        
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost
        })
    
    def get_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        positions_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in current_prices.items()
        )
        return self.capital + positions_value
    
    def calculate_returns(self, prices):
        """Calculate strategy returns"""
        portfolio_values = []
        
        for timestamp in prices.index:
            current_prices = prices.loc[timestamp]
            value = self.get_portfolio_value(current_prices)
            portfolio_values.append(value)
        
        returns = pd.Series(portfolio_values).pct_change()
        return returns
```

## Risk-Adjusted Performance

### Sharpe Ratio

```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
```

### Maximum Drawdown

```python
def max_drawdown(returns):
    """
    Calculate maximum drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

### Sortino Ratio

```python
def sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sortino Ratio (only penalizes downside volatility)
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    return np.sqrt(252) * excess_returns.mean() / downside_std
```

## Strategy Optimization

### Parameter Tuning

```python
def optimize_strategy(data, param_ranges):
    """
    Grid search for optimal parameters
    """
    best_sharpe = -np.inf
    best_params = None
    
    for short_window in param_ranges['short']:
        for long_window in param_ranges['long']:
            if short_window >= long_window:
                continue
            
            signals = moving_average_crossover(
                data['Close'], short_window, long_window
            )
            returns = calculate_strategy_returns(signals, data)
            sharpe = sharpe_ratio(returns)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'short': short_window,
                    'long': long_window
                }
    
    return best_params, best_sharpe
```

## Common Pitfalls

1. **Overfitting**: Optimizing too much on historical data
2. **Look-Ahead Bias**: Using future information
3. **Survivorship Bias**: Only considering successful companies
4. **Transaction Costs**: Ignoring fees and slippage
5. **Market Impact**: Not accounting for order size effects

## Best Practices

1. **Out-of-Sample Testing**: Always test on unseen data
2. **Walk-Forward Analysis**: Rolling window backtesting
3. **Risk Management**: Use stop-losses and position sizing
4. **Diversification**: Don't rely on single strategy
5. **Regular Review**: Monitor and adapt strategies

## Practice Exercises

1. Implement a momentum strategy:
   - Calculate 12-month momentum
   - Rank assets
   - Rebalance monthly

2. Create a mean-reversion strategy:
   - Identify overbought/oversold conditions
   - Set entry and exit rules
   - Calculate returns

3. Build a pairs trading strategy:
   - Find cointegrated pairs
   - Calculate spread
   - Generate trading signals

## Next Steps

- [Risk Management](/docs/financial-markets/risk-management)
- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Portfolio Optimization](/docs/quantitative-analysis/portfolio-optimization)
