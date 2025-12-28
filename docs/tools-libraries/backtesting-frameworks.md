---
sidebar_position: 2
---

# Backtesting Frameworks

Professional backtesting frameworks to test your trading strategies efficiently.

## Popular Frameworks

### 1. Backtrader

Comprehensive and flexible framework.

```python
import backtrader as bt

class MovingAverageCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
    )
    
    def __init__(self):
        # Indicators
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if not self.position:  # Not in market
            if self.crossover > 0:  # Bullish crossover
                self.buy()
        elif self.crossover < 0:  # Bearish crossover
            self.close()

# Setup
cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageCrossStrategy)

# Add data
data = bt.feeds.YahooFinanceData(
    dataname='AAPL',
    fromdate=datetime(2020, 1, 1),
    todate=datetime(2024, 1, 1)
)
cerebro.adddata(data)

# Set initial capital
cerebro.broker.setcash(100000.0)

# Set commission
cerebro.broker.setcommission(commission=0.001)

# Run
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

# Plot
cerebro.plot()
```

### 2. Zipline

Institutional-grade backtesting (used by Quantopian).

```python
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol

def initialize(context):
    """Initialize strategy"""
    context.asset = symbol('AAPL')
    context.i = 0

def handle_data(context, data):
    """Called on each bar"""
    context.i += 1
    
    # Trade every 5 days
    if context.i % 5 == 0:
        # Calculate moving averages
        short_mavg = data.history(context.asset, 'price', 20, '1d').mean()
        long_mavg = data.history(context.asset, 'price', 50, '1d').mean()
        
        # Trading logic
        if short_mavg > long_mavg:
            order_target_percent(context.asset, 1.0)
        elif short_mavg < long_mavg:
            order_target_percent(context.asset, 0.0)

def analyze(context, perf):
    """Analyze results"""
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio Value')
    
    ax2 = fig.add_subplot(212)
    perf.returns.plot(ax=ax2)
    ax2.set_ylabel('Returns')
    plt.show()

# Run backtest
result = run_algorithm(
    start=pd.Timestamp('2020-1-1', tz='utc'),
    end=pd.Timestamp('2024-1-1', tz='utc'),
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze,
    capital_base=100000,
    data_frequency='daily',
    bundle='quandl'
)
```

### 3. VectorBT

High-performance vectorized backtesting.

```python
import vectorbt as vbt

# Download data
data = vbt.YFData.download('AAPL', start='2020-01-01', end='2024-01-01')

# Calculate indicators
fast_ma = vbt.MA.run(data.get('Close'), 20)
slow_ma = vbt.MA.run(data.get('Close'), 50)

# Generate signals
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# Run backtest
portfolio = vbt.Portfolio.from_signals(
    data.get('Close'),
    entries,
    exits,
    init_cash=100000,
    fees=0.001
)

# Performance metrics
print(portfolio.stats())

# Plot
portfolio.plot().show()
```

### 4. QuantConnect

Cloud-based algorithmic trading platform.

```python
from AlgorithmImports import *

class MovingAverageCross(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        self.AddEquity("AAPL", Resolution.Daily)
        
        # Create indicators
        self.fast = self.EMA("AAPL", 20)
        self.slow = self.EMA("AAPL", 50)
        
        # Warm up indicators
        self.SetWarmUp(50)
    
    def OnData(self, data):
        if self.IsWarmingUp:
            return
        
        if not self.Portfolio.Invested:
            if self.fast.Current.Value > self.slow.Current.Value:
                self.SetHoldings("AAPL", 1.0)
        elif self.fast.Current.Value < self.slow.Current.Value:
            self.Liquidate("AAPL")
```

## PyAlgoTrade

Event-driven backtesting framework.

```python
from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.technical import ma

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument
        self.__position = None
        
        # Indicators
        self.__sma20 = ma.SMA(feed[instrument].getCloseDataSeries(), 20)
        self.__sma50 = ma.SMA(feed[instrument].getCloseDataSeries(), 50)
    
    def onBars(self, bars):
        bar = bars[self.__instrument]
        
        if self.__sma20[-1] is None or self.__sma50[-1] is None:
            return
        
        if self.__position is None:
            if self.__sma20[-1] > self.__sma50[-1]:
                shares = int(self.getBroker().getCash() / bar.getClose())
                self.__position = self.enterLong(self.__instrument, shares)
        elif self.__sma20[-1] < self.__sma50[-1]:
            self.__position.exitMarket()
            self.__position = None

# Run
feed = yahoofeed.Feed()
feed.addBarsFromCSV("AAPL", "AAPL-2020.csv")

myStrategy = MyStrategy(feed, "AAPL")
myStrategy.run()
```

## Framework Comparison

### Feature Matrix

| Feature | Backtrader | Zipline | VectorBT | QuantConnect | PyAlgoTrade |
|---------|-----------|---------|----------|--------------|-------------|
| Event-driven | ✓ | ✓ | ✗ | ✓ | ✓ |
| Vectorized | Partial | ✗ | ✓ | ✗ | ✗ |
| Live Trading | Limited | ✗ | Limited | ✓ | Limited |
| Cloud-based | ✗ | ✗ | ✗ | ✓ | ✗ |
| Learning Curve | Medium | High | Low | Medium | Medium |
| Speed | Medium | Medium | Fast | Medium | Medium |

## Custom Backtesting Framework

### Simple Framework

```python
class SimpleBacktester:
    def __init__(self, data, strategy, initial_capital=100000, 
                 commission=0.001):
        self.data = data
        self.strategy = strategy
        self.capital = initial_capital
        self.commission = commission
        self.positions = 0
        self.trades = []
        self.portfolio_values = []
    
    def run(self):
        """Run backtest"""
        for i in range(len(self.data)):
            # Get signal
            signal = self.strategy.generate_signal(self.data.iloc[:i+1])
            price = self.data['Close'].iloc[i]
            
            # Execute trade
            if signal == 1 and self.positions == 0:  # Buy
                shares = int(self.capital / (price * (1 + self.commission)))
                cost = shares * price * (1 + self.commission)
                
                if cost <= self.capital:
                    self.positions = shares
                    self.capital -= cost
                    self.trades.append({
                        'date': self.data.index[i],
                        'action': 'BUY',
                        'shares': shares,
                        'price': price
                    })
            
            elif signal == -1 and self.positions > 0:  # Sell
                proceeds = self.positions * price * (1 - self.commission)
                self.capital += proceeds
                self.trades.append({
                    'date': self.data.index[i],
                    'action': 'SELL',
                    'shares': self.positions,
                    'price': price
                })
                self.positions = 0
            
            # Record portfolio value
            portfolio_value = self.capital + (self.positions * price)
            self.portfolio_values.append(portfolio_value)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        return {
            'total_return': (self.portfolio_values[-1] / 
                           self.portfolio_values[0]) - 1,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_drawdown': self._max_drawdown(),
            'num_trades': len(self.trades)
        }
    
    def _max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative = pd.Series(self.portfolio_values)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

## Best Practices

1. **Out-of-Sample Testing**: Always validate on unseen data
2. **Transaction Costs**: Include realistic costs
3. **Slippage**: Account for execution delays
4. **Position Sizing**: Use proper risk management
5. **Walk-Forward**: Regular re-optimization
6. **Multiple Assets**: Test portfolio strategies
7. **Different Periods**: Test across various market conditions

## Practice Exercises

1. **Implement a Strategy**:
   - Choose a framework
   - Code a moving average strategy
   - Optimize parameters
   - Analyze results

2. **Compare Frameworks**:
   - Implement same strategy in 2+ frameworks
   - Compare performance
   - Evaluate ease of use

3. **Build Custom Framework**:
   - Design event-driven system
   - Add order management
   - Implement metrics

## Next Steps

- [Machine Learning](/docs/tools-libraries/machine-learning)
- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Risk Management](/docs/financial-markets/risk-management)
