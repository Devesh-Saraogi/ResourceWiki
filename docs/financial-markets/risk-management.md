---
sidebar_position: 3
---

# Risk Management

Effective risk management is crucial for long-term success in quantitative trading. Learn how to measure, monitor, and control risk.

## Core Risk Concepts

### Value at Risk (VaR)

VaR estimates the maximum potential loss over a specific time period at a given confidence level.

```python
import numpy as np
from scipy import stats

def historical_var(returns, confidence_level=0.95):
    """
    Calculate VaR using historical simulation
    """
    return np.percentile(returns, (1 - confidence_level) * 100)

def parametric_var(returns, confidence_level=0.95):
    """
    Calculate VaR assuming normal distribution
    """
    mean = np.mean(returns)
    std = np.std(returns)
    z_score = stats.norm.ppf(1 - confidence_level)
    return mean + z_score * std

# Example
returns = np.random.randn(1000) * 0.02  # Daily returns
var_95 = historical_var(returns, 0.95)
print(f"95% VaR: {var_95:.2%}")
```

### Conditional Value at Risk (CVaR/ES)

Expected shortfall - average loss beyond VaR.

```python
def conditional_var(returns, confidence_level=0.95):
    """
    Calculate CVaR (Expected Shortfall)
    """
    var = historical_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Example
cvar_95 = conditional_var(returns, 0.95)
print(f"95% CVaR: {cvar_95:.2%}")
```

## Position Sizing

### Fixed Fractional Method

Risk a fixed percentage of capital per trade.

```python
def fixed_fractional_sizing(capital, risk_per_trade, entry_price, stop_loss):
    """
    Calculate position size based on fixed risk percentage
    
    Parameters:
    - capital: Total trading capital
    - risk_per_trade: Percentage of capital to risk (e.g., 0.02 for 2%)
    - entry_price: Entry price for the trade
    - stop_loss: Stop loss price
    """
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    position_size = risk_amount / price_risk
    
    return int(position_size)

# Example
capital = 100000
position = fixed_fractional_sizing(
    capital=capital,
    risk_per_trade=0.02,  # Risk 2% per trade
    entry_price=100,
    stop_loss=95
)
print(f"Position size: {position} shares")
```

### Kelly Criterion

Optimal position sizing based on edge and odds.

```python
def kelly_criterion(win_prob, win_loss_ratio):
    """
    Calculate optimal position size using Kelly Criterion
    
    Parameters:
    - win_prob: Probability of winning
    - win_loss_ratio: Average win / Average loss
    """
    kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    
    # Use fractional Kelly for safety
    fractional_kelly = kelly * 0.5  # Half Kelly
    
    return max(0, fractional_kelly)

# Example
kelly_size = kelly_criterion(win_prob=0.55, win_loss_ratio=2.0)
print(f"Kelly position size: {kelly_size:.2%} of capital")
```

## Portfolio Risk

### Portfolio Variance

```python
def portfolio_variance(weights, cov_matrix):
    """
    Calculate portfolio variance
    """
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio standard deviation
    """
    return np.sqrt(portfolio_variance(weights, cov_matrix))

# Example
weights = np.array([0.4, 0.3, 0.3])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.01],
    [0.02, 0.01, 0.16]
])

port_vol = portfolio_volatility(weights, cov_matrix)
print(f"Portfolio Volatility: {port_vol:.2%}")
```

### Beta and Correlation

```python
def calculate_beta(asset_returns, market_returns):
    """
    Calculate beta - sensitivity to market movements
    """
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

def calculate_correlation(returns1, returns2):
    """
    Calculate correlation between two assets
    """
    return np.corrcoef(returns1, returns2)[0, 1]
```

## Risk Limits

### Trading Limits Framework

```python
class RiskLimits:
    def __init__(self, max_position_size, max_portfolio_var, 
                 max_concentration, max_leverage):
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.max_concentration = max_concentration
        self.max_leverage = max_leverage
    
    def check_position_limit(self, position_value, portfolio_value):
        """Check if position exceeds size limit"""
        position_pct = position_value / portfolio_value
        return position_pct <= self.max_position_size
    
    def check_var_limit(self, portfolio_var):
        """Check if portfolio VaR exceeds limit"""
        return portfolio_var <= self.max_portfolio_var
    
    def check_concentration(self, sector_exposure, portfolio_value):
        """Check sector concentration"""
        concentration = sector_exposure / portfolio_value
        return concentration <= self.max_concentration
    
    def check_leverage(self, total_positions, capital):
        """Check leverage ratio"""
        leverage = total_positions / capital
        return leverage <= self.max_leverage

# Example usage
limits = RiskLimits(
    max_position_size=0.10,  # 10% per position
    max_portfolio_var=0.02,  # 2% daily VaR
    max_concentration=0.25,  # 25% per sector
    max_leverage=2.0         # 2x leverage
)
```

## Stop Loss Strategies

### Fixed Stop Loss

```python
def apply_stop_loss(entry_price, current_price, stop_loss_pct, position):
    """
    Check if stop loss is triggered
    
    Parameters:
    - entry_price: Entry price
    - current_price: Current market price
    - stop_loss_pct: Stop loss percentage (e.g., 0.05 for 5%)
    - position: 1 for long, -1 for short
    """
    if position == 1:  # Long position
        stop_price = entry_price * (1 - stop_loss_pct)
        return current_price <= stop_price
    else:  # Short position
        stop_price = entry_price * (1 + stop_loss_pct)
        return current_price >= stop_price
```

### Trailing Stop Loss

```python
class TrailingStopLoss:
    def __init__(self, initial_stop_pct, trailing_pct):
        self.initial_stop_pct = initial_stop_pct
        self.trailing_pct = trailing_pct
        self.highest_price = None
        self.stop_price = None
    
    def update(self, entry_price, current_price, position=1):
        """
        Update trailing stop loss
        """
        if self.highest_price is None:
            self.highest_price = entry_price
            self.stop_price = entry_price * (1 - self.initial_stop_pct)
        
        # Update highest price for long position
        if position == 1 and current_price > self.highest_price:
            self.highest_price = current_price
            new_stop = current_price * (1 - self.trailing_pct)
            self.stop_price = max(self.stop_price, new_stop)
        
        return current_price <= self.stop_price
```

## Stress Testing

### Scenario Analysis

```python
def stress_test_portfolio(returns, scenarios):
    """
    Test portfolio performance under stress scenarios
    
    scenarios: dict of {scenario_name: factor}
    """
    results = {}
    
    for scenario, factor in scenarios.items():
        stressed_returns = returns * factor
        portfolio_loss = stressed_returns.sum()
        results[scenario] = portfolio_loss
    
    return results

# Example scenarios
scenarios = {
    'market_crash': -3.0,      # 3x normal volatility
    'black_swan': -5.0,        # 5x normal volatility
    'moderate_stress': -1.5    # 1.5x normal volatility
}

stress_results = stress_test_portfolio(returns, scenarios)
```

### Monte Carlo Simulation

```python
def monte_carlo_var(returns, n_simulations=10000, horizon=10):
    """
    Calculate VaR using Monte Carlo simulation
    """
    mean = returns.mean()
    std = returns.std()
    
    # Generate random scenarios
    simulated_returns = np.random.normal(mean, std, 
                                        (n_simulations, horizon))
    
    # Calculate cumulative returns
    cumulative_returns = simulated_returns.sum(axis=1)
    
    # Calculate VaR at 95% confidence
    var_95 = np.percentile(cumulative_returns, 5)
    
    return var_95
```

## Diversification

### Correlation-Based Diversification

```python
def diversification_ratio(weights, cov_matrix):
    """
    Calculate diversification ratio
    Higher ratio = better diversification
    """
    # Weighted average volatility
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol = np.dot(weights, individual_vols)
    
    # Portfolio volatility
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    
    return weighted_vol / portfolio_vol

# Example
div_ratio = diversification_ratio(weights, cov_matrix)
print(f"Diversification Ratio: {div_ratio:.2f}")
```

## Risk-Adjusted Returns

### Sharpe Ratio

```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
```

### Information Ratio

```python
def information_ratio(strategy_returns, benchmark_returns):
    """
    Calculate Information Ratio
    Measures excess return per unit of tracking error
    """
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = excess_returns.std()
    return np.sqrt(252) * excess_returns.mean() / tracking_error
```

### Calmar Ratio

```python
def calmar_ratio(returns, window=36):
    """
    Calculate Calmar Ratio
    Annualized return / Maximum drawdown
    """
    # Annualized return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    
    return annualized_return / max_dd if max_dd != 0 else 0
```

## Risk Monitoring

### Real-Time Risk Dashboard

```python
class RiskMonitor:
    def __init__(self, portfolio, limits):
        self.portfolio = portfolio
        self.limits = limits
        self.alerts = []
    
    def check_all_limits(self):
        """Check all risk limits"""
        checks = {
            'position_size': self._check_position_sizes(),
            'var_limit': self._check_var(),
            'concentration': self._check_concentration(),
            'leverage': self._check_leverage()
        }
        
        # Generate alerts for violations
        for check, passed in checks.items():
            if not passed:
                self.alerts.append(f"ALERT: {check} limit breached")
        
        return all(checks.values())
    
    def _check_position_sizes(self):
        """Check individual position sizes"""
        for position in self.portfolio.positions:
            if position.value > self.limits.max_position_size:
                return False
        return True
    
    def _check_var(self):
        """Check portfolio VaR"""
        portfolio_var = self.portfolio.calculate_var()
        return portfolio_var <= self.limits.max_portfolio_var
    
    def _check_concentration(self):
        """Check sector concentration"""
        # Implementation depends on portfolio structure
        return True
    
    def _check_leverage(self):
        """Check leverage ratio"""
        leverage = self.portfolio.total_exposure / self.portfolio.capital
        return leverage <= self.limits.max_leverage
```

## Best Practices

1. **Diversify**: Don't put all eggs in one basket
2. **Size Appropriately**: Use position sizing methods
3. **Set Stop Losses**: Protect against large losses
4. **Monitor Continuously**: Regular risk assessment
5. **Stress Test**: Prepare for extreme scenarios
6. **Document**: Keep records of risk decisions
7. **Review Regularly**: Adjust limits as needed

## Common Mistakes

1. **Over-Leveraging**: Using too much borrowed capital
2. **Ignoring Correlations**: Thinking diversification works when assets move together
3. **No Stop Losses**: Letting losses run
4. **Position Too Large**: Single position dominates portfolio
5. **Ignoring Black Swans**: Not preparing for rare events

## Practice Exercises

1. **Calculate VaR**: 
   - Using historical method
   - Using parametric method
   - Using Monte Carlo

2. **Implement Position Sizing**:
   - Fixed fractional
   - Kelly criterion
   - Compare results

3. **Build Risk Monitor**:
   - Check multiple limits
   - Generate alerts
   - Create dashboard

## Next Steps

- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Portfolio Optimization](/docs/quantitative-analysis/portfolio-optimization)
