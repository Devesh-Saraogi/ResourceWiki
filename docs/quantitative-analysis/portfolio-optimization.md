---
sidebar_position: 3
---

# Portfolio Optimization

Learn how to construct optimal portfolios using modern portfolio theory and advanced optimization techniques.

## Modern Portfolio Theory (MPT)

### Mean-Variance Optimization

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Negative Sharpe ratio for minimization"""
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (p_return - risk_free_rate) / p_std
    return -sharpe

def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Optimize for maximum Sharpe ratio
    """
    n_assets = len(mean_returns)
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds (0 to 1 for each weight)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    init_guess = n_assets * [1. / n_assets]
    
    # Optimize
    result = minimize(negative_sharpe, init_guess,
                     args=(mean_returns, cov_matrix, risk_free_rate),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

# Example
returns = pd.DataFrame({
    'Stock_A': np.random.randn(252) * 0.01,
    'Stock_B': np.random.randn(252) * 0.015,
    'Stock_C': np.random.randn(252) * 0.02
})

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

optimal_weights = max_sharpe_portfolio(mean_returns, cov_matrix)
print("Optimal Weights:", optimal_weights)
```

### Efficient Frontier

```python
def efficient_frontier(mean_returns, cov_matrix, n_portfolios=100):
    """
    Generate efficient frontier
    """
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    for i in range(n_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        # Calculate return and risk
        portfolio_return, portfolio_std = portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        
        # Calculate Sharpe ratio
        sharpe = portfolio_return / portfolio_std
        
        # Store results
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = sharpe
        weights_record.append(weights)
    
    return results, weights_record

def plot_efficient_frontier(mean_returns, cov_matrix):
    """Plot the efficient frontier"""
    import matplotlib.pyplot as plt
    
    results, weights = efficient_frontier(mean_returns, cov_matrix, 10000)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], 
               cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    
    # Plot maximum Sharpe ratio portfolio
    max_sharpe_idx = np.argmax(results[2,:])
    plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx],
               marker='*', color='red', s=500, label='Max Sharpe')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Risk Parity

Equal risk contribution from each asset.

```python
def risk_parity_weights(cov_matrix):
    """
    Calculate risk parity portfolio weights
    Each asset contributes equally to portfolio risk
    """
    def risk_budget_objective(weights, cov_matrix):
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        # Minimize difference from equal risk contribution
        target = np.ones(len(weights)) / len(weights)
        return np.sum((risk_contrib - target)**2)
    
    n_assets = cov_matrix.shape[0]
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]
    
    result = minimize(risk_budget_objective, init_guess,
                     args=(cov_matrix,), method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x
```

## Black-Litterman Model

Combines market equilibrium with investor views.

```python
def black_litterman(market_weights, cov_matrix, risk_aversion, 
                   P, Q, omega):
    """
    Black-Litterman asset allocation
    
    Parameters:
    - market_weights: Market capitalization weights
    - cov_matrix: Covariance matrix
    - risk_aversion: Risk aversion coefficient
    - P: Views matrix
    - Q: Views vector
    - omega: Uncertainty in views
    """
    # Implied equilibrium returns
    pi = risk_aversion * np.dot(cov_matrix, market_weights)
    
    # Posterior estimates
    tau = 0.05  # Scaling factor
    
    # Posterior covariance
    M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + 
                              np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
    
    # Posterior returns
    posterior_returns = np.dot(M_inverse,
                              (np.dot(np.linalg.inv(tau * cov_matrix), pi) +
                               np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)))
    
    # Optimal weights
    optimal_weights = np.dot(M_inverse, posterior_returns) / risk_aversion
    
    return optimal_weights, posterior_returns
```

## Minimum Variance Portfolio

```python
def minimum_variance_portfolio(cov_matrix):
    """
    Calculate minimum variance portfolio
    """
    n_assets = cov_matrix.shape[0]
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]
    
    result = minimize(portfolio_variance, init_guess,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

## Maximum Diversification

```python
def maximum_diversification_portfolio(mean_returns, cov_matrix):
    """
    Maximize diversification ratio
    """
    n_assets = len(mean_returns)
    
    def neg_diversification_ratio(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights, asset_vols)
        return -weighted_vol / portfolio_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]
    
    result = minimize(neg_diversification_ratio, init_guess,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

## Hierarchical Risk Parity (HRP)

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def hierarchical_risk_parity(returns):
    """
    Hierarchical Risk Parity portfolio allocation
    """
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Convert correlation to distance
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    
    # Hierarchical clustering
    link = linkage(squareform(dist_matrix.values), method='single')
    
    # Get quasi-diagonal matrix
    sort_ix = _get_quasi_diag(link)
    sort_ix = corr_matrix.index[sort_ix].tolist()
    
    # Calculate weights
    hrp_weights = _get_recursive_bisection(returns[sort_ix].cov())
    
    return hrp_weights

def _get_quasi_diag(link):
    """Get quasi-diagonal order"""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i+1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()

def _get_recursive_bisection(cov, sort_ix=None):
    """Recursive bisection for HRP"""
    if sort_ix is None:
        sort_ix = list(range(cov.shape[0]))
    
    weights = pd.Series(1, index=sort_ix)
    clusters = [sort_ix]
    
    while len(clusters) > 0:
        clusters = [cluster[start:end] 
                   for cluster in clusters 
                   for start, end in ((0, len(cluster)//2), 
                                     (len(cluster)//2, len(cluster)))
                   if len(cluster) > 1]
        
        for i in range(0, len(clusters), 2):
            cluster0 = clusters[i]
            cluster1 = clusters[i+1]
            
            cov0 = cov.iloc[cluster0, cluster0]
            cov1 = cov.iloc[cluster1, cluster1]
            
            w0 = 1 / np.diag(cov0).sum()
            w1 = 1 / np.diag(cov1).sum()
            alpha = w0 / (w0 + w1)
            
            weights[cluster0] *= alpha
            weights[cluster1] *= 1 - alpha
    
    return weights
```

## Constraints and Bounds

### Portfolio with Constraints

```python
def optimized_portfolio_with_constraints(mean_returns, cov_matrix,
                                        target_return=None,
                                        min_weights=None,
                                        max_weights=None,
                                        sector_constraints=None):
    """
    Portfolio optimization with various constraints
    """
    n_assets = len(mean_returns)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints list
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Target return constraint
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.dot(x, mean_returns) - target_return
        })
    
    # Sector constraints
    if sector_constraints is not None:
        for sector_indices, min_weight, max_weight in sector_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=sector_indices: np.sum(x[idx]) - min_weight
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=sector_indices: max_weight - np.sum(x[idx])
            })
    
    # Bounds
    if min_weights is None:
        min_weights = np.zeros(n_assets)
    if max_weights is None:
        max_weights = np.ones(n_assets)
    
    bounds = tuple(zip(min_weights, max_weights))
    
    init_guess = n_assets * [1. / n_assets]
    
    result = minimize(portfolio_variance, init_guess,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

## Rebalancing Strategies

### Periodic Rebalancing

```python
def periodic_rebalancing(returns, target_weights, rebalance_frequency='Q'):
    """
    Simulate periodic rebalancing
    
    Parameters:
    - returns: DataFrame of asset returns
    - target_weights: Target portfolio weights
    - rebalance_frequency: 'D', 'W', 'M', 'Q', 'Y'
    """
    portfolio_values = []
    current_weights = target_weights.copy()
    
    rebalance_dates = returns.resample(rebalance_frequency).last().index
    
    portfolio_value = 1.0
    
    for date in returns.index:
        # Update portfolio value based on returns
        daily_return = np.dot(current_weights, returns.loc[date])
        portfolio_value *= (1 + daily_return)
        
        # Update weights
        current_weights *= (1 + returns.loc[date])
        current_weights /= current_weights.sum()
        
        # Rebalance if needed
        if date in rebalance_dates:
            current_weights = target_weights.copy()
        
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=returns.index)
```

### Threshold Rebalancing

```python
def threshold_rebalancing(returns, target_weights, threshold=0.05):
    """
    Rebalance when drift exceeds threshold
    """
    portfolio_values = []
    current_weights = target_weights.copy()
    portfolio_value = 1.0
    rebalance_count = 0
    
    for date in returns.index:
        # Update portfolio
        daily_return = np.dot(current_weights, returns.loc[date])
        portfolio_value *= (1 + daily_return)
        
        # Update weights
        current_weights *= (1 + returns.loc[date])
        current_weights /= current_weights.sum()
        
        # Check if rebalancing needed
        weight_drift = np.abs(current_weights - target_weights).max()
        
        if weight_drift > threshold:
            current_weights = target_weights.copy()
            rebalance_count += 1
        
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=returns.index), rebalance_count
```

## Best Practices

1. **Diversification**: Don't over-concentrate
2. **Rebalancing**: Regular review and adjustment
3. **Transaction Costs**: Consider costs when rebalancing
4. **Estimation Error**: Returns are harder to estimate than risk
5. **Robustness**: Test across different market conditions
6. **Constraints**: Use realistic constraints

## Practice Exercises

1. **Efficient Frontier**:
   - Calculate for 5 assets
   - Plot the frontier
   - Find maximum Sharpe portfolio

2. **Risk Parity**:
   - Implement risk parity
   - Compare to equal weighting
   - Analyze risk contributions

3. **Constrained Optimization**:
   - Add sector constraints
   - Set min/max weights
   - Compare results

## Next Steps

- [Data Sources](/docs/tools-libraries/data-sources)
- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Risk Management](/docs/financial-markets/risk-management)
