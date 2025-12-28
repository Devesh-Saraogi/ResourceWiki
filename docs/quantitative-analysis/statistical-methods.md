---
sidebar_position: 1
---

# Statistical Methods

Statistical analysis forms the foundation of quantitative trading. Learn the essential statistical methods used in finance.

## Descriptive Statistics

### Basic Measures

```python
import numpy as np
import pandas as pd
from scipy import stats

def calculate_basic_stats(returns):
    """Calculate basic statistical measures"""
    statistics = {
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns),
        'variance': np.var(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns)
    }
    return statistics

# Example
returns = np.random.randn(1000) * 0.01
stats_dict = calculate_basic_stats(returns)
for key, value in stats_dict.items():
    print(f"{key}: {value:.4f}")
```

### Distribution Analysis

```python
def test_normality(returns):
    """
    Test if returns follow normal distribution
    """
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    # Shapiro-Wilk test
    sw_stat, sw_pvalue = stats.shapiro(returns)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(returns, 'norm')
    
    results = {
        'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
        'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pvalue},
        'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_pvalue}
    }
    
    return results
```

## Time Series Analysis

### Stationarity Tests

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series):
    """
    Test if time series is stationary
    """
    # Augmented Dickey-Fuller test
    adf_result = adfuller(series)
    
    # KPSS test
    kpss_result = kpss(series)
    
    results = {
        'adf': {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        },
        'kpss': {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'is_stationary': kpss_result[1] > 0.05
        }
    }
    
    return results
```

### Autocorrelation

```python
def calculate_autocorrelation(series, lags=20):
    """
    Calculate autocorrelation for different lags
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    acf_values = acf(series, nlags=lags)
    pacf_values = pacf(series, nlags=lags)
    
    return {
        'acf': acf_values,
        'pacf': pacf_values
    }
```

### ARIMA Modeling

```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(series, order=(1, 1, 1)):
    """
    Fit ARIMA model to time series
    
    Parameters:
    - order: (p, d, q) where p=AR order, d=differencing, q=MA order
    """
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    
    return {
        'model': fitted_model,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'summary': fitted_model.summary()
    }
```

## Correlation and Cointegration

### Correlation Analysis

```python
def correlation_analysis(returns_df):
    """
    Analyze correlations between multiple assets
    """
    # Pearson correlation
    pearson_corr = returns_df.corr(method='pearson')
    
    # Spearman correlation (rank-based)
    spearman_corr = returns_df.corr(method='spearman')
    
    # Kendall tau correlation
    kendall_corr = returns_df.corr(method='kendall')
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'kendall': kendall_corr
    }
```

### Cointegration Testing

```python
from statsmodels.tsa.stattools import coint

def test_cointegration(series1, series2):
    """
    Test if two time series are cointegrated
    """
    score, pvalue, _ = coint(series1, series2)
    
    is_cointegrated = pvalue < 0.05
    
    return {
        'score': score,
        'p_value': pvalue,
        'is_cointegrated': is_cointegrated
    }

def find_cointegrated_pairs(prices_df, significance=0.05):
    """
    Find all cointegrated pairs in a dataset
    """
    n = prices_df.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = prices_df.columns
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            result = test_cointegration(prices_df[keys[i]], 
                                       prices_df[keys[j]])
            pvalue_matrix[i, j] = result['p_value']
            
            if result['is_cointegrated']:
                pairs.append((keys[i], keys[j], result['p_value']))
    
    return pairs, pvalue_matrix
```

## Regression Analysis

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

def simple_linear_regression(X, y):
    """
    Perform simple linear regression
    """
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    
    predictions = model.predict(X.reshape(-1, 1))
    r_squared = model.score(X.reshape(-1, 1), y)
    
    return {
        'coefficients': {'intercept': model.intercept_, 
                        'slope': model.coef_[0]},
        'r_squared': r_squared,
        'predictions': predictions
    }
```

### Multiple Regression

```python
def multiple_regression(X, y):
    """
    Perform multiple linear regression
    """
    from statsmodels.api import OLS, add_constant
    
    X_with_const = add_constant(X)
    model = OLS(y, X_with_const)
    results = model.fit()
    
    return {
        'coefficients': results.params,
        'p_values': results.pvalues,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'summary': results.summary()
    }
```

### CAPM Beta Estimation

```python
def calculate_capm_beta(asset_returns, market_returns):
    """
    Calculate CAPM beta using regression
    """
    # Remove NaN values
    data = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()
    
    # Regression
    X = data['market'].values.reshape(-1, 1)
    y = data['asset'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    beta = model.coef_[0]
    alpha = model.intercept_
    r_squared = model.score(X, y)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared
    }
```

## Hypothesis Testing

### T-Test

```python
def perform_t_test(sample1, sample2):
    """
    Perform independent two-sample t-test
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Chi-Square Test

```python
def chi_square_test(observed, expected):
    """
    Perform chi-square goodness of fit test
    """
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## Volatility Estimation

### Historical Volatility

```python
def historical_volatility(returns, window=30, annualize=True):
    """
    Calculate rolling historical volatility
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days
    
    return vol
```

### GARCH Model

```python
from arch import arch_model

def fit_garch(returns, p=1, q=1):
    """
    Fit GARCH model to estimate volatility
    
    Parameters:
    - p: GARCH order
    - q: ARCH order
    """
    # Rescale returns to percentage
    returns_pct = returns * 100
    
    model = arch_model(returns_pct, vol='Garch', p=p, q=q)
    fitted_model = model.fit(disp='off')
    
    # Get conditional volatility
    conditional_vol = fitted_model.conditional_volatility / 100
    
    return {
        'model': fitted_model,
        'conditional_volatility': conditional_vol,
        'params': fitted_model.params
    }
```

### Exponentially Weighted Moving Average (EWMA)

```python
def ewma_volatility(returns, lambda_param=0.94):
    """
    Calculate EWMA volatility (RiskMetrics approach)
    """
    var = np.zeros(len(returns))
    var[0] = returns[0]**2
    
    for t in range(1, len(returns)):
        var[t] = lambda_param * var[t-1] + (1 - lambda_param) * returns[t]**2
    
    volatility = np.sqrt(var)
    return volatility
```

## Monte Carlo Simulation

### Basic Monte Carlo

```python
def monte_carlo_simulation(S0, mu, sigma, T, n_simulations=10000, n_steps=252):
    """
    Monte Carlo simulation for stock prices (Geometric Brownian Motion)
    
    Parameters:
    - S0: Initial stock price
    - mu: Expected annual return
    - sigma: Annual volatility
    - T: Time horizon in years
    - n_simulations: Number of simulation paths
    - n_steps: Number of time steps
    """
    dt = T / n_steps
    
    # Generate random returns
    returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        (n_simulations, n_steps)
    )
    
    # Calculate price paths
    price_paths = S0 * np.exp(np.cumsum(returns, axis=1))
    
    return price_paths
```

### Value at Risk via Monte Carlo

```python
def monte_carlo_var(portfolio_value, returns, confidence_level=0.95, 
                    n_simulations=10000, horizon=10):
    """
    Calculate VaR using Monte Carlo simulation
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate future returns
    simulated_returns = np.random.normal(mu, sigma, (n_simulations, horizon))
    
    # Calculate portfolio values
    final_values = portfolio_value * (1 + simulated_returns.sum(axis=1))
    
    # Calculate losses
    losses = portfolio_value - final_values
    
    # Calculate VaR
    var = np.percentile(losses, confidence_level * 100)
    
    return var
```

## Bootstrap Methods

### Bootstrap Confidence Intervals

```python
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, 
                                 confidence=0.95):
    """
    Calculate bootstrap confidence interval
    
    Parameters:
    - data: Original dataset
    - statistic_func: Function to calculate statistic (e.g., np.mean)
    - n_bootstrap: Number of bootstrap samples
    - confidence: Confidence level
    """
    bootstrap_statistics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), 
                                           replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(stat)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
    upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)
    
    return {
        'lower': lower,
        'upper': upper,
        'bootstrap_distribution': bootstrap_statistics
    }
```

## Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

def perform_pca(returns_df, n_components=None):
    """
    Perform PCA on returns data
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(returns_df)
    
    return {
        'principal_components': principal_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_
    }
```

## Best Practices

1. **Check Assumptions**: Verify statistical assumptions before applying methods
2. **Test for Robustness**: Use multiple methods to confirm results
3. **Handle Outliers**: Identify and treat extreme values appropriately
4. **Avoid P-Hacking**: Don't cherry-pick results
5. **Use Appropriate Tests**: Choose tests suitable for your data distribution

## Practice Exercises

1. **Distribution Analysis**:
   - Load stock returns data
   - Test for normality
   - Calculate skewness and kurtosis
   - Compare to normal distribution

2. **Correlation Study**:
   - Find cointegrated pairs
   - Calculate rolling correlations
   - Identify regime changes

3. **Volatility Modeling**:
   - Compare historical, EWMA, and GARCH volatility
   - Forecast future volatility
   - Evaluate model accuracy

## Next Steps

- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Portfolio Optimization](/docs/quantitative-analysis/portfolio-optimization)
- [Machine Learning](/docs/tools-libraries/machine-learning)
