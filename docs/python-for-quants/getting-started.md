---
sidebar_position: 1
---

# Getting Started with Python for Quants

Python has become the de facto language for quantitative finance. This guide will help you set up your Python environment and understand why Python is essential for quant development.

## Why Python for Quantitative Finance?

- **Extensive Libraries**: NumPy, Pandas, SciPy for numerical computations
- **Data Analysis**: Easy data manipulation and analysis
- **Visualization**: Matplotlib, Plotly for creating charts
- **Machine Learning**: scikit-learn, TensorFlow for AI-driven strategies
- **Community**: Large community and extensive resources

## Setting Up Your Environment

### 1. Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

### 2. Install Essential Libraries

```bash
pip install numpy pandas matplotlib scipy
```

### 3. Jupyter Notebook (Recommended)

```bash
pip install jupyter
jupyter notebook
```

## Your First Quant Program

Let's calculate simple returns for a stock:

```python
import pandas as pd
import numpy as np

# Sample stock prices
prices = pd.Series([100, 102, 101, 105, 107, 106])

# Calculate simple returns
returns = prices.pct_change()

print("Prices:")
print(prices)
print("\nReturns:")
print(returns)

# Calculate mean return
mean_return = returns.mean()
print(f"\nAverage Return: {mean_return:.4f}")
```

## Key Concepts to Master

1. **Data Structures**: Lists, tuples, dictionaries, DataFrames
2. **NumPy Arrays**: Fast numerical operations
3. **Pandas DataFrames**: Time series manipulation
4. **Functions**: Writing reusable code
5. **Error Handling**: try/except blocks

## Practice Exercise

Create a function that:
1. Takes a list of stock prices
2. Calculates daily returns
3. Computes mean and standard deviation
4. Returns a dictionary with these statistics

```python
def analyze_returns(prices):
    # Your code here
    pass
```

## Next Steps

Once you're comfortable with Python basics, move on to:
- [NumPy and Pandas](/docs/python-for-quants/numpy-pandas)
- [Data Visualization](/docs/python-for-quants/data-visualization)

## Resources

- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
