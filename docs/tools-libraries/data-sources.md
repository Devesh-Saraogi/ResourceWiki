---
sidebar_position: 1
---

# Data Sources

Access to quality financial data is essential for quantitative analysis. This guide covers various data sources and how to access them.

## Free Data Sources

### 1. Yahoo Finance (yfinance)

Most popular free data source.

```python
import yfinance as yf
import pandas as pd

# Download single stock
ticker = yf.Ticker("AAPL")

# Get historical data
hist = ticker.history(period="1y")
print(hist.head())

# Get multiple stocks
tickers = ["AAPL", "GOOGL", "MSFT"]
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")

# Get fundamental data
info = ticker.info
print(f"Market Cap: {info['marketCap']}")
print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")

# Get financials
financials = ticker.financials
balance_sheet = ticker.balance_sheet
cashflow = ticker.cashflow
```

### 2. Alpha Vantage

Comprehensive API with free tier.

```python
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# Initialize
API_KEY = 'your_api_key_here'
ts = TimeSeries(key=API_KEY, output_format='pandas')
fd = FundamentalData(key=API_KEY, output_format='pandas')

# Get intraday data
data, meta_data = ts.get_intraday(
    symbol='AAPL',
    interval='1min',
    outputsize='full'
)

# Get daily data
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Get company overview
overview, _ = fd.get_company_overview(symbol='AAPL')

# Get income statement
income_statement, _ = fd.get_income_statement_annual(symbol='AAPL')
```

### 3. Pandas DataReader

Access multiple data sources through one interface.

```python
import pandas_datareader as pdr
from datetime import datetime

start = datetime(2020, 1, 1)
end = datetime(2024, 1, 1)

# Yahoo Finance
df_yahoo = pdr.get_data_yahoo('AAPL', start, end)

# FRED (Federal Reserve Economic Data)
gdp = pdr.get_data_fred('GDP', start, end)
unemployment = pdr.get_data_fred('UNRATE', start, end)
interest_rate = pdr.get_data_fred('DFF', start, end)

# World Bank
population = pdr.wb.download(indicator='SP.POP.TOTL', 
                            country=['US'], 
                            start=2010, end=2024)
```

### 4. Quandl

Financial and economic data.

```python
import quandl

# Set API key
quandl.ApiConfig.api_key = 'your_api_key_here'

# Get data
data = quandl.get('WIKI/AAPL', start_date='2020-01-01', end_date='2024-01-01')

# Get multiple tickers
tickers = ['WIKI/AAPL', 'WIKI/GOOGL', 'WIKI/MSFT']
data = quandl.get(tickers, start_date='2020-01-01')

# Get database metadata
quandl.get('FRED/GDP')
```

## Premium Data Sources

### 1. Bloomberg Terminal

Professional-grade data (expensive).

```python
from blpapi import Session, SessionOptions, Name

# Connect to Bloomberg
sessionOptions = SessionOptions()
sessionOptions.setServerHost('localhost')
sessionOptions.setServerPort(8194)
session = Session(sessionOptions)

# Subscribe to real-time data
# (Simplified example)
```

### 2. Refinitiv Eikon

Comprehensive financial data.

```python
import eikon as ek

# Set app key
ek.set_app_key('your_app_key_here')

# Get time series
df = ek.get_timeseries(
    'AAPL.O',
    start_date='2020-01-01',
    end_date='2024-01-01',
    interval='daily'
)

# Get fundamental data
df, err = ek.get_data(
    instruments=['AAPL.O', 'GOOGL.O'],
    fields=['TR.Revenue', 'TR.GrossProfit', 'TR.EBITDA']
)
```

## Cryptocurrency Data

### 1. CoinGecko

Free crypto data API.

```python
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

# Get current price
price = cg.get_price(ids='bitcoin', vs_currencies='usd')

# Get historical data
history = cg.get_coin_market_chart_by_id(
    id='bitcoin',
    vs_currency='usd',
    days=365
)

# Get market data
markets = cg.get_coins_markets(vs_currency='usd')
```

### 2. Binance

Crypto exchange API.

```python
from binance.client import Client

client = Client(api_key='your_api_key', api_secret='your_api_secret')

# Get historical klines
klines = client.get_historical_klines(
    'BTCUSDT',
    Client.KLINE_INTERVAL_1DAY,
    '1 Jan, 2023',
    '1 Jan, 2024'
)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
])
```

## Alternative Data

### 1. News Sentiment

```python
from newsapi import NewsApiClient

# Initialize
newsapi = NewsApiClient(api_key='your_api_key')

# Get headlines
headlines = newsapi.get_everything(
    q='Apple stock',
    from_param='2024-01-01',
    to='2024-01-31',
    language='en',
    sort_by='relevancy'
)

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

for article in headlines['articles']:
    sentiment = analyzer.polarity_scores(article['title'])
    print(f"{article['title']}: {sentiment['compound']}")
```

### 2. Social Media Data

```python
import praw  # Reddit

# Reddit API
reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_client_secret',
    user_agent='your_user_agent'
)

# Get posts from WallStreetBets
subreddit = reddit.subreddit('wallstreetbets')
for submission in subreddit.hot(limit=10):
    print(submission.title, submission.score)
```

## Data Cleaning and Preprocessing

### Handling Missing Data

```python
def clean_financial_data(df):
    """Clean and preprocess financial data"""
    # Forward fill missing prices (market closed days)
    df = df.fillna(method='ffill')
    
    # Drop remaining NaN
    df = df.dropna()
    
    # Remove outliers (3 standard deviations)
    returns = df.pct_change()
    mask = (np.abs(returns - returns.mean()) <= 3 * returns.std())
    df = df[mask.all(axis=1)]
    
    # Adjust for splits and dividends
    df['Adj_Close'] = df['Close'] * (df['Adj Close'] / df['Close'])
    
    return df
```

### Resampling Data

```python
def resample_data(df, frequency='D'):
    """
    Resample OHLCV data to different frequency
    
    frequency: 'D' (day), 'W' (week), 'M' (month)
    """
    resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    return resampled
```

## Building a Data Pipeline

### Complete Data Pipeline

```python
class DataPipeline:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
    
    def fetch_data(self):
        """Fetch data from source"""
        import yfinance as yf
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                self.data[symbol] = ticker.history(
                    start=self.start_date,
                    end=self.end_date
                )
                print(f"✓ Downloaded {symbol}")
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
    
    def clean_data(self):
        """Clean and validate data"""
        for symbol in self.data:
            # Remove missing data
            self.data[symbol] = self.data[symbol].dropna()
            
            # Validate data
            if len(self.data[symbol]) == 0:
                print(f"Warning: No data for {symbol}")
    
    def calculate_features(self):
        """Calculate technical features"""
        for symbol in self.data:
            df = self.data[symbol]
            
            # Returns
            df['Returns'] = df['Close'].pct_change()
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            self.data[symbol] = df
    
    def save_data(self, filepath):
        """Save data to file"""
        for symbol, df in self.data.items():
            df.to_csv(f"{filepath}/{symbol}.csv")
    
    def run(self):
        """Run complete pipeline"""
        print("Starting data pipeline...")
        self.fetch_data()
        self.clean_data()
        self.calculate_features()
        print("Pipeline complete!")
        return self.data

# Usage
pipeline = DataPipeline(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)
data = pipeline.run()
```

## Best Practices

1. **Cache Data**: Save downloaded data to avoid repeated API calls
2. **Rate Limiting**: Respect API rate limits
3. **Error Handling**: Handle network errors and missing data
4. **Data Validation**: Always validate data quality
5. **Version Control**: Track data versions
6. **Documentation**: Document data sources and transformations

## Practice Exercises

1. **Build a Data Fetcher**:
   - Download data for multiple tickers
   - Handle errors gracefully
   - Save to local storage

2. **Create a Data Quality Report**:
   - Check for missing values
   - Identify outliers
   - Validate data ranges

3. **Build an Economic Calendar**:
   - Fetch economic indicators
   - Track release dates
   - Analyze impact on markets

## Next Steps

- [Backtesting Frameworks](/docs/tools-libraries/backtesting-frameworks)
- [Machine Learning](/docs/tools-libraries/machine-learning)
- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
