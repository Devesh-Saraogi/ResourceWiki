---
sidebar_position: 3
---

# Machine Learning for Quant Finance

Apply machine learning techniques to quantitative trading and financial analysis.

## Essential Libraries

### Scikit-Learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare features
def create_features(df):
    """Create ML features from price data"""
    df = df.copy()
    
    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd'] = calculate_macd(df['Close'])
    
    # Lag features
    for i in range(1, 6):
        df[f'return_lag_{i}'] = df['returns'].shift(i)
    
    # Target (next day direction)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    return df.dropna()

# Train model
features = ['sma_20', 'sma_50', 'rsi', 'macd'] + \
           [f'return_lag_{i}' for i in range(1, 6)]

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))
```

### XGBoost

Powerful gradient boosting library.

```python
import xgboost as xgb

# Prepare data
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'eval_metric': 'auc'
}

# Train
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtest, 'test')],
    early_stopping_rounds=10
)

# Predictions
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Feature importance
importance = model.get_score(importance_type='gain')
```

## Deep Learning with TensorFlow/Keras

### LSTM for Time Series

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']].values)

X, y = create_sequences(scaled_data, seq_length=60)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, 
                     input_shape=(X_train.shape[1], 1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(25),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
```

### Transformer for Financial Data

```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

def create_transformer_model(seq_length, n_features):
    """Create Transformer model"""
    inputs = keras.Input(shape=(seq_length, n_features))
    
    # Multi-head attention
    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(inputs, inputs)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Feed forward
    ff = keras.layers.Dense(128, activation='relu')(x)
    ff = keras.layers.Dense(n_features)(ff)
    
    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    
    # Output
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

## Reinforcement Learning

### Q-Learning for Trading

```python
import gym
import numpy as np

class TradingEnvironment(gym.Env):
    """Custom Trading Environment"""
    
    def __init__(self, df):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.positions = 0
        self.capital = 10000
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
    
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.positions = 0
        self.capital = 10000
        return self._get_observation()
    
    def step(self, action):
        """Take action"""
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Execute action
        if action == 1:  # Buy
            shares = int(self.capital / current_price)
            self.positions += shares
            self.capital -= shares * current_price
        elif action == 2:  # Sell
            self.capital += self.positions * current_price
            self.positions = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        portfolio_value = self.capital + (self.positions * current_price)
        reward = portfolio_value - 10000
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get current state"""
        # Return features for current step
        return np.array([
            self.df['Close'].iloc[self.current_step],
            self.df['Volume'].iloc[self.current_step],
            # Add more features...
        ])
```

### Deep Q-Network (DQN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        """Build neural network"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train on batch"""
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

## Feature Engineering

### Advanced Features

```python
def create_advanced_features(df):
    """Create comprehensive feature set"""
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / \
                        (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Trend features
    df['trend_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['trend_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    return df
```

## Model Evaluation

### Walk-Forward Validation

```python
def walk_forward_validation(X, y, model, n_splits=5):
    """
    Walk-forward validation for time series
    """
    results = []
    test_size = len(X) // n_splits
    
    for i in range(n_splits):
        # Split data
        train_end = len(X) - (n_splits - i) * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'fold': i + 1,
            'accuracy': accuracy,
            'test_size': len(y_test)
        })
    
    return pd.DataFrame(results)
```

## Best Practices

1. **Avoid Look-Ahead Bias**: Only use past data
2. **Feature Scaling**: Normalize/standardize features
3. **Cross-Validation**: Use time-series aware validation
4. **Regularization**: Prevent overfitting
5. **Ensemble Methods**: Combine multiple models
6. **Monitor Performance**: Track out-of-sample results
7. **Retrain Regularly**: Update models with new data

## Practice Exercises

1. **Price Prediction**:
   - Build LSTM model
   - Predict next day's price
   - Evaluate accuracy

2. **Direction Classification**:
   - Create binary classifier
   - Predict up/down
   - Optimize features

3. **RL Trading Agent**:
   - Implement Q-learning
   - Train on historical data
   - Test performance

## Next Steps

- [Backtesting](/docs/quantitative-analysis/backtesting)
- [Statistical Methods](/docs/quantitative-analysis/statistical-methods)
- [Data Sources](/docs/tools-libraries/data-sources)
