#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# In[32]:


# Step 1: Fetch NVIDIA stock data
df = yf.download('NVDA', start='2001-01-01', end='2024-12-25')


# In[33]:


df.head()


# In[34]:


# Step 2: Feature engineering
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band


# In[35]:


# Add technical indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
df['RSI'] = compute_rsi(df['Close'])                  # Relative Strength Index
df['Price_Change'] = df['Close'].pct_change()        # Daily percentage change

df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])


# In[36]:


# Define target variable with a threshold for movement
df['Target'] = ((df['Close'].shift(-1) - df['Close']) > 0.01).astype(int)  # Predict significant upward movement


# In[37]:


df.head()


# In[38]:


df = df.dropna()  # Drop rows with NaN values


# In[88]:


df = df.reset_index()


# In[89]:


# Step 3: Train and test the model
features = ['SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
X = df[features]
y = df['Target']


# In[90]:


# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[91]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
y_scaled = y_resampled


# In[92]:


X_scaled


# In[93]:


y_scaled.value_counts()


# In[ ]:


# Perform hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)
print("Best Parameters:", grid_search.best_params_)


# In[ ]:


# Train the model with best parameters
best_model = grid_search.best_estimator_


# In[ ]:


# Use cross-validation to evaluate the model
scores = cross_val_score(best_model, X_scaled, y_resampled, cv=5, scoring='f1')
print("Cross-Validation Accuracy:", scores)
print("Mean Cross-Validation Accuracy:", scores.mean())


# In[ ]:


# Split into training and testing sets for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)


# In[ ]:


# Evaluate the model
y_pred = best_model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))


# In[ ]:


# Step 4: Predict stock movement
latest_data = df.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)
prediction = best_model.predict(X_latest)
print("Prediction for the next day: The stock will go", 'Up' if prediction[0] == 1 else 'Down')


# In[ ]:


# Plot the stock prices with SMA and Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['SMA_10'], label='SMA 10', linestyle='--')
plt.plot(df['SMA_50'], label='SMA 50', linestyle='--')
plt.plot(df['BB_Upper'], label='Bollinger Upper Band', linestyle='--')
plt.plot(df['BB_Lower'], label='Bollinger Lower Band', linestyle='--')
plt.legend()
plt.title("NVIDIA Stock Price with Moving Averages and Bollinger Bands")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[ ]:


# Feature importance visualization
feature_importances = pd.Series(best_model.feature_importances_, index=features)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()


# In[ ]:


import pkg_resources

# Create a requirements.txt file
with open("requirements.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(f"{dist.project_name}=={dist.version}\n")


# In[ ]:


import joblib

# Save the trained model to a file using joblib
joblib.dump(model, 'nvidia.pkl')

print("Model saved as 'nvidia.pkl'")


# In[ ]:


import joblib

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")


# In[ ]:


# Save the DataFrame to a CSV file
df.to_csv("df.csv", index=False)

print("DataFrame saved as 'historical_data.csv'")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "nvidia.ipynb"')


# In[ ]:




