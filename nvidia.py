#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# In[2]:


# Step 1: Fetch NVIDIA stock data
df = yf.download('NVDA', start='2001-01-01', end='2024-12-25')


# In[3]:


df.head()


# In[4]:


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


# In[5]:


# Add technical indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
df['RSI'] = compute_rsi(df['Close'])                  # Relative Strength Index
df['Price_Change'] = df['Close'].pct_change()        # Daily percentage change

df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])


# In[6]:


# Define target variable with a threshold for movement
df['Target'] = ((df['Close'].shift(-1) - df['Close']) > 0.01).astype(int)  # Predict significant upward movement


# In[7]:


df.head()


# In[8]:


df = df.dropna()  # Drop rows with NaN values


# In[9]:


df = df.reset_index()


# In[28]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# In[29]:


df.head()


# In[30]:


# Step 3: Train and test the model
features = ['SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
X = df[features]
y = df['Target']


# In[12]:


# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[13]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
y_scaled = y_resampled


# In[14]:


X_scaled


# In[15]:


y_scaled.value_counts()


# In[19]:


# Perform hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)
print("Best Parameters:", grid_search.best_params_)


# In[20]:


# Train the model with best parameters
best_model = grid_search.best_estimator_


# In[21]:


# Use cross-validation to evaluate the model
scores = cross_val_score(best_model, X_scaled, y_resampled, cv=5, scoring='f1')
print("Cross-Validation Accuracy:", scores)
print("Mean Cross-Validation Accuracy:", scores.mean())


# In[22]:


# Split into training and testing sets for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)


# In[23]:


# Evaluate the model
y_pred = best_model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))


# In[24]:


# Step 4: Predict stock movement
latest_data = df.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)
prediction = best_model.predict(X_latest)
print("Prediction for the next day: The stock will go", 'Up' if prediction[0] == 1 else 'Down')


# In[25]:


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


# In[26]:


# Feature importance visualization
feature_importances = pd.Series(best_model.feature_importances_, index=features)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()


# In[32]:


import pkg_resources

# Create a requirements.txt file
with open("requirements.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(f"{dist.project_name}=={dist.version}\n")


# In[33]:


import joblib

# Save the trained model to a file using joblib
joblib.dump(model, 'nvidia.pkl')

print("Model saved as 'nvidia.pkl'")


# In[34]:


import joblib

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")


# In[35]:


# Save the DataFrame to a CSV file
df.to_csv("df.csv", index=False)

print("DataFrame saved as 'historical_data.csv'")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "nvidia.ipynb"')


# In[27]:


# This will write the content to an app.py file
app_code = """
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model and scaler
try:
    model = joblib.load('nvidia.pkl')  # Load the trained model
    scaler = joblib.load('scaler.pkl')  # Load the scaler used to scale X
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()  # Stop further execution if model/scaler loading fails

# Load the data
try:
    df = pd.read_csv('df.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure 'Date' is datetime
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()  # Stop further execution if data loading fails

# Check the columns of the dataframe
st.write("Data columns:", df.columns)

# Streamlit App
st.title('Stock Prediction App')
st.write('This app predicts whether the stock price will go up or down based on a given date.')

# Step 1: User input for date
input_date = st.date_input("Enter a Date", min_value=datetime(2020, 1, 1), max_value=datetime(2025, 2, 1))

# Step 2: Check if the input date is in the dataset
input_date_str = input_date.strftime('%Y-%m-%d')

# Display message if the date is not available in the dataset
if input_date_str not in df['Date'].values:
    st.error(f"Data for {input_date_str} is not available in the dataset.")
else:
    st.write(f"Data for {input_date_str} is available.")
    
    # Step 3: Feature extraction based on the given date
    data_for_date = df[df['Date'] == input_date_str].drop(columns=['Date', 'Target'])  # Drop date and target
    if data_for_date.empty:
        st.error(f"No data found for {input_date_str}. Please choose a different date.")
    else:
        # Ensure no NaN values before scaling
        if data_for_date.isnull().sum().any():
            st.error("Input data contains missing values. Please try again.")
        else:
            # Feature scaling
            X_input_scaled = scaler.transform(data_for_date)  # Scale the features

            # Step 4: Predict whether the stock will go up or down
            prediction = model.predict(X_input_scaled)
            prediction_label = 'Up' if prediction[0] == 1 else 'Down'

            # Step 5: Display the prediction result
            st.write(f"Prediction for {input_date_str}: The stock will go {prediction_label}.")
"""

# Writing the code to 'app.py'
with open("app.py", "w") as f:
    f.write(app_code)

print("app.py has been written successfully!")


# In[ ]:




