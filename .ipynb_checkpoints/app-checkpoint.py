import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model and scaler
model = joblib.load('nvidia.pkl')  # Load the trained model
scaler = joblib.load('scaler.pkl')  # Load the scaler used to scale X

# Load the data
df = pd.read_csv('df.csv')

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Streamlit App
st.title('NVDA Stock Prediction App')
st.write('This app predicts whether the stock price will go up or down for future dates based on the latest available data.')

# Step 1: User input for date (even future date)
input_date = st.date_input("**Enter a Date**", min_value=datetime(2020, 1, 1), max_value=datetime(2025, 2, 1))

# Check if the input date is in the dataset
input_date_str = input_date.strftime('%Y-%m-%d')

# Get the last available date in the dataset
last_available_date = df['Date'].max()

# Convert both `input_date` and `last_available_date` to datetime.date for comparison
if input_date > last_available_date.date():  # Convert Timestamp to date for comparison
    # Use the last available date for prediction
    data_for_date = df[df['Date'] == last_available_date].drop(columns=['Date', 'Target'])
else:
    # If the input date exists, use the corresponding data
    if input_date_str not in df['Date'].dt.strftime('%Y-%m-%d').values:
        st.error(f"Data for {input_date_str} is not available in the dataset. Please choose a different date.")
    else:
        st.write(f"Data for {input_date_str} is available.")
        data_for_date = df[df['Date'] == input_date_str].drop(columns=['Date', 'Target'])

# Features used for prediction (6 features based on your previous model)
features = ['SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'MACD', 'Signal_Line']

# Select only the relevant columns (6 features)
data_for_date = data_for_date[features]  # Select only the 6 features

# Feature scaling
X_input_scaled = scaler.transform(data_for_date)  # Scale the features

# Step 3: Predict whether the stock will go up or down for the next day
prediction = model.predict(X_input_scaled)
prediction_label = 'Up' if prediction[0] == 1 else 'Down'

# Step 4: Display the prediction result
st.write(f"**Prediction for {input_date_str}: The stock will go {prediction_label}.**")


# Step 7: Display Feature Importance Visualization
st.subheader("Feature Importance Visualization")
# Feature Importance Visualization
feature_importances = pd.Series(model.feature_importances_, index=features)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='teal')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)  # Ensure this is called after plt commands



