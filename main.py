import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Set the page title
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Sidebar
st.sidebar.header("User Input")

# Select a stock
stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL for Apple)", "AAPL")

# Select a date range
start_date = st.sidebar.date_input("Start date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End date", pd.to_datetime('2022-01-01'))

# Download historical data
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

data = load_data(stock_symbol, start_date, end_date)

# Main content
st.title("Stock Price Prediction")

st.subheader("Historical Data")
st.write(data)

# Linear Regression Model
st.subheader("Stock Price Prediction")

# Select the number of days for prediction
n_days = st.sidebar.slider("Select the number of days for prediction", 1, 365, 30)

# Create a feature (X) and target (y) variable
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['OrdinalDate'] = data['Date'].map(lambda x: x.toordinal())

X = data[['OrdinalDate']].values
y = data['Close'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Create future dates for prediction
future_dates = pd.date_range(start=end_date, periods=n_days + 1)
future_ordinal_dates = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)

# Make predictions
predicted_prices = model.predict(future_ordinal_dates)

# Create a DataFrame for predictions
prediction_data = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_prices})
prediction_data.set_index('Date', inplace=True)

# Combine historical and predicted data for plotting
combined_data = pd.concat([data['Close'], prediction_data['Predicted Close']], axis=1)
combined_data.columns = ['Historical Close', 'Predicted Close']

# Plot the historical and predicted data
st.line_chart(combined_data)

# Display the prediction data
st.write("Predicted Price Data")
st.write(prediction_data)
