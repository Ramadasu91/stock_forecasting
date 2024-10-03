import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt

# Function to preprocess the dataset for multiple targets
def preprocess_data(df, target_cols):
    df = df.copy()
    df['Date '] = pd.to_datetime(df['Date '], format='%d-%b-%Y')  # Format change for given data
    df.set_index('Date ', inplace=True)
    df = df.dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[target_cols])
    df = pd.DataFrame(df_scaled, index=df.index, columns=target_cols)
    return df, scaler

# Function to create sequences for LSTM/GRU/XGBoost with multiple target columns
def create_sequences(data, target_cols, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data.iloc[i:i + n_steps].values)  # Capture all columns (3 features: High, Low, Close)
        y.append(data.iloc[i + n_steps][target_cols].values)  # Capture the next step's High, Low, Close
    return np.array(X), np.array(y)

# Generalized function to train models (LSTM/GRU)
def train_model(X, y, model_type="LSTM"):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(100, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
    elif model_type == "GRU":
        model.add(GRU(100, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50, activation='relu'))
    model.add(Dense(3))  # 3 outputs: High, Low, Close
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=300, verbose=0, callbacks=[early_stopping])
    return model

# Function to forecast multiple target variables
def forecast_best_model(best_model, model_name, data, scaler, n_steps, n_days):
    predictions = []
    input_seq = data[-n_steps:].reshape(1, n_steps, 3)  # Reshape to include 3 features (High, Low, Close)
    
    for _ in range(n_days):
        if model_name in ['LSTM', 'GRU']:
            pred = best_model.predict(input_seq)[0]
            input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)
        predictions.append(pred)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 3))

# Streamlit UI
st.set_page_config(page_title="Skavch Multi-Target Forecasting Engine", page_icon=":chart_with_upwards_trend:", layout="wide")

# Add an image to the header
#st.image("bg1.jpg", use_column_width=True)  # Adjust the image path as necessary

st.title("Skavch Multi-Target Sales Forecasting Engine")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
n_days = st.number_input("Enter the number of future days to forecast", min_value=1, max_value=36)
submit_button = st.button(label="Submit")

if submit_button:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        target_cols = ['High ', 'Low ', 'Close ']
        if all(col in df.columns for col in target_cols):
            df, scaler = preprocess_data(df, target_cols)
            st.write("Data Preprocessed Successfully!")

            # Prepare data for models
            n_steps = 12
            X, y = create_sequences(df, target_cols, n_steps)

            # Train models for each target variable
            lstm_model = train_model(X, y, "LSTM")
            gru_model = train_model(X, y, "GRU")

            # Forecast future values with LSTM and GRU
            forecast_lstm = forecast_best_model(lstm_model, "LSTM", df[target_cols].values, scaler, n_steps, n_days)
            forecast_gru = forecast_best_model(gru_model, "GRU", df[target_cols].values, scaler, n_steps, n_days)

            # Inverse transform the entire dataset at once
            df_inverse = scaler.inverse_transform(df.values)

            # Extract the historical values for High, Low, Close
            df_historical = pd.DataFrame(df_inverse, index=df.index, columns=target_cols)

            # Plotting the forecasted and actual data
            st.subheader("Forecast Results")
            plt.figure(figsize=(18, 9))
            for col in target_cols:
                # Plot historical data
                plt.plot(df_historical.index, df_historical[col], label=f'Historical {col}')
            
            # Plot forecasted data
            future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=n_days, freq='D')
            for i, col in enumerate(target_cols):
                plt.plot(future_dates, forecast_lstm[:, i], label=f'Forecasted LSTM {col}', linestyle='dashed')
                plt.plot(future_dates, forecast_gru[:, i], label=f'Forecasted GRU {col}', linestyle='dotted')

            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Time Series Forecast for High, Low, Close')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.error(f"One or more target columns ('High', 'Low', 'Close') not found in the dataset.")
    else:
        st.error("Please upload a CSV file.")
