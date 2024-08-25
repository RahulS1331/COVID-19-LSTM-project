import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model('lstm_covid19_model.h5')

# Load and preprocess the data
data = pd.read_csv("time_series_covid19_deaths_global.csv")
date_columns = pd.to_datetime(data.columns[4:], format='%m/%d/%y', errors='coerce')
data.columns = ['Province/State', 'Country/Region', 'Lat', 'Long'] + list(date_columns)
data_country = data.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).transpose()
data_country.index = pd.to_datetime(data_country.index, format='%m/%d/%y', errors='coerce')

# Streamlit interface
st.title('COVID-19 Daily Deaths Prediction')
country = st.selectbox('Select a country:', data_country.columns)
country_data = data_country[country].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
country_data_scaled = scaler.fit_transform(country_data.reshape(-1, 1))

# Prepare the data for prediction
def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, _ = create_sequences(country_data_scaled, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Predict future values
y_pred = model.predict(X)
y_pred = scaler.inverse_transform(y_pred)

# Plot the results
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(country_data[time_steps:], label='True Values')
ax.plot(y_pred, label='Predicted Values')
ax.set_title(f'COVID-19 Daily Deaths Prediction for {country}')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Deaths')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
