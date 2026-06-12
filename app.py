import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model

# Load
model = load_model(
    "models/weather_lstm.keras"
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

st.title("🌦 Weather Forecast Using LSTM")

st.write(
    "Enter last 24 temperature values."
)

temps = []

for i in range(24):
    value = st.number_input(
        f"Temperature {i+1}",
        value=20.0,
        key=i
    )
    temps.append(value)

if st.button("Predict Next Temperature"):

    temps = np.array(temps).reshape(-1,1)

    temps_scaled = scaler.transform(
        temps
    )

    X = np.array(
        [temps_scaled]
    )

    prediction = model.predict(X)

    result = scaler.inverse_transform(
        prediction
    )

    st.success(
        f"Predicted Temperature: {result[0][0]:.2f} °C"
    )