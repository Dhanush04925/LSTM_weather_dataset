import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Dataset
df = pd.read_csv("data/weatherHistory.csv")

# Select temperature column
data = df[['Temperature (C)']].values

# Scale
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler
pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

# Create sequences
X = []
y = []

window_size = 24

for i in range(window_size, len(data_scaled)):
    X.append(data_scaled[i-window_size:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

# LSTM Model
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=False,
        input_shape=(X.shape[1], 1)
    )
)

model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    y,
    epochs=5,
    batch_size=32
)

# Save model
model.save(
    "models/weather_lstm.keras"
)

print("Model Saved")