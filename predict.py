import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define the model architecture (must match the one used during training)
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Load the trained model weights
model.load_state_dict(torch.load("weather_model.pth"))
model.eval()

# Reconstruct the scalers used during training for consistent scaling
df = pd.read_csv("weather.csv")
X = df[['humidity', 'wind']].values
y = df['temp'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# New input sample: humidity = 52%, wind = 11 km/h
sample = np.array([[52, 11]])
sample_scaled = scaler_X.transform(sample)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# Make prediction using the trained model
with torch.no_grad():
    prediction = model(sample_tensor)
    temp_pred = scaler_y.inverse_transform(prediction.numpy())
    print(f"Predicted Temperature: {temp_pred[0][0]:.2f} °C")
