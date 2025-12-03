# Step 2: Tạo sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(sequence_length=14):
    df = pd.read_csv("data/synthetic_data.csv")
    values = df["inbound"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length])
        y.append(scaled[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler
