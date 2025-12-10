import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

SEQ_LEN = 14

print("🔹 Loading preprocessed data...")
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")

# ---------------- BASELINE DNN ----------------

def build_dnn(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


print("🔹 Training DNN baseline...")
dnn = build_dnn((SEQ_LEN, X_train.shape[2]))
dnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

dnn.save("dnn_model.h5")


# ---------------- LSTM MODEL ----------------

def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


print("🔹 Training LSTM model...")
lstm = build_lstm((SEQ_LEN, X_train.shape[2]))
lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

lstm.save("lstm_model.h5")

print("✅ Training finished!")
