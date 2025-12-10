import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

SEQ_LEN = 14
EPOCHS = 20
BATCH_SIZE = 32

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

history_dnn = dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("💾 Saving DNN model and history...")
dnn.save("dnn_model.h5")
np.save("data/dnn_loss.npy", history_dnn.history)


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

history_lstm = lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("💾 Saving LSTM model and history...")
lstm.save("lstm_model.h5")
np.save("data/lstm_loss.npy", history_lstm.history)

print("✅ Training finished!")
