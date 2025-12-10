import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

print("🔹 Loading data & models...")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

dnn = load_model("dnn_model.h5")
lstm = load_model("lstm_model.h5")

print("🔹 Predicting...")
y_pred_dnn = dnn.predict(X_test)
y_pred_lstm = lstm.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(y_pred_dnn, label="DNN Prediction", alpha=0.7)
plt.plot(y_pred_lstm, label="LSTM Prediction", alpha=0.7)
plt.title("Retail Demand Forecasting (Scaled Values)")
plt.xlabel("Samples")
plt.ylabel("Scaled Demand")
plt.legend()
plt.show()
