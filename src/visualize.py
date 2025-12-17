import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

print("🔹 Loading data & models...")

# Load test sets
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Load trained models
dnn = load_model("dnn_model.h5")
lstm = load_model("lstm_model.h5")

# Make predictions
dnn_pred = dnn.predict(X_test)
lstm_pred = lstm.predict(X_test)

# ------------------------------
# 1) LSTM Train / Validation Loss
# ------------------------------
print("📊 Plotting LSTM Loss Curve...")
lstm_hist = np.load("data/lstm_loss.npy", allow_pickle=True).item()

plt.figure(figsize=(10,4))
plt.plot(lstm_hist["loss"], label="Train Loss")
plt.plot(lstm_hist["val_loss"], label="Validation Loss")
plt.title("LSTM Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 2) DNN Train / Validation Loss
# ------------------------------
print("📊 Plotting DNN Loss Curve...")
dnn_hist = np.load("data/dnn_loss.npy", allow_pickle=True).item()

plt.figure(figsize=(10,4))
plt.plot(dnn_hist["loss"], label="Train Loss")
plt.plot(dnn_hist["val_loss"], label="Validation Loss")
plt.title("DNN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 3) LSTM Actual vs Predicted
# ------------------------------
print("📊 Plotting LSTM Prediction...")
plt.figure(figsize=(12,5))
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(lstm_pred, label="LSTM Prediction", linewidth=1.5)
plt.title("LSTM Forecasting Performance")
plt.xlabel("Samples")
plt.ylabel("Scaled Demand")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 4) DNN Actual vs Predicted
# ------------------------------
print("📊 Plotting DNN Prediction...")
plt.figure(figsize=(12,5))
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(dnn_pred, label="DNN Prediction", linewidth=1.5)
plt.title("DNN Forecasting Performance")
plt.xlabel("Samples")
plt.ylabel("Scaled Demand")
plt.legend()
plt.grid(True)
plt.show()

print("✅ All plots generated!")

# ------------------------------
# 5) Comparison: LSTM vs DNN vs Actual
# ------------------------------
print("📊 Plotting Model Comparison...")

plt.figure(figsize=(14,6))
plt.plot(y_test, label="Actual", linewidth=2, color="black")
plt.plot(dnn_pred, label="DNN Prediction", alpha=0.7)
plt.plot(lstm_pred, label="LSTM Prediction", alpha=0.7)
plt.title("Model Comparison: LSTM vs DNN vs Actual")
plt.xlabel("Samples")
plt.ylabel("Scaled Demand")
plt.legend()
plt.grid(True)
plt.show()

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

dnn = load_model("dnn_model.h5")
lstm = load_model("lstm_model.h5")

y_pred_dnn = dnn.predict(X_test)
y_pred_lstm = lstm.predict(X_test)

mae_dnn = mean_absolute_error(y_test, y_pred_dnn)
rmse_dnn = np.sqrt(mean_squared_error(y_test, y_pred_dnn))

mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

r2_dnn = r2_score(y_test, y_pred_dnn)
r2_lstm = r2_score(y_test, y_pred_lstm)

print("DNN R2 :", r2_dnn)
print("LSTM R2:", r2_lstm)

print("DNN MAE :", mae_dnn)
print("DNN RMSE:", rmse_dnn)
print("LSTM MAE :", mae_lstm)
print("LSTM RMSE:", rmse_lstm)