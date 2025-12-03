# Step 6: Plots
import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    y_test = np.load("data/y_test.npy")
    y_dnn = np.load("data/y_pred_dnn.npy")
    y_lstm = np.load("data/y_pred_lstm.npy")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_dnn, label="Predicted DNN")
    plt.plot(y_lstm, label="Predicted LSTM")
    plt.title("Actual vs Predicted Inbound Orders")
    plt.xlabel("Time")
    plt.ylabel("Orders")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_results()