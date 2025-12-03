# Step 5: Train + Evaluate
from preprocess import load_and_preprocess
from model_dnn import build_dnn
from model_lstm import build_lstm
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

X_train, y_train, X_test, y_test, scaler = load_and_preprocess()

# Models
dnn = build_dnn(X_train.shape[1:])
lstm = build_lstm(X_train.shape[1:])

callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]

# Train DNN
print("\nTraining DNN...")
dnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Train LSTM
print("\nTraining LSTM...")
lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Predict
y_pred_dnn = dnn.predict(X_test)
y_pred_lstm = lstm.predict(X_test)

# Inverse scale
y_pred_dnn = scaler.inverse_transform(y_pred_dnn)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_real = scaler.inverse_transform(y_test)

np.save("data/y_test.npy", y_test_real)
np.save("data/y_pred_dnn.npy", y_pred_dnn)
np.save("data/y_pred_lstm.npy", y_pred_lstm)

print("Training completed!")