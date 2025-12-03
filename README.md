# Forecasting Daily Inbound Orders Using LSTM

//Env
Python 3.10.11
pip 23.x.x

//Setting lib
pip install numpy pandas matplotlib scikit-learn
pip install tensorflow==2.12.0

//Run
pip install -r requirements.txt
python src/data_generator.py
python src/train.py
python src/visualize.py

This project applies Deep Learning to forecast next-day inbound order volumes.

## Models
- Baseline: Deep Neural Network (DNN)
- Main Model: LSTM Network

## Steps
1. Generate or load data
2. Preprocess & create sequences
3. Train DNN baseline
4. Train LSTM model
5. Evaluate using MAE/RMSE
6. Plot results
7. Conclude: LSTM > DNN

## Technology Stack
- Python
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib