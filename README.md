# ⭐ Forecasting Daily Retail Demand Using LSTM

## Dataset Source
The dataset used in this project comes from Kaggle:

Retail Sales Forecasting Dataset  
https://www.kaggle.com/datasets/tevecsystems/retail-sales-forecasting

This dataset contains historical sales, stock levels, and price information used to train the retail demand forecasting model.

This project applies Deep Learning to forecast **next-day retail demand (daily sales)** using multivariate time-series data.  
The goal is to compare a simple **DNN baseline** with a sequence-aware **LSTM model**, commonly used for time-series forecasting.

Accurate demand forecasting helps improve:
- inventory replenishment  
- stock planning  
- stockout / overstock prevention  
- supply chain efficiency  

---

# 📦 Dataset

The dataset (`retail_raw.csv`) includes the following columns:

| Column | Description |
|--------|-------------|
| `sales` | Daily sales (original name: `venda`) |
| `stock` | End-of-day inventory level (original: `estoque`) |
| `price` | Unit price (`preco`) |
| Generated features | calendar features, lag features, rolling window features |

Generated time-series features:
- `day_of_week`, `month`, `is_weekend`
- `lag_1`, `lag_7`, `lag_14`
- `rolling_mean_7`, `rolling_std_7`

---

# 🧠 Models

### **1. Baseline Model**
- Deep Neural Network (DNN)  
- Serves as a non-sequential benchmark.

### **2. Main Model**
- Long Short-Term Memory Network (LSTM)  
- Learns temporal dependencies and typically outperforms DNN for time-series forecasting.

---

# ⚙️ Environment

Python 3.10.11
pip 23.x.x

//Setting lib
pip install numpy pandas matplotlib scikit-learn
pip install tensorflow==2.12.0

//Run
pip install -r requirements.txt
# 1) Preprocess to create npy
python src/preprocess.py

# 2) Train 
python src/train.py

# 3) Result
python src/visualize.py

This displays:

- Actual vs Predicted sales  
- DNN vs LSTM comparison  

---

## 🔁 Pipeline Overview

## 📈 Pipeline Overview

1. **Load & preprocess retail dataset**
2. **Generate calendar, lag, and rolling features**
3. **Scale & reshape data into LSTM sequences**
4. **Train baseline DNN**
5. **Train LSTM model**
6. **Evaluate using MAE / RMSE**
7. **Visualize predictions**
8. **Conclusion: LSTM > DNN**

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

## 🎯 Conclusion

LSTM significantly improves forecasting accuracy compared to a standard DNN baseline  
because it captures temporal dependencies across multiple time-steps.

This project demonstrates a complete workflow for time-series forecasting in retail operations and can be extended to:  
- Inventory planning  
- Replenishment optimization  
- Supply chain analytics  

---

## 📌 Author

Pham Van Loc – Supply Chain & Deep Learning Project  
Changwon National University  

