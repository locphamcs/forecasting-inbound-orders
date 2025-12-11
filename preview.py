import pandas as pd
import matplotlib.pyplot as plt

# Load raw dataset
df = pd.read_csv("data/retail_raw.csv")   # hoặc dataset bạn đang dùng

# Convert date column
df['data'] = pd.to_datetime(df['data'])

# Sort chronologically
df = df.sort_values('data')

# =============================
# ẢNH 1: Dataset Preview
# =============================
print(df.head(10))   # Chụp màn hình đoạn output này


# =============================
# ẢNH 2: Sales Trend Plot
# =============================
plt.figure(figsize=(10,4))
plt.plot(df['data'], df['venda'])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales (venda)")
plt.grid(True)
plt.show()  # Chụp ảnh biểu đồ này
