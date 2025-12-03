# Step 1: Tạo synthetic data
import numpy as np
import pandas as pd

def generate_synthetic_data(num_days=365):
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")

    base = 20
    trend = np.linspace(0, 10, num_days)
    weekday = dates.weekday
    weekly_pattern = np.where(weekday < 5, 5, -3)
    noise = np.random.normal(0, 3, num_days)

    inbound = base + trend + weekly_pattern + noise
    inbound = np.maximum(inbound, 0).round().astype(int)

    df = pd.DataFrame({"date": dates, "inbound": inbound})
    df.to_csv("data/synthetic_data.csv", index=False)
    return df

if __name__ == "__main__":
    generate_synthetic_data()
    print("Synthetic data generated!")
