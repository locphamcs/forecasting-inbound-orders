import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

SEQ_LEN = 14  # số ngày trong 1 sequence


def create_lag_features(df):
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    return df


def create_rolling_features(df):
    df["rolling_mean_7"] = df["sales"].rolling(window=7).mean()
    df["rolling_std_7"] = df["sales"].rolling(window=7).std()
    return df


def create_date_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def create_sequences(X, y, seq_len=14):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def main():
    # Lấy đường dẫn root của project (thư mục chứa folder src/)
    base_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    data_dir = os.path.join(base_dir, "data")
    raw_csv_path = os.path.join(data_dir, "retail_raw.csv")

    print("📂 Project root:", base_dir)
    print("📂 Data folder :", data_dir)
    print("📄 CSV path    :", raw_csv_path)

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file CSV tại: {raw_csv_path}\n"
            f"👉 Hãy chắc chắn bạn đã đặt file CSV vào folder 'data/' và đặt tên là 'retail_raw.csv'"
        )

    print("🔹 Loading dataset...")
    df = pd.read_csv(raw_csv_path)

    # rename cột theo đúng nghĩa
    df = df.rename(columns={"data": "date", "venda": "sales", "estoque": "stock", "preco": "price"})
    df = df.sort_values("date")

    # tạo feature theo thời gian
    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # bỏ các dòng bị NaN do lag/rolling
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "sales", "stock", "price",
        "day_of_week", "month", "is_weekend",
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_std_7"
    ]

    print("🔹 Scaling features...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    X_all = scaled
    y_all = scaled[:, 0]  # cột 0 = 'sales' đã scale

    print("🔹 Creating sequences...")
    X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)

    # train/val/test split theo tỉ lệ
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_val = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]

    X_test = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]

    print("🔹 Saving .npy files...")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print("✅ Preprocess done! Files saved in /data")


if __name__ == "__main__":
    main()
