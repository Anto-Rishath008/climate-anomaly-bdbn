# src/make_train_val.py
import numpy as np
import os

def main():
    in_path = "data_raw/era5_preprocessed.npz"
    out_path = "data_processed/train_val.npz"

    print(f"Loading preprocessed dataset from {in_path} ...")
    data = np.load(in_path)
    X, y = data["X"], data["y"]  # (N, 5, H, W), (N, H, W)

    N = len(X)
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )

    print(f"✅ Created train/val split")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
