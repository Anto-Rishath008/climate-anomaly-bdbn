import numpy as np

def main():
    file_path = "data_raw/era5_preprocessed.npz"
    print(f"Loading preprocessed dataset from {file_path} ...")
    
    data = np.load(file_path)
    X, y = data["X"], data["y"]

    print(f"\n✅ Dataset loaded successfully!")
    print(f"X shape: {X.shape}")  # (samples, time_steps, lat, lon)
    print(f"y shape: {y.shape}")  # (samples, lat, lon)

    # Show basic statistics
    print("\n📊 X statistics:")
    print(f"  Min: {X.min():.4f}")
    print(f"  Max: {X.max():.4f}")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std: {X.std():.4f}")

    print("\n📊 y statistics:")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")

if __name__ == "__main__":
    main()
