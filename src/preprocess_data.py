import xarray as xr
import numpy as np
import os

def preprocess_era5(input_nc, output_npz, seq_len=5):
    """
    Preprocess ERA5 temperature data:
    1. Load ERA5 file
    2. Convert to Celsius
    3. Compute anomalies (relative to mean)
    4. Normalize
    5. Create sliding windows
    6. Save as .npz dataset
    """
    print(f"Loading data from {input_nc} ...")
    ds = xr.open_dataset(input_nc)
    t2m = ds['t2m'].values  # shape: (time, lat, lon)

    # Convert to Celsius
    t2m_celsius = t2m - 273.15

    # Compute anomaly (subtract mean across time)
    climatology = np.mean(t2m_celsius, axis=0)
    anomalies = t2m_celsius - climatology

    # Normalize anomalies (z-score)
    mean = np.mean(anomalies)
    std = np.std(anomalies)
    anomalies_norm = (anomalies - mean) / std

    # Create sliding windows
    X, y = [], []
    for i in range(len(anomalies_norm) - seq_len):
        X.append(anomalies_norm[i:i+seq_len])   # seq_len timesteps
        y.append(anomalies_norm[i+seq_len])     # next step target

    X, y = np.array(X), np.array(y)
    print(f"Created dataset: X shape = {X.shape}, y shape = {y.shape}")

    # Save dataset
    np.savez_compressed(output_npz, X=X, y=y)
    print(f"Saved preprocessed dataset to {output_npz}")

if __name__ == "__main__":
    # Example usage
    input_file = os.path.join("data_raw", "era5_t2m_jan_feb_2020.nc")
    output_file = os.path.join("data_raw", "era5_preprocessed.npz")
    preprocess_era5(input_file, output_file, seq_len=5)
