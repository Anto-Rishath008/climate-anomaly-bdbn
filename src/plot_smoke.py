# src/plot_smoke.py
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Load file
path = Path("data_raw/era5_smoke_t2m_2020-01-01_00.nc")
ds = xr.open_dataset(path)
t2m = ds["t2m"].squeeze()  # remove extra time dimension

# Plot
plt.figure(figsize=(8, 6))
t2m.plot(cmap="coolwarm")
plt.title("ERA5 2m Temperature (2020-01-01)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
