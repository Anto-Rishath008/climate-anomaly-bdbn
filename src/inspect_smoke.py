# src/inspect_smoke.py
import xarray as xr
from pathlib import Path

path = Path("data_raw/era5_smoke_t2m_2020-01-01_00.nc")

ds = xr.open_dataset(path)
print(ds)

# Show basic stats of the variable
t2m = ds["t2m"]
print("\nVariable shape:", t2m.shape)
print("Mean value (K):", float(t2m.mean()))
print("Min value (K):", float(t2m.min()))
print("Max value (K):", float(t2m.max()))
