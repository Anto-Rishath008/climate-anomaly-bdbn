# src/inspect_range.py
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

path = Path("data_raw/era5_t2m_jan_feb_2020.nc")
ds = xr.open_dataset(path)

print(ds)  # full dataset summary

t2m = ds["t2m"]  # 2m air temperature in Kelvin
print("\n--- BASIC STATS ---")
print("Dims:", dict(ds.dims))
print("Time length:", t2m.sizes.get("time", t2m.sizes.get("valid_time", None)))
print("Lat length:", t2m.sizes["latitude"])
print("Lon length:", t2m.sizes["longitude"])
print("Mean (K):", float(t2m.mean()))
print("Min  (K):", float(t2m.min()))
print("Max  (K):", float(t2m.max()))

# If time/valid_time exists, plot the first day's map to verify structure
t_dim = "time" if "time" in t2m.dims else "valid_time"
snap = t2m.isel({t_dim: 0}).squeeze()

plt.figure(figsize=(8, 6))
snap.plot(cmap="coolwarm")
plt.title(f"ERA5 2m Temperature — first timestep ({str(snap.coords[t_dim].values)[:10]})")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.show()
