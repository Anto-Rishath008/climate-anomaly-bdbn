# src/download_era5_smoke.py
from pathlib import Path
import cdsapi

out = Path("data_raw/era5_smoke_t2m_2020-01-01_00.nc")
out.parent.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

# One variable, one timestep, small region (South Asia sub-box)
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "area": [35, 65, 5, 100],  # North, West, South, East
        "format": "netcdf",
    },
    str(out),
)

print("Saved:", out.resolve())
