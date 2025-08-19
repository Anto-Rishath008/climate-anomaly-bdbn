# src/download_era5_range.py
import cdsapi
from pathlib import Path

# Output directory
out_dir = Path("data_raw")
out_dir.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": ["2020"],
        "month": ["01", "02"],   # Jan + Feb for testing
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "format": "netcdf",
        "area": [35, 65, 5, 100],  # N,W,S,E (India region)
    },
    str(out_dir / "era5_t2m_jan_feb_2020.nc")
)
