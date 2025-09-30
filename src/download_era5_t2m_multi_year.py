# download_era5_t2m_multi_year.py
# Requires: pip install cdsapi
import cdsapi, os

# ---- CONFIG ----
YEARS  = [str(y) for y in range(2015, 2021)]  # change as you like (e.g., 2000–2024)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
# 6-hourly to match your current preprocessing (00, 06, 12, 18)
TIMES  = ["00:00", "06:00", "12:00", "18:00"]

OUTDIR = "data_raw/era5_parts"  # monthly chunks land here
os.makedirs(OUTDIR, exist_ok=True)

c = cdsapi.Client()

for y in YEARS:
    for m in MONTHS:
        out_path = os.path.join(OUTDIR, f"era5_t2m_{y}_{m}.nc")
        if os.path.exists(out_path):
            print(f"[skip] {out_path}")
            continue
        print(f"[req] {y}-{m} -> {out_path}")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["2m_temperature"],
                "year": y,
                "month": m,
                "day": DAYS,
                "time": TIMES,
                "format": "netcdf",
            },
            out_path,
        )
print("[ok] All requests submitted/completed.")
