# Climate Anomaly Forecasting using Bayesian Deep Belief Networks

This project explores forecasting of climate anomalies using **Bayesian Deep Belief Networks (DBNs)** with ERA5 reanalysis data.

---

## Project Structure

```
├── .venv/                # Virtual environment (ignored in Git)
├── data_raw/              # Raw ERA5 NetCDF files (NOT pushed to GitHub)
├── src/                   # Python source code
│   ├── download_era5_range.py   # Script to download ERA5 data for a given date range
│   ├── download_era5_smoke.py   # Example "smoke test" data download
│   ├── inspect_range.py         # Inspect downloaded range dataset
│   ├── inspect_smoke.py         # Inspect sample smoke dataset
│   └── plot_smoke.py            # Quick plot utility for smoke dataset
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Data

Raw ERA5 reanalysis data (NetCDF files) are stored locally in the `data_raw/` folder but are **not pushed to GitHub** to keep the repository lightweight.

### Downloading ERA5 Data

To download ERA5 data:

```bash
python src/download_era5_range.py
```

This will fetch ERA5 temperature data into the `data_raw/` folder.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Code

1. Activate your virtual environment:
   ```bash
   .venv\Scripts\activate   # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```

2. Run scripts from the `src/` folder.

Example:
```bash
python src/inspect_range.py
```

---

## Notes

- `data_raw/` is excluded from version control to avoid large file uploads.
- Teachers/teammates can regenerate the data by running the download scripts.
