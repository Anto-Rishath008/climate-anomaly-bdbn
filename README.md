# Climate Anomaly Forecasting using Bayesian Deep Belief Networks

This project is part of **Team 19** at Amrita Vishwa Vidyapeetham.  
It integrates **Deep Learning** and **Probability** concepts to forecast climate anomalies using **ERA5 Reanalysis Data**.

---

## 📂 Project Structure

```
Project/
│── data_raw/                # Raw ERA5 data (NetCDF files)
│   ├── era5_smoke_t2m_2020-01-01_00.nc   # Single-day sample data
│   └── era5_t2m_jan_feb_2020.nc          # 2-month dataset (Jan–Feb 2020)
│
│── src/                     # Python scripts
│   ├── download_era5_smoke.py   # Downloads 1 day of ERA5 data
│   ├── download_era5_range.py   # Downloads ERA5 data for a date range
│   ├── inspect_smoke.py         # Inspects single-day dataset
│   └── inspect_range.py         # Inspects range dataset (basic stats)
│
│── README.md                # Project documentation
│── requirements.txt         # List of dependencies
```

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Anto-Rishath008/climate-anomaly-bayesian-dbns.git
cd climate-anomaly-bayesian-dbns

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # (Windows)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Download single-day sample data**  
   ```bash
   python src/download_era5_smoke.py
   ```

2. **Inspect the dataset**  
   ```bash
   python src/inspect_smoke.py
   ```

3. **Download a date range (e.g., Jan–Feb 2020)**  
   ```bash
   python src/download_era5_range.py
   ```

4. **Inspect the range dataset**  
   ```bash
   python src/inspect_range.py
   ```

---

## 📊 Outputs

- `era5_smoke_t2m_2020-01-01_00.nc` → Single-day dataset (~70 KB)  
- `era5_t2m_jan_feb_2020.nc` → Two-month dataset (~4 MB)  
- Inspection scripts show dataset dimensions, mean, min, max temperatures.

---

## 🎯 Goal

We will extend this pipeline to build a **Bayesian Deep Belief Network** for climate anomaly forecasting,  
combining **probability theory** and **deep learning** principles as part of our course project.

---

## 👥 Team 19

- Vysakh Unnikrishnan (CB.SC.U4AIE23161)  
- Anto Rishath (CB.SC.U4AIE23103)  
- Abhishek Sankaramani (CB.SC.U4AIE23107)
