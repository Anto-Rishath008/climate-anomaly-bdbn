# Climate Anomaly Forecasting using Bayesian DBNs

This repository implements **Bayesian Deep Belief Networks (DBNs)** for forecasting climate anomalies using ERA5 reanalysis data. The project integrates **probabilistic reasoning** to provide not only point predictions but also **uncertainty estimates**.

---

## 📂 Project Structure
```
.
├── src/                     # Training & evaluation scripts
│   ├── train_dbn_conv_nll.py
│   ├── evaluate_and_plots.py
├── data_raw/                # Raw ERA5 data (netCDF/NPZ)
├── data_processed/          # Train/val splits, metrics, predictions
├── figures/                 # Plots & evaluation curves
├── README.md
├── .gitignore
└── .gitattributes
```

---

## 🚀 Setup
```bash
# Clone repo
git clone https://github.com/Anto-Rishath008/climate-anomaly-bayesian-dbns.git
cd climate-anomaly-bayesian-dbns

# Setup venv
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Training and Evaluation

### Train with Negative Log-Likelihood
```bash
python src/train_dbn_conv_nll.py --epochs 150 --anom_k 1.0
```

### Evaluate
```bash
python src/evaluate_and_plots.py --pred data_processed/preds_mean_std_conv_nll.npz
```

---

## 📊 Results

- **NLL model**:
  - RMSE (z-score): ~1.01
  - MAE (z-score): ~0.68
  - Within‑tolerance accuracy (±1σ): ~76%

---

## 🔢 Mathematical Background

We model each prediction as Gaussian:

\[
p(y \mid x, \theta) = \mathcal{N}(y; \mu_\theta(x), \sigma^2_\theta(x))
\]

Training objective (Negative Log-Likelihood, NLL):

\[
\mathcal{L}_{\text{NLL}} = \frac{1}{2} \sum_i \left[
\frac{(y_i - \mu_\theta(x_i))^2}{\sigma^2_\theta(x_i)} + \log \sigma^2_\theta(x_i)
\right]
\]

This ensures learning both **accurate means** and **well-calibrated uncertainties**.

---

## 🌐 Web App Plan

We aim to deploy the trained model into a **Flask/Django + React frontend app** where:
- Users can query anomaly forecasts.
- Predictions will include **confidence intervals**.

---

## 👨‍💻 Authors

Team 19, Amrita Vishwa Vidyapeetham
- Vysakh Unnikrishnan (CB.SC.U4AIE23161)
- Anto Rishath (CB.SC.U4AIE23103)
- Abhishek Sankaramani (CB.SC.U4AIE23107)

---

## 📜 License
MIT License © 2025 Team 19

