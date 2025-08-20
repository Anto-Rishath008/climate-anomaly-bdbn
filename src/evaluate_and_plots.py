# src/evaluate_and_plots.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_preds(pred_path):
    data = np.load(pred_path, allow_pickle=True)
    # prefer flattened shapes (N, HW) if present
    y_pred = data["y_pred"]
    y_std  = data["y_std"]
    if "y_true" in data:
        y_true = data["y_true"]
    else:
        # fallback to separate truth file if not bundled
        tv = np.load("data_processed/train_val.npz")
        y_true_full = tv["y_val"]  # (N, H, W)
        # match shapes
        if y_pred.ndim == 2:  # (N, HW)
            N, HW = y_pred.shape
            H, W = y_true_full.shape[1], y_true_full.shape[2]
            y_true = y_true_full.reshape(N, -1)
        else:
            y_true = y_true_full
    return y_pred, y_std, y_true

def ensure_flat(arr):
    """Return (N, HW)."""
    if arr.ndim == 3:
        N, H, W = arr.shape
        return arr.reshape(N, H*W)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape}")

def corr_coeff(y_true_f, y_pred_f):
    """Pearson r across all pixels and all samples (flattened)."""
    yt = y_true_f.reshape(-1)
    yp = y_pred_f.reshape(-1)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = (yt.std() * yp.std() + 1e-8)
    return float((yt @ yp) / (len(yt) * denom))

def reliability_curve(y_true, y_pred, y_std, nbins=10, out="figures/reliability_curve.png"):
    # requires 3D arrays to show maps later; this curve can work on flat as well
    yt = ensure_flat(y_true)
    yp = ensure_flat(y_pred)
    ys = ensure_flat(y_std)

    conf = np.linspace(0.05, 0.95, nbins)
    emp  = []
    for c in conf:
        # z-interval width for 2-sided normal: c = erf(k/sqrt(2))
        # approximate k from c using inverse CDF
        k = np.sqrt(2) * erfinv(2*c - 1.0)
        within = np.abs(yt - yp) <= k * ys
        emp.append(within.mean())
    emp = np.array(emp)

    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.plot(conf, emp, marker="o", label="Empirical")
    plt.plot([0,1], [0,1], "--", label="Perfect")
    plt.xlabel("Nominal")
    plt.ylabel("Empirical")
    plt.title("Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

# lightweight inverse error function (since SciPy may not be present)
# Chebyshev-like approximation for |x|<=1 (sufficient for plotting)
def erfinv(x):
    a = 0.147  # Winitzki approximation
    ln = np.log(1 - x*x)
    term = 2/(np.pi*a) + ln/2
    return np.sign(x) * np.sqrt( np.sqrt(term*term - ln/a) - term )

def try_save_sample_maps(y_true, y_pred, y_std, idx, outpath):
    """Save three-panel map for one sample if 3D (N,H,W)."""
    if y_true.ndim != 3:
        return False
    yt, yp, ys = y_true[idx], y_pred[idx], y_std[idx]
    vmin = np.percentile(yt, 2)
    vmax = np.percentile(yt, 98)

    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(15,4))
    for i, (title, img) in enumerate([("Truth", yt), ("Pred", yp), ("σ", ys)]):
        ax = plt.subplot(1,3,i+1)
        im = ax.imshow(img, cmap="viridis", vmin=vmin if i<2 else None, vmax=vmax if i<2 else None)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="data_processed/preds_mean_std_conv.npz",
                    help="Path to predictions .npz (must contain y_pred, y_std, optional y_true)")
    args = ap.parse_args()

    print(f"Using predictions from: {args.pred}")
    y_pred, y_std, y_true = load_preds(args.pred)

    # If needed, reshape predictions back to (N,H,W) using train_val shapes
    # for nicer maps
    may_maps = False
    if y_true.ndim == 2:
        tv = np.load("data_processed/train_val.npz")
        if "y_val" in tv:
            N, H, W = tv["y_val"].shape
            if y_pred.shape[0] == N and y_pred.shape[1] == H*W:
                y_pred_map = y_pred.reshape(N, H, W)
                y_std_map  = y_std.reshape(N, H, W)
                y_true_map = tv["y_val"]  # keep original map
                may_maps = True
        else:
            y_pred_map = y_std_map = y_true_map = None
    else:
        y_pred_map, y_std_map, y_true_map = y_pred, y_std, y_true
        may_maps = True

    # Metrics on flattened arrays
    yp_f = ensure_flat(y_pred)
    ys_f = ensure_flat(y_std)
    yt_f = ensure_flat(y_true)

    mse = np.mean((yp_f - yt_f)**2)
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(yp_f - yt_f)))
    corr = corr_coeff(yt_f, yp_f)

    # Accuracy definitions
    accuracy_corr = max(0.0, corr) * 100.0  # correlation as %
    # old tolerance metric: within +/- 1 sigma of *true* variability
    tol = float(np.std(yt_f))
    acc_tol = float(np.mean(np.abs(yp_f - yt_f) <= tol)) * 100.0

    # Skill vs climatology (beat zero-anomaly)
    mse_clim = float(np.mean((yt_f - 0.0)**2))
    skill = (1.0 - mse / (mse_clim + 1e-8)) * 100.0

    print(f"Shapes -> y_pred: {y_pred.shape}, y_std: {y_std.shape}, y_true: {y_true.shape}")
    print(f"RMSE (z-score): {rmse:.4f}")
    print(f"MAE  (z-score): {mae:.4f}")
    print(f"Correlation     : {corr:.4f}  ({corr*100:.2f}%)")
    print(f"Accuracy (corr%): {accuracy_corr:.2f}%")
    print(f"Within‑tolerance accuracy (±1σ of truth): {acc_tol:.2f}%")
    print(f"Average predictive std: {float(np.mean(ys_f)):.4f}")
    print(f"Truth std (z-score space): {float(np.std(yt_f)):.4f}")
    print(f"Skill vs climatology (MSE): {skill:.2f}%")

    # Reliability curve (works with flat too)
    try:
        reliability_curve(y_true, y_pred, y_std, out="figures/reliability_curve.png")
        print("Saved: figures/reliability_curve.png")
    except Exception as e:
        print(f"[Warn] Reliability curve failed: {e}")

    # Example maps (only if we have 3D map tensors)
    saved_any = False
    for i in range(3):
        if may_maps:
            ok = try_save_sample_maps(
                y_true_map if y_true_map is not None else y_true,
                y_pred_map if y_pred_map is not None else y_pred,
                y_std_map  if y_std_map  is not None else y_std,
                idx=min(i, y_pred.shape[0]-1),
                outpath=f"figures/sample_{i:03d}_maps.png"
            )
            saved_any = saved_any or ok
    if saved_any:
        print("Saved: figures/sample_000_maps.png")
        print("Saved: figures/sample_001_maps.png")
        print("Saved: figures/sample_002_maps.png")

if __name__ == "__main__":
    main()
