import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_svi(pred_file):
    data = np.load(pred_file)
    y_pred, y_std, y_true = data["y_pred"], data["y_std"], data["y_true"]

    print(f"Shapes -> y_pred: {y_pred.shape}, y_std: {y_std.shape}, y_true: {y_true.shape}")

    # Flatten
    yp = y_pred.flatten()
    yt = y_true.flatten()
    ys = y_std.flatten()

    # Metrics
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    corr = np.corrcoef(yt, yp)[0, 1]
    acc = corr * 100

    within_tol = np.mean((np.abs(yp - yt) <= ys)) * 100
    avg_pred_std = ys.mean()
    truth_std = yt.std()
    climatology_mse = mean_squared_error(yt, np.zeros_like(yt))
    skill_vs_clim = 100 * (1 - (rmse**2) / climatology_mse)

    print(f"RMSE (z-score): {rmse:.4f}")
    print(f"MAE  (z-score): {mae:.4f}")
    print(f"Correlation     : {corr:.4f}  ({acc:.2f}%)")
    print(f"Accuracy (corr%): {acc:.2f}%")
    print(f"Within-tolerance accuracy (±1σ of truth): {within_tol:.2f}%")
    print(f"Average predictive std: {avg_pred_std:.4f}")
    print(f"Truth std (z-score space): {truth_std:.4f}")
    print(f"Skill vs climatology (MSE): {skill_vs_clim:.2f}%")

    # Save metrics
    with open("data_processed/metrics_svi_eval.txt", "w") as f:
        f.write(f"RMSE={rmse:.4f}\n")
        f.write(f"MAE={mae:.4f}\n")
        f.write(f"Correlation={corr:.4f}\n")
        f.write(f"Accuracy={acc:.2f}%\n")
        f.write(f"WithinTolerance={within_tol:.2f}%\n")
        f.write(f"AvgPredStd={avg_pred_std:.4f}\n")
        f.write(f"TruthStd={truth_std:.4f}\n")
        f.write(f"SkillVsClim={skill_vs_clim:.2f}%\n")

    # --------- PLOTS ----------
    # Reliability curve
    err = np.abs(yp - yt)
    bins = np.linspace(0, np.percentile(ys, 95), 20)
    bin_idx = np.digitize(ys, bins)
    mean_err, mean_std = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 0:
            mean_err.append(err[mask].mean())
            mean_std.append(ys[mask].mean())
    plt.figure()
    plt.plot(mean_std, mean_err, "o-", label="Observed error")
    plt.plot([0, max(mean_std)], [0, max(mean_std)], "k--", label="Ideal")
    plt.xlabel("Predicted std")
    plt.ylabel("Observed error")
    plt.legend()
    plt.title("Reliability curve (SVI)")
    plt.savefig("figures/reliability_curve_svi.png", dpi=150)

    # Histogram of predictive std
    plt.figure()
    plt.hist(ys, bins=50, alpha=0.7, color="blue")
    plt.xlabel("Predictive std")
    plt.ylabel("Frequency")
    plt.title("Distribution of predictive uncertainties (SVI)")
    plt.savefig("figures/predictive_std_hist_svi.png", dpi=150)

    # Sample reconstructions (assume grid shape if matches)
    if y_pred.shape[1] in [121*141, 32*32]:
        grid_h, grid_w = (121, 141) if y_pred.shape[1] == 121*141 else (32, 32)
        for i in range(min(3, y_pred.shape[0])):  # first 3 samples
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(y_true[i].reshape(grid_h, grid_w), cmap="coolwarm")
            plt.title("Truth")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(y_pred[i].reshape(grid_h, grid_w), cmap="coolwarm")
            plt.title("Prediction")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(y_std[i].reshape(grid_h, grid_w), cmap="viridis")
            plt.title("Uncertainty (σ)")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"figures/sample_{i:03d}_maps_svi.png", dpi=150)

    print("Saved: figures/reliability_curve_svi.png, predictive_std_hist_svi.png, sample maps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to preds_mean_std_svi.npz")
    args = parser.parse_args()
    evaluate_svi(args.pred)
