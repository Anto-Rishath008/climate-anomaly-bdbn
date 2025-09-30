import argparse, glob, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data_processed/preds_s*.npz",
                    help="pattern to find member .npz files")
    ap.add_argument("--out", default="data_processed/preds_mean_std_conv_nll.npz")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    assert paths, f"No files match {args.glob}"
    ys, ss, yt = [], [], None
    for p in paths:
        d = np.load(p)
        ys.append(d["y_pred"])
        ss.append(d["y_std"])
        if yt is None: yt = d["y_true"]
    y_pred = np.mean(ys, axis=0)
    # For std, average the predictive variances (σ^2) then sqrt
    var = np.mean([s**2 for s in ss], axis=0)
    y_std = np.sqrt(np.maximum(var, 1e-12))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, y_pred=y_pred, y_std=y_std, y_true=yt)
    print(f"[ok] Blended {len(paths)} members -> {args.out}")

if __name__ == "__main__":
    main()
