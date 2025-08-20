# src/train_dbn_conv.py
import os
import math
import time
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# ----------------------------
# Config
# ----------------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS_RBM1 = 8
EPOCHS_RBM2 = 8
EPOCHS_SUP = 120
EARLY_STOP_PATIENCE = 15
LR_RBM = 1e-3
LR_SUP = 2e-3
VAL_RATIO = 0.2
GRID = 32  # target H=W after resize
SEQ_LEN = 5  # must match your preprocess
DATA_FILE = "data_raw/era5_preprocessed.npz"
OUT_DIR = "data_processed"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Utilities
# ----------------------------
def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)

def resize_bilinear_3d(batch_hw: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    batch_hw: (N, H, W) tensor
    out_hw: (H2, W2)
    returns: (N, H2, W2)
    """
    x = batch_hw.unsqueeze(1)  # (N,1,H,W)
    y = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return y.squeeze(1)

def make_loaders():
    data = np.load(DATA_FILE)
    X = data["X"]                       # (N, T, H, W) z-score anomalies
    Y = data["y"]                       # (N, H, W)     next-step anomaly

    assert X.shape[1] == SEQ_LEN, f"Expected seq len {SEQ_LEN}, got {X.shape[1]}"
    N, T, H, W = X.shape

    # resize to GRID x GRID
    X_t = to_tensor(X)                  # (N,T,H,W)
    Y_t = to_tensor(Y)                  # (N,H,W)
    X_r = resize_bilinear_3d(X_t.view(N*T, H, W), (GRID, GRID)).view(N, T, GRID, GRID)
    Y_r = resize_bilinear_3d(Y_t, (GRID, GRID))

    dataset = TensorDataset(X_r, Y_r)   # all in z-score space

    # deterministic split
    val_len = int(round(N * VAL_RATIO))
    train_len = N - val_len
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader, train_len, val_len

# ----------------------------
# Model parts
# ----------------------------
class ConvEncoder(nn.Module):
    """
    Simple conv feature extractor over the sequence dimension (T as channels)
    Input:  (N, T, H, W)
    Output: (N, C, h, w) with h=w=GRID
    """
    def __init__(self, in_ch=SEQ_LEN, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class RBM(nn.Module):
    """
    Bernoulli-Bernoulli RBM with 1-step Contrastive Divergence.
    We apply it on *normalized* continuous inputs treated as probabilities (simple, effective in practice).
    """
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.W + self.bh)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.bv)
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        ph, h = self.sample_h(v)
        pv, v_recon = self.sample_v(h)
        return v_recon

    def cd1_loss(self, v0):
        # one step CD
        ph0, h0 = self.sample_h(v0)
        pv1, v1 = self.sample_v(h0)
        ph1, h1 = self.sample_h(v1)

        # negative free energy gradient approximation
        dW = (v0.t() @ ph0) - (v1.t() @ ph1)
        dbv = torch.sum(v0 - v1, dim=0)
        dbh = torch.sum(ph0 - ph1, dim=0)

        # a pseudo loss to log
        recon_err = F.mse_loss(pv1, v0)
        return dW, dbv, dbh, recon_err

def pretrain_rbm(rbm, feat_loader, n_vis, n_hid, lr=1e-3, epochs=8, name="RBM"):
    opt = torch.optim.SGD([rbm.W, rbm.bv, rbm.bh], lr=lr)
    for e in range(1, epochs+1):
        losses = []
        for feats in feat_loader:
            (f_batch,) = feats  # (N, C, H, W)
            # flatten to (N, n_vis), normalize to [0,1] via sigmoid squashing
            v0 = torch.sigmoid(f_batch.view(f_batch.size(0), -1)).to(device)

            dW, dbv, dbh, rec = rbm.cd1_loss(v0)

            # manual update (simple)
            rbm.W.data += lr * dW / v0.size(0)
            rbm.bv.data += lr * dbv / v0.size(0)
            rbm.bh.data += lr * dbh / v0.size(0)
            losses.append(rec.item())
        print(f"[{name}] epoch {e}/{epochs} loss={np.mean(losses):.4f}")

class PredictorHead(nn.Module):
    """
    DBN-style: two RBMs -> treat as nonlinear feature transforms, then MLP regressor with MC Dropout
    """
    def __init__(self, flat_dim, hid1=512, hid2=256, p_drop=0.2):
        super().__init__()
        self.hid1 = hid1
        self.hid2 = hid2
        self.rbm1 = RBM(flat_dim, hid1)
        self.rbm2 = RBM(hid1, hid2)
        self.regressor = nn.Sequential(
            nn.Linear(hid2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(512, GRID*GRID),  # mean output
        )

    def forward(self, x_flat):
        # deterministic forward (mean)
        # use sigmoid on RBM hidden probabilities as features
        h1p = torch.sigmoid(x_flat @ self.rbm1.W + self.rbm1.bh)
        h2p = torch.sigmoid(h1p @ self.rbm2.W + self.rbm2.bh)
        mean = self.regressor(h2p)
        return mean

    def mc_forward(self, x_flat, mc_samples=20):
        self.train()  # enable dropout
        outs = []
        for _ in range(mc_samples):
            h1p = torch.sigmoid(x_flat @ self.rbm1.W + self.rbm1.bh)
            h2p = torch.sigmoid(h1p @ self.rbm2.W + self.rbm2.bh)
            mean = self.regressor(h2p)
            outs.append(mean.unsqueeze(0))
        self.eval()
        outs = torch.cat(outs, dim=0)   # (S, N, GRID*GRID)
        return outs.mean(0), outs.std(0)

class ConvDBNModel(nn.Module):
    def __init__(self, in_ch=SEQ_LEN, base=32, p_drop=0.2):
        super().__init__()
        self.encoder = ConvEncoder(in_ch=in_ch, base=base)
        # flat dim after conv: base * GRID * GRID
        flat_dim = base * GRID * GRID
        self.head = PredictorHead(flat_dim, hid1=1024, hid2=512, p_drop=p_drop)

    def forward(self, x):  # x: (N,T,H,W) resized to GRIDxGRID
        f = self.encoder(x)                    # (N, base, GRID, GRID)
        f_flat = f.view(f.size(0), -1)         # (N, flat)
        mean_flat = self.head(f_flat)          # (N, GRID*GRID)
        return mean_flat.view(-1, GRID, GRID)

    def mc_predict(self, x, mc=20):
        f = self.encoder(x)
        f_flat = f.view(f.size(0), -1)
        mean_flat, std_flat = self.head.mc_forward(f_flat, mc_samples=mc)
        return mean_flat.view(-1, GRID, GRID), std_flat.view(-1, GRID, GRID)

# ----------------------------
# Training / Eval
# ----------------------------
def validate(model, loader, loss_fn):
    model.eval()
    losses = []
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)  # (N,T,GRID,GRID)
            yb = yb.to(device)  # (N,GRID,GRID)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses.append(loss.item())
            all_pred.append(pred.cpu())
            all_true.append(yb.cpu())
    preds = torch.cat(all_pred, dim=0)     # (Nv, GRID, GRID)
    truths = torch.cat(all_true, dim=0)    # (Nv, GRID, GRID)
    rmse = torch.sqrt(F.mse_loss(preds, truths)).item()
    return np.mean(losses), rmse, preds.numpy(), truths.numpy()

def train():
    train_loader, val_loader, ntr, nva = make_loaders()
    # A loader of conv features for RBM pretraining (unsupervised): reuse inputs only
    feat_train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            next(iter(train_loader))[0]  # placeholder; we'll replace below
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    # The quick trick above won't iterate over full training data; rebuild feature loader properly:
    all_feats = []
    for xb, _ in train_loader:
        # encoder expects (N,T,GRID,GRID) later; here we just collect (N,GRID,GRID) per timestep
        # For RBM pretrain we’ll flatten encoder outputs directly during pretrain loop
        all_feats.append(xb)  # (N,T,GRID,GRID)
    Xall = torch.cat(all_feats, dim=0)  # (Ntr, T, GRID, GRID)
    feat_train_loader = DataLoader(torch.utils.data.TensorDataset(Xall), batch_size=BATCH_SIZE, shuffle=True)

    model = ConvDBNModel().to(device)

    # ---- RBM1 pretraining over conv features
    print("\n[Stage 1] Pretraining RBM-1 over conv features")
    # Build a tiny encoder to generate features on the fly to keep memory down
    encoder = model.encoder.to(device)
    rbm1 = model.head.rbm1
    # pretrain rbm1 by feeding encoder(X) flattened
    for e in range(1, EPOCHS_RBM1+1):
        losses = []
        for (xb,) in feat_train_loader:
            xb = xb.to(device)                # (N,T,GRID,GRID)
            f = encoder(xb)                   # (N, base, GRID, GRID)
            v0 = torch.sigmoid(f.view(f.size(0), -1))  # (N, flat)
            dW, dbv, dbh, rec = rbm1.cd1_loss(v0)
            rbm1.W.data += LR_RBM * dW / v0.size(0)
            rbm1.bv.data += LR_RBM * dbv / v0.size(0)
            rbm1.bh.data += LR_RBM * dbh / v0.size(0)
            losses.append(rec.item())
        print(f"[RBM-1] epoch {e}/{EPOCHS_RBM1} loss={np.mean(losses):.4f}")

    # ---- RBM2 pretraining in hidden space
    print("\n[Stage 2] Pretraining RBM-2 in hidden space")
    rbm2 = model.head.rbm2
    for e in range(1, EPOCHS_RBM2+1):
        losses = []
        for (xb,) in feat_train_loader:
            xb = xb.to(device)
            f = encoder(xb)                    # (N, base, GRID, GRID)
            v0 = torch.sigmoid(f.view(f.size(0), -1))    # to RBM1 vis
            h1p = torch.sigmoid(v0 @ rbm1.W + rbm1.bh)   # prob hidden1
            dW, dbv, dbh, rec = rbm2.cd1_loss(h1p)
            rbm2.W.data += LR_RBM * dW / v0.size(0)
            rbm2.bv.data += LR_RBM * dbv / v0.size(0)
            rbm2.bh.data += LR_RBM * dbh / v0.size(0)
            losses.append(rec.item())
        print(f"[RBM-2] epoch {e}/{EPOCHS_RBM2} loss={np.mean(losses):.4f}")

    # ---- Supervised fine-tuning (MSE on mean prediction)
    print("\n[Stage 3] Fine-tuning predictor head")
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR_SUP * 0.5},
        {"params": model.head.rbm1.parameters(), "lr": LR_SUP * 0.25},
        {"params": model.head.rbm2.parameters(), "lr": LR_SUP * 0.25},
        {"params": model.head.regressor.parameters(), "lr": LR_SUP},
    ])

    best_rmse = float("inf")
    best_epoch = -1
    impatient = 0
    for epoch in range(1, EPOCHS_SUP+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)                     # (N,GRID,GRID)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_loss, val_rmse, _, _ = validate(model, val_loader, loss_fn)
        print(f"[FineTune] epoch {epoch}/{EPOCHS_SUP} train_loss={np.mean(train_losses):.4f} val_RMSE={val_rmse:.4f}")

        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_epoch = epoch
            impatient = 0
            # snapshot
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "bdbn_conv_predictor.pt"))
        else:
            impatient += 1
            if impatient >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best RMSE={best_rmse:.4f})")
                break

    # ---- Reload best and produce MC predictions on the validation set
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "bdbn_conv_predictor.pt"), map_location=device))
    model.eval()

    # collect val tensors (inputs & targets) to preserve **exact val set**
    Xv, Yv = [], []
    for xb, yb in val_loader:
        Xv.append(xb.to(device))
        Yv.append(yb)
    Xv = torch.cat(Xv, dim=0)  # (Nv,T,GRID,GRID) on device
    Yv = torch.cat(Yv, dim=0)  # (Nv,GRID,GRID) on CPU (z-score)

    with torch.no_grad():
        mean_v, std_v = model.mc_predict(Xv, mc=30)   # (Nv,GRID,GRID) each

    # Save artifacts
    mean_np = mean_v.cpu().numpy()
    std_np = std_v.cpu().numpy()
    y_val_np = Yv.numpy()  # <-- the exact validation truth

    # Write a text summary
    with open(os.path.join(OUT_DIR, "metrics_conv.txt"), "w") as f:
        f.write(f"Validation RMSE (z): {best_rmse:.4f}\n")
        f.write(f"Average predictive std (z): {float(std_np.mean()):.4f}\n")

    np.savez(
        os.path.join(OUT_DIR, "preds_mean_std_conv.npz"),
        mean=mean_np,
        std=std_np,
        y_true=y_val_np,   # <---- IMPORTANT
    )

    print("\n=== Evaluation complete (Conv+DBN) ===")
    print("Saved model to:", os.path.join(OUT_DIR, "bdbn_conv_predictor.pt"))
    print("Saved predictions to:", os.path.join(OUT_DIR, "preds_mean_std_conv.npz"))
    print("Saved metrics to:", os.path.join(OUT_DIR, "metrics_conv.txt"))

if __name__ == "__main__":
    train()
