# src/train_dbn.py
# Bayesian Deep Belief Network (DBN) with RBM pretraining + MC Dropout head
# Forecasts next-day anomaly map from previous 5 anomaly maps.

import os
import math
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------

SEED = 42
SEQ_LEN = 5            # input days
TARGET_STEPS = 1       # predict 1 day ahead
GRID = 32              # downsampled spatial size (H=W=32)
BATCH_SIZE = 8
PRETRAIN_EPOCHS_1 = 5  # RBM-1 epochs
PRETRAIN_EPOCHS_2 = 5  # RBM-2 epochs
FINE_TUNE_EPOCHS = 100  # predictor fine-tuning
LR_PRETRAIN = 1e-3
LR_FINETUNE = 5e-4
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.2
MC_SAMPLES = 20

DATA_NPZ = "data_raw/era5_preprocessed.npz"
OUT_DIR = "data_processed"
MODEL_PATH = "data_processed/bdbn_predictor.pt"
PRED_SAVE = "data_processed/preds_mean_std.npz"
METRICS_SAVE = "data_processed/metrics.txt"

# DBN layer sizes (after flattening 5×32×32 = 5120)
VISIBLE = SEQ_LEN * GRID * GRID           # 5 * 32 * 32 = 5120
HIDDEN1 = 1024
HIDDEN2 = 256
OUTPUT = GRID * GRID                      # predict 32 × 32 map

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ERA5NPZDataset(data.Dataset):
    """
    Loads era5_preprocessed.npz (X: [N, T, H, W], y: [N, H, W])
    Returns X resized to [T, GRID, GRID], y to [GRID, GRID].
    """
    def __init__(self, npz_path, split="train", train_ratio=0.8, grid=32):
        super().__init__()
        d = np.load(npz_path)
        X = d["X"]   # (N, T, H, W)
        y = d["y"]   # (N, H, W)

        # deterministic split
        N = X.shape[0]
        idxs = np.arange(N)
        rng = np.random.RandomState(SEED)
        rng.shuffle(idxs)
        n_train = int(train_ratio * N)
        if split == "train":
            use = idxs[:n_train]
        else:
            use = idxs[n_train:]

        self.X = torch.from_numpy(X[use]).float()    # [n, T, H, W]
        self.y = torch.from_numpy(y[use]).float()    # [n, H, W]
        self.grid = grid

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]  # [T, H, W]
        y = self.y[i]  # [H, W]

        # Resize to GRID×GRID using bilinear
        # reshape to [1,T,H,W] → interpolate on spatial dims → [T,GRID,GRID]
        x_t = x.unsqueeze(0)              # [1, T, H, W]
        x_t = F.interpolate(x_t, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        x_t = x_t.squeeze(0)              # [T, GRID, GRID]

        y_t = y.unsqueeze(0).unsqueeze(0) # [1,1,H,W]
        y_t = F.interpolate(y_t, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        y_t = y_t.squeeze(0).squeeze(0)   # [GRID, GRID]

        return x_t, y_t

# -----------------------------
# Gaussian–Bernoulli RBM
# -----------------------------

class GaussianBernoulliRBM(nn.Module):
    """
    Visible: Gaussian (continuous), Hidden: Bernoulli.
    Simplified with unit variance for visibles.
    CD-1 training.
    """
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weight init
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.b_h = nn.Parameter(torch.zeros(n_hidden))   # hidden bias

    def sample_h_given_v(self, v):
        # v: [B, n_visible]
        # p(h=1|v) = sigmoid(v W + b_h)
        logits = v @ self.W + self.b_h
        prob = torch.sigmoid(logits)
        h_sample = torch.bernoulli(prob)
        return prob, h_sample

    def sample_v_given_h(self, h):
        # Gaussian visibles with unit variance, mean = h W^T + b_v
        mean = h @ self.W.t() + self.b_v
        v_sample = mean + torch.randn_like(mean)  # add N(0,1) noise
        return mean, v_sample

    def free_energy(self, v):
        # F(v) = 0.5 * ||v - b_v||^2 - sum log(1 + exp(b_h + v W))
        vbias_term = 0.5 * torch.sum((v - self.b_v) ** 2, dim=1)
        hidden_term = torch.sum(torch.log1p(torch.exp(self.b_h + v @ self.W)), dim=1)
        return vbias_term - hidden_term

    def forward(self, v, k=1):
        # CD-k (default k=1)
        v0 = v
        ph0, h0 = self.sample_h_given_v(v0)

        vk = v0
        hk = h0
        for _ in range(k):
            _, vk = self.sample_v_given_h(hk)
            phk, hk = self.sample_h_given_v(vk)

        # gradients via energy difference
        loss = torch.mean(self.free_energy(v0) - self.free_energy(vk))
        return loss

    @torch.no_grad()
    def transform(self, v):
        # deterministic hidden probabilities as features
        prob, _ = self.sample_h_given_v(v)
        return prob

# -----------------------------
# Predictor (initialized from RBMs)
# -----------------------------

class PredictorMLP(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.drop1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(p_drop)
        self.fc_out = nn.Linear(h2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc_out(x)
        return x

# -----------------------------
# Training helpers
# -----------------------------

def train_rbm(rbm, loader, epochs=5, lr=1e-3, device="cpu", name="RBM"):
    opt = optim.Adam(rbm.parameters(), lr=lr)
    rbm.train()
    for ep in range(1, epochs + 1):
        losses = []
        for X, _ in loader:
            # flatten input to [B, V]
            B = X.size(0)
            v = X.view(B, -1).to(device)
            opt.zero_grad()
            loss = rbm(v)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[{name}] epoch {ep}/{epochs}  loss={np.mean(losses):.4f}")

def fine_tune_predictor(model, loader_tr, loader_te, epochs=10, lr=1e-3, wd=1e-5, device="cpu", patience=10):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for X, y in loader_tr:
            X = X.to(device)  # [B, T, H, W]
            y = y.to(device)  # [B, H, W]
            B = X.size(0)
            v = X.view(B, -1)
            target = y.view(B, -1)

            opt.zero_grad()
            pred = model(v)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # validation
        val_rmse = evaluate_rmse(model, loader_te, device=device)
        print(f"[FineTune] epoch {ep}/{epochs}  train_loss={np.mean(losses):.4f}  val_RMSE={val_rmse:.4f}")

        # early stopping check
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val RMSE={best_val:.4f})")
                break


def evaluate_rmse(model, loader, device="cpu"):
    model.eval()
    loss_fn = nn.MSELoss(reduction="none")
    mses = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            B = X.size(0)
            v = X.view(B, -1)
            target = y.view(B, -1)
            out = model(v)
            mse = loss_fn(out, target).mean(dim=1)  # per-sample MSE
            mses.append(mse.cpu().numpy())
    mses = np.concatenate(mses, axis=0)
    return float(np.sqrt(mses.mean()))

def mc_dropout_predict(model, X, mc_samples=20, device="cpu"):
    """
    Run MC Dropout: keep dropout ON at eval time by calling model.train()
    """
    model.train()  # IMPORTANT: keep dropout active
    with torch.no_grad():
        preds = []
        for _ in range(mc_samples):
            out = model(X.to(device))  # [B, out_dim]
            preds.append(out.unsqueeze(0).cpu().numpy())
        preds = np.vstack(preds)  # [S, B, out_dim]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std  # [B, out_dim] each

# -----------------------------
# Main
# -----------------------------

def main():
    set_seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets / loaders
    ds_train = ERA5NPZDataset(DATA_NPZ, split="train", grid=GRID)
    ds_test  = ERA5NPZDataset(DATA_NPZ, split="test", grid=GRID)

    # We only need inputs for RBM pretraining (unsupervised)
    # Build loaders that return only X (but y is also available)
    train_loader_unsup = data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader_sup    = data.DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    train_loader_sup   = data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # -----------------------------
    # RBM pretraining (DBN)
    # -----------------------------
    rbm1 = GaussianBernoulliRBM(VISIBLE, HIDDEN1).to(device)
    rbm2 = GaussianBernoulliRBM(HIDDEN1, HIDDEN2).to(device)

    print("\n[Stage 1] Pretraining RBM-1 (visible -> hidden1)")
    train_rbm(rbm1, train_loader_unsup, epochs=PRETRAIN_EPOCHS_1, lr=LR_PRETRAIN, device=device, name="RBM-1")

    # Transform training data through RBM-1 to get hidden1 features for RBM-2
    print("\n[Stage 2] Pretraining RBM-2 (hidden1 -> hidden2)")
    rbm1.eval()
    # Build an iterator that feeds hidden1 to rbm2
    def hidden1_loader():
        for X, _ in train_loader_unsup:
            B = X.size(0)
            v = X.view(B, -1).to(device)
            with torch.no_grad():
                h1 = rbm1.transform(v)  # [B, HIDDEN1]
            yield h1

    # Tiny wrapper to train rbm2 using the hidden1 batches
    opt2 = optim.Adam(rbm2.parameters(), lr=LR_PRETRAIN)
    for ep in range(1, PRETRAIN_EPOCHS_2 + 1):
        losses = []
        for h1 in hidden1_loader():
            opt2.zero_grad()
            loss = rbm2(h1)  # CD-1 in hidden space
            loss.backward()
            opt2.step()
            losses.append(loss.item())
        print(f"[RBM-2] epoch {ep}/{PRETRAIN_EPOCHS_2}  loss={np.mean(losses):.4f}")

    # -----------------------------
    # Build predictor MLP & init from RBMs
    # -----------------------------
    predictor = PredictorMLP(VISIBLE, HIDDEN1, HIDDEN2, OUTPUT, p_drop=DROPOUT_P).to(device)

    # Initialize predictor's weights from RBMs (transpose mapping)
    with torch.no_grad():
        predictor.fc1.weight.copy_(rbm1.W.t())
        predictor.fc1.bias.copy_(rbm1.b_h)
        predictor.fc2.weight.copy_(rbm2.W.t())
        predictor.fc2.bias.copy_(rbm2.b_h)
        # fc_out is random-initialized (maps hidden2 → output map)

    # -----------------------------
    # Fine-tune predictor
    # -----------------------------
    print("\n[Stage 3] Fine-tuning predictor (supervised)")
    fine_tune_predictor(
        predictor,
        loader_tr=train_loader_sup,
        loader_te=test_loader_sup,
        epochs=FINE_TUNE_EPOCHS,
        lr=LR_FINETUNE,
        wd=WEIGHT_DECAY,
        device=device,
        patience=10,
    )

    # Load best checkpoint
    predictor.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # -----------------------------
    # Evaluate with MC Dropout (uncertainty)
    # -----------------------------
    predictor.eval()  # we'll re-enable dropout inside mc function
    all_mean = []
    all_std = []
    all_true = []

    with torch.no_grad():
        for X, y in test_loader_sup:
            B = X.size(0)
            v = X.view(B, -1)
            mean, std = mc_dropout_predict(predictor, v, mc_samples=MC_SAMPLES, device=device)
            all_mean.append(mean)              # [B, OUTPUT]
            all_std.append(std)                # [B, OUTPUT]
            all_true.append(y.view(B, -1).numpy())

    mean_arr = np.vstack(all_mean)   # [N_test, OUTPUT]
    std_arr  = np.vstack(all_std)    # [N_test, OUTPUT]
    true_arr = np.vstack(all_true)   # [N_test, OUTPUT]

    rmse = float(np.sqrt(((mean_arr - true_arr) ** 2).mean()))
    avg_uncert = float(std_arr.mean())

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_SAVE, mean=mean_arr, std=std_arr, truth=true_arr, grid=GRID)

    with open(METRICS_SAVE, "w") as f:
        f.write(f"Validation RMSE (normalized units): {rmse:.4f}\n")
        f.write(f"Average predictive std (MC dropout): {avg_uncert:.4f}\n")
        f.write(f"Grid size: {GRID}x{GRID}\n")
        f.write(f"Seq len: {SEQ_LEN}\n")
        f.write(f"Train epochs: pretrain1={PRETRAIN_EPOCHS_1}, pretrain2={PRETRAIN_EPOCHS_2}, finetune={FINE_TUNE_EPOCHS}\n")

    print("\n=== Evaluation complete ===")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved predictions to: {PRED_SAVE}")
    print(f"Saved metrics to: {METRICS_SAVE}")
    print(f"Validation RMSE (z-score units): {rmse:.4f}")
    print(f"Average predictive std (z-score units): {avg_uncert:.4f}")

if __name__ == "__main__":
    main()
