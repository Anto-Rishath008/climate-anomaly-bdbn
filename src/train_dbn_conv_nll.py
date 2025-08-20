import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Utility: NLL Loss
# =========================
def nll_loss(mu, logvar, target):
    """
    Gaussian Negative Log-Likelihood (NLL) with variance regularization.
    mu: predicted mean
    logvar: predicted log-variance
    target: ground truth
    """
    # Clamp log-variance to avoid exploding/vanishing values
    logvar = torch.clamp(logvar, min=-4.0, max=1.0)
    var = torch.exp(logvar)

    # Gaussian NLL
    nll = 0.5 * (logvar + (target - mu) ** 2 / var)

    # Variance regularization penalty (encourages stability)
    var_penalty = 1e-4 * (logvar ** 2).mean()

    return nll.mean() + var_penalty


# =========================
# Model
# =========================
class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )

    def forward(self, x):
        return self.conv(x)


class DBNHead(nn.Module):
    def __init__(self, in_dim=32 * 32 * 16, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2)  # mean + logvar
        )
        self.out_dim = out_dim

    def forward(self, x):
        h = self.fc(x)
        mu, logvar = torch.split(h, self.out_dim, dim=-1)
        return mu, logvar


class ConvDBNModel(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16, out_dim=1024):
        super().__init__()
        self.features = ConvFeatureExtractor(in_channels, hidden_channels)
        self.head = DBNHead(in_dim=32 * 32 * hidden_channels, hidden_dim=256, out_dim=out_dim)

    def forward(self, x):
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        mu, logvar = self.head(flat)
        return mu, logvar


# =========================
# Training
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    data = np.load("data_processed/train_val.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(y_train.shape[0], -1), dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.reshape(y_val.shape[0], -1), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Model
    model = ConvDBNModel(in_channels=X_train.shape[1], hidden_channels=16, out_dim=y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    best_val_rmse = float("inf")
    patience, patience_counter = 20, 0
    max_epochs = 140

    for epoch in range(1, max_epochs + 1):
        # ---------- Training ----------
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, logvar = model(xb)
            loss = nll_loss(mu, logvar, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---------- Validation ----------
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar = model(xb)
                val_preds.append(mu.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        print(f"[FineTune-NLL] epoch {epoch}/{max_epochs} "
              f"train_NLL={np.mean(train_losses):.4f} val_RMSE={val_rmse:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), "data_processed/bdbn_conv_nll.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best RMSE={best_val_rmse:.4f})")
                break

    # Save predictions
    model.load_state_dict(torch.load("data_processed/bdbn_conv_nll.pt"))
    model.eval()
    preds, preds_std = [], []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            mu, logvar = model(xb)
            var = torch.exp(torch.clamp(logvar, min=-4.0, max=1.0))
            preds.append(mu.cpu().numpy())
            preds_std.append(var.sqrt().cpu().numpy())

    np.savez("data_processed/preds_mean_std_conv_nll.npz",
             y_pred=np.concatenate(preds, axis=0),
             y_std=np.concatenate(preds_std, axis=0),
             y_true=val_targets)


if __name__ == "__main__":
    main()
