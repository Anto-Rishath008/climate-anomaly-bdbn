import argparse, os, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------- utils -----------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def corrcoef_flat(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a*b).sum(dim=1)
    den = a.norm(dim=1)*b.norm(dim=1) + eps
    return num/den

# ----------------- dataset -----------------

class ERA5Windows(Dataset):
    """
    Expects npz with:
      X_* : (N, T, H, W)
      y_* : (N, H, W)
    Optional: resize both to (R,R) with bilinear.
    """
    def __init__(self, X, Y, augment=False, resize=None):
        self.X = X; self.Y = Y
        self.augment = augment
        self.resize = int(resize) if (resize is not None and int(resize)>0) else None
        assert X.ndim==4 and Y.ndim==3 and X.shape[0]==Y.shape[0]

    def __len__(self): return self.X.shape[0]

    def _maybe_aug(self, x, y):
        # x:(T,H,W), y:(H,W)
        if not self.augment: return x,y
        if np.random.rand()<0.5:
            x = x[..., ::-1].copy(); y = y[..., ::-1].copy()   # hflip
        if np.random.rand()<0.5:
            x = x[..., ::-1, :].copy(); y = y[::-1, :].copy()  # vflip
        return x,y

    def __getitem__(self, i):
        x = self.X[i]; y = self.Y[i]
        x,y = self._maybe_aug(x,y)
        x = np.asarray(x, np.float32).copy()      # (T,H,W)
        y = np.asarray(y, np.float32).copy()      # (H,W)

        x = torch.from_numpy(x).unsqueeze(0)      # (1,T,H,W)
        y = torch.from_numpy(y)[None,None,...]    # (1,1,H,W)

        if self.resize is not None:
            # bilinear for both; y is continuous anomaly
            x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(self.resize, self.resize), mode="bilinear", align_corners=False)

        x = x.squeeze(0)              # (T,R,R)
        y = y.squeeze(0)              # (1,R,R)
        return x, y

# ----------------- model -----------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.05):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.g1 = nn.GroupNorm(min(8,out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.g2 = nn.GroupNorm(min(8,out_ch), out_ch)
        self.d = nn.Dropout2d(drop)
    def forward(self, x):
        x = F.relu(self.g1(self.c1(x))); x = self.d(x)
        x = F.relu(self.g2(self.c2(x))); return x

class UNetSmall(nn.Module):
    """Input (B,T,H,W) → (mu, log_sigma) each (B,1,H,W)."""
    def __init__(self, in_ch, base=48, drop=0.05):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base, drop)
        self.d1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.e2 = ConvBlock(base*2, base*2, drop)
        self.d2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.bt = ConvBlock(base*4, base*4, drop)

        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2, drop)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base, drop)
        self.head = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.e1(x)           # (B,b,H,W)
        e2 = self.e2(self.d1(e1)) # (B,2b,H/2,W/2)
        b  = self.bt(self.d2(e2)) # (B,4b,H/4,W/4)

        d2 = self.u2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2,e2], dim=1))

        d1 = self.u1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1,e1], dim=1))

        out = self.head(d1)
        mu, log_sigma = out[:, :1], out[:, 1:]
        return mu, log_sigma

# ----------------- loss -----------------

def het_nll(mu, log_sigma, y, sigma_min, sigma_max, mse_w, sig_reg_w):
    sigma = F.softplus(log_sigma) + 1e-5
    if sigma_min is not None or sigma_max is not None:
        sigma = sigma.clamp(min=sigma_min, max=sigma_max)

    nll = 0.5*((y-mu)**2)/(sigma**2) + torch.log(sigma)
    nll = nll.mean()

    mse = F.mse_loss(mu, y)
    sig_reg = (torch.log(sigma+1e-8)**2).mean()
    loss = nll + mse_w*mse + sig_reg_w*sig_reg
    return loss, nll.item(), mse.item(), sig_reg.item(), sigma.mean().item()

# ----------------- eval/predict -----------------

@torch.no_grad()
def evaluate(model, dl, device, args, ema=None):
    if ema is not None: ema.apply()
    model.eval()
    # FIXED LINE:
    nlls, mses, sigregs, sigmas, cors = [], [], [], [], []
    for xb,yb in dl:
        xb = xb.to(device); yb = yb.to(device)
        mu, log_sigma = model(xb)
        loss, nll, mse, sreg, smean = het_nll(mu, log_sigma, yb,
                                              args.sigma_min, args.sigma_max,
                                              args.mse_weight, args.sigma_reg)
        nlls.append(nll); mses.append(mse); sigregs.append(sreg); sigmas.append(smean)
        cors.append(corrcoef_flat(mu.view(mu.size(0),-1), yb.view(yb.size(0),-1)).mean().item())
    if ema is not None: ema.restore()
    return dict(nll=float(np.mean(nlls)), mse=float(np.mean(mses)),
                sigma_reg=float(np.mean(sigregs)), sigma_mean=float(np.mean(sigmas)),
                corr=float(np.mean(cors)))

@torch.no_grad()
def predict_and_save(model, dl, device, args, ema=None):
    if ema is not None: ema.apply()
    model.eval()
    muL, sdL, yL = [], [], []
    for xb,yb in dl:
        xb = xb.to(device)
        if args.tta:
            outs = []
            def p(x, un=None):
                m, ls = model(x)
                s = (F.softplus(ls)+1e-5).clamp(args.sigma_min, args.sigma_max)
                if un is not None: m, s = un(m), un(s)
                outs.append((m,s))
            p(xb, None)
            p(torch.flip(xb,[-1]), lambda t: torch.flip(t,[-1]))
            p(torch.flip(xb,[-2]), lambda t: torch.flip(t,[-2]))
            p(torch.flip(xb,[-2,-1]), lambda t: torch.flip(t,[-2,-1]))
            mu = torch.stack([o[0] for o in outs]).mean(0)
            sd = torch.stack([o[1] for o in outs]).mean(0)
        else:
            mu, ls = model(xb)
            sd = (F.softplus(ls)+1e-5).clamp(args.sigma_min, args.sigma_max)

        muL.append(mu.squeeze(1).cpu().numpy())
        sdL.append(sd.squeeze(1).cpu().numpy())
        yL.append(yb.squeeze(1).cpu().numpy())
    if ema is not None: ema.restore()

    y_pred = np.concatenate(muL,0); y_std=np.concatenate(sdL,0); y_true=np.concatenate(yL,0)
    os.makedirs("data_processed", exist_ok=True)
    np.savez("data_processed/preds_mean_std_conv_nll.npz", y_pred=y_pred, y_std=y_std, y_true=y_true)
    print("[ok] Saved predictions to data_processed/preds_mean_std_conv_nll.npz")

# ----------------- EMA -----------------

class EMAWrap:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n:p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_((self.decay)).add_(p.data, alpha=1-self.decay)

    def apply(self):
        self.backup = {}
        for n, p in self.shadow.items():
            self.backup[n] = p.clone()

    def restore(self):
        self.backup = {}

# ----------------- train -----------------

def train(args):
    set_seed(args.seed)
    data = np.load("data_processed/train_val.npz")
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    Ntr,T,H,W = Xtr.shape
    print(f"Loaded train: {Xtr.shape}, val: {Xva.shape}. HxW={H}x{W}, T={T}")

    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device.type)

    tr_ds = ERA5Windows(Xtr,ytr,augment=args.augment, resize=args.resize)
    va_ds = ERA5Windows(Xva,yva,augment=False,      resize=args.resize)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    model = UNetSmall(in_ch=T, base=args.base, drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type=="cuda"))
    ema = EMAWrap(model, decay=args.ema_decay) if args.ema else None

    best = -1.0 if args.monitor=="corr" else math.inf
    best_path = "data_processed/bdbn_conv_nll.pt"
    patience = args.patience; no_imp=0; t0=time.time()

    for ep in range(1, args.epochs+1):
        model.train(); tr_nlls=[]
        for xb,yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=(args.use_amp and device.type=="cuda")):
                mu, ls = model(xb)
                loss, nll, mse, sreg, smean = het_nll(mu, ls, yb,
                                    args.sigma_min, args.sigma_max, args.mse_weight, args.sigma_reg)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_nlls.append(nll)
            if ema: ema.update(model)
        sch.step()

        val = evaluate(model, va_ld, device, args, ema=ema)
        print(f"[{ep}/{args.epochs}] Train NLL: {np.mean(tr_nlls):.4f} | "
              f"Val NLL: {val['nll']:.4f} | Val Corr: {val['corr']:.4f} | σ_mean(val): {val['sigma_mean']:.3f}")

        metric = val["corr"] if args.monitor=="corr" else val["nll"]
        improved = (metric>best+1e-5) if args.monitor=="corr" else (metric<best-1e-5)
        if improved:
            best = metric; no_imp=0
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1
            if no_imp>=patience:
                print("[early stop] no val improvement for", patience, "epochs.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print("[ok] Saved best model to", best_path)
    predict_and_save(model, va_ld, device, args, ema=ema)
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

# ----------------- cli -----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--monitor", choices=["corr","nll"], default="corr")
    p.add_argument("--sigma_min", type=float, default=0.15)
    p.add_argument("--sigma_max", type=float, default=1.20)
    p.add_argument("--mse_weight", type=float, default=0.25)
    p.add_argument("--sigma_reg", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--resize", type=int, default=64, help="Resize H,W to this square (e.g., 64 or 32). Use 0 to disable.")
    return p.parse_args()

def main(): train(parse_args())
if __name__=="__main__": main()
