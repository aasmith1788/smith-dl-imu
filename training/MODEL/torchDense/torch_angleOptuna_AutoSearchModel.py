#!/usr/bin/env python3
"""
Optuna‑driven Neural Architecture Search for IMU Angle Prediction
=================================================================
 ‑ Memory‑safe (12 M‑param cap)
 ‑ Correct nRMSE denormalisation (x * scale + min)
 ‑ Patience fixed to 10 epochs to curb over‑fitting
"""

import os, sys, datetime, json, gc
from os.path import join
from pickle import load
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna

# -------------------------------------------------------------------
# (optional) install dependencies the *first* time you run the script
# -------------------------------------------------------------------
if __name__ == "__main__" and "--install" in sys.argv:
    import subprocess
    pkgs = ["torch", "torchvision", "torchaudio", "numpy",
            "tensorboard", "tqdm", "optuna", "optuna-dashboard"]
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

# ------------------------------ CONFIG -----------------------------
DATASET_NAME = "IWALQQ_1st_correction"
DATA_TYPE    = "angle"
BASE_DATA_DIR = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\SAVE_dataSet"
OUTPUT_DIR   = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Training_results\Optuna_AutoSearch"

N_TRIALS          = 100
N_FOLDS           = 5
MAX_EPOCHS_TRIAL  = 100
MAX_EPOCHS_FINAL  = 500
PRUNING_INTERVAL  = 10

MAX_PARAMETERS    = 12_000_000  # 12 M‑param safety cap
EARLY_STOP_PATIENCE =  10     # fixed patience for trials & finals

# ----------------------------- PATHS -------------------------------
DATASET_DIR = join(BASE_DATA_DIR, DATASET_NAME)
MODELS_DIR  = join(OUTPUT_DIR, "models")
LOGS_DIR    = join(OUTPUT_DIR, "logs")
OPTUNA_DIR  = join(OUTPUT_DIR, "optuna_studies")
for d in (OUTPUT_DIR, MODELS_DIR, LOGS_DIR, OPTUNA_DIR):
    os.makedirs(d, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------- UTILITIES -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def total_params(sizes, in_dim=4242, out_dim=303):
    total, prev = 0, in_dim
    for s in sizes:
        total += prev * s + s
        prev = s
    return total + prev * out_dim + out_dim

# ------------------------- EARLY‑STOPPING --------------------------
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=1e-3):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.early_stop = False
        self.max_grad_norm = None  # set externally if needed

    def __call__(self, val_loss):
        improved = val_loss < self.best_loss - self.min_delta
        self.best_loss = min(val_loss, self.best_loss)
        self.counter = 0 if improved else self.counter + 1
        self.early_stop = self.counter >= self.patience
        return self.early_stop

# ----------------------- MODEL DEFINITION --------------------------
class ConfigurableModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flatten = nn.Flatten()
        self.layers, self.bns, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        prev = 4242
        for h, drop in zip(cfg["hidden_dims"], cfg["dropouts"]):
            self.layers.append(nn.Linear(prev, h))
            self.bns.append(nn.BatchNorm1d(h) if cfg["batch_norm"] else None)
            self.drops.append(nn.Dropout(drop))
            prev = h

        self.head = nn.Linear(prev, 303)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _act(self, x):
        act = self.cfg["activation"]
        if act == "relu":   return F.relu(x)
        if act == "gelu":   return F.gelu(x)
        if act == "swish":  return F.silu(x)
        if act == "tanh":   return torch.tanh(x)          # fixed
        if act == "leaky_relu": return F.leaky_relu(x, 0.1)
        return x  # linear

    def forward(self, x):
        x = self.flatten(x)
        for l, bn, dr in zip(self.layers, self.bns, self.drops):
            x = self._act(l(x))
            if bn is not None:
                x = bn(x)
            x = dr(x)
        return self.head(x)

# ------------------------ DATA PIPELINE ----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, dtype, split, fold):
        d = np.load(join(root, f"{fold}_fold_final_{split}.npz"))
        self.X = torch.from_numpy(d[f"final_X_{split}"]).float()
        self.Y = torch.from_numpy(d[f"final_Y_{dtype}_{split}"]).float()

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# --------------------- TRAINING UTILITIES --------------------------
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse, self.eps = nn.MSELoss(), eps
    def forward(self, p, t):
        return torch.sqrt(self.mse(p, t) + self.eps)

def make_loss(name):
    return {"rmse": RMSELoss(), "mse": nn.MSELoss(),
            "mae": nn.L1Loss(), "huber": nn.HuberLoss()}[name]

def make_opt(model, kind, lr, wd):
    if kind == "adam":   return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "nadam":  return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "adamw":  return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "sgd":    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(kind)

def nrmse_axis(pred, true, axis, scaler):
    idx = {"x":0,"y":1,"z":2}[axis]
    nrmse = 0.0
    for p, t in zip(pred, true):
        p_axis = p.view(3, -1).t()[:, idx]
        t_axis = t.view(3, -1).t()[:, idx]

        # denormalise:  x * scale + min
        p_axis = p_axis * scaler.scale_[idx] + scaler.min_[idx]
        t_axis = t_axis * scaler.scale_[idx] + scaler.min_[idx]

        rng  = torch.max(t_axis) - torch.min(t_axis)
        if rng > 0:
            nrmse += 100.0 * torch.sqrt(torch.mean((p_axis - t_axis) ** 2)) / rng
    return nrmse / len(pred)

# --------------------- OPTUNA SUGGESTIONS --------------------------
def suggest_arch(trial):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden   = [trial.suggest_int(f"h{i}", 256, 1024) for i in range(n_layers)]
    if total_params(hidden) > MAX_PARAMETERS:
        raise optuna.exceptions.TrialPruned()

    cfg = dict(
        n_layers   = n_layers,
        hidden_dims= hidden,
        dropouts   = [trial.suggest_float(f"d{i}", 0.1, 0.5) for i in range(n_layers)],
        batch_norm = trial.suggest_categorical("batch_norm", [True, False]),
        activation = trial.suggest_categorical("act", ["relu", "gelu", "swish", "tanh", "leaky_relu"])
    )
    return cfg

def suggest_train(trial):
    return dict(
        lr      = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        batch   = trial.suggest_categorical("batch", [32, 64, 128]),
        opt     = trial.suggest_categorical("opt", ["adam", "adamw", "nadam"]),
        loss    = trial.suggest_categorical("loss", ["rmse", "mse", "mae"]),
        wd      = trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        sched   = trial.suggest_categorical("sched", [True, False]),
        sched_pat = trial.suggest_int("sched_pat", 5, 20),
        sched_fac = trial.suggest_float("sched_fac", 0.1, 0.7),
        clip    = trial.suggest_float("clip", 0.5, 5.0)
    )

# ------------------------- TRAIN LOOP ------------------------------
def train_eval(model, tr_loader, va_loader, cfg_opt, scaler,
               max_epochs, trial=None, logdir=None):
    loss_fn = make_loss(cfg_opt["loss"])
    opt     = make_opt(model, cfg_opt["opt"], cfg_opt["lr"], cfg_opt["wd"])
    sched   = None
    if cfg_opt["sched"]:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg_opt["sched_fac"],
            patience=cfg_opt["sched_pat"], min_lr=1e-7
        )

    es = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    es.max_grad_norm = cfg_opt["clip"]

    tw = SummaryWriter(logdir) if logdir else None
    best, best_state = float("inf"), None

    for epoch in range(max_epochs):
        # --- train ---
        model.train(); tr_loss = 0; tr_x = tr_y = tr_z = 0; n = 0
        for X, y in tqdm(tr_loader, leave=False, desc=f"Epoch {epoch+1}/{max_epochs}"):
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), es.max_grad_norm)
            opt.step()

            bs = X.size(0)
            tr_loss += loss.item() * bs
            n += bs
            tr_x += nrmse_axis(out, y, "x", scaler) * bs
            tr_y += nrmse_axis(out, y, "y", scaler) * bs
            tr_z += nrmse_axis(out, y, "z", scaler) * bs

        tr_loss, tr_x, tr_y, tr_z = tr_loss/n, tr_x/n, tr_y/n, tr_z/n

        # --- validate ---
        model.eval(); va_loss = 0; va_x = va_y = va_z = 0; n=0
        with torch.no_grad():
            for X, y in va_loader:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                loss = loss_fn(out, y)
                bs   = X.size(0)
                va_loss += loss.item() * bs
                n += bs
                va_x += nrmse_axis(out, y, "x", scaler) * bs
                va_y += nrmse_axis(out, y, "y", scaler) * bs
                va_z += nrmse_axis(out, y, "z", scaler) * bs
        va_loss, va_x, va_y, va_z = va_loss/n, va_x/n, va_y/n, va_z/n

        if tw:
            tw.add_scalar("train/loss", tr_loss, epoch)
            tw.add_scalar("val/loss", va_loss, epoch)

        if va_loss < best:
            best, best_state = va_loss, {k:v.cpu() for k,v in model.state_dict().items()}

        if sched:
            sched.step(va_loss)

        if es(va_loss):
            break

        if trial and epoch % PRUNING_INTERVAL == 0:
            trial.report(va_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_state:
        model.load_state_dict(best_state)
    if tw:
        tw.close()
    return best

# ------------------------- OPTUNA OBJECTIVE ------------------------
def objective(trial):
    clear_gpu()
    arch = suggest_arch(trial)
    train_cfg = suggest_train(trial)

    scores = []
    for fold in range(N_FOLDS):
        ds_full = Dataset(DATASET_DIR, DATA_TYPE, "train", fold)
        val_ratio = 0.2
        tr_size   = int(len(ds_full)*(1-val_ratio))
        va_size   = len(ds_full) - tr_size
        tr_ds, va_ds = torch.utils.data.random_split(
            ds_full, [tr_size, va_size],
            generator=torch.Generator().manual_seed(42+fold))

        tr_ld = DataLoader(tr_ds, batch_size=train_cfg["batch"], shuffle=True, drop_last=True)
        va_ld = DataLoader(va_ds, batch_size=train_cfg["batch"], shuffle=False)

        mdl = ConfigurableModel(dict(
            hidden_dims = arch["hidden_dims"],
            dropouts    = arch["dropouts"],
            batch_norm  = arch["batch_norm"],
            activation  = arch["activation"]
        )).to(device)

        scaler = load(open(join(DATASET_DIR, f"{fold}_fold_scaler4Y_{DATA_TYPE}.pkl"), "rb"))
        best = train_eval(mdl, tr_ld, va_ld, train_cfg, scaler, MAX_EPOCHS_TRIAL, trial)
        scores.append(best)

        del mdl, tr_ld, va_ld; clear_gpu()

    return float(np.mean(scores))

# ------------------------------- MAIN ------------------------------
def main():
    print(f"Device: {device} | Param cap: {MAX_PARAMETERS:,}")
    study_name = f"arch_search_{timestamp}"
    storage    = f"sqlite:///{join(OPTUNA_DIR, study_name)}.db"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    with open(join(OPTUNA_DIR, f"results_{study_name}.json"), "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "param_cap": MAX_PARAMETERS
        }, f, indent=2)

if __name__ == "__main__":
    main()
