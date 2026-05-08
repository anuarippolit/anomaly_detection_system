import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from itertools import product
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.metrics import evaluate_predictions, save_metrics

SEED = 42


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16, 8)):
        super().__init__()

        enc = []
        in_dim = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1)]
            in_dim = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1)]
            in_dim = h
        dec += [nn.Linear(in_dim, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _train_ae(config, X_train_s, X_earlystop_s, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    input_dim = X_train_s.shape[1]
    model = DeepAutoencoder(input_dim=input_dim, hidden_dims=config["hidden_dims"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5,
    )
    criterion = nn.MSELoss()

    X_train_gpu   = torch.from_numpy(X_train_s).float().to(device)
    X_earlystop_t = torch.from_numpy(X_earlystop_s).float().to(device)
    n_train       = X_train_gpu.shape[0]
    batch_size    = config["batch_size"]

    MAX_EPOCHS        = 150
    PATIENCE          = 15
    MIN_DELTA         = 1e-5
    best_val          = float("inf")
    best_state        = None
    epochs_no_improve = 0
    last_epoch        = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        last_epoch = epoch

        model.train()
        perm    = torch.randperm(n_train, device=device)
        running = torch.zeros((), device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb  = X_train_gpu[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            running += loss.detach() * xb.size(0)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_earlystop_t), X_earlystop_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val - MIN_DELTA:
            best_val          = val_loss
            best_state        = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, last_epoch, best_val


@torch.no_grad()
def _reconstruction_errors(model, X_np, device, batch_size=8192):
    model.eval()
    X_gpu = torch.from_numpy(X_np).float().to(device)
    out   = []
    for i in range(0, X_gpu.shape[0], batch_size):
        xb  = X_gpu[i:i + batch_size]
        err = ((model(xb) - xb) ** 2).mean(dim=1)
        out.append(err.cpu().numpy())
    return np.concatenate(out)


def _best_f1_threshold(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1s))])


def _run_variant(label, X_train_s, X_earlystop_s, X_val_s, X_test_s,
                 y_val, y_test, config, device, results_path):
    print(f"  Training {label}...")
    model, epochs, _ = _train_ae(config, X_train_s, X_earlystop_s, device)

    val_errors  = _reconstruction_errors(model, X_val_s,  device)
    test_errors = _reconstruction_errors(model, X_test_s, device)

    threshold = _best_f1_threshold(y_val, val_errors)
    metrics   = evaluate_predictions(y_test, test_errors, threshold=threshold)

    save_metrics(metrics, model_name=label, path=results_path)
    print(f"  {label} -> F1={metrics['f1']:.4f}  "
          f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return metrics


def train_autoencoder(data_dir="data/processed", results_path="results/metrics.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Autoencoder using device: {device}")

    X_train_normal = pd.read_csv(f"{data_dir}/X_train_normal.csv").values.astype(np.float32)
    X_val_s        = pd.read_csv(f"{data_dir}/X_val.csv").values.astype(np.float32)
    X_test_s       = pd.read_csv(f"{data_dir}/X_test.csv").values.astype(np.float32)
    y_val          = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
    y_test         = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    # hold out 10% of normal training data for early stopping
    rng           = np.random.default_rng(SEED)
    idx           = rng.permutation(len(X_train_normal))
    n_earlystop   = max(1, int(len(X_train_normal) * 0.1))
    X_earlystop_s = X_train_normal[idx[:n_earlystop]]
    X_train_s     = X_train_normal[idx[n_earlystop:]]

    baseline_config = {
        "hidden_dims":  (64, 32, 16, 8),
        "lr":           1e-3,
        "weight_decay": 1e-5,
        "batch_size":   256,
    }

    param_grid = {
        "hidden_dims":  [(64, 32, 16, 8), (128, 64, 32, 16), (32, 16, 8)],
        "weight_decay": [1e-5, 1e-4],
        "lr":           [1e-3],
        "batch_size":   [256, 2048],
    }

    # baseline
    print("Running baseline (normal-only)...")
    _run_variant(
        "autoencoder_baseline", X_train_s, X_earlystop_s,
        X_val_s, X_test_s, y_val, y_test,
        baseline_config, device, results_path,
    )

    # grid search
    print("Running grid search (normal-only)...")
    keys        = list(param_grid.keys())
    all_configs = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    total       = len(all_configs)
    best_pr_auc = -1.0
    best_config = None

    for i, config in enumerate(all_configs, 1):
        model, _, _ = _train_ae(config, X_train_s, X_earlystop_s, device)
        val_errors  = _reconstruction_errors(model, X_val_s, device)
        pr_auc      = average_precision_score(y_val, val_errors)

        print(f"  [{i:>2}/{total}] arch={str(config['hidden_dims']):<22} "
              f"wd={config['weight_decay']:.0e} bs={config['batch_size']:<4} "
              f"-> val PR-AUC={pr_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_config = config

    print(f"Best config: {best_config}  (val PR-AUC={best_pr_auc:.4f})")

    # tuned
    print("Retraining best config (normal-only)...")
    _run_variant(
        "autoencoder_tuned", X_train_s, X_earlystop_s,
        X_val_s, X_test_s, y_val, y_test,
        best_config, device, results_path,
    )
