import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from detectors import oracle_good_x, oracle_good_x_adaptive_any
from losses import amplitude_loss
from metrics import amplitude_ratio, valid_time, valid_time_x

"""
Файл функциями для оценки прогнозов
"""


@torch.no_grad()
def evaluate_model_selective(model, X, Y, cfg, *, n_eval=256, gate_threshold_log=None, use_adaptive_oracle=None):
    model.eval()

    task = cfg.data.task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gate_threshold_log is None:
        gate_threshold_log = cfg.detector.gate_threshold

    if use_adaptive_oracle is None:
        use_adaptive_oracle = task == "x_delay"

    n_eval = min(n_eval, len(X))

    xb = X[:n_eval].to(device)
    yb = Y[:n_eval].to(device)

    pred, gate, err_hat, pi, mu, sigma = model(
        xb,
        horizon=cfg.data.full_horizon,
        tf_ratio=0.0,
        noise_std=0.0,
        sample_mode=False)

    short_rmse = F.mse_loss(pred[:, :cfg.data.short_k], yb[:, :cfg.data.short_k]).sqrt().item()
    amp_loss = amplitude_loss(pred[:, cfg.data.short_k:], yb[:, cfg.data.short_k:]).item()

    pred_np = pred.detach().cpu().numpy()
    true_np = yb.detach().cpu().numpy()
    gate_np = gate.squeeze(-1).detach().cpu().numpy()

    if task == "x_delay":
        amp_ratio_x = (
            pred[:, cfg.data.short_k:, 0].std(dim=1, correction=0).mean()/(yb[:, cfg.data.short_k:, 0].std(dim=1, correction=0).mean() + 1e-8)).item()
    elif task == "full_phase":
        amp_rat = amplitude_ratio(pred_np[:, cfg.data.short_k:, :], true_np[:, cfg.data.short_k:, :])
        amp_ratio_x = float(amp_rat[0])

    vt = valid_time_x(pred_np, true_np, threshold=0.4, dt=cfg.attractor.dt)

    if use_adaptive_oracle:
        oracle_mask, _, _ = oracle_good_x_adaptive_any(
            pred_np, true_np,
            eps_short=cfg.detector.eps_short,
            eps_long=cfg.detector.eps_long)
    else:
        oracle_mask, _ = oracle_good_x(pred_np, true_np, eps_x=cfg.detector.eps_x)

    y_true_gate = oracle_mask.reshape(-1).astype(int)
    y_score_gate = gate_np.reshape(-1)

    if len(np.unique(y_true_gate)) == 2:
        gate_auc = roc_auc_score(y_true_gate, y_score_gate)
        gate_ap = average_precision_score(y_true_gate, y_score_gate)
    else:
        gate_auc = np.nan
        gate_ap = np.nan

    gate_mask = gate_np >= gate_threshold_log

    tp = np.logical_and(gate_mask, oracle_mask).sum()
    fp = np.logical_and(gate_mask, ~oracle_mask).sum()
    fn = np.logical_and(~gate_mask, oracle_mask).sum()

    gate_precision = tp/max(tp + fp, 1)
    gate_recall = tp/max(tp + fn, 1)
    gate_coverage = gate_mask.mean()

    gate_good_mean = gate_np[oracle_mask].mean() if oracle_mask.any() else np.nan
    gate_bad_mean = gate_np[~oracle_mask].mean() if (~oracle_mask).any() else np.nan
    gate_sep = gate_good_mean - gate_bad_mean

    mean_sigma = sigma.mean().item()
    mean_gate = gate.mean().item()

    print(f"mean_sigma={mean_sigma:.4f} "
        f"mean_gate={mean_gate:.4f} "
        f"gate_auc={gate_auc:.4f} "
        f"gate_ap={gate_ap:.4f} "
        f"gate_sep={gate_sep:.4f} "
        f"gate_cov@{gate_threshold_log:.2f}={gate_coverage:.3f} "
        f"gate_p@{gate_threshold_log:.2f}={gate_precision:.3f} "
        f"gate_r@{gate_threshold_log:.2f}={gate_recall:.3f}")

    gate_diag = {"gate_auc": float(gate_auc),
        "gate_ap": float(gate_ap),
        "gate_sep": float(gate_sep),
        "gate_precision": float(gate_precision),
        "gate_recall": float(gate_recall),
        "gate_coverage": float(gate_coverage),
        "gate_good_mean": float(gate_good_mean),
        "gate_bad_mean": float(gate_bad_mean)}

    return short_rmse, amp_loss, amp_ratio_x, vt, gate_diag


@torch.no_grad()
def evaluate_model_mdnstats(model, X, Y, cfg, *, n_eval=256):
    model.eval()

    task = cfg.data.task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_eval = min(n_eval, len(X))

    xb = X[:n_eval].to(device)
    yb = Y[:n_eval].to(device)

    pred, pi, mu, sigma = model(
        xb,horizon=cfg.data.full_horizon,
        tf_ratio=0.0, noise_std=0.0)

    short_rmse = F.mse_loss(pred[:, :cfg.data.short_k], yb[:, :cfg.data.short_k]).sqrt().item()
    amp = amplitude_loss(pred[:, cfg.data.short_k:], yb[:, cfg.data.short_k:]).item()

    pred_np = pred.detach().cpu().numpy()
    true_np = yb.detach().cpu().numpy()

    vt_x = valid_time_x(pred_np, true_np, threshold=0.4, dt=cfg.attractor.dt)
    vt = valid_time(pred_np, true_np, threshold=0.4, dt=cfg.attractor.dt)

    mean_sigma = sigma.mean().item()

    if task == "x_delay":
        amp_ratio_x = (
            pred[:, cfg.data.short_k:, 0].std(dim=1, correction=0).mean()/(yb[:, cfg.data.short_k:, 0].std(dim=1, correction=0).mean() + 1e-8)).item()
    elif task == "full_phase":
        amp_rat = amplitude_ratio(pred_np[:, cfg.data.short_k:, :], true_np[:, cfg.data.short_k:, :])
        amp_ratio_x = float(amp_rat[0])


    print(f"mean_sigma={mean_sigma:.4f} "
        f"short_rmse={short_rmse:.4f} "
        f"amp={amp:.4f} "
        f"amp_ratio_x={amp_ratio_x:.4f} "
        f"valid_time={vt:.3f} "
        f"valid_time_x={vt_x:.3f}")

    return {"short_rmse": float(short_rmse),
        "amp": float(amp),
        "amp_ratio_x": float(amp_ratio_x),
        "valid_time": float(vt),
        "valid_time_x": float(vt_x),
        "mean_sigma": float(mean_sigma)}


def evaluate_model(model, X, Y, cfg, *, n_eval=256, gate_threshold_log=None, use_adaptive_oracle=None):
    detector_mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    if detector_mode == "selective":
        return evaluate_model_selective(
            model, X, Y, cfg,n_eval=n_eval,
            gate_threshold_log=gate_threshold_log,
            use_adaptive_oracle=use_adaptive_oracle)

    if detector_mode == "posthoc":
        return evaluate_model_mdnstats(model, X, Y, cfg, n_eval=n_eval)


