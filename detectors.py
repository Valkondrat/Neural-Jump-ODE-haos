from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from inference import selective_rollout_numpy
from utils import _get_x_np, _normalize_threshold, _squeeze_gate_np, _squeeze_step_mask_np


"""
Файл с функциями, как либо связанными с детекторами непрогнозируемых точек
"""


def oracle_good_x(pred, true, eps_x):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    err_x = np.abs(pred_x - true_x)
    return err_x < eps_x, err_x


def _adaptive_oracle_good_x_np(pred, true, *, eps_short=0.6, eps_long=1.2):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    H = pred_x.shape[1]
    eps_by_step = np.linspace(eps_short, eps_long, H, dtype=np.float32)
    err_x = np.abs(pred_x - true_x)
    oracle_mask = err_x <= eps_by_step[None, :]
    return oracle_mask, err_x, eps_by_step


def adaptive_eps_x_np(horizon, eps_short=0.6, eps_long=1.2):
    steps = np.arange(horizon)
    frac = steps / max(horizon - 1, 1)
    return eps_short + frac * (eps_long - eps_short)


def oracle_good_x_adaptive_any(pred, true, eps_short=0.6, eps_long=1.2):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    H = pred_x.shape[1]
    eps_by_step = adaptive_eps_x_np(horizon=H, eps_short=eps_short, eps_long=eps_long)
    err_x = np.abs(pred_x - true_x)
    oracle_mask = err_x < eps_by_step.reshape(1, H)
    return oracle_mask, err_x, eps_by_step


def apply_platt_to_gate(gate_np, platt_model):
    gate_np = _squeeze_gate_np(gate_np)
    flat = gate_np.reshape(-1, 1)
    calibrated = platt_model.predict_proba(flat)[:, 1]
    return calibrated.reshape(gate_np.shape)


def calibrate_gate_platt_adaptive_constrained(model, X_val, Y_val, cfg, n_samples=2048,
                                               eps_short=None, eps_long=None, target_precision=None, min_recall=None, min_coverage=None):
    if eps_short is None:
        eps_short = cfg.detector.eps_short
    if eps_long is None:
        eps_long = cfg.detector.eps_long
    if target_precision is None:
        target_precision = cfg.detector.target_precision
    if min_recall is None:
        min_recall = cfg.detector.min_recall
    if min_coverage is None:
        min_coverage = cfg.detector.min_coverage

    pred_np, _, true_np, gate_np, _ = selective_rollout_numpy(model, X_val, cfg,
        Y=Y_val, n_samples=n_samples, horizon=cfg.data.full_horizon, gate_threshold=0.0, return_aux=False, random_sample=True)

    gate_np = _squeeze_gate_np(gate_np)
    y_good, err_x, eps_by_step = oracle_good_x_adaptive_any(pred_np, true_np, eps_short=eps_short, eps_long=eps_long)

    y_flat = y_good.reshape(-1)
    g_flat = gate_np.reshape(-1, 1)

    platt_lr = LogisticRegression(C=1.0, max_iter=1000)
    platt_lr.fit(g_flat, y_flat)
    g_cal = platt_lr.predict_proba(g_flat)[:, 1]
    best = None

    for th in np.linspace(0.01, 0.99, 300):
        mask = g_cal >= th
        coverage = mask.mean()

        if coverage < min_coverage:
            continue

        tp = np.logical_and(mask, y_flat).sum()
        fp = np.logical_and(mask, ~y_flat).sum()
        fn = np.logical_and(~mask, y_flat).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision < target_precision:
            continue

        if recall < min_recall:
            continue

        item = {"threshold": float(th), "precision": float(precision), "recall": float(recall), "coverage": float(coverage),
            "TP": int(tp),"FP": int(fp),"FN": int(fn)}

        if best is None or item["coverage"] > best["coverage"]:
            best = item

    if best is None:
        return cfg.detector.gate_threshold, platt_lr, None

    return best["threshold"], platt_lr, best


def make_oracle_unpredictable_labels(pred, true, cfg=None, *, eps_x=None, adaptive=False, eps_short=None, eps_long=None):
    if cfg is not None:
        if eps_x is None:
            eps_x = cfg.detector.eps_x
        if eps_short is None:
            eps_short = cfg.detector.eps_short
        if eps_long is None:
            eps_long = cfg.detector.eps_long

    if eps_x is None:
        eps_x = 0.7
    if eps_short is None:
        eps_short = 0.4
    if eps_long is None:
        eps_long = 1.2

    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError(f"pred and true must have shape [N,H,D], got {pred.shape} and {true.shape}")

    N, H, _ = pred.shape
    err_x = np.abs(pred[:, :, 0] - true[:, :, 0])

    if adaptive:
        thresholds = np.linspace(eps_short, eps_long, H, dtype=np.float32)[None, :]
        labels = (err_x > thresholds).astype(np.int64)
    else:
        labels = (err_x > eps_x).astype(np.int64)

    return labels, err_x


def build_mdn_stat_features(rollout: Dict[str, np.ndarray]):
    unc = rollout["unc"]
    sigma = rollout["sigma"]
    pi = rollout["pi"]
    N, H, _ = unc.shape

    step = np.linspace(0.0, 1.0, H, dtype=np.float32)[None, :, None]
    step = np.repeat(step, N, axis=0)

    sigma_mean = sigma.mean(axis=(2, 3), keepdims=False)[..., None]
    sigma_max = sigma.max(axis=(2, 3), keepdims=False)[..., None]
    pi_max = pi.max(axis=2, keepdims=True)
    feats = np.concatenate([unc, sigma_mean, sigma_max, pi_max, step], axis=-1)

    return feats.reshape(N * H, feats.shape[-1])


def fit_posthoc_unpredictable_classifier(rollout_val: Dict[str, np.ndarray], cfg, *, eps_x=None, adaptive=False, eps_short=None, eps_long=None):

    labels, err_x = make_oracle_unpredictable_labels(rollout_val["pred"], rollout_val["true"], cfg, eps_x=eps_x, adaptive=adaptive, eps_short=eps_short,
        eps_long=eps_long)

    X_feat = build_mdn_stat_features(rollout_val)
    y = labels.reshape(-1)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Oracle labels contain only one class. "
        "Change eps_x or use adaptive thresholds.")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_feat, y)
    proba = clf.predict_proba(X_feat)[:, 1]

    info = {"positive_rate": float(y.mean()),"auc": float(roc_auc_score(y, proba)), "ap": float(average_precision_score(y, proba))}

    return clf, info


def apply_posthoc_classifier(clf, rollout: Dict[str, np.ndarray]):
    X_feat = build_mdn_stat_features(rollout)
    N, H, _ = rollout["pred"].shape
    proba = clf.predict_proba(X_feat)[:, 1].reshape(N, H)
    return proba


def _apply_platt_torch(gate_raw, *, platt_model=None, platt_coef=None, platt_intercept=None):
    if platt_model is not None:
        g_np = gate_raw.detach().cpu().numpy().reshape(-1, 1)
        p_np = platt_model.predict_proba(g_np)[:, 1]
        return torch.as_tensor(p_np.reshape(gate_raw.shape), device=gate_raw.device, dtype=gate_raw.dtype)

    if platt_coef is not None and platt_intercept is not None:
        return torch.sigmoid(float(platt_coef) * gate_raw + float(platt_intercept))

    return gate_raw


def _step_posthoc_features_torch(pi, mu, sigma, step: int, horizon: int):

    from utils import mdn_uncertainty_features_torch

    B = pi.size(0)
    unc = mdn_uncertainty_features_torch(pi, mu, sigma)
    sigma_mean = sigma.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
    sigma_max = sigma.amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
    pi_max = pi.max(dim=1, keepdim=True).values
    step_norm = torch.full((B, 1), step / max(horizon - 1, 1), device=pi.device, dtype=pi.dtype)
    return torch.cat([unc, sigma_mean, sigma_max, pi_max, step_norm], dim=-1)


def _predict_posthoc_unpred_proba_torch(clf, features_torch):
    feat_np = features_torch.detach().cpu().numpy()
    p_np = clf.predict_proba(feat_np)[:, 1]
    return torch.as_tensor(p_np[:, None], device=features_torch.device, dtype=features_torch.dtype)


def _score_to_predictable_mask(score, cfg, threshold, mode=None):
    score = _squeeze_step_mask_np(score)

    if mode is None:
        mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    threshold = _normalize_threshold(threshold)

    if mode == "selective":
        return score >= threshold
    if mode == "posthoc":
        return score < threshold

    raise ValueError("mode must be 'selective' or 'posthoc'")
