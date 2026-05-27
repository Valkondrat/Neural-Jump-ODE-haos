from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from detectors import _adaptive_oracle_good_x_np, _score_to_predictable_mask
from utils import _get_x_np, _normalize_threshold, _squeeze_step_mask_np, _to_numpy


"""
Файл для визуализаций  
"""


def _rmse_x_by_step(pred, true):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    return np.sqrt(((pred_x - true_x) ** 2).mean(axis=0))


def _rmse_x_by_step_partial(pred, true, mask):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    mask = _squeeze_step_mask_np(mask).astype(bool)
    H = pred_x.shape[1]
    out = np.full(H, np.nan)
    err2 = (pred_x - true_x) ** 2

    for t in range(H):
        m = mask[:, t]
        if m.any():
            out[t] = np.sqrt(err2[m, t].mean())

    return out


def plot_prediction_example(pred, true, cfg, *, idx=0, score=None, gate=None, proba_unpred=None, threshold=None, mode=None, corrected_pred=None, oracle_mask=None, eps_short=None, eps_long=None, show_oracle=True, show_step_panels=True, show_reports=False, name=None, figsize=None):
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    corrected_pred = _to_numpy(corrected_pred)

    if mode is None:
        mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    if threshold is None:
        threshold = cfg.detector.gate_threshold if mode == "selective" else 0.5

    threshold = _normalize_threshold(threshold)

    if eps_short is None:
        eps_short = cfg.detector.eps_short
    if eps_long is None:
        eps_long = cfg.detector.eps_long

    if score is None:
        if gate is not None:
            score = gate
        elif proba_unpred is not None:
            score = proba_unpred

    score = _squeeze_step_mask_np(score)

    H = min(pred.shape[1], true.shape[1])
    if corrected_pred is not None:
        H = min(H, corrected_pred.shape[1])
    if score is not None:
        H = min(H, score.shape[1])

    pred = pred[:, :H]
    true = true[:, :H]

    if corrected_pred is not None:
        corrected_pred = corrected_pred[:, :H]

    if score is not None:
        score = score[:, :H]
        predictable_mask = _score_to_predictable_mask(score, cfg, threshold, mode=mode)
    else:
        predictable_mask = None

    idx = int(idx)
    if idx < 0 or idx >= pred.shape[0]:
        raise IndexError(f"idx={idx} out of range for N={pred.shape[0]}")

    if oracle_mask is None and show_oracle:
        oracle_mask, _, eps_by_step = _adaptive_oracle_good_x_np(pred, true, eps_short=eps_short, eps_long=eps_long)
    elif oracle_mask is not None:
        oracle_mask = _squeeze_step_mask_np(oracle_mask).astype(bool)[:, :H]
        eps_by_step = None
    else:
        oracle_mask = None
        eps_by_step = None

    if show_reports:
        from metrics import print_metrics
        print_metrics(
            pred=pred,true=true,
            cfg=cfg,
            gate=score if mode == "selective" else None,
            gate_threshold=threshold if mode == "selective" else None,
            name=f"{name or 'MODEL RESULTS'} | METRICS")

    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    corr_x = None if corrected_pred is None else _get_x_np(corrected_pred)

    tt = np.arange(H)
    short_k = min(cfg.data.short_k, H)

    if name is None:
        name = f"{cfg.attractor.name.upper()} {cfg.data.task} {mode} | idx={idx}"

    if show_step_panels:
        if figsize is None:
            figsize = (12, 6)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.25])
        ax = fig.add_subplot(gs[0, :])
        ax_rmse = fig.add_subplot(gs[1, 0])
        ax_cov = fig.add_subplot(gs[1, 1])
    else:
        if figsize is None:
            figsize = (15, 5)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax_rmse = None
        ax_cov = None

    ax.axvline(short_k, color="gray", ls=":", lw=1)
    ax.plot(tt, true_x[idx], lw=2.2, label="true x")
    ax.plot(tt, pred_x[idx], lw=1.7, label="raw pred x")

    if corr_x is not None:
        ax.plot(tt, corr_x[idx], lw=1.8, ls="--", label="corrected pred x")

    if predictable_mask is not None:
        rejected = ~predictable_mask[idx]
        if rejected.any():
            ax.scatter(tt[rejected], pred_x[idx, rejected], s=28, marker="x", alpha=0.85, label="rejected raw pred")

    vals = [true_x[idx], pred_x[idx]]
    if corr_x is not None:
        vals.append(corr_x[idx])

    vals = np.concatenate([v.reshape(-1) for v in vals])
    y_lim = max(1.0, np.nanmax(np.abs(vals)) * 1.12)

    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("forecast step")
    ax.set_ylabel("x")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    raw_rmse = _rmse_x_by_step(pred, true)
    corr_rmse = None if corrected_pred is None else _rmse_x_by_step(corrected_pred, true)
    partial_rmse = None
    oracle_rmse = None
    coverage_by_step = None
    oracle_coverage_by_step = None

    if show_step_panels and ax_rmse is not None:
        ax_rmse.plot(raw_rmse, lw=2, label="raw RMSE-x")

        if corr_rmse is not None:
            ax_rmse.plot(corr_rmse, lw=2, ls="--", label="corrected RMSE-x")

        if predictable_mask is not None:
            partial_rmse = _rmse_x_by_step_partial(pred, true, predictable_mask)
            ax_rmse.plot(partial_rmse, lw=2, ls=":", label="partial RMSE-x by mask")

        if oracle_mask is not None:
            oracle_rmse = _rmse_x_by_step_partial(pred, true, oracle_mask)
            ax_rmse.plot(oracle_rmse, lw=2, ls="-.", label="oracle RMSE-x")

        ax_rmse.axvline(short_k, color="gray", ls=":", lw=1)
        ax_rmse.set_title("RMSE-x by forecast step")
        ax_rmse.set_xlabel("forecast step")
        ax_rmse.set_ylabel("RMSE-x")
        ax_rmse.grid(alpha=0.3)
        ax_rmse.legend(fontsize=9)

    if show_step_panels and ax_cov is not None:
        if predictable_mask is not None:
            coverage_by_step = predictable_mask.mean(axis=0)
            ax_cov.plot(coverage_by_step, lw=2, label="predictable coverage")

        if oracle_mask is not None:
            oracle_coverage_by_step = oracle_mask.mean(axis=0)
            ax_cov.plot(oracle_coverage_by_step, lw=2, ls="--", label="oracle predictable coverage")

        ax_cov.axvline(short_k, color="gray", ls=":", lw=1)
        ax_cov.set_ylim(-0.05, 1.05)
        ax_cov.set_title("Coverage by forecast step")
        ax_cov.set_xlabel("forecast step")
        ax_cov.set_ylabel("coverage")
        ax_cov.grid(alpha=0.3)
        ax_cov.legend(fontsize=9)

    fig.suptitle(name, fontsize=15)
    fig.tight_layout()
    plt.show()

    return {
        "idx": idx,
        "mode": mode,
        "threshold": threshold,
        "predictable_mask": predictable_mask,
        "oracle_mask": oracle_mask,
        "coverage_by_step": coverage_by_step,
        "oracle_coverage_by_step": oracle_coverage_by_step,
        "rmse_x_raw": raw_rmse,
        "rmse_x_corrected": corr_rmse,
        "rmse_x_partial": partial_rmse,
        "rmse_x_oracle": oracle_rmse,
        "eps_by_step": eps_by_step}

