from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from detectors import make_oracle_unpredictable_labels

"""
Файл для вывода метрик, оценки моделей и создания финальных таблиц с метриками 
"""


def _to_numpy(a):
    if a is None:
        return None
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _get_x_np(arr):
    arr = _to_numpy(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        return arr[:, :, 0]

    raise ValueError(f"Unsupported shape: {arr.shape}")


def _squeeze_gate_np(gate):
    gate = _to_numpy(gate)

    if gate.ndim == 3:
        gate = gate.squeeze(-1)

    return gate


def rmse_by_step(pred, true):
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return np.sqrt(((pred - true) ** 2).mean(axis=(0, 2)))

def rmse_by_step_x(pred, true):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    return np.sqrt(((pred_x - true_x) ** 2).mean(axis=0))


def rmse_by_step_x_partial(pred, true, mask):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    mask = _squeeze_gate_np(mask)
    err2 = (pred_x - true_x) ** 2
    denom = np.maximum(mask.sum(axis=0), 1)
    return np.sqrt((err2 * mask).sum(axis=0) / denom)


def coord_rmse(pred, true):
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return np.sqrt(((pred - true) ** 2).mean(axis=(0, 1)))


def amplitude_ratio(pred, true):
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    std_p = pred.std(axis=1).mean(axis=0)
    std_t = true.std(axis=1).mean(axis=0)
    return std_p / (std_t + 1e-8)


def count_switches_1d(arr):
    s = np.sign(arr)
    return np.mean(s[1:] != s[:-1])


def switching_ratio(pred_x, true_x):
    pred_x = _to_numpy(pred_x)
    true_x = _to_numpy(true_x)
    sw_p = np.mean([count_switches_1d(a) for a in pred_x])
    sw_t = np.mean([count_switches_1d(a) for a in true_x])
    return sw_p, sw_t, sw_p / (sw_t + 1e-8)


def spectral_distance_np(pred_x, true_x):
    pred_x = _to_numpy(pred_x)
    true_x = _to_numpy(true_x)
    fp = np.abs(np.fft.rfft(pred_x, axis=1)).mean(axis=0) + 1e-8
    ft = np.abs(np.fft.rfft(true_x, axis=1)).mean(axis=0) + 1e-8
    return np.mean((np.log(fp) - np.log(ft)) ** 2)


def valid_time(pred, true, threshold=0.4, dt=0.05):
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    err = np.sqrt(((pred - true) ** 2).mean(axis=-1))
    exceed = err > threshold
    first_exceed = exceed.argmax(axis=1)
    never_exceeded = exceed.sum(axis=1) == 0
    first_exceed[never_exceeded] = err.shape[1]
    return first_exceed.mean() * dt


def valid_time_x(pred, true, threshold=0.4, dt=0.05):
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    err = np.abs(pred_x - true_x)
    exceed = err > threshold
    first_exceed = exceed.argmax(axis=1)
    never_exceeded = exceed.sum(axis=1) == 0
    first_exceed[never_exceeded] = err.shape[1]
    return first_exceed.mean() * dt


def _compute_x_metrics(pred, true, short_k: int, dt: float) -> dict[str, Any]:
    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    short_k = min(short_k, pred_x.shape[1])

    pred_short = pred_x[:, :short_k]
    true_short = true_x[:, :short_k]
    pred_long = pred_x[:, short_k:]
    true_long = true_x[:, short_k:]

    short_rmse_x = np.sqrt(((pred_short - true_short) ** 2).mean())
    full_rmse_x = np.sqrt(((pred_x - true_x) ** 2).mean())

    if pred_long.shape[1] > 0:
        long_rmse_x = np.sqrt(((pred_long - true_long) ** 2).mean())
        std_p = pred_long.std(axis=1).mean()
        std_t = true_long.std(axis=1).mean()
        amp_ratio_x = std_p / (std_t + 1e-8)
        sw_p, sw_t, sw_ratio = switching_ratio(pred_long, true_long)
        spec_d = spectral_distance_np(pred_long, true_long)
    else:
        long_rmse_x = np.nan
        amp_ratio_x = np.nan
        sw_p = np.nan
        sw_t = np.nan
        sw_ratio = np.nan
        spec_d = np.nan

    return {"short_rmse_x": float(short_rmse_x),
        "long_rmse_x": float(long_rmse_x),
        "full_rmse_x": float(full_rmse_x),
        "amp_ratio_x": float(amp_ratio_x),
        "switching_true": float(sw_t),
        "switching_pred": float(sw_p),
        "switching_ratio": float(sw_ratio),
        "spectral_distance_x": float(spec_d),
        "valid_time_x_02": float(valid_time_x(pred, true, threshold=0.2, dt=dt)),
        "valid_time_x_04": float(valid_time_x(pred, true, threshold=0.4, dt=dt)),
        "valid_time_x_08": float(valid_time_x(pred, true, threshold=0.8, dt=dt)),
        "rmse_by_step_x": rmse_by_step_x(pred, true)}


def _compute_full_phase_metrics(pred, true, short_k: int, dt: float) -> dict[str, Any]:
    pred = _to_numpy(pred)
    true = _to_numpy(true)

    if pred.ndim != 3 or pred.shape[-1] < 3:
        return {}

    short_k = min(short_k, pred.shape[1])

    pred_short = pred[:, :short_k]
    true_short = true[:, :short_k]
    pred_long = pred[:, short_k:]
    true_long = true[:, short_k:]

    short_rmse = np.sqrt(((pred_short - true_short) ** 2).mean())
    full_rmse = np.sqrt(((pred - true) ** 2).mean())

    coord_err = coord_rmse(pred, true)
    rmse_step = rmse_by_step(pred, true)
    rmse_step_coord = rmse_by_step_x(pred, true)

    if pred_long.shape[1] > 0:
        long_rmse = np.sqrt(((pred_long - true_long) ** 2).mean())
        amp_rat = amplitude_ratio(pred_long, true_long)
        sw_p, sw_t, sw_ratio = switching_ratio(pred_long[:, :, 0], true_long[:, :, 0])
        spec_d = spectral_distance_np(pred_long[:, :, 0], true_long[:, :, 0])
    else:
        long_rmse = np.nan
        amp_rat = np.full(pred.shape[-1], np.nan)
        sw_p = np.nan
        sw_t = np.nan
        sw_ratio = np.nan
        spec_d = np.nan

    return {"short_rmse": float(short_rmse),
        "long_rmse": float(long_rmse),
        "full_rmse": float(full_rmse),
        "coord_rmse": coord_err,
        "amp_ratio": amp_rat,
        "switching_true_full": float(sw_t),
        "switching_pred_full": float(sw_p),
        "switching_ratio_full": float(sw_ratio),
        "spectral_distance_x_full": float(spec_d),
        "valid_time_02": float(valid_time(pred, true, threshold=0.2, dt=dt)),
        "valid_time_04": float(valid_time(pred, true, threshold=0.4, dt=dt)),
        "valid_time_08": float(valid_time(pred, true, threshold=0.8, dt=dt)),
        "rmse_by_step": rmse_step,
        "rmse_by_step_per_coord": rmse_step_coord}


def _compute_selective_metrics(pred, true, gate, *, gate_threshold: float, eps_short: float, eps_long: float) -> dict[str, Any]:
    from detectors import oracle_good_x_adaptive_any

    gate = _squeeze_gate_np(gate)
    mask = gate >= gate_threshold
    oracle_mask, err_x, eps_by_step = oracle_good_x_adaptive_any(pred, true, eps_short=eps_short, eps_long=eps_long)

    err2_x = err_x ** 2
    full_rmse_x = np.sqrt(err2_x.mean())
    partial_rmse_x = np.sqrt((err2_x * mask).sum() / max(mask.sum(), 1))
    oracle_rmse_x = np.sqrt((err2_x * oracle_mask).sum() / max(oracle_mask.sum(), 1))

    tp = np.logical_and(mask, oracle_mask).sum()
    fp = np.logical_and(mask, ~oracle_mask).sum()
    fn = np.logical_and(~mask, oracle_mask).sum()
    tn = np.logical_and(~mask, ~oracle_mask).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {"gate_threshold": float(gate_threshold),
        "eps_short": float(eps_short),
        "eps_long": float(eps_long),
        "selective_full_rmse_x": float(full_rmse_x),
        "selective_partial_rmse_x": float(partial_rmse_x),
        "coverage": float(mask.mean()),
        "nonpredictable_share": float(1.0 - mask.mean()),
        "daemon_coverage": float(oracle_mask.mean()),
        "daemon_rmse_x": float(oracle_rmse_x),
        "precision": float(precision),
        "recall": float(recall),
        "daemon_mask": oracle_mask,
        "err_x": err_x,
        "eps_by_step": eps_by_step,
        "mask": mask,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)}


def print_metrics(pred, true, cfg, *, gate=None, gate_threshold=None, eps_short=None, eps_long=None, name=None, print_output: bool = True) -> dict[str, Any]:
    short_k = cfg.data.short_k
    dt = cfg.attractor.dt

    if gate_threshold is None:
        gate_threshold = cfg.detector.gate_threshold
    if eps_short is None:
        eps_short = cfg.detector.eps_short
    if eps_long is None:
        eps_long = cfg.detector.eps_long

    if name is None:
        mode = getattr(getattr(cfg, "detector", None), "mode", "unknown")
        name = f"{cfg.attractor.name.upper()} {cfg.data.task} {mode} METRICS"

    pred_np = _to_numpy(pred)
    true_np = _to_numpy(true)

    metrics: dict[str, Any] = {}

    x_metrics = _compute_x_metrics(pred_np, true_np, short_k=short_k, dt=dt)
    metrics.update(x_metrics)

    full_phase_metrics = _compute_full_phase_metrics(pred_np, true_np, short_k=short_k, dt=dt)
    metrics.update(full_phase_metrics)

    if gate is not None:
        selective_metrics = _compute_selective_metrics(pred_np,true_np,
            gate,gate_threshold=gate_threshold,
            eps_short=eps_short,eps_long=eps_long)
        metrics.update(selective_metrics)

    if print_output:
        print(name)

        print("\n[X metrics]")
        print(f"Short RMSE-x (0:{short_k})             : {metrics['short_rmse_x']:.4f}")
        print(f"Long  RMSE-x ({short_k}:end)           : {metrics['long_rmse_x']:.4f}")
        print(f"Full  RMSE-x                           : {metrics['full_rmse_x']:.4f}")
        print(f"Amplitude ratio x                      : {metrics['amp_ratio_x']:.4f}")
        print(
            "Switching true / pred / ratio          : "
            f"{metrics['switching_true']:.4f} / "
            f"{metrics['switching_pred']:.4f} / "
            f"{metrics['switching_ratio']:.4f}")
        print(f"Spectral distance on x-long            : {metrics['spectral_distance_x']:.6f}")
        print(
            "Valid Time-x @0.2/0.4/0.8              : "
            f"{metrics['valid_time_x_02']:.3f} / "
            f"{metrics['valid_time_x_04']:.3f} / "
            f"{metrics['valid_time_x_08']:.3f}")

        if full_phase_metrics:
            print("\n[Full-phase metrics]")
            print(f"Short RMSE all coords (0:{short_k})     : {metrics['short_rmse']:.4f}")
            print(f"Long  RMSE all coords ({short_k}:end)   : {metrics['long_rmse']:.4f}")
            print(f"Full  RMSE all coords                  : {metrics['full_rmse']:.4f}")
            print(f"Coord RMSE                             : {metrics['coord_rmse']}")
            print(f"Amplitude ratio                        : {metrics['amp_ratio']}")
            print(
                "Switching true / pred / ratio          : "
                f"{metrics['switching_true_full']:.4f} / "
                f"{metrics['switching_pred_full']:.4f} / "
                f"{metrics['switching_ratio_full']:.4f}")
            print(f"Spectral distance on x-long            : {metrics['spectral_distance_x_full']:.6f}")
            print(
                "Valid Time all @0.2/0.4/0.8            : "
                f"{metrics['valid_time_02']:.3f} / "
                f"{metrics['valid_time_04']:.3f} / "
                f"{metrics['valid_time_08']:.3f}")

        if gate is not None:
            print("\n[Selective / gate metrics]")
            print(f"Gate threshold                         : {metrics['gate_threshold']:.4f}")
            print("Adaptive eps x                         : " f"{metrics['eps_short']:.3f} → {metrics['eps_long']:.3f}")
            print("Full recursive RMSE-x, all points      : " f"{metrics['selective_full_rmse_x']:.4f}")
            print("Partial RMSE-x, gate-selected points   : " f"{metrics['selective_partial_rmse_x']:.4f}")
            print(f"Coverage                               : {metrics['coverage']:.3f}")
            print(f"Nonpredictable share                   : {metrics['nonpredictable_share']:.3f}")
            print(f"Adaptive daemon coverage               : {metrics['daemon_coverage']:.3f}")
            print(f"Adaptive daemon partial RMSE-x         : {metrics['daemon_rmse_x']:.4f}")
            print(f"Gate precision vs adaptive daemon      : {metrics['precision']:.4f}")
            print(f"Gate recall vs adaptive daemon         : {metrics['recall']:.4f}")
            print("TP FP FN TN                             : " f"{metrics['TP']} {metrics['FP']} {metrics['FN']} {metrics['TN']}")

    return metrics


def print_posthoc_detector_metrics(proba_unpred, pred, true, cfg, *, threshold=0.5, eps_x=None, adaptive=False, eps_short=None, eps_long=None):
    labels, err_x = make_oracle_unpredictable_labels(
        pred,true,
        cfg,eps_x=eps_x,
        adaptive=adaptive,
        eps_short=eps_short,
        eps_long=eps_long)

    oracle_predictable = labels == 0
    oracle_rmse_x = np.sqrt((err_x[oracle_predictable] ** 2).mean()) if oracle_predictable.any() else np.nan
    oracle_coverage = oracle_predictable.mean()

    y = labels.reshape(-1)
    p = proba_unpred.reshape(-1)

    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
    else:
        auc, ap = np.nan, np.nan

    mask_unpred = p >= threshold

    pred_x = _get_x_np(pred)
    true_x = _get_x_np(true)
    predictable_mask = (~mask_unpred).reshape(pred_x.shape)
    partial_rmse_x = np.sqrt(((pred_x - true_x) ** 2)[predictable_mask].mean()) if predictable_mask.any() else np.nan

    tp = ((mask_unpred == 1) & (y == 1)).sum()
    fp = ((mask_unpred == 1) & (y == 0)).sum()
    fn = ((mask_unpred == 0) & (y == 1)).sum()

    precision = tp/max(tp + fp, 1)
    recall = tp/max(tp + fn, 1)
    coverage_predictable = 1.0 - mask_unpred.mean()

    print("POST-HOC UNPREDICTABLE DETECTOR")

    print(f"AUC / AP                       : {auc:.4f} / {ap:.4f}")
    print(f"threshold                      : {threshold:.3f}")
    print(f"precision unpredictable         : {precision:.4f}")
    print(f"recall unpredictable            : {recall:.4f}")
    print(f"partial RMSE-x predicted reliable : {partial_rmse_x:.4f}")
    print(f"predictable coverage            : {coverage_predictable:.4f}")
    print(f"daemon predictable coverage     : {oracle_coverage:.4f}")
    print(f"daemon unpredictable rate       : {y.mean():.4f}")
    print(f"daemon RMSE-x predictable       : {oracle_rmse_x:.4f}")

    return {"auc": float(auc),
        "ap": float(ap),
        "threshold": float(threshold),
        "precision_unpredictable": float(precision),
        "recall_unpredictable": float(recall),
        "predictable_coverage": float(coverage_predictable),
        "daemon_coverage": float(oracle_coverage),
        "daemon_rmse_x": float(oracle_rmse_x),
        "daemon_unpredictable_rate": float(y.mean()),
        "partial_rmse_x": float(partial_rmse_x),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn)}


REQUIRED_METRIC_COLUMNS = ["short_rmse_x",
    "long_rmse_x",
    "full_rmse_x",
    "partial_rmse_x",
    "amp_ratio_x",
    "switching_true",
    "switching_pred",
    "switching_ratio",
    "spectral_distance_x",
    "valid_time_x_02",
    "valid_time_x_04",
    "valid_time_x_08",
    "coverage",
    "daemon_coverage",
    "daemon_rmse_x",
    "precision",
    "recall"]


def _to_scalar(value):
    if value is None:
        return np.nan

    if isinstance(value, (int, float, bool, str)):
        return value

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    arr = np.asarray(value)

    if arr.ndim == 0:
        return arr.item()

    return np.nan


def _normalize_posthoc_detector_keys(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)

    if "coverage" not in out and "predictable_coverage" in out:
        out["coverage"] = out["predictable_coverage"]

    if "daemon_coverage" not in out and "daemon_unpredictable_rate" in out:
        out["daemon_coverage"] = 1.0 - out["daemon_unpredictable_rate"]

    if "precision" not in out and "precision_unpredictable" in out:
        out["precision"] = out["precision_unpredictable"]

    if "recall" not in out and "recall_unpredictable" in out:
        out["recall"] = out["recall_unpredictable"]

    if "partial_rmse_x" not in out and "selective_partial_rmse_x" in out:
        out["partial_rmse_x"] = out["selective_partial_rmse_x"]

    if "full_rmse_x" not in out and "selective_full_rmse_x" in out:
        out["full_rmse_x"] = out["selective_full_rmse_x"]

    return out


def merge_metric_dicts(*dicts: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for d in dicts:
        if d is None:
            continue
        out.update(dict(d))

    return _normalize_posthoc_detector_keys(out)


def make_required_metrics_table(results: Mapping[str, Mapping[str, Any] | Iterable[Mapping[str, Any] | None]] | Iterable[tuple[str, Mapping[str, Any] | Iterable[Mapping[str, Any] | None]]], *, sort_by: str | None = None, ascending: bool = True, round_digits: int | None = 4) -> pd.DataFrame:
    items = results.items() if isinstance(results, Mapping) else list(results)
    rows = []

    for model_name, metric_value in items:
        if isinstance(metric_value, Mapping):
            metrics = merge_metric_dicts(metric_value)
        else:
            metrics = merge_metric_dicts(*metric_value)

        row = {"model": model_name}

        for col in REQUIRED_METRIC_COLUMNS:
            row[col] = _to_scalar(metrics.get(col, np.nan))

        rows.append(row)

    df = pd.DataFrame(rows, columns=["model"] + REQUIRED_METRIC_COLUMNS)

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    if round_digits is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(round_digits)

    return df.reset_index(drop=True)
