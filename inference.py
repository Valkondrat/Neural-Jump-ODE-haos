import torch
import numpy as np
from utils import mdn_uncertainty_features_sequence, _normalize_threshold, _to_numpy, _get_z_h_for_decoder, _mdn_map_component_torch, _squeeze_step_mask_np


"""
Файл с функциями для rollout
"""


@torch.no_grad()
def selective_rollout_numpy(model, X, cfg, Y=None, n_samples=300, horizon=None, gate_threshold=None, sample_mode=False, return_aux=True, random_sample=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if horizon is None:
        horizon = cfg.data.full_horizon

    if gate_threshold is None:
        gate_threshold = cfg.detector.gate_threshold

    if isinstance(gate_threshold, tuple):
        gate_threshold = gate_threshold[0]
    if isinstance(gate_threshold, dict):
        gate_threshold = gate_threshold["threshold"]

    gate_threshold = float(gate_threshold)
    n_samples = min(n_samples, len(X))

    if random_sample:
        idx = torch.randperm(len(X))[:n_samples]
        xb = X[idx].to(device)
        true_np = None if Y is None else Y[idx].cpu().numpy()
    else:
        xb = X[:n_samples].to(device)
        true_np = None if Y is None else Y[:n_samples].cpu().numpy()

    out = model(xb, horizon=horizon, tf_ratio=0.0,
        noise_std=0.0, sample_mode=sample_mode)

    pred, gate, err_hat, pi, mu, sigma = out[:6]

    pred_np = pred.cpu().numpy()
    gate_np = gate.squeeze(-1).cpu().numpy()
    err_hat_np = err_hat.squeeze(-1).cpu().numpy()

    mask_np = gate_np >= gate_threshold

    partial_pred = pred_np.copy()
    partial_pred[~mask_np] = np.nan

    if not return_aux:
        return pred_np, partial_pred, true_np, gate_np, mask_np

    aux = {"err_hat": err_hat_np,
        "pi": pi.cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "sigma": sigma.cpu().numpy()}

    return pred_np, partial_pred, true_np, gate_np, mask_np, aux


@torch.no_grad()
def forced_rollout_numpy(model, X, cfg, Y=None, n_samples=300, horizon=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if horizon is None:
        horizon = cfg.data.full_horizon

    n_samples = min(n_samples, len(X))

    xb = X[:n_samples].to(device)
    true_np = None if Y is None else Y[:n_samples].cpu().numpy()

    pred, pi, mu, sigma = model(
        xb,horizon=horizon,
        tf_ratio=0.0,
        noise_std=0.0)

    unc = mdn_uncertainty_features_sequence(pi, mu, sigma)

    return {"pred": pred.cpu().numpy(),
        "true": true_np,
        "pi": pi.cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "sigma": sigma.cpu().numpy(),
        "unc": unc.cpu().numpy()}


@torch.no_grad()
def corrected_rollout_numpy(model, X, cfg, *, Y=None, n_samples=None, horizon=None, threshold=0.5, mode=None, random_sample=False, replacement="mdn_map", platt_model=None, platt_coef=None, platt_intercept=None, posthoc_clf=None, return_aux=True):
    from detectors import _apply_platt_torch, _predict_posthoc_unpred_proba_torch, _step_posthoc_features_torch

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if horizon is None:
        horizon = cfg.data.full_horizon

    if mode is None:
        mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    threshold = _normalize_threshold(threshold)

    if n_samples is None:
        n_samples = len(X)

    n_samples = min(n_samples, len(X))

    if random_sample:
        idx = torch.randperm(len(X))[:n_samples]
        xb = X[idx].to(device)
        true_np = None if Y is None else Y[idx].detach().cpu().numpy()
    else:
        xb = X[:n_samples].to(device)
        true_np = None if Y is None else Y[:n_samples].detach().cpu().numpy()

    task = cfg.data.task
    is_x_delay = task == "x_delay"
    is_full_phase = task == "full_phase"

    decoder = model.decoder

    if not hasattr(decoder, "rnn") or not hasattr(decoder, "head"):
        raise ValueError("corrected_rollout_numpy expects model.decoder to have .rnn and .head")

    z, h = _get_z_h_for_decoder(model, xb)
    last_sign, wing_time = decoder._init_wing_state(xb)

    corrected_feeds = []
    raw_context_preds = []
    scores = []
    predictable_masks = []
    all_pi, all_mu, all_sigma = [], [], []
    all_err_hat = []

    if is_x_delay:
        scalar_buffer = [xb[:, t, 0:1].detach() for t in range(xb.size(1))]
        required = (decoder.delay_dim - 1) * decoder.delay_tau + 1

        x_prev = scalar_buffer[-1]
        x_prev_prev = scalar_buffer[-2]
        last_accepted = x_prev.clone()
        steps_on_wing = wing_time.clone()

        for step in range(horizon):
            delay_prev = decoder._build_delay_from_buffer(scalar_buffer)
            inp_parts = [delay_prev]

            if mode == "posthoc":
                inp_parts.append(z)

            inp_parts.extend([wing_time, last_sign])
            inp = decoder.input_norm(torch.cat(inp_parts, dim=-1))
            h = decoder.rnn(inp, h)

            if mode == "selective":
                prox = torch.exp(-x_prev.abs() / 0.5)
                dx = (x_prev - x_prev_prev).abs()

                if len(scalar_buffer) >= 3:
                    v1 = scalar_buffer[-1] - scalar_buffer[-2]
                    v0 = scalar_buffer[-2] - scalar_buffer[-3]
                    ddx = (v1 - v0).abs()
                else:
                    ddx = torch.zeros_like(dx)

                phys = torch.cat([steps_on_wing, prox, dx, ddx], dim=-1)
                x_next, pi, mu, sigma, gate_raw, err_hat = decoder.head(h, phys, step=step)
                score = _apply_platt_torch(gate_raw, platt_model=platt_model, platt_coef=platt_coef, platt_intercept=platt_intercept)
                accept = score >= threshold
                all_err_hat.append(err_hat.unsqueeze(1))
            else:
                x_next, pi, mu, sigma = decoder.head(h)
                feat = _step_posthoc_features_torch(pi, mu, sigma, step=step, horizon=horizon)
                score = _predict_posthoc_unpred_proba_torch(posthoc_clf, feat)
                accept = score < threshold

            mdn_map = _mdn_map_component_torch(pi, mu, sigma)

            if replacement == "mdn_map":
                repl = mdn_map
            elif replacement == "last_accepted":
                repl = last_accepted
            else:
                repl = x_prev

            x_feed = torch.where(accept, x_next, repl).detach()
            last_accepted = torch.where(accept, x_next.detach(), last_accepted)

            corrected_feeds.append(x_feed.unsqueeze(1))
            raw_context_preds.append(x_next.unsqueeze(1))
            scores.append(score.unsqueeze(1))
            predictable_masks.append(accept.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            x_prev_prev = x_prev
            x_prev = x_feed
            scalar_buffer.append(x_feed)

            new_sign = torch.sign(x_feed)
            switched = (new_sign != last_sign).float()

            wing_scale = getattr(decoder, "wing_scale", float(cfg.data.short_k))
            wing_time = (wing_time + 1.0 / wing_scale) * (1.0 - switched)
            steps_on_wing = (steps_on_wing + 1.0 / wing_scale) * (1.0 - switched)
            last_sign = new_sign

    elif is_full_phase:
        xyz_prev = xb[:, -1, :]
        xyz_prev_prev = xb[:, -2, :]
        last_accepted = xyz_prev.clone()
        steps_on_wing = wing_time.clone()

        for step in range(horizon):
            inp_parts = [xyz_prev]

            if mode == "posthoc":
                inp_parts.append(z)

            inp_parts.extend([wing_time, last_sign])
            inp = decoder.input_norm(torch.cat(inp_parts, dim=-1))
            h = decoder.rnn(inp, h)

            if mode == "selective":
                prox = torch.exp(-xyz_prev[:, 0:1].abs() / 0.5)
                dx = (xyz_prev[:, 0:1] - xyz_prev_prev[:, 0:1]).abs()
                phys = torch.cat([steps_on_wing, prox, xyz_prev[:, 2:3], dx], dim=-1)

                xyz_next, pi, mu, sigma, gate_raw, err_hat = decoder.head(h, phys, step=step)
                score = _apply_platt_torch(gate_raw, platt_model=platt_model, platt_coef=platt_coef, platt_intercept=platt_intercept)
                accept = score >= threshold
                all_err_hat.append(err_hat.unsqueeze(1))
            else:
                xyz_next, pi, mu, sigma = decoder.head(h)
                feat = _step_posthoc_features_torch(pi, mu, sigma, step=step, horizon=horizon)
                score = _predict_posthoc_unpred_proba_torch(posthoc_clf, feat)
                accept = score < threshold

            mdn_map = _mdn_map_component_torch(pi, mu, sigma)

            if replacement == "mdn_map":
                repl = mdn_map
            elif replacement == "last_accepted":
                repl = last_accepted
            else:
                repl = xyz_prev

            xyz_feed = torch.where(accept, xyz_next, repl).detach()
            last_accepted = torch.where(accept, xyz_next.detach(), last_accepted)

            corrected_feeds.append(xyz_feed.unsqueeze(1))
            raw_context_preds.append(xyz_next.unsqueeze(1))
            scores.append(score.unsqueeze(1))
            predictable_masks.append(accept.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            xyz_prev_prev = xyz_prev
            xyz_prev = xyz_feed

            new_sign = torch.sign(xyz_prev[:, 0:1])
            switched = (new_sign != last_sign).float()

            wing_scale = getattr(decoder, "wing_scale", float(cfg.data.short_k))
            wing_time = (wing_time + 1.0 / wing_scale) * (1.0 - switched)
            steps_on_wing = (steps_on_wing + 1.0 / wing_scale) * (1.0 - switched)
            last_sign = new_sign

    corrected_pred_np = torch.cat(corrected_feeds, dim=1).cpu().numpy()
    raw_context_pred_np = torch.cat(raw_context_preds, dim=1).cpu().numpy()
    score_np = torch.cat(scores, dim=1).squeeze(-1).cpu().numpy()
    mask_np = torch.cat(predictable_masks, dim=1).squeeze(-1).cpu().numpy().astype(bool)

    aux = {"raw_context_pred": raw_context_pred_np,
        "score": score_np,
        "predictable_mask": mask_np,
        "pi": torch.cat(all_pi, dim=1).cpu().numpy(),
        "mu": torch.cat(all_mu, dim=1).cpu().numpy(),
        "sigma": torch.cat(all_sigma, dim=1).cpu().numpy(),
        "replacement": replacement,
        "mode": mode,
        "threshold": threshold}

    if all_err_hat:
        aux["err_hat"] = torch.cat(all_err_hat, dim=1).squeeze(-1).cpu().numpy()

    if return_aux:
        return corrected_pred_np, true_np, score_np, aux

    return corrected_pred_np, true_np, score_np


def corrected_from_scores_numpy(pred, scores, cfg, *, threshold=0.5, mode=None, replacement="nan"):
    pred = _to_numpy(pred).copy()
    scores = _squeeze_step_mask_np(scores)

    if mode is None:
        mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    threshold = _normalize_threshold(threshold)

    if mode == "selective":
        predictable = scores >= threshold
    elif mode == "posthoc":
        predictable = scores < threshold
    else:
        raise ValueError("mode must be 'selective' or 'posthoc'")

    corrected = pred.copy()

    if replacement == "nan":
        corrected[~predictable] = np.nan
    elif replacement == "hold":
        for i in range(corrected.shape[0]):
            for t in range(corrected.shape[1]):
                if not predictable[i, t]:
                    if t > 0:
                        corrected[i, t] = corrected[i, t - 1]
                    else:
                        corrected[i, t] = np.nan
    else:
        raise ValueError("replacement must be one of: 'nan', 'hold'")

    return corrected, predictable
