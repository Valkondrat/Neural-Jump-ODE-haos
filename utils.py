import numpy as np
import torch

"""
Файл со служебными функциями.  
"""


def rk4_step_torch(f, z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def sample_batch(X, Y, cfg, device=None):
    idx = torch.randint(0, len(X), (cfg.train.batch_size,))
    xb = X[idx]
    yb = Y[idx]

    if device is not None:
        xb = xb.to(device)
        yb = yb.to(device)

    return xb, yb


def wing_phase_features_x_torch(x_seq):
    x = x_seq[:, :, 0]
    last_sign = torch.sign(x[:, -1:])
    signs = torch.sign(x)
    steps = torch.zeros(x.shape[0], device=x.device)

    for t in range(x.shape[1] - 1, -1, -1):
        still_same = (signs[:, t] == last_sign[:, 0]).float()
        steps = steps * still_same + still_same

    steps_on_wing = (steps / 20.0).unsqueeze(1)
    proximity = torch.exp(-x[:, -1:].abs() / 0.5)
    dx = (x[:, -1:] - x[:, -2:-1]).abs()

    if x.size(1) >= 3:
        v1 = x[:, -1:] - x[:, -2:-1]
        v0 = x[:, -2:-1] - x[:, -3:-2]
        ddx = (v1 - v0).abs()
    else:
        ddx = torch.zeros_like(dx)

    return torch.cat([steps_on_wing, proximity, dx, ddx], dim=-1)


def mdn_uncertainty_features_torch(pi, mu, sigma, eps=1e-8):
    mean = (pi.unsqueeze(-1) * mu).sum(dim=1)
    aleatoric_var = (pi.unsqueeze(-1) * sigma.pow(2)).sum(dim=1)
    epistemic_var = (pi.unsqueeze(-1) * (mu - mean.unsqueeze(1)).pow(2)).sum(dim=1)
    total_var = aleatoric_var + epistemic_var

    aleatoric_std = torch.sqrt(aleatoric_var.mean(dim=-1, keepdim=True) + eps)
    epistemic_std = torch.sqrt(epistemic_var.mean(dim=-1, keepdim=True) + eps)
    total_std = torch.sqrt(total_var.mean(dim=-1, keepdim=True) + eps)
    entropy = -(pi * torch.log(pi + eps)).sum(dim=-1, keepdim=True)
    pmax = pi.max(dim=-1, keepdim=True).values
    one_minus_pmax = 1.0 - pmax

    return torch.cat([aleatoric_std, epistemic_std, total_std, entropy, one_minus_pmax], dim=-1)


def mdn_uncertainty_features_sequence(pi, mu, sigma):
    B, H, K = pi.shape
    feats = []

    for t in range(H):
        feats.append(mdn_uncertainty_features_torch(pi[:, t], mu[:, t], sigma[:, t]).unsqueeze(1))

    return torch.cat(feats, dim=1)


def coverage_curriculum(epoch, cfg, warmup_end=None, ramp_end=None, lambda_max=None):
    if warmup_end is None:
        warmup_end = cfg.detector.coverage_warmup_end
    if ramp_end is None:
        ramp_end = cfg.detector.coverage_ramp_end
    if lambda_max is None:
        lambda_max = cfg.detector.coverage_lambda_max

    if epoch < warmup_end:
        return 0.0
    if epoch >= ramp_end:
        return lambda_max

    frac = (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
    return lambda_max * frac


def coverage_target_curriculum(epoch, cfg, start_cov=None, end_cov=None, ramp_start=None, ramp_end=None):
    if start_cov is None:
        start_cov = cfg.detector.coverage_start
    if end_cov is None:
        end_cov = cfg.detector.coverage_end
    if ramp_start is None:
        ramp_start = cfg.detector.coverage_target_ramp_start
    if ramp_end is None:
        ramp_end = cfg.detector.coverage_target_ramp_end

    if epoch < ramp_start:
        return start_cov
    if epoch >= ramp_end:
        return end_cov

    frac = (epoch - ramp_start) / max(ramp_end - ramp_start, 1)
    return start_cov + frac * (end_cov - start_cov)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def current_tf_ratio(epoch, cfg):
    if epoch < cfg.train.tf_full_end:
        return 1.0

    if epoch >= cfg.train.tf_zero_epoch:
        return 0.0

    frac = (epoch - cfg.train.tf_full_end) / max(cfg.train.tf_zero_epoch - cfg.train.tf_full_end, 1)
    return max(0.0, 1.0 - frac)


def current_horizon(epoch, cfg):
    frac = min(1.0, epoch / max(cfg.train.curriculum_end_epoch, 1))
    return min(cfg.data.full_horizon, cfg.data.short_k + int((cfg.data.full_horizon - cfg.data.short_k) * frac))


def current_noise_std(epoch, cfg, decay_epochs=None):
    if decay_epochs is None:
        decay_epochs = cfg.train.n_epochs

    return max(0.0, cfg.train.input_noise_max * (1.0 - epoch / max(decay_epochs, 1)))


def _normalize_threshold(th):
    if isinstance(th, tuple):
        th = th[0]
    elif isinstance(th, dict):
        th = th["threshold"]

    return float(th)


def _mdn_map_component_torch(pi, mu, sigma):
    B = pi.size(0)
    idx = torch.multinomial(pi, 1).squeeze(1)
    b = torch.arange(B, device=pi.device)
    mu_sampled = mu[b, idx]
    sigma_sampled = sigma[b, idx]
    return mu_sampled + sigma_sampled * torch.randn_like(mu_sampled)


def _get_z_h_for_decoder(model, xb):
    enc = model.encode(xb)

    if isinstance(enc, tuple):
        if len(enc) != 2:
            raise ValueError("Unsupported encode() tuple output.")
        z, h = enc
        return z, h

    z = enc

    if hasattr(model, "z2h"):
        h = model.z2h(z)
        return z, h

    return None, z


def _squeeze_step_mask_np(mask):
    mask = _to_numpy(mask)

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask.squeeze(-1)

    return mask
