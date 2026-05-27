from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

"""
Файл c функциями потерь
"""


def velocity_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)

    vp = pred[:, 1:] - pred[:, :-1]
    vt = true[:, 1:] - true[:, :-1]

    denom = vt.std().detach().clamp_min(1e-4)
    return ((vp - vt) ** 2).mean() / (denom ** 2)


def spectral_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)

    losses = []
    for c in range(pred.size(-1)):
        fp = torch.fft.rfft(pred[:, :, c], dim=1).abs() + 1e-6
        ft = torch.fft.rfft(true[:, :, c], dim=1).abs() + 1e-6
        losses.append(((torch.log(fp) - torch.log(ft)) ** 2).mean())

    return torch.stack(losses).mean()


def amplitude_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)

    std_p = pred.std(dim=1, correction=0)
    std_t = true.std(dim=1, correction=0).detach()
    ratio = std_p / (std_t + 1e-6)
    return ((ratio - 1.0) ** 2).mean()


def amplitude_loss_z(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)

    if pred.size(-1) < 3:
        raise ValueError(f"amplitude_loss_z expects last dimension >= 3, got {pred.shape}")

    std_p = pred[:, :, 2].std(dim=1, correction=0)
    std_t = true[:, :, 2].std(dim=1, correction=0).detach()
    ratio = std_p / (std_t + 1e-6)
    return ((ratio - 1.0) ** 2).mean()


def mean_x_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    mp = pred[:, :, 0].mean(dim=1)
    mt = true[:, :, 0].mean(dim=1).detach()
    return ((mp - mt) ** 2).mean()


def mean_z_loss_from_long(pred_long: torch.Tensor, true_long: torch.Tensor) -> torch.Tensor:
    if pred_long.size(1) < 1:
        return torch.tensor(0.0, device=pred_long.device)

    if pred_long.size(-1) < 3:
        raise ValueError(f"mean_z_loss_from_long expects last dimension >= 3, got {pred_long.shape}")

    mp = pred_long[:, :, 2].mean(dim=1)
    mt = true_long[:, :, 2].mean(dim=1).detach()
    return ((mp - mt) ** 2).mean()


def wing_switch_loss(pred_x: torch.Tensor, true_x: torch.Tensor) -> torch.Tensor:
    if pred_x.size(1) < 2:
        return torch.tensor(0.0, device=pred_x.device)

    beta = 5.0
    sp = torch.tanh(beta * pred_x)
    st = torch.tanh(beta * true_x).detach()

    sw_p = (sp[:, 1:] - sp[:, :-1]).abs().mean(dim=1)
    sw_t = (st[:, 1:] - st[:, :-1]).abs().mean(dim=1)

    diff = sw_p - sw_t
    return (diff ** 2).mean()


def spectral_loss_xyz(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return spectral_loss(pred, true)


def amplitude_loss_xyz(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return amplitude_loss(pred, true)


def mean_z_loss_local(pred_long: torch.Tensor, true_long: torch.Tensor) -> torch.Tensor:
    return mean_z_loss_from_long(pred_long, true_long)


def mdn_nll_mean(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_exp = target.unsqueeze(1).expand_as(mu)

    log_prob = -0.5 * (((target_exp - mu) / sigma) ** 2
        + 2.0 * sigma.log()
        + math.log(2.0 * math.pi)).sum(dim=-1)

    log_mix = torch.logsumexp(log_prob + torch.log(pi + 1e-8), dim=-1)
    return -log_mix.mean()


def mdn_nll_loss(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mdn_nll_mean(pi, mu, sigma, target)


def mdn_nll_sequence(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, true: torch.Tensor, max_steps: Optional[int] = None) -> torch.Tensor:
    H = true.size(1)
    if max_steps is not None:
        H = min(H, max_steps)

    total = torch.tensor(0.0, device=true.device)

    for t in range(H):
        total = total + mdn_nll_mean(pi[:, t], mu[:, t], sigma[:, t], true[:, t])

    return total / max(H, 1)


def sigma_regularization_loss(sigma: torch.Tensor, min_sigma: float = 0.15) -> torch.Tensor:
    return torch.clamp(min_sigma - sigma, min=0.0).mean()


def sigma_smoothness_loss(sigma: torch.Tensor) -> torch.Tensor:
    if sigma.dim() < 4 or sigma.size(1) < 2:
        return torch.tensor(0.0, device=sigma.device)

    return ((sigma[:, 1:] - sigma[:, :-1]) ** 2).mean()


def soft_cdf_loss_1d(pred_x: torch.Tensor, true_x: torch.Tensor, grid: Optional[torch.Tensor] = None, temp: float = 8.0, mode: str = "adaptive") -> torch.Tensor:
    if pred_x.size(1) < 1:
        return torch.tensor(0.0, device=pred_x.device)

    if grid is None:
        if mode == "fixed_lorenz":
            grid = torch.cat(
                [torch.linspace(-1.8, -0.3, 5, device=pred_x.device),
                    torch.linspace(-0.3, 0.3, 7, device=pred_x.device),
                    torch.linspace(0.3, 1.8, 5, device=pred_x.device)])
        elif mode == "adaptive":
            with torch.no_grad():
                q = torch.linspace(0.02, 0.98, 21, device=pred_x.device)
                grid = torch.quantile(true_x.detach().reshape(-1), q)
                grid = torch.unique(grid)
        else:
            raise ValueError(f"Unknown CDF mode: {mode!r}")

    total = torch.tensor(0.0, device=pred_x.device)

    for tau in grid:
        fp = torch.sigmoid(temp * (tau - pred_x)).mean(dim=1)
        ft = torch.sigmoid(temp * (tau - true_x.detach())).mean(dim=1)
        total = total + ((fp - ft) ** 2).mean()

    return total / max(len(grid), 1)


def soft_bin_probs_1d(x: torch.Tensor, edges: Sequence[float], temp: float = 10.0) -> torch.Tensor:
    bins = []

    left = torch.sigmoid(temp * (edges[0] - x))
    bins.append(left.mean(dim=1))

    for lo, hi in zip(edges[:-1], edges[1:]):
        cdf_hi = torch.sigmoid(temp * (hi - x))
        cdf_lo = torch.sigmoid(temp * (lo - x))
        bins.append((cdf_hi - cdf_lo).mean(dim=1))

    right = 1.0 - torch.sigmoid(temp * (edges[-1] - x))
    bins.append(right.mean(dim=1))

    return torch.stack(bins, dim=1)


def x_bin_occupancy_loss(pred_x: torch.Tensor, true_x: torch.Tensor, edges: Optional[Sequence[float]] = None, temp: float = 10.0, mode: str = "adaptive") -> torch.Tensor:
    if pred_x.size(1) < 1:
        return torch.tensor(0.0, device=pred_x.device)

    if edges is None:
        if mode == "fixed_lorenz":
            edges = (-1.75, -0.8, -0.2, 0.2, 0.8, 1.75)
        elif mode == "adaptive":
            with torch.no_grad():
                q = torch.tensor([0.05, 0.20, 0.40, 0.60, 0.80, 0.95], device=pred_x.device)
                edges_t = torch.quantile(true_x.detach().reshape(-1), q)
                edges = tuple(edges_t.detach().cpu().tolist())
        else:
            raise ValueError(f"Unknown CDF mode: {mode!r}")

    p = soft_bin_probs_1d(pred_x, edges, temp=temp)
    t = soft_bin_probs_1d(true_x.detach(), edges, temp=temp)

    return ((p - t) ** 2).mean()


def coverage_penalty_asymmetric(mean_g: torch.Tensor, target: float, alpha_low: float = 2.0, alpha_high: float = 0.5) -> torch.Tensor:
    diff = mean_g - target
    return torch.where(diff < 0, alpha_low * diff ** 2, alpha_high * diff ** 2)


def adaptive_eps_x(step: int, horizon: int, eps_short: float = 0.4, eps_long: float = 1.2) -> float:
    frac = step / max(horizon - 1, 1)
    return eps_short + frac * (eps_long - eps_short)


def focal_gate_loss(gate: torch.Tensor, gate_target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy(gate, gate_target, reduction="none")
    p_t = gate * gate_target + (1.0 - gate) * (1.0 - gate_target)
    focal_w = (1.0 - p_t) ** gamma
    return (focal_w * bce).mean()


def gate_aux_loss_x_from_pred(pred: torch.Tensor, gate: torch.Tensor, err_hat: torch.Tensor, target: torch.Tensor, coverage_target: float = 0.55, lambda_cov: float = 0.5, lambda_gate: float = 1.0, lambda_err: float = 0.1, eps_x: float = 0.7, temp_err: float = 0.15) -> torch.Tensor:
    with torch.no_grad():
        real_err_x = torch.abs(pred[:, 0:1] - target[:, 0:1])
        gate_target = torch.sigmoid((eps_x - real_err_x) / temp_err)

    l_gate = focal_gate_loss(gate, gate_target, gamma=2.0)
    l_err = F.huber_loss(err_hat, real_err_x)
    cov_penalty = coverage_penalty_asymmetric(gate.mean(), coverage_target)
    return lambda_gate * l_gate + lambda_err * l_err + lambda_cov * cov_penalty


def _require_loss_cfg(loss_cfg):
    if loss_cfg is None:
        raise ValueError("loss_cfg must be provided, e.g. loss_cfg=cfg.loss")


def _stat_ramp_weight(epoch: int, loss_cfg) -> float:
    ramp_denom = max(loss_cfg.stat_ramp_epochs, 1)
    return min(1.0, max(0.0, (epoch - loss_cfg.stat_start_epoch) / ramp_denom))


def _forecast_loss(pred_tf: torch.Tensor, pi_tf: torch.Tensor, mu_tf: torch.Tensor, sigma_tf: torch.Tensor, true: torch.Tensor, short_k: int, loss_cfg) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l_nll_short = torch.tensor(0.0, device=true.device)

    for t in range(short_k):
        l_nll_short = l_nll_short + mdn_nll_mean(pi_tf[:, t], mu_tf[:, t], sigma_tf[:, t], true[:, t])

    l_nll_short = l_nll_short / max(short_k, 1)
    l_mse_short = F.mse_loss(pred_tf[:, :short_k], true[:, :short_k])
    l_vel = velocity_loss(pred_tf[:, :short_k], true[:, :short_k])
    forecast = loss_cfg.mdn_nll_weight * l_nll_short + 1.00 * l_mse_short + 0.50 * l_vel

    parts = {
        "nll_short": l_nll_short,
        "mse_short": l_mse_short,
        "vel": l_vel
    }

    return forecast, parts


def _gate_aux_rollout_loss(pred_fr: torch.Tensor, gate_fr: torch.Tensor, err_hat_fr: torch.Tensor, true: torch.Tensor, coverage_target: float, lambda_cov: float, eps_x: float, use_adaptive_oracle: bool, eps_short: float, eps_long: float) -> torch.Tensor:
    H = true.size(1)
    l_gate_aux = torch.tensor(0.0, device=true.device)

    for t in range(H):
        if use_adaptive_oracle:
            eps_t = adaptive_eps_x(t, H, eps_short=eps_short, eps_long=eps_long)
        else:
            eps_t = eps_x

        l_gate_aux = l_gate_aux + gate_aux_loss_x_from_pred(
            pred=pred_fr[:, t], gate=gate_fr[:, t],
            err_hat=err_hat_fr[:, t], target=true[:, t],
            coverage_target=coverage_target, lambda_cov=lambda_cov,
            lambda_gate=1.0, lambda_err=0.1,
            eps_x=eps_t,temp_err=0.15)

    return l_gate_aux / max(H, 1)


def _selective_debug_dict(forecast_loss: torch.Tensor, forecast_parts: Dict[str, torch.Tensor], gate_loss: torch.Tensor, stat_loss: torch.Tensor, gate_tf: torch.Tensor, gate_fr: torch.Tensor) -> Dict[str, float]:
    return {"forecast_loss": forecast_loss.detach().item(),
        "nll_short": forecast_parts["nll_short"].detach().item(),
        "mse_short": forecast_parts["mse_short"].detach().item(),
        "vel": forecast_parts["vel"].detach().item(),
        "gate_loss": gate_loss.detach().item(),
        "stat_loss": stat_loss.detach().item(),
        "mean_gate_tf": gate_tf.detach().mean().item(),
        "mean_gate_fr": gate_fr.detach().mean().item()}


def _mdnstats_debug_dict(loss: torch.Tensor, forecast_loss: torch.Tensor, stat_loss: torch.Tensor, mse_short: torch.Tensor, vel: torch.Tensor, mdn_nll: torch.Tensor, sigma_reg: torch.Tensor, sigma_smooth: torch.Tensor, sigma: torch.Tensor) -> Dict[str, float]:
    return {"loss": loss.detach().item(),
        "forecast_loss": forecast_loss.detach().item(),
        "mse_short": mse_short.detach().item(),
        "vel": vel.detach().item(),
        "mdn_nll": mdn_nll.detach().item(),
        "sigma_reg": sigma_reg.detach().item(),
        "sigma_smooth": sigma_smooth.detach().item(),
        "stat_loss": stat_loss.detach().item(),
        "mean_sigma": sigma.detach().mean().item()}


def combined_selective_loss_xonly(pred_tf: torch.Tensor, gate_tf: torch.Tensor, err_hat_tf: torch.Tensor, pi_tf: torch.Tensor, mu_tf: torch.Tensor, sigma_tf: torch.Tensor, pred_fr: torch.Tensor, gate_fr: torch.Tensor, err_hat_fr: torch.Tensor, pi_fr: torch.Tensor, mu_fr: torch.Tensor, sigma_fr: torch.Tensor, true: torch.Tensor, short_k: int, epoch: int, n_epochs: int, coverage_target: float, lambda_cov: float, eps_x: float = 0.7, use_adaptive_oracle: bool = False, eps_short: float = 0.6, eps_long: float = 1.2, *, loss_cfg=None) -> tuple[torch.Tensor, Dict[str, float]]:
    _require_loss_cfg(loss_cfg)

    H = true.size(1)
    short_k = min(short_k, H)

    forecast_loss, forecast_parts = _forecast_loss(
        pred_tf=pred_tf, pi_tf=pi_tf, mu_tf=mu_tf,
        sigma_tf=sigma_tf, true=true,
        short_k=short_k, loss_cfg=loss_cfg)

    gate_loss = _gate_aux_rollout_loss(
        pred_fr=pred_fr, gate_fr=gate_fr,
        err_hat_fr=err_hat_fr, true=true,
        coverage_target=coverage_target,
        lambda_cov=lambda_cov, eps_x=eps_x,
        use_adaptive_oracle=use_adaptive_oracle,
        eps_short=eps_short, eps_long=eps_long)

    l_sig_r = sigma_regularization_loss(sigma_fr, min_sigma=0.15)

    if H <= short_k + 1:
        stat_loss = loss_cfg.sigma_reg_weight * l_sig_r
    else:
        pred_long = pred_fr[:, short_k:]
        true_long = true[:, short_k:]

        stat_w = _stat_ramp_weight(epoch, loss_cfg)

        l_spec = spectral_loss(pred_long, true_long)
        l_amp = amplitude_loss(pred_long, true_long)
        l_wing = wing_switch_loss(pred_fr[:, :, 0], true[:, :, 0])
        l_dist = soft_cdf_loss_1d(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_bins = x_bin_occupancy_loss(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_meanx = mean_x_loss(pred_fr, true)

        stat_loss = (
            loss_cfg.sigma_reg_weight * l_sig_r
            + stat_w
            * (
                loss_cfg.spec_w * l_spec
                + loss_cfg.amp_w * l_amp
                + loss_cfg.wing_w * l_wing
                + loss_cfg.dist_w * l_dist
                + loss_cfg.bins_w * l_bins
                + loss_cfg.meanx_w * l_meanx))

    loss = forecast_loss + gate_loss + stat_loss

    debug = _selective_debug_dict(
        forecast_loss=forecast_loss, forecast_parts=forecast_parts,
        gate_loss=gate_loss, stat_loss=stat_loss,
        gate_tf=gate_tf, gate_fr=gate_fr)

    return loss, debug


def combined_selective_loss(pred_tf: torch.Tensor, gate_tf: torch.Tensor, err_hat_tf: torch.Tensor, pi_tf: torch.Tensor, mu_tf: torch.Tensor, sigma_tf: torch.Tensor, pred_fr: torch.Tensor, gate_fr: torch.Tensor, err_hat_fr: torch.Tensor, pi_fr: torch.Tensor, mu_fr: torch.Tensor, sigma_fr: torch.Tensor, true: torch.Tensor, short_k: int, epoch: int, n_epochs: int, coverage_target: float, lambda_cov: float, eps_x: float = 0.7, use_adaptive_oracle: bool = False, eps_short: float = 0.6, eps_long: float = 1.2, *, loss_cfg=None) -> tuple[torch.Tensor, Dict[str, float]]:
    _require_loss_cfg(loss_cfg)

    H = true.size(1)
    short_k = min(short_k, H)

    forecast_loss, forecast_parts = _forecast_loss(
        pred_tf=pred_tf,pi_tf=pi_tf,
        mu_tf=mu_tf,sigma_tf=sigma_tf,
        true=true,short_k=short_k,
        loss_cfg=loss_cfg)

    gate_loss = _gate_aux_rollout_loss(
        pred_fr=pred_fr,gate_fr=gate_fr,
        err_hat_fr=err_hat_fr,true=true,
        coverage_target=coverage_target,
        lambda_cov=lambda_cov,
        eps_x=eps_x,use_adaptive_oracle=use_adaptive_oracle,
        eps_short=eps_short,eps_long=eps_long)

    l_sig_r = sigma_regularization_loss(sigma_fr, min_sigma=0.15)

    if H <= short_k + 1:
        stat_loss = loss_cfg.sigma_reg_weight * l_sig_r
    else:
        pred_long = pred_fr[:, short_k:]
        true_long = true[:, short_k:]

        stat_w = _stat_ramp_weight(epoch, loss_cfg)

        l_spec = spectral_loss(pred_long, true_long)
        l_amp = amplitude_loss(pred_long, true_long)
        l_amp_z = amplitude_loss_z(pred_long, true_long)
        l_wing = wing_switch_loss(pred_fr[:, :, 0], true[:, :, 0])
        l_dist = soft_cdf_loss_1d(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_bins = x_bin_occupancy_loss(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_meanx = mean_x_loss(pred_fr, true)
        l_meanz = mean_z_loss_from_long(pred_long, true_long)

        stat_loss = (
            loss_cfg.sigma_reg_weight * l_sig_r
            + stat_w
            * (
                loss_cfg.spec_w * l_spec
                + loss_cfg.amp_w * l_amp
                + loss_cfg.amp_z_w * l_amp_z
                + loss_cfg.wing_w * l_wing
                + loss_cfg.dist_w * l_dist
                + loss_cfg.bins_w * l_bins
                + loss_cfg.meanx_w * l_meanx
                + loss_cfg.meanz_w * l_meanz))

    loss = forecast_loss + gate_loss + stat_loss

    debug = _selective_debug_dict(
        forecast_loss=forecast_loss,forecast_parts=forecast_parts,
        gate_loss=gate_loss,stat_loss=stat_loss,
        gate_tf=gate_tf,gate_fr=gate_fr)

    return loss, debug


def combined_mdnstats_loss(pred: torch.Tensor, true: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, short_k: int, epoch: int, n_epochs: int, *, loss_cfg=None) -> tuple[torch.Tensor, Dict[str, float]]:
    _require_loss_cfg(loss_cfg)

    H = true.size(1)
    D = true.size(-1)
    short_k = min(short_k, H)

    pred_short = pred[:, :short_k]
    true_short = true[:, :short_k]

    l_mse_short = F.mse_loss(pred_short, true_short)
    l_vel = velocity_loss(pred_short, true_short)
    l_mdn = mdn_nll_sequence(pi=pi, mu=mu, sigma=sigma, true=true, max_steps=H)
    l_sigma_reg = sigma_regularization_loss(sigma, min_sigma=0.15)
    l_sigma_smooth = sigma_smoothness_loss(sigma)

    forecast_loss = 1.00 * l_mse_short + 0.50 * l_vel + loss_cfg.mdn_nll_weight * l_mdn
    base_sigma_loss = loss_cfg.sigma_reg_weight * l_sigma_reg + loss_cfg.sigma_smooth_weight * l_sigma_smooth

    if H <= short_k + 1:
        stat_loss = base_sigma_loss
    else:
        pred_long = pred[:, short_k:]
        true_long = true[:, short_k:]

        stat_w = _stat_ramp_weight(epoch, loss_cfg)

        l_spec = spectral_loss(pred_long, true_long)
        l_amp = amplitude_loss(pred_long, true_long)
        l_wing = wing_switch_loss(pred[:, :, 0], true[:, :, 0])
        l_dist = soft_cdf_loss_1d(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_bins = x_bin_occupancy_loss(pred_long[:, :, 0], true_long[:, :, 0], mode=loss_cfg.cdf_mode)
        l_meanx = mean_x_loss(pred, true)

        stat_terms = (
            loss_cfg.spec_w * l_spec
            + loss_cfg.amp_w * l_amp
            + loss_cfg.wing_w * l_wing
            + loss_cfg.dist_w * l_dist
            + loss_cfg.bins_w * l_bins
            + loss_cfg.meanx_w * l_meanx
        )

        if D >= 3:
            l_amp_z = amplitude_loss_z(pred_long, true_long)
            l_meanz = mean_z_loss_from_long(pred_long, true_long)
            stat_terms = stat_terms + loss_cfg.amp_z_w * l_amp_z + loss_cfg.meanz_w * l_meanz

        stat_loss = base_sigma_loss + stat_w * stat_terms

    loss = forecast_loss + stat_loss

    debug = _mdnstats_debug_dict(
        loss=loss,
        forecast_loss=forecast_loss,
        stat_loss=stat_loss,
        mse_short=l_mse_short,
        vel=l_vel,
        mdn_nll=l_mdn,
        sigma_reg=l_sigma_reg,
        sigma_smooth=l_sigma_smooth,
        sigma=sigma)

    return loss, debug
