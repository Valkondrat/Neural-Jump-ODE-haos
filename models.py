import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mdn_uncertainty_features_torch, rk4_step_torch

"""
Файл со всеми моделями. 
HybridNJODEEncoderGRUModel* - модели с детектором непрогнозируемых точек внутри. 
HybridNJODEEncoderGRUMDNStatsModel* - модели с детектором непрогнозируемых точек после обучения.  
"""

def _model_cfg(cfg):
    return cfg.model if cfg is not None and hasattr(cfg, "model") else cfg


def _data_cfg(cfg):
    return cfg.data if cfg is not None and hasattr(cfg, "data") else None


def _attractor_cfg(cfg):
    return cfg.attractor if cfg is not None and hasattr(cfg, "attractor") else None


def _cfg_value(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


class ODEFunc(nn.Module):
    def __init__(self, z_dim, hidden_mult=3, res_scale=0.1):
        super().__init__()
        hdim = hidden_mult * z_dim
        self.net = nn.Sequential(nn.Linear(z_dim, hdim), nn.SiLU(), nn.Linear(hdim, hdim), nn.SiLU(), nn.Linear(hdim, z_dim))
        self.res_scale = res_scale

    def forward(self, z):
        return self.net(z) + self.res_scale * z


class GatedJump(nn.Module):
    def __init__(self, z_dim, hidden_mult=2):
        super().__init__()
        hdim = hidden_mult * z_dim
        self.gate_net = nn.Sequential(nn.Linear(2 * z_dim, hdim), nn.SiLU(), nn.Linear(hdim, z_dim))
        self.jump_net = nn.Sequential(nn.Linear(2 * z_dim, hdim), nn.SiLU(), nn.Linear(hdim, z_dim))

    def forward(self, z, z_obs):
        x = torch.cat([z, z_obs], dim=-1)
        gate = torch.sigmoid(self.gate_net(x))
        cand = self.jump_net(x)
        return gate * z + (1.0 - gate) * cand


class SelectiveHead(nn.Module):
    def __init__(self, hidden_dim: int, phys_dim: int = 4, n_components: int = 3, out_dim: int = 1, max_horizon=200, detach_gate_backbone=True):
        super().__init__()
        self.K = n_components
        self.D = out_dim
        self.max_horizon = max_horizon
        self.detach_gate_backbone = detach_gate_backbone
        self.det_out = nn.Sequential(nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, out_dim))
        self.step_embed = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16))
        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * out_dim)
        self.log_sigma = nn.Linear(hidden_dim, n_components * out_dim)
        gate_in = hidden_dim + phys_dim + 16 + 5
        self.gate = nn.Sequential(nn.Linear(gate_in, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.err_pred = nn.Sequential(nn.Linear(gate_in, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, h, phys, step=0):
        B = h.size(0)
        pred_det = self.det_out(h)
        pi = F.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(B, self.K, self.D)
        sigma = torch.exp(self.log_sigma(h).view(B, self.K, self.D)).clamp(1e-3, 2.0)
        unc = mdn_uncertainty_features_torch(pi, mu, sigma)
        t_norm = torch.full((B, 1), step/self.max_horizon, device=h.device)
        t_emb = self.step_embed(t_norm)

        if self.detach_gate_backbone:
            h_gate = h.detach()
            phys_gate = phys.detach()
            unc_gate = unc.detach()
        else:
            h_gate = h
            phys_gate = phys
            unc_gate = unc

        gate_input = torch.cat([h_gate, phys_gate, t_emb, unc_gate], dim=-1)
        g = self.gate(gate_input)
        err_hat = self.err_pred(gate_input)
        return pred_det, pi, mu, sigma, g, err_hat

    def mdn_mean(self, pi, mu):
        return (pi.unsqueeze(-1) * mu).sum(dim=1)

    def mdn_sample(self, pi, mu, sigma):
        B = pi.size(0)
        idx = torch.multinomial(pi, 1).squeeze(1)
        b = torch.arange(B, device=pi.device)
        return mu[b, idx] + sigma[b, idx] * torch.randn_like(mu[b, idx])


class PureSelectiveGRUDecoder1D(nn.Module):
    def __init__(self, hidden_dim: int, n_mdn_components: int = 3, phys_dim: int = 4, delay_dim: int = 12, delay_tau: int = 2, max_horizon: int = 200, wing_scale: float = 20.0):
        super().__init__()
        self.delay_dim = delay_dim
        self.delay_tau = delay_tau
        self.wing_scale = wing_scale
        self.input_norm = nn.LayerNorm(delay_dim + 2)
        self.rnn = nn.GRUCell(input_size=delay_dim + 2, hidden_size=hidden_dim)
        self.head = SelectiveHead(hidden_dim=hidden_dim, phys_dim=phys_dim, n_components=n_mdn_components, out_dim=1, max_horizon=max_horizon)

    def _init_wing_state(self, x_seq):
        x = x_seq[:, :, 0]
        last_sign = torch.sign(x[:, -1:])
        signs = torch.sign(x)
        wing_time = torch.zeros(x.size(0), 1, device=x_seq.device)

        for t in range(x.size(1) - 1, -1, -1):
            same = (signs[:, t:t + 1] == last_sign).float()
            wing_time = wing_time * same + same

        wing_time = wing_time / self.wing_scale
        return last_sign, wing_time

    def _update_delay_vector(self, delay_vec, x_next):
        return torch.cat([x_next, delay_vec[:, :-1]], dim=-1)

    def _build_delay_from_buffer(self, scalar_buffer):
        vals = []

        for k in range(self.delay_dim):
            lag = k * self.delay_tau
            vals.append(scalar_buffer[-1 - lag])

        return torch.cat(vals, dim=-1)

    def decode(self, h0, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False):
        B = x_seq.size(0)
        h = h0
        delay_prev = x_seq[:, -1, :]
        x_prev = delay_prev[:, 0:1]
        x_prev_prev = x_seq[:, -2, 0:1]
        last_sign, wing_time = self._init_wing_state(x_seq)
        steps_on_wing = wing_time.clone()
        scalar_buffer = [x_seq[:, t, 0:1].detach() for t in range(x_seq.size(1))]
        required = (self.delay_dim - 1) * self.delay_tau + 1

        if len(scalar_buffer) < required:
            raise ValueError(
                f"WINDOW={len(scalar_buffer)} too small for "
                f"delay_dim={self.delay_dim}, delay_tau={self.delay_tau}. "
                f"Need at least {required}.")

        preds, gates, err_hats = [], [], []
        all_pi, all_mu, all_sigma = [], [], []

        for step in range(horizon):
            delay_prev = self._build_delay_from_buffer(scalar_buffer)
            noisy_delay = delay_prev

            if self.training and noise_std > 0:
                noisy_delay = noisy_delay + noise_std * torch.randn_like(noisy_delay)

            inp = self.input_norm(torch.cat([noisy_delay, wing_time, last_sign], dim=-1))
            h = self.rnn(inp, h)
            prox = torch.exp(-x_prev.abs() / 0.5)
            dx = (x_prev - x_prev_prev).abs()

            if len(scalar_buffer) >= 3:
                v1 = scalar_buffer[-1] - scalar_buffer[-2]
                v0 = scalar_buffer[-2] - scalar_buffer[-3]
                ddx = (v1 - v0).abs()
            else:
                ddx = torch.zeros_like(dx)

            phys = torch.cat([steps_on_wing, prox, dx, ddx], dim=-1)
            x_next, pi, mu, sigma, g, err_hat = self.head(h, phys, step=step)
            preds.append(x_next.unsqueeze(1))
            gates.append(g.unsqueeze(1))
            err_hats.append(err_hat.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            if x_true is not None and tf_ratio > 0.0:
                tf_mask = (torch.rand(B, 1, device=x_seq.device) < tf_ratio).float()
                x_feed = tf_mask * x_true[:, step, :] + (1.0 - tf_mask) * x_next.detach()
            else:
                x_feed = x_next.detach()

            x_prev_prev = x_prev
            x_prev = x_feed
            scalar_buffer.append(x_feed)
            new_sign = torch.sign(x_feed)
            switched = (new_sign != last_sign).float()
            wing_time = (wing_time + 1.0 / self.wing_scale) * (1.0 - switched)
            steps_on_wing = (steps_on_wing + 1.0 / self.wing_scale) * (1.0 - switched)
            last_sign = new_sign

        return (torch.cat(preds, dim=1),
            torch.cat(gates, dim=1),
            torch.cat(err_hats, dim=1),
            torch.cat(all_pi, dim=1),
            torch.cat(all_mu, dim=1),
            torch.cat(all_sigma, dim=1))


class HybridNJODEEncoderGRUModel1D(nn.Module):
    def __init__(self, input_dim: int | None = None, cfg=None, z_dim: int | None = None, hidden_dim: int | None = None, seg_ode: int | None = None, dt: float | None = None, n_mdn_components: int | None = None, phys_dim: int = 4, delay_tau: int | None = None, use_layernorm_z2h: bool | None = None):
        super().__init__()
        model_cfg = _model_cfg(cfg)
        data_cfg = _data_cfg(cfg)
        attractor_cfg = _attractor_cfg(cfg)

        if input_dim is None:
            input_dim = _cfg_value(data_cfg, "delay_dim", 12)
        if delay_tau is None:
            delay_tau = _cfg_value(data_cfg, "delay_tau", 2)

        z_dim = _cfg_value(model_cfg, "z_dim", 128) if z_dim is None else z_dim
        hidden_dim = _cfg_value(model_cfg, "hidden_dim", 256) if hidden_dim is None else hidden_dim
        seg_ode = _cfg_value(model_cfg, "seg_ode", 4) if seg_ode is None else seg_ode
        dt = _cfg_value(attractor_cfg, "dt", 0.05) if dt is None else dt
        n_mdn_components = _cfg_value(model_cfg, "n_mdn_components", 3) if n_mdn_components is None else n_mdn_components
        use_layernorm_z2h = _cfg_value(model_cfg, "use_layernorm_z2h", True) if use_layernorm_z2h is None else use_layernorm_z2h
        ode_hidden_mult = _cfg_value(model_cfg, "ode_hidden_mult", 3)
        ode_res_scale = _cfg_value(model_cfg, "ode_res_scale", 0.10)
        jump_hidden_mult = _cfg_value(model_cfg, "jump_hidden_mult", 2)
        max_horizon = _cfg_value(data_cfg, "full_horizon", 200)
        wing_scale = float(_cfg_value(data_cfg, "short_k", 20))
        self.input_dim = input_dim
        self.seg_ode = seg_ode
        self.dt = dt
        self.obs2z = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, z_dim))
        self.ode_func = ODEFunc(z_dim=z_dim, hidden_mult=ode_hidden_mult, res_scale=ode_res_scale)
        self.jump = GatedJump(z_dim=z_dim, hidden_mult=jump_hidden_mult)

        if use_layernorm_z2h:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        else:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.Tanh())

        self.decoder = PureSelectiveGRUDecoder1D(
            hidden_dim=hidden_dim, n_mdn_components=n_mdn_components,
            phys_dim=phys_dim,delay_dim=input_dim,
            delay_tau=delay_tau,max_horizon=max_horizon,
            wing_scale=wing_scale)

    def latent_rollout_step(self, z):
        dt_sub = self.dt/self.seg_ode

        for _ in range(self.seg_ode):
            z = rk4_step_torch(self.ode_func, z, dt_sub)

        return z

    def encode(self, x_seq):
        B, W, C = x_seq.shape
        z = self.obs2z(x_seq[:, 0, :])

        for t in range(1, W):
            z = self.latent_rollout_step(z)
            z_obs = self.obs2z(x_seq[:, t, :])
            z = self.jump(z, z_obs)

        return z

    def forward(self, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False, max_sigma_scale=1.0, return_div=False, div_n_samples=1):
        z = self.encode(x_seq)
        h0 = self.z2h(z)
        pred, gate, err_hat, pi, mu, sigma = self.decoder.decode(h0=h0, x_seq=x_seq, horizon=horizon, x_true=x_true, tf_ratio=tf_ratio, noise_std=noise_std, sample_mode=sample_mode)

        if return_div:
            div_loss = torch.tensor(0.0, device=x_seq.device)
            return pred, gate, err_hat, pi, mu, sigma, div_loss

        return pred, gate, err_hat, pi, mu, sigma


class PureSelectiveGRUDecoder(nn.Module):
    def __init__(self, hidden_dim: int, n_mdn_components: int = 3, phys_dim: int = 4, max_horizon: int = 200, wing_scale: float = 20.0):
        super().__init__()
        self.wing_scale = wing_scale
        self.input_norm = nn.LayerNorm(3 + 2)
        self.rnn = nn.GRUCell(input_size=3 + 2, hidden_size=hidden_dim)
        self.head = SelectiveHead(hidden_dim=hidden_dim, phys_dim=phys_dim, n_components=n_mdn_components, out_dim=3, max_horizon=max_horizon)

    def _init_wing_state(self, x_seq):
        B = x_seq.size(0)
        x = x_seq[:, :, 0]
        last_sign = torch.sign(x[:, -1:])
        signs = torch.sign(x)
        wing_time = torch.zeros(B, 1, device=x_seq.device)

        for t in range(x.size(1) - 1, -1, -1):
            same = (signs[:, t:t + 1] == last_sign).float()
            wing_time = wing_time * same + same

        wing_time = wing_time/self.wing_scale
        return last_sign, wing_time

    def decode(self, h0, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False):
        B = x_seq.size(0)
        xyz_prev = x_seq[:, -1, :]
        xyz_prev_prev = x_seq[:, -2, :]
        last_sign, wing_time = self._init_wing_state(x_seq)
        steps_on_wing = wing_time.clone()
        h = h0
        preds, gates, err_hats = [], [], []
        all_pi, all_mu, all_sigma = [], [], []

        for step in range(horizon):
            noisy = xyz_prev

            if self.training and noise_std > 0:
                noisy = noisy + noise_std * torch.randn_like(noisy)

            inp = self.input_norm(torch.cat([noisy, wing_time, last_sign], dim=-1))
            h = self.rnn(inp, h)
            prox = torch.exp(-xyz_prev[:, 0:1].abs() / 0.5)
            dx = (xyz_prev[:, 0:1] - xyz_prev_prev[:, 0:1]).abs()
            phys = torch.cat([steps_on_wing, prox, xyz_prev[:, 2:3], dx], dim=-1)
            xyz_next, pi, mu, sigma, g, err_hat = self.head(h, phys, step=step)
            preds.append(xyz_next.unsqueeze(1))
            gates.append(g.unsqueeze(1))
            err_hats.append(err_hat.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            if x_true is not None and tf_ratio > 0.0:
                tf_mask = (torch.rand(B, 1, device=x_seq.device) < tf_ratio).float()
                xyz_feed = tf_mask * x_true[:, step, :] + (1.0 - tf_mask) * xyz_next.detach()
            else:
                xyz_feed = xyz_next.detach()

            xyz_prev_prev = xyz_prev
            xyz_prev = xyz_feed
            new_sign = torch.sign(xyz_prev[:, 0:1])
            switched = (new_sign != last_sign).float()
            wing_time = (wing_time + 1.0 / self.wing_scale) * (1.0 - switched)
            steps_on_wing = (steps_on_wing + 1.0 / self.wing_scale) * (1.0 - switched)
            last_sign = new_sign

        return (torch.cat(preds, dim=1),
            torch.cat(gates, dim=1),
            torch.cat(err_hats, dim=1),
            torch.cat(all_pi, dim=1),
            torch.cat(all_mu, dim=1),
            torch.cat(all_sigma, dim=1))


class HybridNJODEEncoderGRUModel(nn.Module):
    def __init__(self, input_dim: int | None = None, cfg=None, z_dim: int | None = None, hidden_dim: int | None = None, seg_ode: int | None = None, dt: float | None = None, n_mdn_components: int | None = None, phys_dim: int = 4, use_layernorm_z2h: bool | None = None):
        super().__init__()
        model_cfg = _model_cfg(cfg)
        data_cfg = _data_cfg(cfg)
        attractor_cfg = _attractor_cfg(cfg)

        if input_dim is None:
            input_dim = 3

        z_dim = _cfg_value(model_cfg, "z_dim", 64) if z_dim is None else z_dim
        hidden_dim = _cfg_value(model_cfg, "hidden_dim", 128) if hidden_dim is None else hidden_dim
        seg_ode = _cfg_value(model_cfg, "seg_ode", 4) if seg_ode is None else seg_ode
        dt = _cfg_value(attractor_cfg, "dt", 0.05) if dt is None else dt
        n_mdn_components = _cfg_value(model_cfg, "n_mdn_components", 3) if n_mdn_components is None else n_mdn_components
        use_layernorm_z2h = _cfg_value(model_cfg, "use_layernorm_z2h", True) if use_layernorm_z2h is None else use_layernorm_z2h
        ode_hidden_mult = _cfg_value(model_cfg, "ode_hidden_mult", 3)
        ode_res_scale = _cfg_value(model_cfg, "ode_res_scale", 0.10)
        jump_hidden_mult = _cfg_value(model_cfg, "jump_hidden_mult", 2)
        max_horizon = _cfg_value(data_cfg, "full_horizon", 200)
        wing_scale = float(_cfg_value(data_cfg, "short_k", 20))
        self.seg_ode = seg_ode
        self.dt = dt
        self.obs2z = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, z_dim))
        self.ode_func = ODEFunc(z_dim=z_dim, hidden_mult=ode_hidden_mult, res_scale=ode_res_scale)
        self.jump = GatedJump(z_dim=z_dim, hidden_mult=jump_hidden_mult)

        if use_layernorm_z2h:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        else:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.Tanh())

        self.decoder = PureSelectiveGRUDecoder(hidden_dim=hidden_dim, n_mdn_components=n_mdn_components, phys_dim=phys_dim, max_horizon=max_horizon, wing_scale=wing_scale)

    def latent_rollout_step(self, z):
        dt_sub = self.dt/self.seg_ode

        for _ in range(self.seg_ode):
            z = rk4_step_torch(self.ode_func, z, dt_sub)

        return z

    def encode(self, x_seq):
        B, W, C = x_seq.shape
        z = self.obs2z(x_seq[:, 0, :])

        for t in range(1, W):
            z = self.latent_rollout_step(z)
            z_obs = self.obs2z(x_seq[:, t, :])
            z = self.jump(z, z_obs)

        return z

    def forward(self, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False, max_sigma_scale=1.0, return_div=False, div_n_samples=1):
        z = self.encode(x_seq)
        h0 = self.z2h(z)
        pred, gate, err_hat, pi, mu, sigma = self.decoder.decode(h0=h0, x_seq=x_seq, horizon=horizon, x_true=x_true, tf_ratio=tf_ratio, noise_std=noise_std, sample_mode=sample_mode)

        if return_div:
            div_loss = torch.tensor(0.0, device=x_seq.device)
            return pred, gate, err_hat, pi, mu, sigma, div_loss

        return pred, gate, err_hat, pi, mu, sigma


class ForecastMDNStatsHead(nn.Module):
    def __init__(self, hidden_dim: int, n_components: int = 3, out_dim: int = 1):
        super().__init__()
        self.K = n_components
        self.D = out_dim
        self.det_out = nn.Sequential(nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, out_dim))
        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components * out_dim)
        self.log_sigma = nn.Linear(hidden_dim, n_components * out_dim)

    def forward(self, h):
        B = h.size(0)
        pred_det = self.det_out(h)
        pi = F.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(B, self.K, self.D)
        sigma = torch.exp(self.log_sigma(h).view(B, self.K, self.D)).clamp(1e-3, 2.0)
        return pred_det, pi, mu, sigma

    @staticmethod
    def mdn_mean(pi, mu):
        return (pi.unsqueeze(-1) * mu).sum(dim=1)

    @staticmethod
    def mdn_sample(pi, mu, sigma):
        B = pi.size(0)
        idx = torch.multinomial(pi, 1).squeeze(1)
        b = torch.arange(B, device=pi.device)
        return mu[b, idx] + sigma[b, idx] * torch.randn_like(mu[b, idx])


class GRUDecoderNoLatentODE1D(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, delay_dim: int, delay_tau: int = 1, n_mdn_components: int = 3, wing_scale: float = 20.0):
        super().__init__()
        self.delay_dim = delay_dim
        self.delay_tau = delay_tau
        self.wing_scale = wing_scale
        self.input_norm = nn.LayerNorm(delay_dim + z_dim + 2)
        self.rnn = nn.GRUCell(input_size=delay_dim + z_dim + 2, hidden_size=hidden_dim)
        self.head = ForecastMDNStatsHead(hidden_dim=hidden_dim, n_components=n_mdn_components, out_dim=1)

    def _init_wing_state(self, x_seq):
        x = x_seq[:, :, 0]
        last_sign = torch.sign(x[:, -1:])
        signs = torch.sign(x)
        wing_time = torch.zeros(x.size(0), 1, device=x_seq.device)

        for t in range(x.size(1) - 1, -1, -1):
            same = (signs[:, t:t + 1] == last_sign).float()
            wing_time = wing_time * same + same

        wing_time = wing_time/self.wing_scale
        return last_sign, wing_time

    def _build_delay_from_buffer(self, scalar_buffer):
        vals = []

        for k in range(self.delay_dim):
            lag = k * self.delay_tau
            vals.append(scalar_buffer[-1 - lag])

        return torch.cat(vals, dim=-1)

    def decode(self, z0, h0, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False):
        B = x_seq.size(0)
        z_context = z0
        h = h0
        last_sign, wing_time = self._init_wing_state(x_seq)
        scalar_buffer = [x_seq[:, t, 0:1].detach() for t in range(x_seq.size(1))]
        required = (self.delay_dim - 1) * self.delay_tau + 1

        preds = []
        all_pi, all_mu, all_sigma = [], [], []

        for step in range(horizon):
            delay_prev = self._build_delay_from_buffer(scalar_buffer)
            noisy_delay = delay_prev

            if self.training and noise_std > 0.0:
                noisy_delay = noisy_delay + noise_std * torch.randn_like(noisy_delay)

            inp = self.input_norm(torch.cat([noisy_delay, z_context, wing_time, last_sign], dim=-1))
            h = self.rnn(inp, h)
            x_next, pi, mu, sigma = self.head(h)
            preds.append(x_next.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            if x_true is not None and tf_ratio > 0.0:
                tf_mask = (torch.rand(B, 1, device=x_seq.device) < tf_ratio).float()
                x_feed = tf_mask * x_true[:, step, :] + (1.0 - tf_mask) * x_next.detach()
            else:
                x_feed = x_next.detach()

            scalar_buffer.append(x_feed)
            new_sign = torch.sign(x_feed)
            switched = (new_sign != last_sign).float()
            wing_time = (wing_time + 1.0 / self.wing_scale) * (1.0 - switched)
            last_sign = new_sign

        return (torch.cat(preds, dim=1),
            torch.cat(all_pi, dim=1),
            torch.cat(all_mu, dim=1),
            torch.cat(all_sigma, dim=1))


class GRUDecoderNoLatentODE(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, n_mdn_components: int = 3, wing_scale: float = 20.0):
        super().__init__()
        self.wing_scale = wing_scale
        self.input_norm = nn.LayerNorm(3 + z_dim + 2)
        self.rnn = nn.GRUCell(input_size=3 + z_dim + 2, hidden_size=hidden_dim)
        self.head = ForecastMDNStatsHead(hidden_dim=hidden_dim, n_components=n_mdn_components, out_dim=3)

    def _init_wing_state(self, x_seq):
        B = x_seq.size(0)
        x = x_seq[:, :, 0]
        last_sign = torch.sign(x[:, -1:])
        signs = torch.sign(x)
        wing_time = torch.zeros(B, 1, device=x_seq.device)

        for t in range(x.size(1) - 1, -1, -1):
            same = (signs[:, t:t + 1] == last_sign).float()
            wing_time = wing_time * same + same

        wing_time = wing_time/self.wing_scale
        return last_sign, wing_time

    def decode(self, z0, h0, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False):
        B = x_seq.size(0)
        z_context = z0
        h = h0
        xyz_prev = x_seq[:, -1, :]
        last_sign, wing_time = self._init_wing_state(x_seq)
        preds = []
        all_pi, all_mu, all_sigma = [], [], []

        for step in range(horizon):
            noisy_prev = xyz_prev

            if self.training and noise_std > 0.0:
                noisy_prev = noisy_prev + noise_std * torch.randn_like(noisy_prev)

            inp = self.input_norm(torch.cat([noisy_prev, z_context, wing_time, last_sign], dim=-1))
            h = self.rnn(inp, h)
            xyz_next, pi, mu, sigma = self.head(h)
            preds.append(xyz_next.unsqueeze(1))
            all_pi.append(pi.unsqueeze(1))
            all_mu.append(mu.unsqueeze(1))
            all_sigma.append(sigma.unsqueeze(1))

            if x_true is not None and tf_ratio > 0.0:
                tf_mask = (torch.rand(B, 1, device=x_seq.device) < tf_ratio).float()
                xyz_feed = tf_mask * x_true[:, step, :] + (1.0 - tf_mask) * xyz_next.detach()
            else:
                xyz_feed = xyz_next.detach()

            xyz_prev = xyz_feed
            new_sign = torch.sign(xyz_prev[:, 0:1])
            switched = (new_sign != last_sign).float()
            wing_time = (wing_time + 1.0 / self.wing_scale) * (1.0 - switched)
            last_sign = new_sign

        return (torch.cat(preds, dim=1),
            torch.cat(all_pi, dim=1),
            torch.cat(all_mu, dim=1),
            torch.cat(all_sigma, dim=1))


class HybridNJODEEncoderGRUMDNStatsModel1D(nn.Module):
    def __init__(self, input_dim: int | None = None, cfg=None, z_dim: int | None = None, hidden_dim: int | None = None, seg_ode: int | None = None, dt: float | None = None, n_mdn_components: int | None = None, delay_tau: int | None = None, use_layernorm_z2h: bool | None = None):
        super().__init__()
        model_cfg = _model_cfg(cfg)
        data_cfg = _data_cfg(cfg)
        attractor_cfg = _attractor_cfg(cfg)

        if input_dim is None:
            input_dim = _cfg_value(data_cfg, "delay_dim", 24)
        if delay_tau is None:
            delay_tau = _cfg_value(data_cfg, "delay_tau", 1)

        z_dim = _cfg_value(model_cfg, "z_dim", 128) if z_dim is None else z_dim
        hidden_dim = _cfg_value(model_cfg, "hidden_dim", 256) if hidden_dim is None else hidden_dim
        seg_ode = _cfg_value(model_cfg, "seg_ode", 4) if seg_ode is None else seg_ode
        dt = _cfg_value(attractor_cfg, "dt", 0.05) if dt is None else dt
        n_mdn_components = _cfg_value(model_cfg, "n_mdn_components", 3) if n_mdn_components is None else n_mdn_components
        use_layernorm_z2h = _cfg_value(model_cfg, "use_layernorm_z2h", True) if use_layernorm_z2h is None else use_layernorm_z2h
        ode_hidden_mult = _cfg_value(model_cfg, "ode_hidden_mult", 3)
        ode_res_scale = _cfg_value(model_cfg, "ode_res_scale", 0.10)
        jump_hidden_mult = _cfg_value(model_cfg, "jump_hidden_mult", 2)
        wing_scale = float(_cfg_value(data_cfg, "short_k", 20))
        self.input_dim = input_dim
        self.seg_ode = seg_ode
        self.dt = dt
        self.obs2z = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, z_dim))
        self.ode_func = ODEFunc(z_dim=z_dim, hidden_mult=ode_hidden_mult, res_scale=ode_res_scale)
        self.jump = GatedJump(z_dim=z_dim, hidden_mult=jump_hidden_mult)

        if use_layernorm_z2h:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        else:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.Tanh())

        self.decoder = GRUDecoderNoLatentODE1D(z_dim=z_dim, hidden_dim=hidden_dim, delay_dim=input_dim, delay_tau=delay_tau, n_mdn_components=n_mdn_components, wing_scale=wing_scale)

    def latent_rollout_step(self, z):
        dt_sub = self.dt/self.seg_ode

        for _ in range(self.seg_ode):
            z = rk4_step_torch(self.ode_func, z, dt_sub)

        return z

    def encode(self, x_seq):
        B, W, C = x_seq.shape
        z = self.obs2z(x_seq[:, 0, :])

        for t in range(1, W):
            z = self.latent_rollout_step(z)
            z_obs = self.obs2z(x_seq[:, t, :])
            z = self.jump(z, z_obs)

        h = self.z2h(z)
        return z, h

    def forward(self, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False, max_sigma_scale=1.0, return_div=False, div_n_samples=1):
        z, h = self.encode(x_seq)
        pred, pi, mu, sigma = self.decoder.decode(z0=z, h0=h, x_seq=x_seq, horizon=horizon, x_true=x_true, tf_ratio=tf_ratio, noise_std=noise_std, sample_mode=sample_mode)

        if return_div:
            div_loss = torch.tensor(0.0, device=x_seq.device)
            return pred, pi, mu, sigma, div_loss

        return pred, pi, mu, sigma


class HybridNJODEEncoderGRUMDNStatsModel(nn.Module):
    def __init__(self, input_dim: int | None = None, cfg=None, z_dim: int | None = None, hidden_dim: int | None = None, seg_ode: int | None = None, dt: float | None = None, n_mdn_components: int | None = None, use_layernorm_z2h: bool | None = None):
        super().__init__()
        model_cfg = _model_cfg(cfg)
        data_cfg = _data_cfg(cfg)
        attractor_cfg = _attractor_cfg(cfg)

        if input_dim is None:
            input_dim = 3

        z_dim = _cfg_value(model_cfg, "z_dim", 128) if z_dim is None else z_dim
        hidden_dim = _cfg_value(model_cfg, "hidden_dim", 256) if hidden_dim is None else hidden_dim
        seg_ode = _cfg_value(model_cfg, "seg_ode", 4) if seg_ode is None else seg_ode
        dt = _cfg_value(attractor_cfg, "dt", 0.05) if dt is None else dt
        n_mdn_components = _cfg_value(model_cfg, "n_mdn_components", 3) if n_mdn_components is None else n_mdn_components
        use_layernorm_z2h = _cfg_value(model_cfg, "use_layernorm_z2h", True) if use_layernorm_z2h is None else use_layernorm_z2h
        ode_hidden_mult = _cfg_value(model_cfg, "ode_hidden_mult", 3)
        ode_res_scale = _cfg_value(model_cfg, "ode_res_scale", 0.10)
        jump_hidden_mult = _cfg_value(model_cfg, "jump_hidden_mult", 2)
        wing_scale = float(_cfg_value(data_cfg, "short_k", 20))
        self.seg_ode = seg_ode
        self.dt = dt
        self.obs2z = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, z_dim))
        self.ode_func = ODEFunc(z_dim=z_dim, hidden_mult=ode_hidden_mult, res_scale=ode_res_scale)
        self.jump = GatedJump(z_dim=z_dim, hidden_mult=jump_hidden_mult)

        if use_layernorm_z2h:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        else:
            self.z2h = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.Tanh())

        self.decoder = GRUDecoderNoLatentODE(z_dim=z_dim, hidden_dim=hidden_dim, n_mdn_components=n_mdn_components, wing_scale=wing_scale)

    def latent_rollout_step(self, z):
        dt_sub = self.dt/self.seg_ode

        for _ in range(self.seg_ode):
            z = rk4_step_torch(self.ode_func, z, dt_sub)

        return z

    def encode(self, x_seq):
        B, W, C = x_seq.shape
        z = self.obs2z(x_seq[:, 0, :])

        for t in range(1, W):
            z = self.latent_rollout_step(z)
            z_obs = self.obs2z(x_seq[:, t, :])
            z = self.jump(z, z_obs)

        h = self.z2h(z)
        return z, h

    def forward(self, x_seq, horizon, x_true=None, tf_ratio=0.0, noise_std=0.0, sample_mode=False, max_sigma_scale=1.0, return_div=False, div_n_samples=1):
        z, h = self.encode(x_seq)
        pred, pi, mu, sigma = self.decoder.decode(z0=z, h0=h, x_seq=x_seq, horizon=horizon, x_true=x_true, tf_ratio=tf_ratio, noise_std=noise_std, sample_mode=sample_mode)

        if return_div:
            div_loss = torch.tensor(0.0, device=x_seq.device)
            return pred, pi, mu, sigma, div_loss

        return pred, pi, mu, sigma


def build_selective_model_from_cfg(cfg):
    if cfg.data.task == "x_delay":
        return HybridNJODEEncoderGRUModel1D(cfg=cfg)

    if cfg.data.task == "full_phase":
        return HybridNJODEEncoderGRUModel(cfg=cfg)


def build_mdn_stats_model_from_cfg(cfg):
    if cfg.data.task == "x_delay":
        return HybridNJODEEncoderGRUMDNStatsModel1D(cfg=cfg)

    if cfg.data.task == "full_phase":
        return HybridNJODEEncoderGRUMDNStatsModel(cfg=cfg)


def build_model_from_cfg(cfg):
    detector_mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    if detector_mode == "selective":
        return build_selective_model_from_cfg(cfg)

    if detector_mode == "posthoc":
        return build_mdn_stats_model_from_cfg(cfg)
