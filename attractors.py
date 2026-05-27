from typing import Any
import numpy as np


"""
Файл для генерауии данных из аттракторов для обучения моделей.
"""

def lorenz_rhs_np(xyz: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)


def rossler_rhs_np(xyz: np.ndarray, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> np.ndarray:
    x, y, z = xyz
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_step_np(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _get_attractor_cfg(cfg: Any) -> Any:
    return cfg.attractor if hasattr(cfg, "attractor") else cfg


def simulate_attractor(cfg: Any, *, params: dict[str, float] | None = None, dt: float | None = None, t_total: float | None = None, t_transient: float | None = None, y0: list[float] | np.ndarray | None = None) -> np.ndarray:
    attractor_cfg = _get_attractor_cfg(cfg)
    attractor_name = attractor_cfg.name

    params = attractor_cfg.params if params is None else params
    dt = attractor_cfg.dt if dt is None else dt
    t_total = attractor_cfg.t_total if t_total is None else t_total
    t_transient = attractor_cfg.t_transient if t_transient is None else t_transient

    y0 = attractor_cfg.y0 if y0 is None else y0
    n_steps = int(t_total / dt)
    xs = np.zeros((n_steps, 3), dtype=np.float64)
    state = np.asarray(y0, dtype=np.float64).copy()

    if attractor_name == "lorenz":
        def rhs(s: np.ndarray) -> np.ndarray:
            return lorenz_rhs_np(s, **params)
    elif attractor_name == "rossler":
        def rhs(s: np.ndarray) -> np.ndarray:
            return rossler_rhs_np(s, **params)
    else:
        raise ValueError(f"Unknown attractor_name={attractor_name!r}")

    for i in range(n_steps):
        xs[i] = state
        state = rk4_step_np(rhs, state, dt)

    skip = int(t_transient / dt)
    return xs[skip:]
