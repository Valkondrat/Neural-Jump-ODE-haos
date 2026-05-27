import numpy as np
import matplotlib.pyplot as plt

"""
В файле приводятся функции, необходимые для визуализации аттактора в блоке Данные в ВКР
"""

def rk4_step_np(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def lorenz_rhs_np(xyz, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)


def simulate_lorenz(sigma=10.0,rho=28.0,beta=8.0 / 3.0,dt=0.05,t_total=1000.0,t_transient=100.0,y0=None):
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    n_steps = int(t_total / dt)
    xs = np.zeros((n_steps, 3), dtype=np.float64)
    state = y0.copy()

    def rhs(s):
        return lorenz_rhs_np(s, sigma=sigma, rho=rho, beta=beta)

    for i in range(n_steps):
        xs[i] = state
        state = rk4_step_np(rhs, state, dt)
    skip = int(t_transient / dt)
    return xs[skip:]


def rossler_rhs_np(xyz, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz], dtype=np.float64)


def simulate_rossler(a=0.2,b=0.2,c=5.7,dt=0.01,t_total=1000.0,t_transient=100.0,y0=None):
    if y0 is None:
        y0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    n_steps = int(t_total / dt)
    xs = np.zeros((n_steps, 3), dtype=np.float64)
    state = y0.copy()

    def rhs(s):
        return rossler_rhs_np(s, a=a, b=b, c=c)

    for i in range(n_steps):
        xs[i] = state
        state = rk4_step_np(rhs, state, dt)
    skip = int(t_transient / dt)
    return xs[skip:]


def plot_attractor_with_timeseries(data, dt, n_ts, n_3d, title, filename, view_elev=25, view_azim=-60):
    ts_data = data[:n_ts]
    attractor_data = data[:n_3d]
    t = np.arange(n_ts) * dt
    x_ts, y_ts, z_ts = ts_data[:, 0], ts_data[:, 1], ts_data[:, 2]
    x_3d, y_3d, z_3d = attractor_data[:, 0], attractor_data[:, 1], attractor_data[:, 2]
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.15, 1.65], hspace=0.35, wspace=0.05)
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0])
    ax_z = fig.add_subplot(gs[2, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    ax_x.plot(t, x_ts, linewidth=1.2)
    ax_x.set_ylabel("x(t)")
    ax_x.grid(True, alpha=0.3)
    ax_y.plot(t, y_ts, linewidth=1.2)
    ax_y.set_ylabel("y(t)")
    ax_y.grid(True, alpha=0.3)
    ax_z.plot(t, z_ts, linewidth=1.2)
    ax_z.set_ylabel("z(t)")
    ax_z.set_xlabel("t")
    ax_z.grid(True, alpha=0.3)
    ax_3d.plot(x_3d, y_3d, z_3d, linewidth=0.35, alpha=0.9)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.view_init(elev=view_elev, azim=view_azim)

    ax_3d.set_box_aspect([np.ptp(x_3d), np.ptp(y_3d), np.ptp(z_3d)])

    fig.suptitle(title, fontsize=16)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
