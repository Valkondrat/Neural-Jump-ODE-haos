import numpy as np
import torch
from attractors import simulate_attractor


"""
Файл с сборки датасетов
"""

def make_delay_embedding_1d(x, delay_dim=12, tau=2):
    x = np.asarray(x).reshape(-1)
    max_lag = (delay_dim - 1) * tau
    rows = []

    for t in range(max_lag, len(x)):
        rows.append([x[t - k * tau] for k in range(delay_dim)])

    return np.asarray(rows, dtype=np.float32)


def make_windows_delay_x(x_delay, x_target, window, horizon):
    X, Y = [], []
    total = len(x_delay)

    for i in range(total - window - horizon):
        X.append(x_delay[i:i + window])
        Y.append(x_target[i + window:i + window + horizon, None])

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.tensor(np.asarray(Y), dtype=torch.float32)
    return X, Y


def standardize_fit(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def standardize_apply(arr, mean, std):
    return (arr - mean) / std


def build_dataset_xonly_delay(cfg):
    xyz_raw = simulate_attractor(cfg)
    n = len(xyz_raw)
    i1 = int(cfg.data.train_ratio * n)
    i2 = int((cfg.data.train_ratio + cfg.data.val_ratio) * n)

    train_raw = xyz_raw[:i1]
    val_raw = xyz_raw[i1 + cfg.data.gap:i2]
    test_raw = xyz_raw[i2 + cfg.data.gap:]

    mean, std = standardize_fit(train_raw)

    xyz_train = standardize_apply(train_raw, mean, std)
    xyz_val = standardize_apply(val_raw, mean, std)
    xyz_test = standardize_apply(test_raw, mean, std)

    x_train = xyz_train[:, 0]
    x_val = xyz_val[:, 0]
    x_test = xyz_test[:, 0]

    max_lag = (cfg.data.delay_dim - 1) * cfg.data.delay_tau
    Xd_train_np = make_delay_embedding_1d(x_train, cfg.data.delay_dim, cfg.data.delay_tau)
    Xd_val_np = make_delay_embedding_1d(x_val, cfg.data.delay_dim, cfg.data.delay_tau)
    Xd_test_np = make_delay_embedding_1d(x_test, cfg.data.delay_dim, cfg.data.delay_tau)

    xt_train_np = x_train[max_lag:]
    xt_val_np = x_val[max_lag:]
    xt_test_np = x_test[max_lag:]

    X_train_x, Y_train_x = make_windows_delay_x(Xd_train_np, xt_train_np, cfg.data.window, cfg.data.full_horizon)
    X_val_x, Y_val_x = make_windows_delay_x(Xd_val_np, xt_val_np, cfg.data.window, cfg.data.full_horizon)
    X_test_x, Y_test_x = make_windows_delay_x(Xd_test_np, xt_test_np, cfg.data.window, cfg.data.full_horizon)

    return X_train_x, Y_train_x, X_val_x, Y_val_x, X_test_x, Y_test_x, mean, std


def make_windows_xyz(xyz, window, horizon):
    X, Y = [], []
    total = len(xyz)

    for i in range(total - window - horizon):
        X.append(xyz[i:i + window])
        Y.append(xyz[i + window:i + window + horizon])

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.tensor(np.asarray(Y), dtype=torch.float32)
    return X, Y


def build_dataset(cfg):
    xyz_raw = simulate_attractor(cfg)
    n = len(xyz_raw)
    i1 = int(cfg.data.train_ratio * n)
    i2 = int((cfg.data.train_ratio + cfg.data.val_ratio) * n)

    train_raw = xyz_raw[:i1]
    val_raw = xyz_raw[i1 + cfg.data.gap:i2]
    test_raw = xyz_raw[i2 + cfg.data.gap:]

    mean, std = standardize_fit(train_raw)

    xyz_train = standardize_apply(train_raw, mean, std)
    xyz_val = standardize_apply(val_raw, mean, std)
    xyz_test = standardize_apply(test_raw, mean, std)

    X_train, Y_train = make_windows_xyz(xyz_train, cfg.data.window, cfg.data.full_horizon)
    X_val, Y_val = make_windows_xyz(xyz_val, cfg.data.window, cfg.data.full_horizon)
    X_test, Y_test = make_windows_xyz(xyz_test, cfg.data.window, cfg.data.full_horizon)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std
