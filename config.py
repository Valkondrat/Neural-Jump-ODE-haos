from dataclasses import dataclass, field
from typing import Literal, Any

from dataclasses import replace


"""
Файл с общей конфигурацией для всех моделей.
"""


AttractorName = Literal["lorenz", "rossler"]
TaskName = Literal["full_phase", "x_delay"]
DetectorName = Literal["posthoc", "selective"]


@dataclass
class AttractorConfig:
    name: AttractorName
    params: dict[str, float]
    y0: list[float]
    dt: float
    t_total: float
    t_transient: float


@dataclass
class DataConfig:
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    gap: int = 300

    window: int = 60
    short_k: int = 20
    full_horizon: int = 200

    task: TaskName = "full_phase"
    delay_dim: int | None = None
    delay_tau: int | None = None

    @property
    def long_horizon(self) -> int:
        return self.full_horizon - self.short_k


@dataclass
class ModelConfig:
    z_dim: int = 128
    hidden_dim: int = 256
    seg_ode: int = 4
    ode_hidden_mult: int = 3
    ode_res_scale: float = 0.10
    jump_hidden_mult: int = 2
    n_mdn_components: int = 3
    use_layernorm_z2h: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 128
    n_epochs: int = 1600
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    input_noise_max: float = 0.03

    tf_full_end: int = 400
    tf_zero_epoch: int = 1400
    curriculum_end_epoch: int = 700


@dataclass
class LossConfig:
    mdn_nll_weight: float = 0.10
    sigma_reg_weight: float = 0.50
    sigma_smooth_weight: float = 0.02

    stat_start_epoch: int = 150
    stat_ramp_epochs: int = 600

    spec_w: float = 0.10
    amp_w: float = 4.00
    amp_z_w: float = 1.50
    wing_w: float = 3.00
    dist_w: float = 0.50
    bins_w: float = 0.60
    meanx_w: float = 0.15
    meanz_w: float = 0.10

    cdf_mode: Literal["fixed_lorenz", "adaptive"] = "fixed_lorenz"


@dataclass
class DetectorConfig:
    mode: DetectorName = "posthoc"

    eps_x: float = 0.7
    eps_short: float = 0.6
    eps_long: float = 1.2

    coverage_warmup_end: int = 300
    coverage_ramp_end: int = 900
    coverage_lambda_max: float = 0.7

    coverage_start: float = 0.95
    coverage_end: float = 0.65
    coverage_target_ramp_start: int = 300
    coverage_target_ramp_end: int = 1000

    gate_threshold: float = 0.5
    target_precision: float = 0.80
    min_recall: float = 0.45
    min_coverage: float = 0.10


@dataclass
class ExperimentConfig:
    attractor: AttractorConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
    detector: DetectorConfig

    save_dir: str = "./checkpoints"
    save_stem: str = "experiment"

    @property
    def best_weights_path(self) -> str:
        return f"{self.save_dir}/{self.save_stem}_best.pt"

    @property
    def best_vtx_path(self) -> str:
        return f"{self.save_dir}/{self.save_stem}_best_vtx.pt"


ATTRACTOR_PRESETS = {
    "lorenz": AttractorConfig(
        name="lorenz",
        params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        y0=[1.0, 1.0, 1.0],
        dt=0.05,
        t_total=4000.0,
        t_transient=100.0),
    "rossler": AttractorConfig(
        name="rossler",
        params={"a": 0.2, "b": 0.2, "c": 5.7},
        y0=[1.0, 0.0, 0.0],
        dt=0.05,
        t_total=4000.0,
        t_transient=200.0)}

TASK_PRESETS = {
    "full_phase": DataConfig(
        task="full_phase",
        window=60,
        short_k=20,
        full_horizon=200,
        delay_dim=None,
        delay_tau=None),
    "x_delay": DataConfig(
        task="x_delay",
        window=60,
        short_k=20,
        full_horizon=200,
        delay_dim=24,
        delay_tau=1)}

ROSSSLER_TASK_OVERRIDES = {
    "full_phase": {
        "window": 120,
        "short_k": 30,
        "full_horizon": 300},
    "x_delay": {
        "window": 120,
        "short_k": 30,
        "full_horizon": 300,
        "delay_dim": 32,
        "delay_tau": 2}}


DETECTOR_PRESETS = {
    "posthoc": DetectorConfig(
        mode="posthoc",
        eps_x=0.7,
        eps_short=0.6,
        eps_long=1.2),
    "selective": DetectorConfig(
        mode="selective",
        eps_x=0.7,
        eps_short=0.6,
        eps_long=1.2,
        coverage_lambda_max=0.7,
        coverage_start=0.95,
        coverage_end=0.65,)}

LOSS_PRESETS = {
    "lorenz": LossConfig(
        mdn_nll_weight=0.10,
        sigma_reg_weight=0.50,
        sigma_smooth_weight=0.02,
        spec_w=0.10,
        amp_w=4.00,
        amp_z_w=1.50,
        wing_w=3.00,
        dist_w=0.50,
        bins_w=0.60,
        meanx_w=0.15,
        meanz_w=0.10,
        cdf_mode="fixed_lorenz"),
    "rossler": LossConfig(
        mdn_nll_weight=0.005,
        sigma_reg_weight=0.20,
        sigma_smooth_weight=0.005,
        stat_start_epoch=300,
        stat_ramp_epochs=700,
        spec_w=0.10,
        amp_w=1.50,
        amp_z_w=1.00,
        wing_w=0.00,
        dist_w=0.20,
        bins_w=0.25,
        meanx_w=0.05,
        meanz_w=0.05,
        cdf_mode="adaptive",),}

def apply_overrides(obj, overrides: dict):
    if not overrides:
        return obj
    return replace(obj, **overrides)


def make_config(attractor: AttractorName,task: TaskName,detector: DetectorName,save_dir: str = "./checkpoints",) -> ExperimentConfig:
    attr_cfg = ATTRACTOR_PRESETS[attractor]
    data_cfg = TASK_PRESETS[task]
    loss_cfg = LOSS_PRESETS[attractor]
    detector_cfg = DETECTOR_PRESETS[detector]

    if attractor == "rossler":
        data_cfg = apply_overrides(data_cfg,ROSSSLER_TASK_OVERRIDES.get(task, {}))

    if detector == "selective":
        train_cfg = TrainConfig(n_epochs=1300 if task == "x_delay" else 1700, tf_full_end=100,
        tf_zero_epoch=800, curriculum_end_epoch=500)
    else:
        train_cfg = TrainConfig(
            n_epochs=1600 if attractor == "lorenz" else 1200, tf_full_end=400 if attractor == "lorenz" else 100,
            tf_zero_epoch=1400 if attractor == "lorenz" else 700, curriculum_end_epoch=700 if attractor == "lorenz" else 800)

    model_cfg = ModelConfig()

    save_stem = f"{attractor}_{task}_{detector}_njode"

    return ExperimentConfig(attractor=attr_cfg, data=data_cfg, model=model_cfg, train=train_cfg,
        loss=loss_cfg, detector=detector_cfg, save_dir=save_dir, save_stem=save_stem)

