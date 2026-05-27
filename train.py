import torch
import torch.nn as nn
from tqdm import tqdm

from evaluate import evaluate_model_mdnstats, evaluate_model_selective
from losses import combined_mdnstats_loss, combined_selective_loss, combined_selective_loss_xonly
from utils import coverage_curriculum, coverage_target_curriculum, current_horizon, current_noise_std, current_tf_ratio, sample_batch

"""
Файл с цифклами обучения. 
"""

def _resolve_selective_loss_fn(cfg, loss_fn=None):
    if loss_fn is not None:
        return loss_fn

    if cfg.data.task == "x_delay":
        return combined_selective_loss_xonly

    if cfg.data.task == "full_phase":
        return combined_selective_loss


def _save_checkpoint(path, *, epoch, model, opt=None, sched=None, best_score=None, score=None, val_metrics=None, cfg=None):
    payload = {"epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_score": best_score,
        "score": score,
        "cfg": cfg}

    if opt is not None:
        payload["optimizer_state_dict"] = opt.state_dict()

    if sched is not None:
        payload["scheduler_state_dict"] = sched.state_dict()

    if val_metrics is not None:
        payload["val_metrics"] = val_metrics

        for key, value in val_metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                payload[f"val_{key}"] = value

    torch.save(payload, path)


def train_model_selective(model, X_train, Y_train, X_val, Y_val, cfg, *, loss_fn=None, log_every=100, n_eval=256, use_adaptive_oracle=False, div_weight=0.0, score_amp_weight=0.50, score_vt_weight=0.05, eta_min=1e-5):
    loss_fn = _resolve_selective_loss_fn(cfg, loss_fn=loss_fn)

    n_epochs = cfg.train.n_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=eta_min)

    best_score = float("inf")
    best_state = None
    history = []

    for epoch in tqdm(range(n_epochs), desc=f"training {cfg.attractor.name} {cfg.data.task} selective"):
        model.train()

        tf_ratio = current_tf_ratio(epoch, cfg)
        noise_std = current_noise_std(epoch, cfg)
        curr_horizon = current_horizon(epoch, cfg)
        eff_short_k = min(cfg.data.short_k, curr_horizon)

        cov_target = coverage_target_curriculum(epoch, cfg)
        lam_cov = coverage_curriculum(epoch, cfg)

        xb, yb = sample_batch(X_train, Y_train, cfg, device=device)
        target = yb[:, :curr_horizon]

        pred_tf, gate_tf, err_hat_tf, pi_tf, mu_tf, sigma_tf = model(
            xb,horizon=curr_horizon,
            x_true=target,tf_ratio=tf_ratio,
            noise_std=noise_std,
            sample_mode=False,return_div=False)

        pred_fr, gate_fr, err_hat_fr, pi_fr, mu_fr, sigma_fr, div_loss = model(
            xb,horizon=curr_horizon,
            x_true=None,tf_ratio=0.0,
            noise_std=0.0,sample_mode=False,
            return_div=True,div_n_samples=1)

        loss, dbg = loss_fn(
            pred_tf=pred_tf,
            gate_tf=gate_tf[:, :curr_horizon],
            err_hat_tf=err_hat_tf[:, :curr_horizon],
            pi_tf=pi_tf[:, :curr_horizon],
            mu_tf=mu_tf[:, :curr_horizon],
            sigma_tf=sigma_tf[:, :curr_horizon],
            pred_fr=pred_fr,
            gate_fr=gate_fr[:, :curr_horizon],
            err_hat_fr=err_hat_fr[:, :curr_horizon],
            pi_fr=pi_fr[:, :curr_horizon],
            mu_fr=mu_fr[:, :curr_horizon],
            sigma_fr=sigma_fr[:, :curr_horizon],
            true=target,
            short_k=eff_short_k,
            epoch=epoch,
            n_epochs=n_epochs,
            coverage_target=cov_target,
            lambda_cov=lam_cov,
            eps_x=cfg.detector.eps_x,
            use_adaptive_oracle=use_adaptive_oracle,
            eps_short=cfg.detector.eps_short,
            eps_long=cfg.detector.eps_long,
            loss_cfg=cfg.loss)

        div_value = 0.0
        if div_loss is not None and div_weight > 0:
            loss = loss + div_weight * div_loss
            div_value = div_loss.detach().item()

        opt.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        opt.step()
        sched.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            val_out = evaluate_model_selective(
                model,
                X_val,
                Y_val,
                cfg,
                n_eval=n_eval,
                gate_threshold_log=cfg.detector.gate_threshold,
                use_adaptive_oracle=use_adaptive_oracle
            )

            val_rmse, val_amp_loss, val_amp_ratio_x, val_vt, gate_diag = val_out
            score = val_rmse + score_amp_weight * abs(val_amp_ratio_x - 1.0) - score_vt_weight * val_vt

            row = {"epoch": epoch,
                "tf_ratio": tf_ratio,
                "horizon": curr_horizon,
                "noise_std": noise_std,
                "coverage_target": cov_target,
                "lambda_cov": lam_cov,
                "div": div_value,
                "train_loss": loss.detach().item(),
                **dbg,
                "val_short_rmse": val_rmse,
                "val_amp_loss": val_amp_loss,
                "val_amp_ratio_x": val_amp_ratio_x,
                "val_vt": val_vt,
                **{f"val_{k}": v for k, v in gate_diag.items()},
                "score": score}
            history.append(row)

            print(
                f"[{epoch:4d}] loss={loss.item():.5f} "
                f"tf={tf_ratio:.2f} "
                f"horizon={curr_horizon:3d} "
                f"noise={noise_std:.3f} "
                f"cov_tgt={cov_target:.2f} "
                f"lam_cov={lam_cov:.2f} "
                f"div={div_value:.5f} "
                f"val_short_rmse={val_rmse:.4f} "
                f"val_amp_loss={val_amp_loss:.4f} "
                f"val_amp_ratio_x={val_amp_ratio_x:.4f} "
                f"val_vt@0.4={val_vt:.3f} "
                f"gate_auc={gate_diag['gate_auc']:.3f} "
                f"gate_sep={gate_diag['gate_sep']:.3f} "
                f"gate_p@{cfg.detector.gate_threshold:.2f}={gate_diag['gate_precision']:.3f} "
                f"gate_r@{cfg.detector.gate_threshold:.2f}={gate_diag['gate_recall']:.3f} "
                f"gate_cov@{cfg.detector.gate_threshold:.2f}={gate_diag['gate_coverage']:.3f} "
                f"| forecast={dbg['forecast_loss']:.4f} "
                f"gate_loss={dbg['gate_loss']:.4f} "
                f"stat={dbg['stat_loss']:.4f} "
                f"gate_tf={dbg['mean_gate_tf']:.3f} "
                f"gate_fr={dbg['mean_gate_fr']:.3f}")

            epoch_weights_path = f"{cfg.save_dir}/{cfg.save_stem}_{epoch}.pt"
            val_metrics = {"short_rmse": val_rmse,
                "amp_loss": val_amp_loss,
                "amp_ratio_x": val_amp_ratio_x,
                "valid_time_x": val_vt,
                **gate_diag}

            _save_checkpoint(
                epoch_weights_path,
                epoch=epoch,model=model,
                opt=opt,sched=sched,
                best_score=best_score,
                score=score,val_metrics=val_metrics,
                cfg=cfg)

            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                _save_checkpoint(
                    cfg.best_weights_path,
                    epoch=epoch,
                    model=model,
                    opt=opt,
                    sched=sched,
                    best_score=best_score,
                    score=score,
                    val_metrics=val_metrics,
                    cfg=cfg)

                print(f"new best saved to: {cfg.best_weights_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

        torch.save(
            {"model_state_dict": model.state_dict(),
                "best_score": best_score,
                "cfg": cfg,
                "history": history},
            cfg.best_weights_path)

        print("final best weights saved to:", cfg.best_weights_path)

    return model, history


def train_model_mdnstats(model, X_train, Y_train, X_val, Y_val, cfg, *, loss_fn=None, log_every=100, n_eval=512, score_amp_weight=0.30, score_vt_weight=0.05, eta_min=1e-5, save_best_vtx=True):
    if loss_fn is None:
        loss_fn = combined_mdnstats_loss

    n_epochs = cfg.train.n_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=eta_min)

    best_score = float("inf")
    best_state = None
    best_vt_x = -float("inf")
    best_vt_state = None
    history = []

    for epoch in tqdm(range(n_epochs), desc=f"training {cfg.attractor.name} {cfg.data.task} posthoc"):
        model.train()

        tf_ratio = current_tf_ratio(epoch, cfg)
        noise_std = current_noise_std(epoch, cfg)
        curr_horizon = current_horizon(epoch, cfg)
        eff_short_k = min(cfg.data.short_k, curr_horizon)

        xb, yb = sample_batch(X_train, Y_train, cfg, device=device)
        target = yb[:, :curr_horizon]

        pred, pi, mu, sigma = model(
            xb,horizon=curr_horizon,
            x_true=target,
            tf_ratio=tf_ratio,
            noise_std=noise_std)

        loss, debug = loss_fn(
            pred=pred, true=target,
            pi=pi, mu=mu, sigma=sigma,
            short_k=eff_short_k,
            epoch=epoch, n_epochs=n_epochs,
            loss_cfg=cfg.loss)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        opt.step()
        sched.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            val_metrics = evaluate_model_mdnstats(model, X_val, Y_val, cfg, n_eval=n_eval)
            score = val_metrics["short_rmse"] + score_amp_weight * val_metrics["amp"] - score_vt_weight * val_metrics["valid_time_x"]

            row = {"epoch": epoch,
                "tf_ratio": tf_ratio,
                "horizon": curr_horizon,
                "noise_std": noise_std,
                "train_loss": debug["loss"],
                **debug,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "score": score}
            history.append(row)

            print(f"\nEpoch {epoch:04d} | "
                f"H={curr_horizon:3d} | tf={tf_ratio:.3f} | "
                f"loss={debug['loss']:.4f} | "
                f"mse={debug['mse_short']:.4f} | "
                f"mdn={debug['mdn_nll']:.4f} | "
                f"val_short={val_metrics['short_rmse']:.4f} | "
                f"val_amp={val_metrics['amp']:.4f} | "
                f"vt={val_metrics['valid_time']:.3f} | "
                f"vt_x={val_metrics['valid_time_x']:.3f} | "
                f"sigma={val_metrics['mean_sigma']:.4f}")

            epoch_weights_path = f"{cfg.save_dir}/{cfg.save_stem}_{epoch}.pt"

            _save_checkpoint(
                epoch_weights_path,
                epoch=epoch, model=model,
                opt=opt, sched=sched,
                best_score=best_score,
                score=score, val_metrics=val_metrics,
                cfg=cfg)

            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                _save_checkpoint(
                    cfg.best_weights_path,
                    epoch=epoch,
                    model=model, opt=opt,
                    sched=sched, best_score=best_score,
                    score=score, val_metrics=val_metrics,
                    cfg=cfg)

                print(f"  saved best -> {cfg.best_weights_path} | score={best_score:.4f}")

            if save_best_vtx and val_metrics["valid_time_x"] > best_vt_x:
                best_vt_x = val_metrics["valid_time_x"]
                best_vt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                vt_path = cfg.best_weights_path.replace(".pt", "_best_vtx.pt")

                torch.save(
                    {"epoch": epoch,
                        "model_state_dict": best_vt_state,
                        "best_vt_x": best_vt_x,
                        "val_metrics": val_metrics,
                        "cfg": cfg,
                        "history": history},
                    vt_path)

                print(f"  saved best vt_x -> {vt_path} | vt_x={best_vt_x:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

        torch.save(
            {"model_state_dict": model.state_dict(),
                "best_score": best_score,
                "cfg": cfg,
                "history": history},
            cfg.best_weights_path)

        print("final best weights saved to:", cfg.best_weights_path)

    return model, history


def train_model(model, X_train, Y_train, X_val, Y_val, cfg, **kwargs):
    detector_mode = getattr(getattr(cfg, "detector", None), "mode", "selective")

    if detector_mode == "selective":
        return train_model_selective(model, X_train, Y_train, X_val, Y_val, cfg, **kwargs)

    if detector_mode == "posthoc":
        return train_model_mdnstats(model, X_train, Y_train, X_val, Y_val, cfg, **kwargs)

