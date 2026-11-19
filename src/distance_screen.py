"""Training-support distance screen for autoregressive rollouts.

Implements the deployment screen of the paper (appendix "Deployment-Screen Definitions,
Calibration, and Robustness"):

    d_M(y)  = sqrt((y - mu)^T Sigma^+ (y - mu))   state-level Mahalanobis distance over the
              continuous channels; mu, Sigma from the training CSV only, Sigma + 1e-6 I,
              pseudo-inverse (PPS and PZRP are collinear, so Sigma is ill-conditioned)
    d_self  = mean_t d_M(yhat_t)   the surrogate's own rollout -> available before deciding
                                   whether to request a system-code (MAAP) run
    d_retro = mean_t d_M(y_t)      the ground-truth trajectory -> retrospective diagnosis only
    tau     = (1 - f) quantile of d_self over a calibration half of the scenarios; no failure
              labels enter the calibration. f is the flag budget.

A rollout with d_self >= tau is flagged for confirmation with the system code. The held-out half
is used only to report recall and the residual failure rate (per-scenario pooled R^2 < 0) among
accepted rollouts. tau is checkpoint-specific: recalibrate it for any other checkpoint,
normalization, plant or accident class (f = 0.35 is the paper's example for OPR1000 dt15min_seq10).

Usage (from src/):
    python distance_screen.py --checkpoint ../weights/OPR1000/dt15min_seq10/model.ckpt \
        --config ../weights/OPR1000/dt15min_seq10/config_used.yaml --flag-budget 0.35
The config's data.data_path (training CSV) and data.test_data_path must point at your CSVs.
Writes one row per test scenario: scenario, d_self, d_retro, RMSE, R2, split, flagged.
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
import yaml

import utils
from predict import load_weights
from predict_batched import autoregressive_predictions_absolute_batched

RIDGE = 1e-6


def fit_support(train_csv, num_continuous, chunksize=500_000):
    """Mean and pseudo-inverse covariance of the continuous channels over all training rows.

    The continuous channels are the first `num_continuous` columns after scenario_number/TIME,
    the same layout the dataset feeds the model."""
    cols = [c for c in pd.read_csv(train_csv, nrows=0).columns
            if c not in ("scenario_number", "TIME")][:num_continuous]
    n, s1, s2 = 0, np.zeros(num_continuous), np.zeros((num_continuous, num_continuous))
    for chunk in pd.read_csv(train_csv, usecols=cols, chunksize=chunksize):
        x = chunk[cols].to_numpy(np.float64)
        x = x[~np.isnan(x).any(axis=1)]
        n += len(x); s1 += x.sum(0); s2 += x.T @ x
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu) + RIDGE * np.eye(num_continuous)
    return mu, np.linalg.pinv(cov)


def mahalanobis(x, mu, cov_inv):
    """State-level distance d_M: states [..., C] -> distances [...]."""
    d = np.asarray(x, np.float64) - mu
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, cov_inv, d), 0.0))


def score_rollouts(pred_dict, true_dict, mu, cov_inv):
    """Per-scenario d_self, d_retro, RMSE (README definition) and the paper's failure-label R^2.

    R2 is pooled over all steps and channels against each channel's time mean (paper Eq. r2;
    a rollout with R2 < 0 counts as failed). It is not the per-step R^2 that predict.py reports."""
    rows = []
    for s in sorted(pred_dict):
        P = np.vstack(pred_dict[s]).astype(np.float64)
        Y = np.vstack(true_dict[s]).astype(np.float64)
        se = (P - Y) ** 2
        ss_tot = ((Y - Y.mean(0)) ** 2).sum()
        rows.append(dict(scenario=s,
                         d_self=mahalanobis(P, mu, cov_inv).mean(),
                         d_retro=mahalanobis(Y, mu, cov_inv).mean(),
                         RMSE=np.sqrt(se.mean(1)).mean(),
                         R2=1.0 - se.sum() / ss_tot if ss_tot > 0 else np.nan))
    return pd.DataFrame(rows)


def coverage_threshold(d_cal, flag_budget):
    """Label-free threshold that flags a fraction `flag_budget` of the calibration rollouts."""
    return float(np.quantile(d_cal, 1.0 - flag_budget))


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description="Mahalanobis distance screen for AR rollouts.")
    ap.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt).")
    ap.add_argument("--config", required=True, help="The checkpoint's config_used.yaml.")
    ap.add_argument("--flag-budget", type=float, default=0.35,
                    help="Fraction of rollouts to send to the system code (default 0.35).")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed of the calibration / held-out split of the test scenarios.")
    ap.add_argument("--output", default="distance_screen.csv")
    a = ap.parse_args()
    if not 0.0 < a.flag_budget < 1.0:
        ap.error("--flag-budget must be in (0, 1)")

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data, nc = cfg["data"], cfg["model"]["backbone_kwargs"]["num_continuous"]

    model = load_weights(cfg, a.checkpoint)
    test = utils.get_dataset(data["test_data_path"], data["sequence_length"],
                             data["prediction_length"], data["prediction_type"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred, true = autoregressive_predictions_absolute_batched(model, test, device=device,
                                                             num_continuous=nc)
    mu, cov_inv = fit_support(data["data_path"], nc)
    df = score_rollouts(pred, true, mu, cov_inv)

    # label-free calibration on one half of the scenarios, evaluation on the other half
    perm = np.random.default_rng(a.split_seed).permutation(len(df))
    cal = np.zeros(len(df), bool)
    cal[perm[: len(df) // 2]] = True
    tau = coverage_threshold(df.d_self[cal], a.flag_budget)
    df["split"] = np.where(cal, "calibration", "held_out")
    df["flagged"] = df.d_self >= tau
    df.to_csv(a.output, index=False)

    h = df[~cal]
    fail, acc = (h.R2 < 0).to_numpy(), (~h.flagged).to_numpy()
    k, n = int((fail & acc).sum()), int(acc.sum())
    lo, hi = wilson(k, n)
    print(f"tau = {tau:.3f}  (flag budget {a.flag_budget:.0%}, calibration n = {int(cal.sum())})")
    print(f"held-out n = {len(h)}: flagged {h.flagged.mean():.1%} | "
          f"recall of R2<0 failures {int((fail & ~acc).sum())}/{int(fail.sum())} | "
          f"residual failure rate among accepted {k}/{n} = {k / max(n, 1):.2%} "
          f"[{lo:.2%}, {hi:.2%}] | without screen {fail.mean():.2%}")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
