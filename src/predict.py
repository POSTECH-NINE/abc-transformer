import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # real FP32 kernels

import re
from pathlib import Path
from model_selector import ModelSelector
import utils
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
import matplotlib.pyplot as plt
import tqdm
import yaml
from typing import Dict

logging.basicConfig(level=logging.INFO)


def load_weights(config, model_path):
    model_name = config["model"]["name"]
    backbone_kwargs = config["model"]["backbone_kwargs"]
    lightning_kwargs = config["model"].get("lightning_kwargs", {})
    base_model, lit_model = ModelSelector(
        model_name, backbone_kwargs=backbone_kwargs, lightning_kwargs=lightning_kwargs
    )
    lit_model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
    logging.info(f"Loaded model from {model_path}")
    return lit_model


def get_next_train_data(past_values: torch.Tensor, next_value: torch.Tensor):
    """
    What this should do:
    1. Remove the first value from past_values
    2. append next_value to past_values (at last position)

    past values shape: batch_size, seq_len, input_size
    next_value shape: batch_size, input_size

    Returns:
        next_train_data: shape: batch_size, seq_len, input_size
    """
    return torch.cat((past_values[:, 1:, :], next_value.unsqueeze(1)), dim=1)


@torch.inference_mode()
def autoregressive_predictions_absolute(model, dataset, exp_dir, training_log_dir=None):
    from collections import defaultdict
    import tqdm
    import torch
    from pathlib import Path

    exp_dir = Path(exp_dir)
    model = model.to(dtype=torch.float32).eval()

    predictions_dict = defaultdict(list)
    true_dict = defaultdict(list)

    pooling_history_dict = defaultdict(list)
    scenario_errors = {}  # {scenario_id: {"mae_steps": ndarray, "rmse_steps": ndarray}}

    def _finalize_scenario(sc_id, t_seq, p_seq, win):
        # per-timestep MAE/RMSE (computed over variables) for finished scenario
        valid_true = np.array(t_seq[win:], dtype=float)
        valid_pred_list = [p for p in p_seq[win:] if p is not None]
        if len(valid_pred_list) == 0 or valid_true.shape[0] == 0:
            return
        valid_pred = np.array(valid_pred_list, dtype=float)
        n = min(valid_true.shape[0], valid_pred.shape[0])
        if n == 0:
            return
        diff = valid_pred[:n] - valid_true[:n]
        mae_steps = np.mean(np.abs(diff), axis=1)
        rmse_steps = np.sqrt(np.mean(diff ** 2, axis=1))
        scenario_errors[sc_id] = {"mae_steps": mae_steps, "rmse_steps": rmse_steps}

    # --- Initialize with the first data point ---
    data_point = dataset[0]

    # 1. Determine the window size (typically 30)
    window_size = data_point["past_values"].shape[0]

    # 2. Pre-fill the buffers with the ground-truth data of the initial window.
    # past_values has shape (window_size, V); converting to a list yields per-timestep (V,) arrays.
    past_values_raw = data_point["past_values"].cpu().numpy()

    true_seq = [v for v in past_values_raw]      # ground truth for the initial window
    pred_seq = [None] * window_size              # no predictions for the initial window
    y_time_seq = [i * 0.25 for i in range(window_size)]  # time axis for the initial window
    current_scenario_id = None

    for i in tqdm.tqdm(range(1, (len(dataset)))):
        # 1. Move data to GPU and unpack variables
        d_cuda = {k: v.unsqueeze(0).cuda().float() for k, v in data_point.items()}
        past_values = d_cuda["past_values"]
        true_bin_y = d_cuda["binary_y"]
        true_cont_y = d_cuda["continuous_y"]
        metadata = d_cuda["y_metadata"]

        scenario_number = int(metadata.view(-1)[0].item())
        y_time = metadata.view(-1)[1].item()
        current_scenario_id = scenario_number

        # 2. Model prediction
        # Unpack according to the number of model outputs
        outputs = model.backbone(past_values, return_attn=True, return_weights=True)
        pred, attn, p_weights = outputs[0], outputs[1], outputs[2]

        # 3. Accumulate results
        pooling_history_dict[scenario_number].append(p_weights.detach().cpu().squeeze().numpy())
        p_np = pred.squeeze(1).detach().cpu().numpy().reshape(-1)
        t_np = true_cont_y.squeeze(1).detach().cpu().numpy().reshape(-1)

        true_seq.append(t_np)
        pred_seq.append(p_np)
        y_time_seq.append(y_time)

        # 4. Autoregressive update
        full_output = torch.cat([pred.squeeze(1), true_bin_y], dim=1)
        updated_values = get_next_train_data(past_values, full_output)

        # 5. Load the next sample and check for a scenario change
        next_data_point = dataset[i]
        next_scenario = int(next_data_point["y_metadata"][0].item())

        if next_scenario == scenario_number:
            next_data_point["past_values"] = updated_values.squeeze(0).cpu()
            data_point = next_data_point
        else:
            # Store per-timestep MAE/RMSE for the finished scenario
            _finalize_scenario(scenario_number, true_seq, pred_seq, window_size)

            # Reset buffers and load the initial window of the new scenario
            print(f"\n[Scenario Change] {scenario_number} -> {next_scenario}. Resetting buffer.")
            data_point = next_data_point
            past_values_init = data_point["past_values"].cpu().numpy()
            true_seq = [v for v in past_values_init]
            pred_seq = [None] * window_size
            y_time_seq = [i * 0.25 for i in range(window_size)]

    # Finalize the last scenario being processed
    if current_scenario_id is not None and current_scenario_id not in scenario_errors:
        _finalize_scenario(current_scenario_id, true_seq, pred_seq, window_size)

    # Save combined summary: micro/macro MAE/RMSE plus train/val MSE from metrics.csv
    _log_dir = training_log_dir if training_log_dir is not None else exp_dir
    write_eval_summary(scenario_errors, _log_dir)

    return predictions_dict, true_dict


def make_experiment_dir(cfg, root="runs"):
    """
    Create an experiment directory automatically from the config.
    Example: runs/seq3_15min/
    """
    data_path = Path(cfg["data"]["data_path"])
    seq_len = cfg["data"]["sequence_length"]

    # Extract a token like 15min / 30min from the file name
    m = re.search(r"(\d+min)", data_path.name)
    freq = m.group(1) if m else "unknown"

    exp_name = f"seq{seq_len}_{freq}"
    exp_dir = Path(root) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def plot_stepwise_error_mean(
    predictions_dict,
    true_dict,
    variable_names,
    out_dir="plots/mean_step_error",
    file_name="mean_step_error.png",
    reduction="mse",  # default: MSE
):
    """
    Compute per-step errors for each scenario, then average across scenarios
    at each timestep to plot per-variable 'Mean Step Error (MAE/MSE)' curves.

    No accumulation (cumsum); plots the mean of err(t) as-is.

    Returns:
        mean_step_err: (T_max, V)  # per-timestep, per-variable mean error across scenarios
        step_err_list: list of (T_s, V)  # per-scenario step errors
    """

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)

    step_err_list = []
    V_resolved = None

    # 1) Per-scenario step errors
    for sid in sorted(predictions_dict.keys()):
        preds_seq = predictions_dict[sid]
        trues_seq = true_dict[sid]

        if len(preds_seq) == 0:
            continue

        # (T_s, V)
        preds = np.stack([np.asarray(p).reshape(-1) for p in preds_seq], axis=0)
        trues = np.stack([np.asarray(t).reshape(-1) for t in trues_seq], axis=0)

        if V_resolved is None:
            V_resolved = preds.shape[1]

        if reduction == "mae":
            err = np.abs(preds - trues)          # (T_s, V)
        elif reduction == "mse":
            err = (preds - trues) ** 2          # (T_s, V)
        else:
            raise ValueError("reduction must be 'mae' or 'mse'")

        step_err_list.append(err)

    # 2) NaN padding (align scenarios of different lengths)
    T_max = max(arr.shape[0] for arr in step_err_list)
    S = len(step_err_list)
    V = V_resolved

    stack = np.full((S, T_max, V), np.nan, dtype=float)
    for i, arr in enumerate(step_err_list):
        T_s = arr.shape[0]
        stack[i, :T_s, :] = arr   # trailing entries stay NaN

    # 3) Mean across scenarios per timestep
    mean_step_err = np.nanmean(stack, axis=0)  # (T_max, V)

    # 4) Save plot
    x = np.arange(1, T_max + 1)
    red_name = reduction.upper()

    plt.figure(figsize=(12, 7))
    for v in range(V):
        plt.plot(x, mean_step_err[:, v], linewidth=2, label=variable_names[v])

    plt.title(f"Mean Step {red_name} Across Scenarios", fontsize=14)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel(f"{red_name} per Step", fontsize=12)
    plt.ylim(0, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[INFO] Mean step {red_name} plot saved -> {out_path}")

    return mean_step_err, step_err_list


def compute_report_accuracy(predictions_dict, true_dict, norm_type="l2"):
    """
    Report formula:
        Err_i = (1/T) * sum_t || y_pred(t,i) - y_true(t,i) ||
        Accuracy = (1 - (1/N)*sum_i Err_i) * 100

    norm_type:
        "l2" -> ||e||_2 (Euclidean norm)
        "l1" -> ||e||_1 (sum of absolute values)
    """

    scenario_errors = []
    rows = []

    for sid in sorted(set(predictions_dict.keys()) & set(true_dict.keys())):
        preds_seq = predictions_dict[sid]
        trues_seq = true_dict[sid]

        T = min(len(preds_seq), len(trues_seq))
        if T == 0:
            continue

        # Convert to (T, V) arrays
        preds = np.stack([np.asarray(p).reshape(-1) for p in preds_seq[:T]], axis=0)
        trues = np.stack([np.asarray(t).reshape(-1) for t in trues_seq[:T]], axis=0)

        # Vector error e(t) per timestep
        err = preds - trues  # (T, V)

        if norm_type == "l2":
            # ||e||_2 per timestep
            step_norm = np.linalg.norm(err, axis=1)   # (T,)
        elif norm_type == "l1":
            # ||e||_1 per timestep
            step_norm = np.sum(np.abs(err), axis=1)   # (T,)
        else:
            raise ValueError("norm_type must be 'l2' or 'l1'")

        # Per-scenario mean error (1/T * sum_t norm)
        scenario_err = step_norm.mean()
        scenario_errors.append(scenario_err)

        rows.append({
            "scenario": sid,
            "T": T,
            "scenario_mean_error": scenario_err
        })

    # Overall accuracy
    N = len(scenario_errors)
    if N == 0:
        raise ValueError("No valid scenarios to compute accuracy.")

    avg_err = float(np.mean(scenario_errors))
    accuracy = (1.0 - avg_err) * 100.0

    df = pd.DataFrame(rows).sort_values("scenario")
    return accuracy, avg_err, df


def write_eval_summary(scenario_errors: Dict[int, Dict[str, float]],
                       training_log_dir,
                       out_path=None):
    """
    Aggregate per-scenario MAE/RMSE and merge with train/val MSE from metrics.csv.

    scenario_errors: {scenario_id: {"mae_steps": np.ndarray, "rmse_steps": np.ndarray}}
        - mae_steps[t]: MAE at timestep t for that scenario (computed over variables)
        - rmse_steps[t]: RMSE at timestep t for that scenario
    training_log_dir: experiment dir containing csv_logs/version_0/metrics.csv
    out_path: where to write the summary. If None, defaults to training_log_dir/eval_summary.txt
    """
    training_log_dir = Path(training_log_dir)
    metrics_csv = training_log_dir / "csv_logs" / "version_0" / "metrics.csv"

    if out_path is None:
        out_path = training_log_dir / "eval_summary.txt"
    out_path = Path(out_path)

    # ---- 1) Aggregate prediction errors ----
    per_scen_means = []
    all_mae_steps = []
    all_rmse_steps = []
    for sid in sorted(scenario_errors):
        mae_arr = np.asarray(scenario_errors[sid]["mae_steps"], dtype=float)
        rmse_arr = np.asarray(scenario_errors[sid]["rmse_steps"], dtype=float)
        if mae_arr.size == 0:
            continue
        per_scen_means.append({
            "scenario": sid,
            "MAE": float(np.mean(mae_arr)),
            "RMSE": float(np.mean(rmse_arr)),
            "T": int(mae_arr.size),
        })
        all_mae_steps.append(mae_arr)
        all_rmse_steps.append(rmse_arr)

    if all_mae_steps:
        all_mae = np.concatenate(all_mae_steps)
        all_rmse = np.concatenate(all_rmse_steps)
        micro_mae = float(np.mean(all_mae))
        micro_rmse = float(np.mean(all_rmse))
        macro_mae = float(np.mean([d["MAE"] for d in per_scen_means]))
        macro_rmse = float(np.mean([d["RMSE"] for d in per_scen_means]))
    else:
        micro_mae = micro_rmse = macro_mae = macro_rmse = float("nan")

    # ---- 2) Read training metrics.csv ----
    train_val_lines = []
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            val_df = df.dropna(subset=["val_loss"])[["epoch", "step", "val_loss"]].copy()
            tr_df = df.dropna(subset=["train_loss"])[["epoch", "step", "train_loss"]].copy()
            tr_epoch_mean = tr_df.groupby("epoch")["train_loss"].mean()

            if len(val_df) > 0:
                best_row = val_df.loc[val_df["val_loss"].idxmin()]
                best_epoch = int(best_row["epoch"])
                best_val = float(best_row["val_loss"])
                best_train = float(tr_epoch_mean.get(best_epoch, float("nan")))

                last_epoch = int(val_df["epoch"].max())
                last_val = float(val_df[val_df["epoch"] == last_epoch]["val_loss"].iloc[-1])
                last_train = float(tr_epoch_mean.get(last_epoch, float("nan")))

                train_val_lines = [
                    f"Best epoch (min val_loss): epoch={best_epoch}  train_MSE={best_train:.8f}  val_MSE={best_val:.8f}",
                    f"Last epoch              : epoch={last_epoch}  train_MSE={last_train:.8f}  val_MSE={last_val:.8f}",
                ]
            else:
                train_val_lines = ["[Warn] metrics.csv has no rows with val_loss."]
        except Exception as e:
            train_val_lines = [f"[Error] Failed to read metrics.csv: {e}"]
    else:
        train_val_lines = [f"[Warn] metrics.csv not found at {metrics_csv}"]

    # ---- 3) Write summary ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Autoregressive Absolute Prediction Summary ===\n")
        f.write(f"Training log dir: {training_log_dir}\n\n")

        f.write("-- Prediction Errors (autoregressive, absolute) --\n")
        f.write(f"Scenarios used (with predictions): {len(per_scen_means)}\n")
        f.write(f"Micro (mean over all scenario-timesteps): MAE={micro_mae:.8f}  RMSE={micro_rmse:.8f}\n")
        f.write(f"Macro (mean of per-scenario means)      : MAE={macro_mae:.8f}  RMSE={macro_rmse:.8f}\n\n")

        f.write("-- Training Losses (from metrics.csv, MSE) --\n")
        for line in train_val_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("-- Per-Scenario Means --\n")
        for row in sorted(per_scen_means, key=lambda r: str(r["scenario"])):
            f.write(f"Scenario {row['scenario']}: T={row['T']:4d}  MAE={row['MAE']:.8f}  RMSE={row['RMSE']:.8f}\n")

    print("\n" + "=" * 60)
    print(f"[Eval Summary] Saved to: {out_path}")
    print(f"  Micro MAE = {micro_mae:.8f}   Micro RMSE = {micro_rmse:.8f}")
    print(f"  Macro MAE = {macro_mae:.8f}   Macro RMSE = {macro_rmse:.8f}")
    for line in train_val_lines:
        print("  " + line)
    print("=" * 60 + "\n")

    return {
        "micro_mae": micro_mae, "micro_rmse": micro_rmse,
        "macro_mae": macro_mae, "macro_rmse": macro_rmse,
        "per_scen_means": per_scen_means,
        "train_val_lines": train_val_lines,
    }


def scenario_wise_metrics(predictions_dict, true_dict, debug=False, out_dir: str = None):
    """
    For each scenario, returns a DataFrame with per-timestep metrics (MAE, RMSE, R2).

    Args:
        predictions_dict (dict): scenario -> iterable of prediction arrays per timestep
        true_dict (dict): scenario -> iterable of ground-truth arrays per timestep
        debug (bool): if True, warns on existing plot dirs, pauses, and saves plots

    Returns:
        scenario_metrics_dfs (List[pd.DataFrame]): one per scenario with columns:
            ["timestep", "MAE", "RMSE", "R2", "scenario"]

    Side effects:
        - Writes a human-readable summary txt at plots/metrics_summary.txt with:
            * Overall micro and macro averages
            * Per-scenario means (MAE, RMSE, R2)
        - When debug=True, saves per-scenario line plots for each metric.
    """
    if out_dir is None:
        plots_dir = "plots"
    else:
        plots_dir = os.path.join(out_dir, "plots")
    out_txt_path = os.path.join(plots_dir, "metrics_summary.txt")

    if debug:
        if os.path.exists(plots_dir):
            warnings.warn(f"Directory '{plots_dir}' already exists.")
        else:
            os.makedirs(plots_dir, exist_ok=True)

    scenario_metrics_dfs = []
    scenario_means = []

    for scenario in sorted(predictions_dict):  # Sorted for consistent order
        preds = predictions_dict[scenario]
        trues = true_dict[scenario]

        data = {"timestep": [], "MAE": [], "RMSE": [], "R2": []}
        for t, (p, y) in enumerate(zip(preds, trues)):
            p_flat = np.ravel(p)
            y_flat = np.ravel(y)

            data["timestep"].append(t)
            data["MAE"].append(mean_absolute_error(y_flat, p_flat))
            data["RMSE"].append(mean_squared_error(y_flat, p_flat) ** 0.5)
            # Note: r2_score may warn for constant y; we let the NaN/negative reflect that case.
            data["R2"].append(r2_score(y_flat, p_flat))

        df = pd.DataFrame(data)
        df["scenario"] = scenario
        scenario_metrics_dfs.append(df)

        scenario_means.append(
            {
                "scenario": scenario,
                "MAE": df["MAE"].mean(),
                "RMSE": df["RMSE"].mean(),
                "R2": df["R2"].mean(),
            }
        )

    print(len(scenario_metrics_dfs), "scenarios processed.")

    if debug:
        _ = input("Press Enter to continue...")

    if debug:
        for df in scenario_metrics_dfs:
            scenario = df["scenario"].iloc[0]
            scenario_dir = os.path.join(plots_dir, str(scenario))
            if os.path.exists(scenario_dir):
                warnings.warn(f"Directory '{scenario_dir}' already exists.")
            else:
                os.makedirs(scenario_dir, exist_ok=True)

            for metric in ["MAE", "RMSE", "R2"]:
                plt.figure()
                plt.plot(df["timestep"], df[metric], marker="o")
                plt.title(f"Scenario {scenario} - {metric}")
                plt.xlabel("Timestep")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(os.path.join(scenario_dir, f"{metric}.jpg"))
                plt.close()

    # Overall summaries
    all_df = pd.concat(scenario_metrics_dfs, ignore_index=True)
    micro_mae = all_df["MAE"].mean()
    micro_rmse = all_df["RMSE"].mean()
    micro_r2 = all_df["R2"].mean()

    means_df = pd.DataFrame(scenario_means)
    macro_mae = means_df["MAE"].mean()
    macro_rmse = means_df["RMSE"].mean()
    macro_r2 = means_df["R2"].mean()

    # Write summary txt
    os.makedirs(plots_dir, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Metrics Summary ===\n")
        f.write("\n-- Overall Averages --\n")
        f.write(
            f"Micro (weighted by timesteps): MAE={micro_mae:.6f}, RMSE={micro_rmse:.6f}, R2={micro_r2:.6f}\n"
        )
        f.write(
            f"Macro (mean of per-scenario means): MAE={macro_mae:.6f}, RMSE={macro_rmse:.6f}, R2={macro_r2:.6f}\n"
        )
        f.write("\n-- Per-Scenario Means --\n")
        for row in means_df.sort_values(
            by="scenario", key=lambda s: s.astype(str)
        ).itertuples(index=False):
            f.write(
                f"Scenario {row.scenario}: MAE={row.MAE:.6f}, RMSE={row.RMSE:.6f}, R2={row.R2:.6f}\n"
            )

    print(f"Summary written to: {out_txt_path}")
    return scenario_metrics_dfs


# --- batched TF/AR evaluation passes (impl in predict_batched.py) ---
try:
    from predict_batched import (
        regressive_predictions_absolute_batched,
        autoregressive_predictions_absolute_batched,
        compute_micro_macro,
    )
except Exception as _e:  # keep predict.py importable even if the module is absent
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoregressive inference and evaluation for ABC-Transformer."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the trained model checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "transformer_decoder.yaml"),
        help="Path to the model config YAML (default: configs/transformer_decoder.yaml)."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for evaluation outputs. Default: auto-created under runs/ from the config."
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded config from {args.config}: {config}")

    lit_model = load_weights(config, args.checkpoint)
    lit_model.cuda()

    test_dataset = utils.get_dataset(
        config["data"]["test_data_path"],
        config["data"]["sequence_length"],
        config["data"]["prediction_length"],
        config["data"]["prediction_type"],
    )

    # [Info] Results are written to a new experiment folder
    if args.output_dir is not None:
        exp_dir = Path(args.output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = make_experiment_dir(config, root="runs")

    print("[INFO] exp_dir =", exp_dir)

    # Run autoregressive prediction; write_eval_summary is called inside
    # and writes eval_summary.txt into exp_dir.
    predictions_dict, true_dict = autoregressive_predictions_absolute(
        lit_model,
        test_dataset,
        exp_dir=exp_dir,
    )
