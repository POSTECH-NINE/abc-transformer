"""Shared configuration for the layer8 multi-seed sweep.

Fixed model config (per request):
    d_model=64, nhead=4, num_layers=8, dropout=0.1
    prediction_length=1, prediction_type='absolute', target='continuous_y'

Sequence lengths (k) swept to fill Table 7 (teacher-forcing) and Table 8
(autoregressive): k in {3, 10, 30}.  Table 6 (ABC-Transformer row only,
teacher-forcing) corresponds to the k=3 TF column.

Experiment matrix: 4 intervals x 3 seq-lengths x 5 seeds = 60 runs.
Each run is trained, then evaluated with BOTH teacher-forcing and
autoregressive (rollout); across seeds we report mean / std / variance.
"""
import os

# Project root on the A100 server (override with SWEEP_ROOT to run elsewhere).
ROOT = os.environ.get(
    "SWEEP_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

# interval label -> (train_csv, test_csv) relative to ROOT (default: the repo root).
# Place the datasets (available on request) under <repo>/data/<interval>/,
# or point SWEEP_ROOT at the directory that holds them.
INTERVAL_FILES = {
    "60min": ("data/60min/combined_sorted_data_normalization_60min_with_CT_PZ_train.csv",
              "data/60min/combined_sorted_data_normalization_60min_with_CT_PZ_test.csv"),
    "30min": ("data/30min/combined_sorted_data_normalization_30min_with_CT_PZ_train.csv",
              "data/30min/combined_sorted_data_normalization_30min_with_CT_PZ_test.csv"),
    "15min": ("data/15min/combined_sorted_data_normalization_15min_with_CT_PZ_train.csv",
              "data/15min/combined_sorted_data_normalization_15min_with_CT_PZ_test.csv"),
    "5min":  ("data/5min/combined_sorted_data_normalization_5min_with_CT_PZ_train.csv",
              "data/5min/combined_sorted_data_normalization_5min_with_CT_PZ_test.csv"),
}

# Smallest -> largest dataset so quick runs validate the pipeline first.
INTERVALS = ["60min", "30min", "15min", "5min"]
SEQ_LENS = [3, 10, 30]
# Representative seeds commonly used in DL reproducibility studies.
# 42 first so the first full run can be checked against the paper's Table 7/8.
SEEDS = [42, 0, 1, 2, 3]

# Fixed hyperparameters
BACKBONE_KWARGS = dict(
    input_size=20,
    num_continuous=10,
    d_model=64,
    nhead=4,
    num_layers=8,
    dropout=0.1,
)
LIGHTNING_KWARGS = dict(tolerance=0.05, y_type="continuous_y")

PRED_LEN = 1
PREDICTION_TYPE = "absolute"
BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
TRAIN_SPLIT = 0.9

# Where sweep logs/checkpoints/metrics go (under ROOT/src)
LOG_ROOT_NAME = "training_logs_layer8"

# Continuous variable names (first 10 feature columns; order identical across intervals)
VARIABLE_NAMES = [
    "PPS", "TGRCS(10)", "TGRCS(15)", "ZWV", "PSGGEN(1)",
    "ZWDC2SG(1)S", "MAX_CET", "CTMTP", "PZRP", "PZRWL",
]
NUM_CONTINUOUS = 10


def abs_paths(interval):
    tr, te = INTERVAL_FILES[interval]
    return os.path.join(ROOT, tr), os.path.join(ROOT, te)


def run_dir(interval, k, seed):
    return os.path.join(ROOT, "src", LOG_ROOT_NAME, interval, f"seq{k}", f"seed{seed}")
