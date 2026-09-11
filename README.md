# ABC-Transformer

**A**ccident **B**inary–**C**ontinuous Transformer — a decoder-only Transformer surrogate for
severe-accident progression in pressurized water reactors. Given a short window of past
thermal-hydraulic (TH) states and the prescribed schedule of safety-system / SAMG actuations, the
model autoregressively forecasts plant behaviour for horizons of up to 72 hours, thousands of
times faster than the system code (MAAP) it emulates.

Two plants and four accident classes are covered by the documented datasets and released weights:

| Plant | Accident class | Datasets | Weights |
|---|---|---|---|
| **OPR1000** | TLOCCW — total loss of component cooling water | Δt ∈ {5, 15, 30, 60} min | 12 configs (Δt × lookback) + 4-seed set |
| **APR1400** | TLOFW — total loss of feedwater (CSP / ECSBS mitigation variants) | Δt = 5 min, two scalings | 4 configs |
| **APR1400** | LLOCA — large-break LOCA (CSP / ECSBS) | Δt = 5 min | 2 configs |
| **APR1400** | SBO — station blackout | Δt = 5 min | 1 config |

Model weights: **[Google Drive folder](https://drive.google.com/drive/folders/16rnc8fnhlLg5FmHAX1Oq9zS3kiD2mlJz?usp=drive_link)**
(mirrors the `weights/` layout described below; see `weights_manifest.csv` for the full index).

---

## 1. Architecture

Decoder-only Transformer with causal self-attention and temporal attention pooling.
Continuous TH channels and binary actuation channels are embedded jointly (the "hybrid
binary–continuous" input); the readout is an attention-pooled summary of the window.
Training is one-step-ahead (teacher forcing); deployment is autoregressive rollout, feeding each
prediction back while the binary actuation schedule is supplied as known conditioning.

| Series | d_model | heads | layers | params | used for |
|---|---|---|---|---|---|
| OPR1000 (all) | 64 | 4 | 8 | ≈ 0.2 M | paper configuration |
| APR1400 TLOFW | 128 | 8 | 4 | ≈ 1.0 M | later APR1400 studies |
| APR1400 LLOCA | 128 | 8 | 10 | ≈ 2.5 M | later APR1400 studies |
| APR1400 SBO | 128 | 4 | 8 | ≈ 2.0 M | later APR1400 studies |

Exact per-configuration hyperparameters: `weights_manifest.csv` and each folder's `config_used.yaml`.

---

## 2. Datasets

### 2.1 Common format

CSV, long format: one row per `(scenario_number, TIME)`.
`TIME` is in **seconds** (rows advance in steps of the dataset's Δt; raw MAAP dumps advance in
≈ 3610 s increments). Continuous channels are min–max normalized to [0, 1] per dataset. The
min–max bounds needed to recover physical units ship in `weights/_normalization/`; other scaler
statistics (e.g. for the standard-scaled variants) are distributed with the datasets on request.
Binary channels encode component/SAMG state as **0.8 = active, 0.2 = inactive** (0.5 threshold);
they are inputs, not prediction targets.

Scenarios are Monte-Carlo samples over component failure times and operator-action (SAMG)
timings; each scenario file name encodes its sampled parameter vector.

**Data availability**: the MAAP-generated CSVs are not redistributed via git or Drive; they are
available from the corresponding authors on reasonable request (see Citation).

### 2.2 OPR1000 · TLOCCW

11,000 MAAP scenarios (9,900 train / 1,100 test), 72 h horizon. The split is scenario-level,
shuffled with a fixed seed (`numpy.random.default_rng(seed=42)`, 90/10) — reproducible from the
combined CSV with `src/make_split.py`.

**Continuous channels (10)** — prediction targets:

| Column | Description | Unit |
|---|---|---|
| `PPS` | RCS pressure | Pa |
| `TGRCS(10)` | Loop-1 cold-leg gas temperature | K |
| `TGRCS(15)` | Loop-1 hot-leg gas temperature | K |
| `ZWV` | Boiled-up water level from RPV bottom | m |
| `PSGGEN(1)` | SG-1 pressure | Pa |
| `ZWDC2SG(1)S` | SG-1 downcomer water level | m |
| `MAX_CET` | Maximum core-exit temperature | K |
| `CTMTP` | Containment pressure | Pa |
| `PZRP` | Pressurizer pressure | Pa |
| `PZRWL` | Pressurizer water level | m |

Note: `PPS` and `PZRP` are almost perfectly correlated (r = 1.000); the training covariance is
correspondingly ill-conditioned — relevant if you build Mahalanobis-type diagnostics on top.

**Binary channels (10)** — known conditioning inputs:
`RCP_pump`, `HX` (values inverted relative to the others), `HPI`, `LPI`, `CNMT_Spray`, `MDAFW`,
`Charging_pump`, `SAMG_1`, `SAMG_2`, `SAMG_3`. Each is derived from the scenario's sampled
operation / disable / failure times (e.g. `HPI` is active between `HPI_operation` and
`HPI_disabled`; injection cutoffs coincide with RWST depletion).

### 2.3 APR1400 · TLOFW (CSP / ECSBS)

Total-loss-of-feedwater sets in two mitigation variants — containment spray pump (**CSP**) versus
emergency containment spray backup system (**ECSBS**) — at Δt = 5 min. Fifteen input channels:
**10 continuous** (`PPS`, `TGRCS(10)`, `TGRCS(15)`, `PSGGEN(1)`, `ZWDC2SG(1)`, `ZWRB(1)`,
`PEX0(17)`, `TWSG(1)`, `TGRB(17)`, `ZWRB(6)`) and the same **5 SAMG binaries**. Lookback k = 50.
Each variant ships in two scalings — 0.1–0.9 min–max and standard-scaled. The min–max bounds
are in `weights/_normalization/`; the standard-scaler statistics are available on request.

### 2.4 APR1400 · LLOCA and SBO

Large-break LOCA (CSP / ECSBS variants) and station blackout sets, Δt = 5 min, lookback k = 50.
Fourteen input channels: the same 10 continuous channels as TLOFW plus **4 SAMG binaries** (one
fewer than TLOFW — see each `config_used.yaml`). LLOCA additionally includes multi-step
prediction heads trained as an output-horizon ablation (pred_len ∈ {1, 3, 100, 800}); the
released weights cover pred_len = 1.

---

## 3. Released weights

Layout of the Drive folder — download it and place it as `weights/` next to `src/`.
Every leaf folder contains the best checkpoint (`epoch=…-val_loss=….ckpt`, lowest validation loss) and the exact
`config_used.yaml` it was trained with.

```
weights/
├── OPR1000/                                     # TLOCCW (single accident class)
│   ├── dt{05,15,30,60}min_seq{03,10,30}/        # 12 = 4 intervals x 3 lookbacks
│   └── dt15min_seq10_multiseed/seed{0,1,2,42}/  # reproducibility set, headline config
├── APR1400/                                     # one folder per accident type x mitigation
│   ├── TLOFW_CSP/dt05min_seq50_{minmax,std}/
│   ├── TLOFW_ECSBS/dt05min_seq50_{minmax,std}/
│   ├── LLOCA_CSP/seq50_pred1/
│   ├── LLOCA_ECSBS/seq50_pred1/
│   └── SBO/seq50_pred1/
└── _normalization/                              # min-max bounds (OPR1000, TLOFW minmax)
```

Full index with validation losses, architecture fields and file sizes: `weights_manifest.csv`.
Checkpoints are PyTorch-Lightning files; the backbone state dict is under `state_dict` with a
`backbone.` prefix. Checkpoints are small: 2.6–10.2 MB (see `size_MB` in `weights_manifest.csv`).
(The LLOCA multi-step ablation heads — `pred3`/`pred100`/`pred800` — are not part of the release;
available on request.)

**Headline configuration** (used in the RESS paper): `OPR1000/dt15min_seq10`.

---

## 4. Repository layout & usage

```
src/
├── train.py                   # set `selected_model`, point the config at your CSVs
├── predict.py                 # autoregressive inference / evaluation (CLI)
├── predict_batched.py         # batched teacher-forcing / autoregressive evaluation
├── make_split.py              # reproduce the paper's train/test split (scenario-level, seed 42)
├── models/
│   ├── model_lightning.py
│   ├── trnasformer_decoder/   # ABC-Transformer backbone (sic)
│   └── rnn/, lstm/            # paper baseline models (Table 6 rows)
└── configs/                   # transformer_decoder / rnn / lstm YAMLs
```

Requires Python >= 3.10 (developed on 3.11): `pip install -r requirements.txt`.

1. Edit `configs/transformer_decoder.yaml` (data paths, `sequence_length`, backbone kwargs — or
   start from the released `config_used.yaml` of the configuration you want to reproduce).
2. `python train.py` — logs, checkpoints and plots land in `training_logs/`. Baselines: set
   `selected_model` in `train.py` to `rnn` or `lstm`. (The `training.scheduler` entries in the
   configs are informational; `configure_optimizers` in `model_lightning.py` defines the actual
   optimizer — AdamW, lr 1e-3, StepLR step 2 / gamma 0.1.)
3. `python predict.py --checkpoint <path/to/epoch=...ckpt>` for the autoregressive
   rollout + evaluation, or use `predict_batched.py` for full-test-set
   teacher-forcing / autoregressive metrics.

Loading a released checkpoint directly:

```python
import torch, re
from models.trnasformer_decoder.model import SimpleDecoderOnlyTransformer

model = SimpleDecoderOnlyTransformer(input_size=20, num_continuous=10,
                                     d_model=64, nhead=4, num_layers=8, dropout=0.1)
sd = torch.load(CKPT, map_location="cpu")["state_dict"]
model.load_state_dict({re.sub(r"^backbone\.", "", k): v for k, v in sd.items()}, strict=False)
```

(For the APR1400 d128 series use the `backbone_kwargs` from its `config_used.yaml`.)

---

## 5. Reference results (OPR1000 · TLOCCW, headline Δt = 15 min, k = 10)

Averaged over the 10 continuous channels and 1,100 held-out scenarios (normalized units).

| Mode | MAE | RMSE |
|---|---|---|
| Teacher forcing (one-step) | 0.0030 | 0.0055 |
| Autoregressive rollout (72 h) | 0.0413 | 0.0697 |

Per-scenario R² over the rollout: median 0.966; 1.2 % of scenarios fall below R² = 0.

Seed sensitivity (4 seeds, identical protocol): teacher forcing is seed-stable to four decimals
(0.0030 ± 0.0000 / 0.0055 ± 0.0000) while the autoregressive metrics vary run-to-run
(0.0467 ± 0.0026 / 0.0764 ± 0.0033) — single-seed AR numbers should be read with that spread in
mind. The four seeds are released in `dt15min_seq10_multiseed/`.

### Metric definitions

With C = 10 continuous channels, per scenario (T rollout steps):

```
MAE   = 1/(T·C) · Σ_t Σ_c |ŷ_{t,c} − y_{t,c}|
RMSE  = 1/T · Σ_t sqrt( 1/C · Σ_c (ŷ_{t,c} − y_{t,c})² )      # root inside the time average
R²(t) = 1 − Σ_c (ŷ_{t,c} − y_{t,c})² / Σ_c (y_{t,c} − ȳ_t)² ,  ȳ_t = 1/C · Σ_c y_{t,c}
R²    = 1/T · Σ_t R²(t)
```

Table values average the per-scenario numbers over scenarios ("macro"). Note the R² baseline is
the cross-channel mean at each instant, not a per-channel time mean.

---

## 6. Scope and caveats

- Each model is a **single-accident-class emulator** for its plant; no cross-accident or
  cross-plant generalization is claimed.
- Binary actuation schedules are **supplied, not predicted**: the task is conditional trajectory
  emulation given a prescribed mitigation sequence.
- Autoregressive error grows with horizon (exposure bias); reliability should be assessed
  per scenario (distribution-level), not from aggregate means alone.
- Normalized errors translate to physical units via the min–max bounds in
  `weights/_normalization/` (standard-scaled variants: statistics on request).

## License

Code and released weights are distributed under the MIT License (see `LICENSE`).

## Citation

If you use this code or the weights, please cite (see `CITATION.cff`):

> W. Jeong, S. Khanal, J. Lee, S. Lee, J. Jeon, *Enhancing Resolution and Reliability
> with Attention Mechanisms in Nuclear Accident Surrogate Modeling*,
> Reliability Engineering & System Safety (under revision — full citation on acceptance).
