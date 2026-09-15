"""Scheduled-sampling training for the ABC-Transformer (exposure-bias mitigation).

Identical to the teacher-forcing baseline (same architecture, optimizer, data
pipeline and seed) except that, with an annealed per-sample probability p, the
most recent continuous input row is replaced by the model's own detached
one-step prediction (Bengio et al., NeurIPS 2015). The prediction target and
loss are unchanged; validation uses pure teacher forcing.

Released scheduled-sampling weights (OPR1000_SS/ on the weights Drive) were
trained with this recipe: p annealed 0 -> 0.5 over 8 epochs then held, up to
25 epochs with early stopping (patience 6), batch 256, AdamW lr 1e-3 with
StepLR(5, 0.2), seed 0.

Usage:
    python train_ss.py [--config configs/transformer_decoder.yaml]
                       [--p-max 0.5] [--anneal-epochs 8] [--epochs 25]
                       [--out-dir training_logs_ss]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from dataset import BaseDataset, TransformerDataset
from model_selector import ModelSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(Path(__file__).parent / "configs" / "transformer_decoder.yaml"))
    ap.add_argument("--p-max", type=float, default=0.5,
                    help="upper bound of the scheduled-sampling probability")
    ap.add_argument("--anneal-epochs", type=int, default=8,
                    help="epochs over which p rises linearly from 0 to p-max")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="training_logs_ss")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg = cfg["data"], cfg["model"]
    nc = mcfg["backbone_kwargs"]["num_continuous"]
    seq_len = dcfg["sequence_length"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.out_dir,
                           f"{mcfg['name']}_ss_seq{seq_len}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(0)
    base = BaseDataset([dcfg["data_path"]], seq_len=seq_len, pred_len=1,
                       log_dir=os.path.join(run_dir, "logs"),
                       cache_dir=os.path.join(run_dir, "cache"))
    ds = TransformerDataset(base, dataset_type=dcfg["prediction_type"])
    n = len(ds); nval = int(0.1 * n)
    tr, va = random_split(ds, [n - nval, nval],
                          generator=torch.Generator().manual_seed(0))
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                       pin_memory=True, persistent_workers=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=1024, shuffle=False, num_workers=2,
                       pin_memory=True, persistent_workers=True)
    print(f"[data] windows={n:,} train={n-nval:,} val={nval:,} seq_len={seq_len}")

    backbone, _ = ModelSelector(mcfg["name"],
                                backbone_kwargs=mcfg["backbone_kwargs"],
                                lightning_kwargs=mcfg.get("lightning_kwargs", {}))
    model = backbone.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.2)
    crit = nn.MSELoss()
    best_val, best_ep, bad = float("inf"), -1, 0
    best_path = os.path.join(run_dir, "ss_best.pt")

    for ep in range(args.epochs):
        p = min(args.p_max, args.p_max * ep / max(1, args.anneal_epochs))
        model.train(); tl = 0.0; nb = 0; t0 = time.time()
        for batch in dl_tr:
            x = batch["past_values"].float().to(dev)
            y = batch["continuous_y"].float().to(dev)
            if p > 0:
                # preview the model's own one-step prediction for the last row
                with torch.no_grad():
                    pred_last = model(x[:, :-1, :])
                mask = (torch.rand(x.size(0), device=dev) < p).float().unsqueeze(1)
                x = x.clone()
                x[:, -1, :nc] = mask * pred_last + (1 - mask) * x[:, -1, :nc]
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        sch.step()

        model.eval(); vl = 0.0; vnb = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["past_values"].float().to(dev)
                y = batch["continuous_y"].float().to(dev)
                vl += crit(model(x), y).item(); vnb += 1
        va_l = vl / max(vnb, 1)
        print(f"epoch {ep:2d}  p={p:.2f}  train={tl/max(nb,1):.6f}  "
              f"val={va_l:.6f}  ({time.time()-t0:.0f}s)")
        if va_l < best_val:
            best_val, best_ep, bad = va_l, ep, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ep{ep}"); break

    print(f"[done] best val={best_val:.6f} @ep{best_ep} -> {best_path}")
    json.dump({"best_val_mse": best_val, "best_epoch": best_ep,
               "p_max": args.p_max, "anneal_epochs": args.anneal_epochs,
               "seq_len": seq_len},
              open(os.path.join(run_dir, "train_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
