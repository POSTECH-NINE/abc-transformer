"""Reproduce the paper's scenario-level train/test split.

The released OPR1000 TLOCCW split (9,900 train / 1,100 test) is obtained by splitting the
combined, normalized CSV at scenario level with a fixed seed:

    python make_split.py combined_sorted_data_normalization_15min_with_CT_PZ.csv

which writes <input>_train.csv and <input>_test.csv next to the input file
(split=0.9, shuffle=True, random_state=42 — the paper configuration).
"""
import argparse
import os

import numpy as np
import pandas as pd


def train_test_split_by_scenario(df: pd.DataFrame, split: float = 0.9,
                                 shuffle: bool = False, random_state: int = None):
    """Split the dataframe into train and test sets by scenario_number.
    Entire scenarios go either into train or test.
    """
    scenarios = df["scenario_number"].unique()

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(scenarios)

    n_train = int(len(scenarios) * split)

    train_scenarios = scenarios[:n_train]
    test_scenarios = scenarios[n_train:]

    train_df = df[df["scenario_number"].isin(train_scenarios)]
    test_df = df[df["scenario_number"].isin(test_scenarios)]

    return train_df, test_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv", help="combined long-format CSV with a scenario_number column")
    ap.add_argument("--split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    train_df, test_df = train_test_split_by_scenario(
        df, split=args.split, shuffle=not args.no_shuffle, random_state=args.seed)

    base, ext = os.path.splitext(args.input_csv)
    train_path, test_path = base + "_train" + ext, base + "_test" + ext
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"train: {train_df['scenario_number'].nunique()} scenarios -> {train_path}")
    print(f"test:  {test_df['scenario_number'].nunique()} scenarios -> {test_path}")
