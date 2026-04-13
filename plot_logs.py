from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_csv(path: str):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        y = r.get(key, None)
        if y is not None:
            xs.append(r["step"])
            ys.append(y)
    return xs, ys


def plot_metric(logs, metric: str, outdir: str):
    plt.figure(figsize=(7, 4.5))
    for name, rows in logs.items():
        xs, ys = series(rows, metric)
        if xs:
            plt.plot(xs, ys, label=name)
    plt.xlabel("Training step")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"{metric}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logdir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV logs found in {args.logdir}")

    logs = {os.path.splitext(os.path.basename(p))[0]: load_csv(p) for p in paths}
    outdir = os.path.join(args.logdir, "plots")
    os.makedirs(outdir, exist_ok=True)

    metrics = [
        "train_len_mean",
        "train_reward_mean",
        "train_entropy_mean",
        "grad_norm",
        "reward_var",
        "adv_var",
        "val_len_greedy",
        "val_len_sample",
    ]
    saved = [plot_metric(logs, metric, outdir) for metric in metrics]
    print("Saved plots:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
