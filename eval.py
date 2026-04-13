from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import torch

from data import (
    sample_clustered_tsp,
    sample_structured_tsp,
    sample_uniform_tsp,
    tour_length,
)
from models import TSPPolicy


@torch.no_grad()
def evaluate_distribution(model, sampler, batch_size: int, n_cities: int, device: torch.device, repeats: int = 10):
    greedy_lengths = []
    sample_lengths = []
    for _ in range(repeats):
        coords = sampler(batch_size, n_cities, device)
        tour_greedy, _, _ = model(coords, decode_type="greedy")
        tour_sample, _, _ = model(coords, decode_type="sample")
        greedy_lengths.append(tour_length(coords, tour_greedy).mean().item())
        sample_lengths.append(tour_length(coords, tour_sample).mean().item())
    return {
        "greedy_mean": sum(greedy_lengths) / len(greedy_lengths),
        "sample_mean": sum(sample_lengths) / len(sample_lengths),
    }


def load_model(ckpt_path: str, model_name: str, device: torch.device) -> TSPPolicy:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = TSPPolicy(
        model_type=model_name,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "transformer", "linear_transformer"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.model, device)

    test_sizes = [20, 50]
    results: Dict[str, Dict[str, float]] = {}

    for n in test_sizes:
        results[f"uniform_n{n}"] = evaluate_distribution(
            model, sample_uniform_tsp, args.batch_size, n, device
        )
        results[f"clustered_n{n}"] = evaluate_distribution(
            model, sample_clustered_tsp, args.batch_size, n, device
        )
        for kind in ["circle", "two_lines", "grid_outliers", "one_cluster_outliers"]:
            results[f"{kind}_n{n}"] = evaluate_distribution(
                model,
                lambda b, nn, d, kind=kind: sample_structured_tsp(b, nn, d, kind=kind),
                args.batch_size,
                n,
                device,
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
