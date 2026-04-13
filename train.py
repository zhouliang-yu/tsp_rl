from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig
from data import sample_uniform_tsp, tour_length
from models import TSPPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMABaseline:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value = None

    def update(self, x: torch.Tensor) -> torch.Tensor:
        mean_x = x.mean().detach()
        if self.value is None:
            self.value = mean_x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * mean_x
        return self.value


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def validate(model: nn.Module, n_cities: int, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    coords = sample_uniform_tsp(batch_size, n_cities, device)
    tour_greedy, _, _ = model(coords, decode_type="greedy")
    tour_sample, _, _ = model(coords, decode_type="sample")
    len_greedy = tour_length(coords, tour_greedy).mean().item()
    len_sample = tour_length(coords, tour_sample).mean().item()
    return {
        "val_len_greedy": len_greedy,
        "val_len_sample": len_sample,
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer", choices=["rnn", "transformer", "linear_transformer"])
    parser.add_argument("--n-cities", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-baseline-beta", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return TrainConfig(
        n_cities=args.n_cities,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        steps=args.steps,
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        ema_baseline_beta=args.ema_baseline_beta,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        model=args.model,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    model = TSPPolicy(
        model_type=cfg.model,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    baseline = EMABaseline(beta=cfg.ema_baseline_beta)

    log_path = os.path.join("logs", f"{cfg.model}_n{cfg.n_cities}_seed{cfg.seed}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "train_reward_mean",
                "train_len_mean",
                "train_entropy_mean",
                "grad_norm",
                "reward_var",
                "adv_var",
                "val_len_greedy",
                "val_len_sample",
            ],
        )
        writer.writeheader()

        for step in range(1, cfg.steps + 1):
            model.train()
            coords = sample_uniform_tsp(cfg.batch_size, cfg.n_cities, device)
            tour, log_probs, entropies = model(coords, decode_type="sample")
            lengths = tour_length(coords, tour)
            rewards = -lengths

            baseline_value = baseline.update(rewards)
            advantage = rewards - baseline_value

            log_prob_sum = log_probs.sum(dim=1)
            entropy_mean = entropies.mean()
            pg_loss = -(advantage.detach() * log_prob_sum).mean()
            entropy_loss = -cfg.entropy_coef * entropy_mean
            loss = pg_loss + entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            optimizer.step()

            row = {
                "step": step,
                "train_reward_mean": rewards.mean().item(),
                "train_len_mean": lengths.mean().item(),
                "train_entropy_mean": entropies.mean().item(),
                "grad_norm": grad_norm,
                "reward_var": rewards.var(unbiased=False).item(),
                "adv_var": advantage.var(unbiased=False).item(),
                "val_len_greedy": "",
                "val_len_sample": "",
            }

            if step % cfg.val_every == 0:
                metrics = validate(model, cfg.n_cities, cfg.val_batch_size, device)
                row.update(metrics)
                print(
                    f"[{cfg.model}] step={step} train_len={row['train_len_mean']:.4f} "
                    f"val_greedy={metrics['val_len_greedy']:.4f} entropy={row['train_entropy_mean']:.4f}"
                )

            if step % cfg.log_every == 0 or step % cfg.val_every == 0:
                writer.writerow(row)
                f.flush()

            if step % cfg.save_every == 0 or step == cfg.steps:
                ckpt_path = os.path.join("checkpoints", f"{cfg.model}_step{step}.pt")
                torch.save(
                    {
                        "config": asdict(cfg),
                        "model_state": model.state_dict(),
                    },
                    ckpt_path,
                )

    print(f"Training finished. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
