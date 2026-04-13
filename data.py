from __future__ import annotations

import math
from typing import Literal

import torch


def sample_uniform_tsp(batch_size: int, n_cities: int, device: torch.device) -> torch.Tensor:
    """Sample i.i.d. Euclidean TSP instances from [0, 1]^2.

    Returns:
        coords: [B, N, 2]
    """
    return torch.rand(batch_size, n_cities, 2, device=device)


def sample_clustered_tsp(
    batch_size: int,
    n_cities: int,
    device: torch.device,
    num_clusters: int = 3,
    cluster_std: float = 0.06,
) -> torch.Tensor:
    """Sample clustered TSP instances.

    Cities are drawn from several Gaussian clusters whose centers lie in [0,1]^2.
    """
    centers = torch.rand(batch_size, num_clusters, 2, device=device)
    assignments = torch.randint(0, num_clusters, (batch_size, n_cities), device=device)
    coords = centers.gather(1, assignments[..., None].expand(-1, -1, 2))
    coords = coords + cluster_std * torch.randn_like(coords)
    return coords.clamp_(0.0, 1.0)


def sample_structured_tsp(
    batch_size: int,
    n_cities: int,
    device: torch.device,
    kind: Literal["circle", "two_lines", "grid_outliers", "one_cluster_outliers"] = "circle",
) -> torch.Tensor:
    """Construct simple but representative toy distributions."""
    if kind == "circle":
        angles = torch.rand(batch_size, n_cities, device=device) * 2 * math.pi
        radius = 0.35 + 0.03 * torch.randn(batch_size, n_cities, device=device)
        x = 0.5 + radius * torch.cos(angles)
        y = 0.5 + radius * torch.sin(angles)
        coords = torch.stack([x, y], dim=-1)
        return coords.clamp_(0.0, 1.0)

    if kind == "two_lines":
        half = n_cities // 2
        x1 = torch.rand(batch_size, half, device=device) * 0.8 + 0.1
        x2 = torch.rand(batch_size, n_cities - half, device=device) * 0.8 + 0.1
        y1 = torch.full_like(x1, 0.3) + 0.02 * torch.randn_like(x1)
        y2 = torch.full_like(x2, 0.7) + 0.02 * torch.randn_like(x2)
        coords = torch.cat([
            torch.stack([x1, y1], dim=-1),
            torch.stack([x2, y2], dim=-1),
        ], dim=1)
        return coords.clamp_(0.0, 1.0)

    if kind == "grid_outliers":
        grid_side = int((n_cities - 4) ** 0.5)
        grid_side = max(grid_side, 2)
        xs = torch.linspace(0.2, 0.8, grid_side, device=device)
        ys = torch.linspace(0.2, 0.8, grid_side, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        grid = grid[: max(n_cities - 4, 1)]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        grid = grid + 0.01 * torch.randn_like(grid)
        outliers = torch.tensor([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]], device=device)
        outliers = outliers.unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.cat([grid, outliers], dim=1)
        if coords.size(1) < n_cities:
            pad = torch.rand(batch_size, n_cities - coords.size(1), 2, device=device)
            coords = torch.cat([coords, pad], dim=1)
        return coords[:, :n_cities].clamp_(0.0, 1.0)

    if kind == "one_cluster_outliers":
        main = 0.5 + 0.08 * torch.randn(batch_size, n_cities - 3, 2, device=device)
        outliers = torch.tensor([[0.1, 0.1], [0.9, 0.2], [0.8, 0.9]], device=device)
        outliers = outliers.unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.cat([main, outliers], dim=1)
        return coords.clamp_(0.0, 1.0)

    raise ValueError(f"Unknown structured kind: {kind}")


@torch.no_grad()
def tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Compute the total length of a tour.

    Args:
        coords: [B, N, 2]
        tour: [B, N] permutation of city indices
    Returns:
        lengths: [B]
    """
    ordered = coords.gather(1, tour[..., None].expand(-1, -1, 2))
    shifted = torch.roll(ordered, shifts=-1, dims=1)
    return torch.norm(ordered - shifted, dim=-1).sum(dim=1)
