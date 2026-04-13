from dataclasses import dataclass


@dataclass
class TrainConfig:
    n_cities: int = 20
    batch_size: int = 256
    val_batch_size: int = 512
    steps: int = 5000
    lr: float = 1e-4
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    entropy_coef: float = 1e-3
    grad_clip: float = 1.0
    ema_baseline_beta: float = 0.9
    log_every: int = 50
    val_every: int = 250
    save_every: int = 1000
    seed: int = 42
    device: str = "cuda"
    model: str = "transformer"
