from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.log_softmax(logits, dim=dim)


class NodeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.proj(coords)


class RNNEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LinearSelfAttention(nn.Module):
    """A simple kernelized linear attention block.

    This is not intended as an exact reproduction of any specific paper,
    but as a clean, course-project-friendly linear-attention baseline.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q)
        k = self._feature_map(k)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(self.dropout(out))


class LinearTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LinearTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.init_token = nn.Parameter(torch.randn(embed_dim))
        self.step_rnn = nn.GRUCell(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = 1.0 / math.sqrt(embed_dim)

    def forward(
        self,
        enc: torch.Tensor,
        decode_type: str = "sample",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a tour.

        Args:
            enc: [B, N, D]
            decode_type: `sample` or `greedy`

        Returns:
            tour: [B, N]
            log_probs: [B, N]
            entropies: [B, N]
        """
        bsz, n_nodes, dim = enc.shape
        device = enc.device
        keys = self.key_proj(enc)

        mask = torch.ones(bsz, n_nodes, device=device, dtype=torch.bool)
        state = enc.mean(dim=1)
        prev_embed = self.init_token.unsqueeze(0).expand(bsz, -1)

        tours = []
        logps = []
        entropies = []

        for _ in range(n_nodes):
            state = self.step_rnn(prev_embed, state)
            query = self.query_proj(state).unsqueeze(1)
            logits = (query * keys).sum(dim=-1) * self.logit_scale
            logp = masked_log_softmax(logits, mask, dim=-1)
            probs = logp.exp()
            entropy = -(probs * logp.masked_fill(~mask, 0.0)).sum(dim=-1)

            if decode_type == "greedy":
                idx = probs.argmax(dim=-1)
            elif decode_type == "sample":
                idx = torch.distributions.Categorical(probs=probs).sample()
            else:
                raise ValueError(f"Unknown decode_type: {decode_type}")

            step_logp = logp.gather(1, idx[:, None]).squeeze(1)
            tours.append(idx)
            logps.append(step_logp)
            entropies.append(entropy)

            prev_embed = enc.gather(1, idx[:, None, None].expand(-1, 1, dim)).squeeze(1)
            mask.scatter_(1, idx[:, None], False)

        tour = torch.stack(tours, dim=1)
        log_probs = torch.stack(logps, dim=1)
        entropies = torch.stack(entropies, dim=1)
        return tour, log_probs, entropies


class TSPPolicy(nn.Module):
    def __init__(
        self,
        model_type: str,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.model_type = model_type
        self.embed = NodeEmbedding(embed_dim)

        if model_type == "rnn":
            self.encoder = RNNEncoder(embed_dim, num_layers, dropout)
        elif model_type == "transformer":
            self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        elif model_type == "linear_transformer":
            self.encoder = LinearTransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.decoder = PointerDecoder(embed_dim)

    def forward(self, coords: torch.Tensor, decode_type: str = "sample"):
        x = self.embed(coords)
        enc = self.encoder(x)
        return self.decoder(enc, decode_type=decode_type)
