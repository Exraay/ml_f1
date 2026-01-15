from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # noqa: BLE001
    raise ImportError("PyTorch is required for src/torch_tabular_models.py") from exc


def default_embedding_dim(cardinality: int, *, emb_multiplier: float = 1.0) -> int:
    base = min(50, max(4, int(round(cardinality**0.5))))
    dim = max(2, int(round(base * float(emb_multiplier))))
    return int(dim)


def init_kaiming(module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EmbeddingWithDropout(nn.Module):
    """
    Drop entire embedding vectors (per feature) during training.
    """

    def __init__(self, num_emb: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(num_emb), int(emb_dim))
        self.dropout = float(dropout)

    def forward(self, x):
        e = self.emb(x)
        if self.dropout > 0 and self.emb.training:
            keep = 1.0 - self.dropout
            mask = torch.bernoulli(torch.full(e.shape[:2], keep, device=e.device))
            e = e * mask.unsqueeze(-1) / keep
        return e


def _activation(name: str):
    name_l = str(name).lower()
    if name_l == "gelu":
        return nn.GELU()
    if name_l == "silu":
        return nn.SiLU()
    return nn.ReLU()


@dataclass(frozen=True)
class ModelOutputSpec:
    out_dim: int = 1  # 1 for point prediction, >1 for multi-quantile, etc.


class TabularEncoderBase(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_numeric = int(n_numeric)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.emb_dims = [default_embedding_dim(c, emb_multiplier=emb_multiplier) for c in self.cat_cardinalities]

        # Use ModuleList of Embedding modules (with optional embedding dropout)
        self.embeddings = nn.ModuleList()
        for card, dim in zip(self.cat_cardinalities, self.emb_dims):
            emb = nn.Embedding(card, dim)
            self.embeddings.append(emb)

        self.emb_dropout = float(emb_dropout)
        self._emb_drop = nn.Dropout(self.emb_dropout) if self.emb_dropout > 0 else None

    def embed(self, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if self._emb_drop is not None:
            embedded = [self._emb_drop(e) for e in embedded]
        return embedded

    @property
    def embedding_out_dim(self) -> int:
        return int(sum(self.emb_dims))


class PlainMLP(nn.Module):
    """
    LayerNorm + GELU MLP for tabular data with categorical embeddings.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.1,
        activation: str = "gelu",
        out_spec: ModelOutputSpec = ModelOutputSpec(1),
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoderBase(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            emb_multiplier=emb_multiplier,
            emb_dropout=emb_dropout,
        )
        self.out_dim = int(out_spec.out_dim)

        in_dim = int(n_numeric) + self.encoder.embedding_out_dim
        act = _activation(activation)
        layers: List[nn.Module] = []
        dim = in_dim
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.LayerNorm(dim),
                    nn.Linear(dim, int(h)),
                    act,
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = int(h)
        layers.extend([nn.LayerNorm(dim), nn.Linear(dim, self.out_dim)])
        self.net = nn.Sequential(*layers)

        self.net.apply(init_kaiming)

    def forward(self, x_num, x_cat):
        emb = self.encoder.embed(x_cat)
        x = torch.cat([x_num] + emb, dim=1)
        out = self.net(x)
        return out


class LegacyBatchNormMLP(nn.Module):
    """
    Baseline matching the originally described architecture:
    Linear -> BatchNorm -> ReLU -> Dropout(0.2) stacks.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.0,
        out_spec: ModelOutputSpec = ModelOutputSpec(1),
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoderBase(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            emb_multiplier=emb_multiplier,
            emb_dropout=emb_dropout,
        )
        self.out_dim = int(out_spec.out_dim)
        if self.out_dim != 1:
            raise ValueError("LegacyBatchNormMLP supports out_dim=1 only.")

        in_dim = int(n_numeric) + self.encoder.embedding_out_dim
        layers: List[nn.Module] = []
        dim = in_dim
        for h in hidden_sizes:
            h = int(h)
            layers.extend(
                [
                    nn.Linear(dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(init_kaiming)

    def forward(self, x_num, x_cat):
        emb = self.encoder.embed(x_cat)
        x = torch.cat([x_num] + emb, dim=1)
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.net.apply(init_kaiming)

    def forward(self, x):
        return x + self.gate * self.net(x)


class ResidualMLP(nn.Module):
    """
    Input projection -> N gated residual blocks -> head.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        width: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.35,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.1,
        out_spec: ModelOutputSpec = ModelOutputSpec(1),
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoderBase(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            emb_multiplier=emb_multiplier,
            emb_dropout=emb_dropout,
        )
        self.out_dim = int(out_spec.out_dim)

        in_dim = int(n_numeric) + self.encoder.embedding_out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList([ResidualBlock(int(width), dropout=float(dropout)) for _ in range(int(n_blocks))])
        self.head = nn.Sequential(
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, self.out_dim),
        )

        self.proj.apply(init_kaiming)
        self.head.apply(init_kaiming)

    def forward(self, x_num, x_cat):
        emb = self.encoder.embed(x_cat)
        x = torch.cat([x_num] + emb, dim=1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TwoTowerMLP(nn.Module):
    """
    Separate towers: categorical context vs numeric dynamics, then merge.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        cat_idx: Sequence[int] = (0, 1, 4),  # default: Driver, Team, Circuit
        width_cat: int = 128,
        width_num: int = 128,
        dropout: float = 0.3,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.1,
        out_spec: ModelOutputSpec = ModelOutputSpec(1),
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoderBase(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            emb_multiplier=emb_multiplier,
            emb_dropout=emb_dropout,
        )
        self.out_dim = int(out_spec.out_dim)
        self.cat_idx = [int(i) for i in cat_idx]

        # Cat tower consumes selected embeddings only
        cat_dim = int(sum(self.encoder.emb_dims[i] for i in self.cat_idx))
        self.cat_tower = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, int(width_cat)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(width_cat)),
            nn.Linear(int(width_cat), int(width_cat)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # Numeric tower
        self.num_tower = nn.Sequential(
            nn.LayerNorm(int(n_numeric)),
            nn.Linear(int(n_numeric), int(width_num)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(width_num)),
            nn.Linear(int(width_num), int(width_num)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        merged = int(width_cat + width_num)
        self.head = nn.Sequential(
            nn.LayerNorm(merged),
            nn.Linear(merged, 128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, self.out_dim),
        )

        self.cat_tower.apply(init_kaiming)
        self.num_tower.apply(init_kaiming)
        self.head.apply(init_kaiming)

    def forward(self, x_num, x_cat):
        emb_all = self.encoder.embed(x_cat)
        cat_vec = torch.cat([emb_all[i] for i in self.cat_idx], dim=1)
        cat_h = self.cat_tower(cat_vec)
        num_h = self.num_tower(x_num)
        x = torch.cat([cat_h, num_h], dim=1)
        return self.head(x)


class _NumericTokenEmbedding(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_token = int(d_token)
        # Per-feature scale and bias to map scalar -> token
        self.weight = nn.Parameter(torch.randn(self.n_features, self.d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.n_features, self.d_token))

    def forward(self, x_num):
        # x_num: (B, n_num)
        x = x_num.unsqueeze(-1)  # (B, n_num, 1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FeatureAttentionMLP(nn.Module):
    """
    Attention over feature tokens (numeric tokens + categorical embedding tokens + CLS).
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        *,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
        emb_multiplier: float = 1.0,
        emb_dropout: float = 0.1,
        out_spec: ModelOutputSpec = ModelOutputSpec(1),
    ) -> None:
        super().__init__()

        self.encoder = TabularEncoderBase(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            emb_multiplier=emb_multiplier,
            emb_dropout=emb_dropout,
        )
        self.out_dim = int(out_spec.out_dim)
        self.d_token = int(d_token)

        self.num_tok = _NumericTokenEmbedding(n_features=int(n_numeric), d_token=self.d_token)
        self.cat_proj = nn.ModuleList([nn.Linear(dim, self.d_token) for dim in self.encoder.emb_dims])
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_token,
            nhead=int(n_heads),
            dim_feedforward=self.d_token * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_token),
            nn.Linear(self.d_token, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, self.out_dim),
        )

        self.cat_proj.apply(init_kaiming)
        self.head.apply(init_kaiming)

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        num_tokens = self.num_tok.forward(x_num)  # (B, n_num, d)

        emb = self.encoder.embed(x_cat)
        cat_tokens = [proj(e).unsqueeze(1) for proj, e in zip(self.cat_proj, emb)]  # each (B, 1, d)
        cat_tokens = torch.cat(cat_tokens, dim=1)  # (B, n_cat, d)

        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=1)
        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0, :]
        return self.head(cls_out)


def build_model(
    arch: str,
    *,
    n_numeric: int,
    cat_cardinalities: Sequence[int],
    out_dim: int = 1,
    hidden_dim: int = 256,
    n_layers: int = 4,
    dropout: float = 0.3,
    emb_multiplier: float = 1.0,
    emb_dropout: float = 0.1,
    activation: str = "gelu",
) -> object:
    """
    Factory for model variants.
    arch: 'baseline', 'residual', 'attention', 'two_tower'
    """
    out_spec = ModelOutputSpec(out_dim=int(out_dim))
    arch_l = str(arch).lower()
    if arch_l in {"legacy"}:
        model = LegacyBatchNormMLP(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            hidden_sizes=(int(hidden_dim), int(hidden_dim // 2), int(hidden_dim // 4)),
            dropout=0.2 if dropout is None else float(dropout),
            emb_multiplier=float(emb_multiplier),
            emb_dropout=0.0,
            out_spec=out_spec,
        )
        return model
    if arch_l in {"baseline", "plain"}:
        model = PlainMLP(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            hidden_sizes=(int(hidden_dim), int(hidden_dim // 2), int(hidden_dim // 4)),
            dropout=float(dropout),
            emb_multiplier=float(emb_multiplier),
            emb_dropout=float(emb_dropout),
            activation=activation,
            out_spec=out_spec,
        )
        return model
    if arch_l in {"residual", "res"}:
        model = ResidualMLP(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            width=int(hidden_dim),
            n_blocks=int(n_layers),
            dropout=float(dropout),
            emb_multiplier=float(emb_multiplier),
            emb_dropout=float(emb_dropout),
            out_spec=out_spec,
        )
        return model
    if arch_l in {"attention", "attn"}:
        model = FeatureAttentionMLP(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            d_token=32 if hidden_dim <= 256 else 48,
            n_heads=4,
            n_layers=max(1, int(n_layers // 2)),
            dropout=float(dropout),
            emb_multiplier=float(emb_multiplier),
            emb_dropout=float(emb_dropout),
            out_spec=out_spec,
        )
        return model
    if arch_l in {"two_tower", "two-tower", "twotower"}:
        model = TwoTowerMLP(
            n_numeric=n_numeric,
            cat_cardinalities=cat_cardinalities,
            width_cat=int(hidden_dim // 2),
            width_num=int(hidden_dim // 2),
            dropout=float(dropout),
            emb_multiplier=float(emb_multiplier),
            emb_dropout=float(emb_dropout),
            out_spec=out_spec,
        )
        return model

    raise ValueError(f"Unknown arch: {arch}")
