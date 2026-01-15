from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("PyTorch is required for uncertainty estimation (MC dropout).") from exc
    return torch, nn


def enable_dropout_only(model) -> None:
    """
    Enable dropout layers during inference while keeping batch norm in eval mode.

    This is the standard approach for MC Dropout in networks that include BatchNorm.
    """
    torch, nn = _require_torch()
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        if isinstance(module, nn.BatchNorm1d):
            module.eval()


def _predict_loader_once(model, loader, *, device) -> np.ndarray:
    torch, _ = _require_torch()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                inputs = batch[:-1]  # assume last element is y
            else:
                inputs = (batch,)

            inputs = tuple(x.to(device) for x in inputs)
            out = model(*inputs)
            preds.append(out.detach().cpu().numpy().ravel())
    return np.concatenate(preds)


@dataclass(frozen=True)
class McDropoutResult:
    samples: np.ndarray  # shape (n_samples, n_points)
    mean: np.ndarray
    std: np.ndarray

    def interval(self, level: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
        """
        Empirical prediction interval from MC samples.
        """
        alpha = 1.0 - float(level)
        low = np.quantile(self.samples, alpha / 2, axis=0)
        high = np.quantile(self.samples, 1.0 - alpha / 2, axis=0)
        return low, high


def mc_dropout_predict(
    model,
    loader,
    *,
    n_samples: int = 30,
    device: Optional[object] = None,
    seed: int = 42,
) -> McDropoutResult:
    """
    Run MC Dropout inference and return per-point mean/std.
    """
    torch, _ = _require_torch()
    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    enable_dropout_only(model)
    all_samples = []
    for _ in range(int(n_samples)):
        all_samples.append(_predict_loader_once(model, loader, device=device))

    samples = np.stack(all_samples, axis=0)
    return McDropoutResult(samples=samples, mean=samples.mean(axis=0), std=samples.std(axis=0))


def calibration_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    covered = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
    return float(covered.mean())


def nominal_vs_observed_coverage(
    samples: np.ndarray,
    y_true: np.ndarray,
    *,
    levels: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> np.ndarray:
    """
    Return array of shape (len(levels), 2): (nominal, observed) coverage.
    """
    y = np.asarray(y_true, dtype=float)
    out = []
    for lvl in levels:
        alpha = 1.0 - float(lvl)
        lo = np.quantile(samples, alpha / 2, axis=0)
        hi = np.quantile(samples, 1.0 - alpha / 2, axis=0)
        obs = calibration_coverage(y, lo, hi)
        out.append((float(lvl), float(obs)))
    return np.asarray(out, dtype=float)


def coverage_by_uncertainty_bins(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pred_std: np.ndarray,
    *,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Bin by predicted std (quantile bins) and compute observed coverage per bin.
    Returns array of shape (bins, 4): (bin_center_std, n, coverage, mean_abs_err_proxy)
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    std = np.asarray(pred_std, dtype=float)

    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(std)
    y = y[mask]
    lo = lo[mask]
    hi = hi[mask]
    std = std[mask]

    if len(y) == 0:
        return np.empty((0, 4), dtype=float)

    # quantile bins
    edges = np.quantile(std, np.linspace(0, 1, int(n_bins) + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        # not enough spread
        covered = (y >= lo) & (y <= hi)
        return np.asarray([[float(std.mean()), float(len(y)), float(covered.mean()), float(np.mean(np.abs((lo + hi) / 2 - y)))]])

    bins = np.digitize(std, edges[1:-1], right=True)
    rows = []
    for b in range(int(bins.min()), int(bins.max()) + 1):
        m = bins == b
        if not m.any():
            continue
        covered = (y[m] >= lo[m]) & (y[m] <= hi[m])
        center = float(std[m].mean())
        rows.append(
            (
                center,
                float(m.sum()),
                float(covered.mean()),
                float(np.mean(np.abs((lo[m] + hi[m]) / 2 - y[m]))),
            )
        )
    return np.asarray(rows, dtype=float)

