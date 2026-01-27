from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


def set_global_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorch MLP Regressor with early stopping and LR scheduling.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (256, 128, 64, 32),
        dropout: float = 0.3,
        batch_norm: bool = True,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 256,
        patience: int = 20,
        min_delta: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: bool = True,
        validation_fraction: float = 0.1,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool | int = False,
        log_every: int = 1,
        log_batch_every: int = 0,
        live_plot_every: int = 0,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.log_every = log_every
        self.log_batch_every = log_batch_every
        self.live_plot_every = live_plot_every

    def _require_torch(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except (ImportError, OSError) as exc:  # noqa: BLE001
            raise ImportError(
                "PyTorch is required for TorchMLPRegressor. Install torch to use this model."
            ) from exc
        return torch, nn, DataLoader, TensorDataset

    def _build_model(self, n_features: int, nn_module):
        layers = []
        in_features = n_features
        for i, hidden_units in enumerate(self.hidden_layers):
            layers.append(nn_module.Linear(in_features, hidden_units))
            if self.batch_norm:
                layers.append(nn_module.BatchNorm1d(hidden_units))
            layers.append(nn_module.LeakyReLU(0.1))
            drop_rate = self.dropout * (1 - i / len(self.hidden_layers) * 0.5)
            layers.append(nn_module.Dropout(drop_rate))
            in_features = hidden_units
        layers.append(nn_module.Linear(in_features, 1))
        return nn_module.Sequential(*layers)

    def _select_device(self, torch):
        if self.device:
            try:
                device = torch.device(self.device)
                if device.type == "cuda":
                    try:
                        _ = torch.cuda.current_device()
                    except Exception as exc:  # noqa: BLE001
                        print(f"CUDA requested but failed to initialize ({exc}); falling back to CPU.")
                        return torch.device("cpu")
                return device
            except Exception as exc:  # noqa: BLE001
                print(f"Requested device '{self.device}' failed ({exc}); falling back to CPU.")
                return torch.device("cpu")

        try:
            if torch.cuda.is_available():
                try:
                    _ = torch.cuda.current_device()
                    return torch.device("cuda")
                except Exception as exc:  # noqa: BLE001
                    print(f"CUDA available but failed to initialize ({exc}); falling back to CPU.")
                    return torch.device("cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"CUDA check failed ({exc}); falling back to CPU.")

        return torch.device("cpu")

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        torch, nn_module, DataLoader, TensorDataset = self._require_torch()
        set_global_seed(self.random_state)
        torch.manual_seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_samples = len(X_np)
        n_val = max(1, int(n_samples * self.validation_fraction))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        device = self._select_device(torch)
        self.device_ = str(device)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_tensor_x = torch.from_numpy(X_val).to(device)
        val_tensor_y = torch.from_numpy(y_val).to(device)

        self.model_ = self._build_model(X_np.shape[1], nn_module)
        self.model_.to(device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = None
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
            )

        loss_fn = nn_module.SmoothL1Loss()
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        self.training_history_ = {"train_loss": [], "val_loss": [], "lr": []}

        # Progress bar if verbose
        use_tqdm = bool(self.verbose)
        if use_tqdm:
            try:
                from tqdm.auto import tqdm
                epoch_iter = tqdm(range(self.epochs), desc="MLP epochs", leave=True)
            except Exception:  # noqa: BLE001
                epoch_iter = range(self.epochs)
                use_tqdm = False
        else:
            epoch_iter = range(self.epochs)

        if self.verbose:
            device_msg = f"Using device: {self.device_}"
            if torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0)
                    device_msg = f"{device_msg} ({device_name})"
                except Exception:  # noqa: BLE001
                    device_msg = f"{device_msg} (CUDA available)"
            if use_tqdm:
                try:
                    from tqdm.auto import tqdm as _tqdm
                    _tqdm.write(device_msg)
                except Exception:  # noqa: BLE001
                    print(device_msg)
            else:
                print(device_msg)

        live_plot = None
        if self.live_plot_every and self.verbose:
            try:
                import plotly.graph_objects as go
                from IPython.display import clear_output, display
                live_plot = {"go": go, "clear_output": clear_output, "display": display}
            except Exception:  # noqa: BLE001
                live_plot = None

        try:
            for epoch in epoch_iter:
                self.model_.train()
                train_losses = []
                for batch_idx, (batch_x, batch_y) in enumerate(train_loader, start=1):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    preds = self.model_(batch_x)
                    loss = loss_fn(preds, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())
                    if self.verbose and self.log_batch_every and (batch_idx % int(self.log_batch_every) == 0):
                        msg = f"  batch {batch_idx:04d} | loss={loss.item():.4f}"
                        if use_tqdm:
                            try:
                                from tqdm.auto import tqdm as _tqdm
                                _tqdm.write(msg)
                            except Exception:  # noqa: BLE001
                                print(msg)
                        else:
                            print(msg)

                avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

                self.model_.eval()
                with torch.no_grad():
                    val_preds = self.model_(val_tensor_x)
                    val_loss = loss_fn(val_preds, val_tensor_y).item()

                if scheduler:
                    scheduler.step(val_loss)

                current_lr = optimizer.param_groups[0]["lr"]
                self.training_history_["train_loss"].append(avg_train_loss)
                self.training_history_["val_loss"].append(val_loss)
                self.training_history_["lr"].append(current_lr)

                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if use_tqdm and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        train=f"{avg_train_loss:.4f}",
                        val=f"{val_loss:.4f}",
                        lr=f"{current_lr:.1e}",
                    )
                    if isinstance(self.verbose, int) and self.verbose >= 2:
                        from tqdm.auto import tqdm as _tqdm
                        if self.log_every and (epoch + 1) % int(self.log_every) == 0:
                            _tqdm.write(
                                f"Epoch {epoch + 1}/{self.epochs} - "
                                f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.1e}"
                            )
                elif self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.1e}"
                    )

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if live_plot and self.live_plot_every and (epoch + 1) % int(self.live_plot_every) == 0:
                    go = live_plot["go"]
                    fig = go.Figure()
                    fig.add_scatter(
                        x=list(range(1, len(self.training_history_["train_loss"]) + 1)),
                        y=self.training_history_["train_loss"],
                        mode="lines",
                        name="train",
                    )
                    fig.add_scatter(
                        x=list(range(1, len(self.training_history_["val_loss"]) + 1)),
                        y=self.training_history_["val_loss"],
                        mode="lines",
                        name="val",
                    )
                    fig.update_layout(
                        title="Deep MLP Training (live)",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=360,
                        template="plotly_white",
                    )
                    live_plot["clear_output"](wait=True)
                    live_plot["display"](fig)
        finally:
            if use_tqdm and hasattr(epoch_iter, "close"):
                epoch_iter.close()

        if best_model_state:
            self.model_.load_state_dict(best_model_state)
            self.model_.to(device)

        self.best_val_loss_ = best_val_loss
        return self

    def predict(self, X):
        torch, _, _, _ = self._require_torch()
        X_np = self._to_numpy(X)

        device = self._select_device(torch)

        self.model_.eval()
        with torch.no_grad():
            preds = (
                self.model_(torch.from_numpy(X_np).to(device)).cpu().numpy().ravel()
            )
        return preds


def make_linear_preprocessor(numeric_features: Iterable[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ]
    )


def make_mlp_preprocessor(numeric_features: Iterable[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_tree_preprocessor(numeric_features: Iterable[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


def make_pipeline(
    model: object,
    numeric_features: Iterable[str],
    *,
    preprocessor: Literal["auto", "linear", "mlp", "tree"] = "auto",
) -> Pipeline:
    if preprocessor == "auto":
        if model.__class__.__name__ in {"XGBRegressor", "XGBRFRegressor"} or model.__class__.__module__.startswith("xgboost"):
            preprocessor = "tree"
        elif isinstance(model, TorchMLPRegressor):
            preprocessor = "mlp"
        else:
            preprocessor = "linear"

    if preprocessor == "tree":
        prep = make_tree_preprocessor(numeric_features)
    elif preprocessor == "mlp":
        prep = make_mlp_preprocessor(numeric_features)
    else:
        prep = make_linear_preprocessor(numeric_features)

    return Pipeline(steps=[("preprocess", prep), ("model", model)])


def make_linear_model(random_state: int, *, alpha: float | None = None) -> Ridge:
    return Ridge(alpha=1.0 if alpha is None else float(alpha), random_state=random_state)


def make_xgboost_model(random_state: int, **params: Any) -> object:
    try:
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("xgboost is required for XGBoost model.") from exc

    defaults = {
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    defaults.update(params)
    return xgb.XGBRegressor(**defaults)


def make_mlp_model(
    random_state: int,
    *,
    verbose: bool | int = True,
    log_every: int = 1,
    log_batch_every: int = 0,
    live_plot_every: int = 0,
    hidden_layers: tuple[int, ...] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    dropout: float | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    device: str | None = None,
) -> TorchMLPRegressor:
    return TorchMLPRegressor(
        hidden_layers=hidden_layers or (256, 128, 64, 32),
        dropout=0.3 if dropout is None else float(dropout),
        epochs=150 if epochs is None else int(epochs),
        batch_size=256 if batch_size is None else int(batch_size),
        patience=20,
        lr=1e-3 if lr is None else float(lr),
        weight_decay=1e-4 if weight_decay is None else float(weight_decay),
        random_state=random_state,
        device=device,
        verbose=verbose,
        log_every=log_every,
        log_batch_every=log_batch_every,
        live_plot_every=live_plot_every,
    )


def make_model_registry(
    numeric_features: List[str],
    *,
    random_state: int,
) -> Dict[str, Pipeline]:
    return {
        "Linear": make_pipeline(make_linear_model(random_state), numeric_features),
        "XGBoost": make_pipeline(make_xgboost_model(random_state), numeric_features),
        "Deep MLP": make_pipeline(make_mlp_model(random_state), numeric_features),
    }


def build_search(
    model_name: str,
    pipeline: Pipeline,
    *,
    random_state: int,
    mode: Literal["off", "fast", "full"] = "fast",
    param_grid: Dict[str, List[Any]] | None = None,
    n_splits: int = 4,
    n_iter: int | None = None,
    cv_splits: int | None = None,
    search_verbose: int = 1,
) -> object:
    if mode == "off":
        return pipeline

    cv = TimeSeriesSplit(n_splits=cv_splits or n_splits)

    if model_name == "Linear":
        default_param_grid = {
            "model__alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
            "preprocess__poly__degree": [1, 2],
        }
    elif model_name == "XGBoost":
        default_param_grid = {
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0.0, 0.5, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [1.0, 2.0, 5.0],
            "model__n_estimators": [300, 600, 900],
        }
    elif model_name == "Deep MLP":
        default_param_grid = {
            "model__hidden_layers": [(128, 64, 32), (256, 128, 64), (256, 128, 64, 32)],
            "model__dropout": [0.2, 0.3, 0.4],
            "model__lr": [5e-4, 1e-3, 2e-3],
            "model__batch_size": [128, 256],
            "model__epochs": [100, 150],
            "model__weight_decay": [0.0, 1e-4, 5e-4],
        }
    else:
        return pipeline

    param_grid = param_grid or default_param_grid

    if mode == "full":
        return GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=search_verbose,
            refit=True,
        )

    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter or min(12, sum(len(v) for v in param_grid.values())),
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=random_state,
        verbose=search_verbose,
        refit=True,
    )
