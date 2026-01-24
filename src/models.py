from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_global_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)


def make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ]
    )


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorch MLP Regressor with advanced training features.

    Features:
    - Configurable hidden layer architecture (default: 4 layers)
    - Batch normalization and dropout for regularization
    - Early stopping to prevent overfitting
    - Learning rate scheduling (ReduceLROnPlateau)
    - Gradient clipping for training stability
    - Validation-based model selection

    Args:
        hidden_layers: Tuple of hidden layer sizes.
        dropout: Dropout probability (0-1).
        batch_norm: Whether to use batch normalization.
        lr: Initial learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum improvement for early stopping.
        weight_decay: L2 regularization strength.
        lr_scheduler: Whether to use learning rate scheduling.
        validation_fraction: Fraction of training data for validation.
        random_state: Random seed.
        device: PyTorch device ("cpu", "cuda", or None for auto).
        verbose: Print training progress.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (256, 128, 64, 32),
        dropout: float = 0.3,
        batch_norm: bool = True,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        min_delta: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: bool = True,
        validation_fraction: float = 0.1,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = False,
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

    def _require_torch(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "PyTorch is required for TorchMLPRegressor. Install torch to use this model."
            ) from exc
        return torch, nn, DataLoader, TensorDataset

    def _build_model(self, n_features: int, nn_module):
        """Build MLP with BatchNorm, ReLU, and Dropout."""
        layers = []
        in_features = n_features

        for i, hidden_units in enumerate(self.hidden_layers):
            layers.append(nn_module.Linear(in_features, hidden_units))
            if self.batch_norm:
                layers.append(nn_module.BatchNorm1d(hidden_units))
            layers.append(nn_module.LeakyReLU(0.1))  # LeakyReLU for better gradients
            # Reduce dropout in later layers
            drop_rate = self.dropout * (1 - i / len(self.hidden_layers) * 0.5)
            layers.append(nn_module.Dropout(drop_rate))
            in_features = hidden_units

        # Output layer
        layers.append(nn_module.Linear(in_features, 1))
        return nn_module.Sequential(*layers)

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        """Fit the MLP regressor with early stopping and LR scheduling."""
        torch, nn_module, DataLoader, TensorDataset = self._require_torch()
        set_global_seed(self.random_state)
        torch.manual_seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # Train/validation split
        n_samples = len(X_np)
        n_val = max(1, int(n_samples * self.validation_fraction))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        # Setup device
        if self.device:
            device = torch.device(self.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_tensor_x = torch.from_numpy(X_val).to(device)
        val_tensor_y = torch.from_numpy(y_val).to(device)

        # Build model
        self.model_ = self._build_model(X_np.shape[1], nn_module)
        self.model_.to(device)

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = None
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
            )

        loss_fn = nn_module.SmoothL1Loss()  # Huber loss for robustness

        # Early stopping
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        self.training_history_ = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                preds = self.model_(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            self.model_.eval()
            with torch.no_grad():
                val_preds = self.model_(val_tensor_x)
                val_loss = loss_fn(val_preds, val_tensor_y).item()

            # Record history
            current_lr = optimizer.param_groups[0]["lr"]
            self.training_history_["train_loss"].append(avg_train_loss)
            self.training_history_["val_loss"].append(val_loss)
            self.training_history_["lr"].append(current_lr)

            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state:
            self.model_.load_state_dict(best_model_state)
            self.model_.to(device)

        self.best_val_loss_ = best_val_loss
        self.n_epochs_trained_ = epoch + 1

        return self

    def predict(self, X):
        """Predict using the trained MLP."""
        torch, _, _, _ = self._require_torch()
        X_np = self._to_numpy(X)

        if self.device:
            device = torch.device(self.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_.eval()
        with torch.no_grad():
            preds = (
                self.model_(torch.from_numpy(X_np).to(device)).cpu().numpy().ravel()
            )
        return preds

    def get_training_history(self) -> Dict[str, List[float]]:
        """Return training history for plotting."""
        return getattr(self, "training_history_", {})


def make_linear_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    regularization: Literal["none", "ridge", "elasticnet"] = "ridge",
    alpha: float = 1.0,
) -> Pipeline:
    """
    Create linear regression pipeline with optional regularization.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        regularization: Type of regularization ("none", "ridge", "elasticnet").
        alpha: Regularization strength (higher = more regularization).

    Returns:
        Sklearn Pipeline with preprocessor and linear model.
    """
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    if regularization == "ridge":
        model = Ridge(alpha=alpha)
    elif regularization == "elasticnet":
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000)
    else:
        model = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def make_random_forest_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_xgboost_search(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    n_splits: int = 5,
    search_mode: Literal["quick", "full"] = "full",
) -> GridSearchCV:
    """
    Create XGBoost GridSearchCV with TimeSeriesSplit cross-validation.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        random_state: Random seed for reproducibility.
        n_splits: Number of CV splits for TimeSeriesSplit.
        search_mode: "quick" for fast search, "full" for comprehensive search.

    Returns:
        GridSearchCV object ready for fitting.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "xgboost is required for the tuned ensemble model. Install xgboost to use it."
        ) from exc

    preprocessor = make_preprocessor(numeric_features, categorical_features)

    # Base model with early stopping support
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",  # Faster training
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    if search_mode == "quick":
        # Quick search for initial exploration
        param_grid = {
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__min_child_weight": [1, 3],
        }
    else:
        # Full search for optimal performance
        param_grid = {
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_alpha": [0, 0.1, 0.5],  # L1 regularization
            "model__reg_lambda": [1, 2, 5],  # L2 regularization
        }

    cv = TimeSeriesSplit(n_splits=n_splits)

    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def make_xgboost_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    **xgb_params: Any,
) -> Pipeline:
    """
    Create XGBoost pipeline with specified parameters (no GridSearch).

    Useful for using best parameters found from GridSearchCV.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        random_state: Random seed for reproducibility.
        **xgb_params: Additional XGBoost parameters.

    Returns:
        Sklearn Pipeline with preprocessor and XGBoost model.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "xgboost is required. Install xgboost to use it."
        ) from exc

    preprocessor = make_preprocessor(numeric_features, categorical_features)

    default_params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 2,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    default_params.update(xgb_params)

    model = xgb.XGBRegressor(**default_params)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_mlp_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    architecture: Literal["standard", "deep", "wide"] = "deep",
    **mlp_params: Any,
) -> Pipeline:
    """
    Create MLP pipeline with configurable architecture.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        random_state: Random seed for reproducibility.
        architecture: Predefined architecture ("standard", "deep", "wide").
        **mlp_params: Additional parameters for TorchMLPRegressor.

    Returns:
        Sklearn Pipeline with preprocessor and MLP model.
    """
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    # Predefined architectures optimized for F1 lap time prediction
    architectures = {
        "standard": {
            "hidden_layers": (128, 64, 32),
            "dropout": 0.2,
            "epochs": 100,
        },
        "deep": {
            # Deep network for capturing complex tire/fuel interactions
            "hidden_layers": (256, 128, 64, 32),
            "dropout": 0.3,
            "epochs": 200,
            "patience": 25,
        },
        "wide": {
            # Wide network for learning feature interactions
            "hidden_layers": (512, 256, 128),
            "dropout": 0.4,
            "epochs": 150,
        },
    }

    model_params = {
        "random_state": random_state,
        "batch_norm": True,
        "lr_scheduler": True,
        "verbose": False,
    }
    model_params.update(architectures.get(architecture, architectures["deep"]))
    model_params.update(mlp_params)

    model = TorchMLPRegressor(**model_params)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_model_registry(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
    xgboost_mode: Literal["quick", "full", "tuned"] = "tuned",
    mlp_architecture: Literal["standard", "deep", "wide"] = "deep",
) -> Dict[str, object]:
    """
    Create model registry with all available models.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        random_state: Random seed for reproducibility.
        xgboost_mode: "quick" for fast search, "full" for GridSearch, "tuned" for preset.
        mlp_architecture: MLP architecture preset ("standard", "deep", "wide").

    Returns:
        Dictionary mapping model names to fitted pipelines/estimators.
    """
    registry = {
        "linear_ols": make_linear_pipeline(
            numeric_features, categorical_features, regularization="none"
        ),
        "linear_ridge": make_linear_pipeline(
            numeric_features, categorical_features, regularization="ridge", alpha=1.0
        ),
        "random_forest": make_random_forest_pipeline(
            numeric_features, categorical_features, random_state
        ),
        "mlp": make_mlp_pipeline(
            numeric_features, categorical_features, random_state,
            architecture=mlp_architecture
        ),
    }

    # XGBoost: GridSearch or pre-tuned pipeline
    if xgboost_mode == "tuned":
        registry["xgboost"] = make_xgboost_pipeline(
            numeric_features, categorical_features, random_state
        )
    else:
        registry["xgboost"] = make_xgboost_search(
            numeric_features, categorical_features, random_state,
            search_mode=xgboost_mode
        )

    return registry


def make_comparison_models(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Dict[str, object]:
    """
    Create minimal model set for scientific comparison.

    Returns only:
    - Linear Regression (Ridge regularized)
    - XGBoost (pre-tuned)
    - MLP (deep architecture)

    This is the core comparison for the thesis.
    """
    return {
        "Linear Regression": make_linear_pipeline(
            numeric_features, categorical_features, regularization="ridge"
        ),
        "XGBoost": make_xgboost_pipeline(
            numeric_features, categorical_features, random_state
        ),
        "Deep MLP": make_mlp_pipeline(
            numeric_features, categorical_features, random_state, architecture="deep"
        ),
    }
