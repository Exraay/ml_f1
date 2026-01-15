from __future__ import annotations

import random
from typing import Dict, Iterable, List

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
        batch_norm: bool = True,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 128,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

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
        layers = []
        in_features = n_features
        for hidden_units in self.hidden_layers:
            layers.append(nn_module.Linear(in_features, hidden_units))
            if self.batch_norm:
                layers.append(nn_module.BatchNorm1d(hidden_units))
            layers.append(nn_module.ReLU())
            layers.append(nn_module.Dropout(self.dropout))
            in_features = hidden_units
        layers.append(nn_module.Linear(in_features, 1))
        return nn_module.Sequential(*layers)

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):  # noqa: D401
        torch, nn_module, DataLoader, TensorDataset = self._require_torch()
        set_global_seed(self.random_state)
        torch.manual_seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        device = torch.device(self.device) if self.device else torch.device("cpu")
        dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model(X_np.shape[1], nn_module)
        self.model_.to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn_module.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = self.model_(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):  # noqa: D401
        torch, _, _, _ = self._require_torch()
        X_np = self._to_numpy(X)
        device = torch.device(self.device) if self.device else torch.device("cpu")
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.from_numpy(X_np).to(device)).cpu().numpy().ravel()
        return preds


def make_linear_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
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
) -> GridSearchCV:
    try:
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "xgboost is required for the tuned ensemble model. Install xgboost to use it."
        ) from exc

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    param_grid = {
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    cv = TimeSeriesSplit(n_splits=3)
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )


def make_mlp_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = TorchMLPRegressor(random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_model_registry(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Dict[str, object]:
    """
    Return model objects with a shared interface.
    """
    return {
        "linear": make_linear_pipeline(numeric_features, categorical_features),
        "random_forest": make_random_forest_pipeline(
            numeric_features, categorical_features, random_state
        ),
        "xgboost": make_xgboost_search(numeric_features, categorical_features, random_state),
        "mlp": make_mlp_pipeline(numeric_features, categorical_features, random_state),
    }
