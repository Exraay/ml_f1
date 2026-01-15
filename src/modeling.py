from __future__ import annotations

from typing import Dict, Iterable, List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def make_model_pipelines(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Dict[str, Pipeline]:
    """
    Return model pipelines with a shared preprocessor.
    Includes:
    - Linear baseline
    - Regularized linear (Ridge, Lasso)
    - Tree-based ensembles (RandomForest, HistGradientBoosting)
    - Feed-forward MLP
    """
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.001, max_iter=5000, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gboost": HistGradientBoostingRegressor(
            max_depth=None,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate="adaptive",
            max_iter=400,
            early_stopping=True,
            random_state=random_state,
        ),
    }

    pipelines: Dict[str, Pipeline] = {
        name: Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        for name, model in models.items()
    }
    return pipelines
