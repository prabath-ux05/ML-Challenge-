"""LightGBM entity matching binary classifier implementation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd

from business_entity_resolution.features.pair_features import ALL_FEATURE_NAMES
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("matching.model")


class EntityMatchClassifier:
    """Configurable LightGBM pairwise match classifier for business entity resolution.

    Optimized for ranking and thresholding on extreme class imbalance with
    custom decision thresholds and F0.5 optimization.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        feature_names: Optional[List[str]] = None,
    ):
        self.params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "objective": "binary",
            "verbosity": -1,
        }
        self.feature_names = feature_names or ALL_FEATURE_NAMES
        self.model: Optional[lgb.LGBMClassifier] = None
        self.booster_: Optional[lgb.Booster] = None
        self.best_threshold_: float = 0.5
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[List[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]]] = None,
        early_stopping_rounds: Optional[int] = 30,
        verbose: bool = False,
    ) -> "EntityMatchClassifier":
        """Fit the LightGBM classifier on candidate pair features.

        Args:
            X: Training feature matrix (n_samples, n_features).
            y: Binary match labels (0 or 1).
            eval_set: Optional validation tuple(s) [(X_val, y_val)] for monitoring.
            early_stopping_rounds: Number of rounds for early stopping on eval_set.
            verbose: LightGBM training verbosity.

        Returns:
            self
        """
        callbacks = []
        if eval_set is not None and early_stopping_rounds is not None and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))

        self.model = lgb.LGBMClassifier(**self.params)

        X_input = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        eval_tuples = None
        if eval_set is not None:
            eval_tuples = []
            for ev_x, ev_y in eval_set:
                ev_x_inp = ev_x[self.feature_names] if isinstance(ev_x, pd.DataFrame) else ev_x
                eval_tuples.append((ev_x_inp, ev_y))

        logger.info(
            "Fitting LightGBM classifier: %d samples, %d features, pos_ratio=%.3f",
            len(X_input),
            X_input.shape[1],
            float(np.mean(y)),
        )

        self.model.fit(
            X_input,
            y,
            eval_set=eval_tuples,
            eval_metric="binary_logloss",
            callbacks=callbacks if callbacks else None,
        )
        self.booster_ = self.model.booster_
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict match probability for candidate pairs.

        Args:
            X: Feature matrix.

        Returns:
            1D array of match probabilities (class 1).
        """
        if not self.is_fitted_ or self.booster_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_input = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        probas = self.booster_.predict(X_input)
        return probas.astype(np.float32)

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary match labels at specified or optimal threshold.

        Args:
            X: Feature matrix.
            threshold: Probability threshold (defaults to self.best_threshold_).

        Returns:
            1D array of binary predictions (0 or 1).
        """
        thresh = threshold if threshold is not None else self.best_threshold_
        probas = self.predict_proba(X)
        return (probas >= thresh).astype(np.int32)

    def get_feature_importances(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Get ranked feature importance DataFrame.

        Args:
            importance_type: 'gain' or 'split'.

        Returns:
            DataFrame sorted by importance descending.
        """
        if not self.is_fitted_ or self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")

        importances = self.booster_.feature_importance(importance_type=importance_type)
        names = self.feature_names

        df_imp = pd.DataFrame({
            "feature": names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        total = df_imp["importance"].sum()
        df_imp["normalized_pct"] = (
            (df_imp["importance"] / total * 100.0).round(2) if total > 0 else 0.0
        )
        return df_imp

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist model and configuration to disk.

        Args:
            filepath: Destination file path (e.g. 'models/lgbm_matcher.txt' or '.json').
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save LightGBM booster text representation
        booster_path = path.with_suffix(".txt")
        self.model.booster_.save_model(str(booster_path))

        # 2. Save metadata and hyperparams config
        meta = {
            "params": self.params,
            "feature_names": self.feature_names,
            "best_threshold": self.best_threshold_,
            "booster_file": booster_path.name,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved model booster to %s and metadata to %s", booster_path, meta_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EntityMatchClassifier":
        """Load persisted model and configuration from disk.

        Args:
            filepath: Path to saved .json or .txt model artifact.

        Returns:
            Loaded EntityMatchClassifier instance.
        """
        path = Path(filepath)
        meta_path = path.with_suffix(".json")
        booster_path = path.with_suffix(".txt")

        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata file not found at {meta_path}")
        if not booster_path.exists():
            raise FileNotFoundError(f"Model booster file not found at {booster_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            feature_names=meta.get("feature_names", ALL_FEATURE_NAMES),
            **{k: v for k, v in meta["params"].items() if k not in ("objective", "verbosity")},
        )
        instance.booster_ = lgb.Booster(model_file=str(booster_path))
        instance.best_threshold_ = meta.get("best_threshold", 0.5)
        instance.is_fitted_ = True

        logger.info("Successfully loaded model from %s", booster_path)
        return instance
