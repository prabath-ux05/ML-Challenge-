"""Model classes, kept in their own module so pickled models load from any entry point."""
import os

import numpy as np

try:  # LightGBM (MIT) when its OpenMP runtime is available; otherwise sklearn's histogram GBDT
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # ImportError, or OSError when libomp is missing
    HAVE_LGB = False
    lgb = None


class LGBModel:
    """LightGBM booster with early stopping on a random 8% hold-out; sklearn-like predict_proba."""

    def __init__(self, seed):
        self.seed = seed
        self.params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
                           feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                           max_bin=255, verbose=-1, seed=seed, num_threads=os.cpu_count() or 4)

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        va = rng.random(len(y)) < 0.08
        w = sample_weight if sample_weight is not None else np.ones(len(y), np.float32)
        dtr = lgb.Dataset(X[~va], y[~va], weight=w[~va], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr)
        self.booster = lgb.train(self.params, dtr, num_boost_round=4000, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
        self.n_iter_ = self.booster.best_iteration
        return self

    def predict_proba(self, X):
        p = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        return np.c_[1 - p, p]


class _Booster:
    """Wraps a trained lgb.Booster with the predict_proba interface used by predict_oof/predict_avg."""

    def __init__(self, booster):
        self.booster, self.n_iter_ = booster, booster.best_iteration

    def predict_proba(self, X):
        p = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        return np.c_[1 - p, p]
