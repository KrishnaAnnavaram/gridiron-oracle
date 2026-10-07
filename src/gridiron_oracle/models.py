"""Home-win probability models and the walk-forward evaluation by season.

Target: ``home_win`` = 1 if the home team won, 0 if it lost. Ties (rare) are left out of training and
scoring, and the report counts them. Every learned model is an sklearn ``Pipeline``: imputation,
scaling and feature selection are fit on the training seasons only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import feature_columns

LABELS = {1: "home win", 0: "away win"}
EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_p": float(p[m].mean()), "home_win_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float((t.n * (t.mean_p - t.home_win_rate).abs()).sum() / t.n.sum()) if len(t) else 0.0


class Model:
    name = "base"
    params: dict = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "Model":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class HomeTeam(Model):
    """Always the home team, with the home-win rate of the training games as the probability."""

    name = "home-team"

    def fit(self, train, val=None):
        self.rate = float(train.home_win.mean())
        return self

    def predict_proba(self, df):
        return np.full(len(df), self.rate)


class EloOnly(Model):
    name = "elo"

    def predict_proba(self, df):
        return df.elo_p_home.to_numpy(dtype=float)


class SklearnModel(Model):
    """A pipeline whose hyperparameters are chosen by log-loss on the validation season."""

    def __init__(self, name: str, make: Callable[..., Pipeline], grid: dict[str, list]):
        self.name, self.make, self.grid = name, make, grid

    def fit(self, train, val=None):
        # Many OpenMP threads make these small fits slower on hosts with many cores.
        with threadpool_limits(4):
            return self._fit(train, val)

    def _fit(self, train, val=None):
        cols = feature_columns(train)
        self.cols = cols
        Xtr, ytr = train[cols], train.home_win.to_numpy()
        combos = [dict(zip(self.grid, v)) for v in product(*self.grid.values())]
        best = (np.inf, combos[0])
        if val is not None and len(val) and len(combos) > 1:
            for params in combos:
                p = self.make(**params).fit(Xtr, ytr).predict_proba(val[cols])[:, 1]
                loss = log_loss(val.home_win.to_numpy(), p)
                if loss < best[0]:
                    best = (loss, params)
        self.params = best[1]
        full = pd.concat([train, val]) if val is not None and len(val) else train
        self.pipe = self.make(**self.params).fit(full[cols], full.home_win.to_numpy())
        return self

    def predict_proba(self, df):
        return self.pipe.predict_proba(df[self.cols])[:, 1]


def logistic(k: int | str = "all", C: float = 1.0, seed: int = 0) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("drop_constant", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("model", LogisticRegression(C=C, max_iter=2000))])


def boosting(max_depth: int = 3, learning_rate: float = 0.05, seed: int = 0) -> Pipeline:
    return Pipeline([("model", HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                                              max_iter=200, l2_regularization=1.0,
                                                              random_state=seed))])


def default_models(seed: int = 0) -> list[Model]:
    return [
        HomeTeam(),
        EloOnly(),
        SklearnModel("logistic", lambda k, C: logistic(k, C, seed), {"k": [10, 25, "all"], "C": [0.01, 0.1, 1.0]}),
        SklearnModel("boosting", lambda max_depth, learning_rate: boosting(max_depth, learning_rate, seed),
                     {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]}),
    ]


@dataclass
class SeasonResult:
    season: int
    model: str
    n: int
    ties_left_out: int
    accuracy: float
    log_loss: float
    brier: float
    params: dict = field(default_factory=dict)


@dataclass
class WalkForward:
    rows: list[SeasonResult]
    predictions: pd.DataFrame  # game_id, season, model, p_home, home_win

    def pooled(self) -> pd.DataFrame:
        out = []
        for name, g in self.predictions.groupby("model", sort=False):
            y, p = g.home_win.to_numpy(), g.p_home.to_numpy()
            out.append({"model": name, "n": len(g), "accuracy": accuracy(y, p), "log_loss": log_loss(y, p),
                        "brier": brier(y, p), "ece": ece(y, p)})
        return pd.DataFrame(out)


def walk_forward(df: pd.DataFrame, test_seasons: list[int], models_factory: Callable[[], list[Model]] | None = None) -> WalkForward:
    """For each test season s: select on season s-1 (trained on < s-1), refit on < s, score on s."""
    factory = models_factory or default_models
    rows, preds = [], []
    for s in test_seasons:
        played = df[df.home_win.notna()]
        train = played[played.season < s - 1]
        val = played[played.season == s - 1]
        test_all = df[df.season == s]
        test = test_all[test_all.home_win.notna()]
        if train.empty or val.empty or test.empty:
            raise ValueError(f"season {s} needs earlier train and validation seasons and test games")
        if train.game_date.max() >= val.game_date.min() or val.game_date.max() >= test.game_date.min():
            raise AssertionError("walk-forward order is broken")
        for model in factory():
            model.fit(train, val)
            p = model.predict_proba(test)
            y = test.home_win.to_numpy()
            rows.append(SeasonResult(s, model.name, len(test), int(len(test_all) - len(test)), accuracy(y, p),
                                     log_loss(y, p), brier(y, p), dict(getattr(model, "params", {}))))
            preds.append(pd.DataFrame({"game_id": test.game_id.to_numpy(), "season": s, "model": model.name,
                                       "p_home": p, "home_win": y}))
    return WalkForward(rows, pd.concat(preds, ignore_index=True))


def bootstrap_diff(pred: pd.DataFrame, a: str, b: str, metric=log_loss, n_boot: int = 1000, seed: int = 0):
    """metric(a) - metric(b) on the same games, with a 95 % percentile interval (games resampled)."""
    pa = pred[pred.model == a].set_index("game_id")
    pb = pred[pred.model == b].set_index("game_id").loc[pa.index]
    y, p1, p2 = pa.home_win.to_numpy(), pa.p_home.to_numpy(), pb.p_home.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(metric(y[i], p1[i]) - metric(y[i], p2[i]))
    return metric(y, p1) - metric(y, p2), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
