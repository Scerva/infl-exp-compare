"""Time-series cross-validation primitives (spec section 15: "Use
time-series cross-validation, NEVER random shuffled train/test splits").

Every out-of-sample R^2 computed anywhere in this project's model suite
goes through ``expanding_window_splits`` or ``rolling_window_splits``
below -- never sklearn's default KFold/train_test_split, which would leak
future information into training folds for time-ordered data.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def expanding_window_splits(
    n: int, n_splits: int = 5, min_train_fraction: float = 0.4
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) with an ever-growing training window
    and contiguous, non-overlapping test blocks -- assumes ``n`` rows are
    already sorted chronologically by the caller.
    """
    min_train = max(1, int(n * min_train_fraction))
    remaining = n - min_train
    if remaining <= 0 or n_splits <= 0:
        return
    fold_size = max(1, remaining // n_splits)
    idx = np.arange(n)
    train_end = min_train
    for _ in range(n_splits):
        test_end = min(train_end + fold_size, n)
        if train_end >= test_end:
            break
        yield idx[:train_end], idx[train_end:test_end]
        train_end = test_end


def rolling_window_splits(
    n: int, n_splits: int = 5, train_fraction: float = 0.4
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Like expanding_window_splits but the training window slides
    forward instead of growing -- a fixed-size lookback rather than
    all history.
    """
    train_size = max(1, int(n * train_fraction))
    remaining = n - train_size
    if remaining <= 0 or n_splits <= 0:
        return
    fold_size = max(1, remaining // n_splits)
    idx = np.arange(n)
    train_start = 0
    train_end = train_size
    for _ in range(n_splits):
        test_end = min(train_end + fold_size, n)
        if train_end >= test_end:
            break
        yield idx[train_start:train_end], idx[train_end:test_end]
        train_start = test_end - train_size if (test_end - train_size) > 0 else 0
        train_end = test_end


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    """Out-of-sample R^2 against the TRAINING mean as the naive
    benchmark (Campbell-Thompson convention) -- not the test-fold mean,
    which would itself use look-ahead information.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot
