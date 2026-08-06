import numpy as np

from src.models.time_series_cv import expanding_window_splits, oos_r2, rolling_window_splits


def test_expanding_window_splits_train_never_overlaps_test():
    for train_idx, test_idx in expanding_window_splits(100, n_splits=5, min_train_fraction=0.4):
        assert train_idx.max() < test_idx.min()


def test_expanding_window_splits_train_grows_each_fold():
    sizes = [len(train_idx) for train_idx, _ in expanding_window_splits(100, n_splits=5, min_train_fraction=0.4)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1


def test_expanding_window_splits_test_indices_are_contiguous_and_chronological():
    prev_max = -1
    for _, test_idx in expanding_window_splits(50, n_splits=4, min_train_fraction=0.3):
        assert test_idx.min() > prev_max
        prev_max = test_idx.max()


def test_rolling_window_splits_train_size_roughly_constant():
    sizes = [len(train_idx) for train_idx, _ in rolling_window_splits(100, n_splits=5, train_fraction=0.3)]
    assert max(sizes) - min(sizes) <= 30  # allows the fixed-size lookback with edge effects


def test_rolling_window_splits_never_overlaps():
    for train_idx, test_idx in rolling_window_splits(100, n_splits=5, train_fraction=0.3):
        assert train_idx.max() < test_idx.min()


def test_splits_handle_small_n_gracefully():
    splits = list(expanding_window_splits(3, n_splits=5, min_train_fraction=0.4))
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0


def test_oos_r2_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert oos_r2(y, y, y_train_mean=2.5) == 1.0


def test_oos_r2_naive_mean_prediction_gives_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, 2.5)
    assert abs(oos_r2(y, pred, y_train_mean=2.5)) < 1e-9


def test_oos_r2_worse_than_naive_is_negative():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([10.0, -10.0, 10.0, -10.0])
    assert oos_r2(y, pred, y_train_mean=2.5) < 0


def test_oos_r2_zero_variance_target_returns_nan():
    y = np.array([2.0, 2.0, 2.0])
    result = oos_r2(y, y, y_train_mean=2.0)
    assert np.isnan(result)
