import numpy as np

from mmm_ridge.splits import holdout_split, rolling_origin_splits
from mmm_ridge.transforms import geometric_adstock


def test_rolling_origin_folds_only_train_on_the_past():
    folds = rolling_origin_splits(60, n_folds=3, horizon=5)
    assert len(folds) == 3
    for train, test in folds:
        assert train.max() < test.min()
        np.testing.assert_array_equal(test, np.arange(test[0], test[0] + 5))
    test_starts = [test[0] for _, test in folds]
    assert test_starts == sorted(test_starts)


def test_holdout_is_after_all_rolling_folds_when_run_on_training_window():
    train, test = holdout_split(100)
    folds = rolling_origin_splits(len(train), n_folds=3, horizon=13)
    last_test = folds[-1][1].max()
    assert last_test < len(train)
    assert test.min() > last_test


def test_adstock_decay_on_impulse():
    np.testing.assert_allclose(geometric_adstock([0, 2, 0], 0.5), [0, 2, 1])
