import numpy as np


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42
):
    """
    Split dataset into train / validation / test
    while keeping class distribution roughly the same.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    rng = np.random.default_rng(seed)

    classes = np.unique(y)

    train_idx = []
    val_idx = []
    test_idx = []

    for cls in classes:
        # take indices of current class
        idx = np.where(y == cls)[0]

        # shuffle them
        rng.shuffle(idx)

        n = len(idx)

        n_test = int(np.round(n * test_frac))
        n_val = int(np.round(n * val_frac))

        test_part = idx[:n_test]
        val_part = idx[n_test:n_test + n_val]
        train_part = idx[n_test + n_val:]

        test_idx.extend(test_part)
        val_idx.extend(val_part)
        train_idx.extend(train_part)

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # shuffle again so classes are mixed
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )


def stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Create stratified K-fold splits.

    Returns:
        [(X_train, y_train, X_val, y_val), ...]
    """

    rng = np.random.default_rng(random_state)

    classes = np.unique(y)
    class_splits = {c: [] for c in classes}

    for c in classes:

        indices = np.where(y == c)[0]
        rng.shuffle(indices)

        class_splits[c] = np.array_split(
            indices,
            n_splits
        )

    folds = []

    for fold in range(n_splits):

        train_idx = []
        val_idx = []

        for c in classes:

            val_idx.extend(class_splits[c][fold])

            for i in range(n_splits):
                if i != fold:
                    train_idx.extend(class_splits[c][i])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        folds.append(
            (
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx]
            )
        )

    return folds