import numpy as np


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)  # shuffle class samples

        n = len(idx)

        n_test = int(n * test_frac)
        n_val = int(n * val_frac)

        # split per class
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # final shuffle so classes mix
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
    rng = np.random.default_rng(random_state)

    classes = np.unique(y)

    # store splits per class
    per_class_splits = {c: [] for c in classes}

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)

        # split indices into k folds
        per_class_splits[c] = np.array_split(idx, n_splits)

    folds = []

    for i in range(n_splits):
        train_idx, val_idx = [], []

        for c in classes:
            parts = per_class_splits[c]

            # current fold = validation
            val_idx.extend(parts[i])

            # others = train
            for j in range(n_splits):
                if j != i:
                    train_idx.extend(parts[j])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        folds.append(
            (
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx]
            )
        )

    return folds