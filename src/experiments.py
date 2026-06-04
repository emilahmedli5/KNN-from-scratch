import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.knn import KNN
from src.metrics import accuracy, precision, recall, f1_score, roc_auc
from src.splits import stratified_kfold

np.random.seed(42)

# project-level figures folder
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def run_computational_benchmark(
    X_train,
    y_train,
    X_test,
    k=5,
    m_test=100
):
    """
    Measures prediction time vs training size.
    """

    n_values = [100, 500, 1000, 5000, 10000]
    times = []

    X_test_fixed = X_test[:m_test]

    for n in n_values:

        # handle small datasets by repeating
        if n > len(X_train):
            repeat = int(np.ceil(n / len(X_train)))

            X_subset = np.tile(X_train, (repeat, 1))[:n]
            y_subset = np.tile(y_train, repeat)[:n]
        else:
            X_subset = X_train[:n]
            y_subset = y_train[:n]

        model = KNN(k=k, metric="euclidean")
        model.fit(X_subset, y_subset)

        start = time.perf_counter()
        model.predict(X_test_fixed)
        end = time.perf_counter()

        times.append(end - start)

    plt.figure(figsize=(8, 6))

    plt.loglog(n_values, times, marker="o", linewidth=2, label="Measured")

    baseline = [
        times[0] * (n / n_values[0]) for n in n_values
    ]

    plt.loglog(n_values, baseline, "--", color="gray", label="O(N) reference")

    plt.title(f"KNN Prediction Time vs Training Size (M={m_test})")
    plt.xlabel("Training samples (N)")
    plt.ylabel("Time (seconds)")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(FIGURES_DIR / "benchmark_2_5.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return n_values, times


def run_hyperparameter_sweep(X_train, y_train, X_val, y_val):
    """
    Evaluate different k values and pick best based on F1.
    """

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    train_scores = {m: [] for m in metrics}
    val_scores = {m: [] for m in metrics}

    for k in k_values:

        model = KNN(k=k)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        # training metrics
        train_scores["Accuracy"].append(accuracy(y_train, y_train_pred))
        train_scores["Precision"].append(precision(y_train, y_train_pred))
        train_scores["Recall"].append(recall(y_train, y_train_pred))
        train_scores["F1"].append(f1_score(y_train, y_train_pred))
        train_scores["AUC-ROC"].append(roc_auc(y_train, y_train_prob))

        # validation metrics
        val_scores["Accuracy"].append(accuracy(y_val, y_val_pred))
        val_scores["Precision"].append(precision(y_val, y_val_pred))
        val_scores["Recall"].append(recall(y_val, y_val_pred))
        val_scores["F1"].append(f1_score(y_val, y_val_pred))
        val_scores["AUC-ROC"].append(roc_auc(y_val, y_val_prob))

    best_k = k_values[np.argmax(val_scores["F1"])]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, m in enumerate(metrics):
        ax = axes[i]

        ax.plot(k_values, train_scores[m], marker="o", label="train")
        ax.plot(k_values, val_scores[m], marker="s", label="val")

        if m == "F1":
            ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")

        ax.set_title(f"{m} vs k")
        ax.set_xlabel("k")
        ax.set_ylabel(m)

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sweep_3_3.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return best_k


def evaluate_baselines(X_train, y_train, X_val, y_val, best_k):
    """
    Compare custom KNN vs sklearn vs majority baseline.
    """

    custom = KNN(k=best_k)
    custom.fit(X_train, y_train)

    y_pred_custom = custom.predict(X_val)
    y_prob_custom = custom.predict_proba(X_val)[:, 1]

    sk = KNeighborsClassifier(n_neighbors=best_k)
    sk.fit(X_train, y_train)

    y_pred_sk = sk.predict(X_val)
    y_prob_sk = sk.predict_proba(X_val)[:, 1]

    # majority baseline
    classes, counts = np.unique(y_train, return_counts=True)
    majority = classes[np.argmax(counts)]

    y_pred_major = np.full(len(y_val), majority)
    y_prob_major = np.full(len(y_val), 1.0 if majority == 1 else 0.0)

    results = {
        "Model": [
            f"Custom KNN (k={best_k})",
            f"Sklearn KNN (k={best_k})",
            "Majority baseline"
        ],
        "Accuracy": [
            accuracy(y_val, y_pred_custom),
            accuracy(y_val, y_pred_sk),
            accuracy(y_val, y_pred_major)
        ],
        "Precision": [
            precision(y_val, y_pred_custom),
            precision(y_val, y_pred_sk),
            precision(y_val, y_pred_major)
        ],
        "Recall": [
            recall(y_val, y_pred_custom),
            recall(y_val, y_pred_sk),
            recall(y_val, y_pred_major)
        ],
        "F1 Score": [
            f1_score(y_val, y_pred_custom),
            f1_score(y_val, y_pred_sk),
            f1_score(y_val, y_pred_major)
        ],
        "AUC-ROC": [
            roc_auc(y_val, y_prob_custom),
            roc_auc(y_val, y_prob_sk),
            roc_auc(y_val, y_prob_major)
        ],
    }

    return pd.DataFrame(results).set_index("Model")


def evaluate_test_set(X_train, y_train, X_test, y_test, best_k):
    """
    Final evaluation on held-out test set.
    """

    model = KNN(k=best_k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
        "Test Score": [
            accuracy(y_test, y_pred),
            precision(y_test, y_pred),
            recall(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc(y_test, y_prob)
        ]
    }

    return pd.DataFrame(results).set_index("Metric")


def run_cross_validation_sweep(X_dev, y_dev, n_splits=5):
    """
    Cross-validation over k values.
    """

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]

    folds = stratified_kfold(
        X_dev,
        y_dev,
        n_splits=n_splits,
        random_state=42
    )

    mean_scores, std_scores = [], []

    for k in k_values:

        fold_scores = []

        for X_tr, y_tr, X_val, y_val in folds:

            model = KNN(k=k)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred))

        mean_scores.append(np.mean(fold_scores))
        std_scores.append(np.std(fold_scores))

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)

    best_k = k_values[np.argmax(mean_scores)]

    plt.figure(figsize=(8, 6))

    plt.plot(k_values, mean_scores, marker="o", label="Mean CV F1")
    plt.fill_between(
        k_values,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        label="±1 std"
    )

    plt.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")

    plt.title(f"{n_splits}-Fold Cross Validation")
    plt.xlabel("k")
    plt.ylabel("F1 Score")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(FIGURES_DIR / "cv_4_2.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return best_k, mean_scores[np.argmax(mean_scores)]


def run_scaling_experiment(X_dev, y_dev, best_k, n_splits=5):
    """
    CV with feature scaling (no data leakage).
    """

    folds = stratified_kfold(X_dev, y_dev, n_splits=n_splits, random_state=42)

    scores = []

    for X_tr, y_tr, X_val, y_val in folds:

        scaler = StandardScaler()

        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        model = KNN(k=best_k)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred))

    return np.mean(scores)