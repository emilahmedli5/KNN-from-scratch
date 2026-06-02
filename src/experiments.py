import time
import numpy as np
import matplotlib.pyplot as plt

from src.knn import KNN


def run_computational_benchmark(
    X_train,
    y_train,
    X_test,
    k=5,
    m_test=100
):
    """
    Measures how prediction time scales with training size N.
    """

    n_values = [100, 500, 1000, 5000, 10000]
    times = []

    X_test_fixed = X_test[:m_test]

    for n in n_values:

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
        times[0] * (n / n_values[0])
        for n in n_values
    ]

    plt.loglog(
        n_values,
        baseline,
        "--",
        color="gray",
        label="O(N) reference"
    )

    plt.title(f"KNN Prediction Time vs Training Size (M={m_test})")
    plt.xlabel("Training samples (N)")
    plt.ylabel("Time (seconds)")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig("../figures/benchmark_2_5.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return n_values, times





import numpy as np
import matplotlib.pyplot as plt

from src.knn import KNN
from src.metrics import (
    accuracy,
    precision,
    recall,
    f1_score_metric,
    roc_auc
)


def run_hyperparameter_sweep(
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Tests different k values and selects best k based on F1.
    """

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    train_scores = {m: [] for m in metric_names}
    val_scores = {m: [] for m in metric_names}

    for k in k_values:

        model = KNN(k=k)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)

        train_scores["Accuracy"].append(accuracy(y_train, y_train_pred))
        train_scores["Precision"].append(precision(y_train, y_train_pred))
        train_scores["Recall"].append(recall(y_train, y_train_pred))
        train_scores["F1"].append(f1_score_metric(y_train, y_train_pred))
        train_scores["AUC-ROC"].append(roc_auc(y_train, y_train_prob))

        val_scores["Accuracy"].append(accuracy(y_val, y_val_pred))
        val_scores["Precision"].append(precision(y_val, y_val_pred))
        val_scores["Recall"].append(recall(y_val, y_val_pred))
        val_scores["F1"].append(f1_score_metric(y_val, y_val_pred))
        val_scores["AUC-ROC"].append(roc_auc(y_val, y_val_prob))

    best_k = k_values[np.argmax(val_scores["F1"])]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, m in enumerate(metric_names):

        ax = axes[i]

        ax.plot(k_values, train_scores[m], marker="o", label="train")
        ax.plot(k_values, val_scores[m], marker="s", label="val")

        if m == "F1":
            ax.axvline(
                best_k,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"best k = {best_k}"
            )

        ax.set_title(f"KNN {m} vs k")
        ax.set_xlabel("k")
        ax.set_ylabel(m)

        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/sweep_3_3.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return best_k



import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from src.knn import KNN
from src.metrics import accuracy, precision, recall, f1_score_metric, roc_auc


def evaluate_baselines(
    X_train,
    y_train,
    X_val,
    y_val,
    best_k
):
    """
    Compare custom KNN vs sklearn KNN vs majority baseline.
    """

    custom = KNN(k=best_k)
    custom.fit(X_train, y_train)

    y_pred_custom = custom.predict(X_val)
    y_prob_custom = custom.predict_proba(X_val)

    sk = KNeighborsClassifier(n_neighbors=best_k)
    sk.fit(X_train, y_train)

    y_pred_sk = sk.predict(X_val)
    y_prob_sk = sk.predict_proba(X_val)[:, 1]

    classes, counts = np.unique(y_train, return_counts=True)
    majority = classes[np.argmax(counts)]

    y_pred_major = np.full(len(y_val), majority)

    prob_major = 1.0 if majority == 1 else 0.0
    y_prob_major = np.full(len(y_val), prob_major)

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
            f1_score_metric(y_val, y_pred_custom),
            f1_score_metric(y_val, y_pred_sk),
            f1_score_metric(y_val, y_pred_major)
        ],
        "AUC-ROC": [
            roc_auc(y_val, y_prob_custom),
            roc_auc(y_val, y_prob_sk),
            roc_auc(y_val, y_prob_major)
        ],
    }

    return pd.DataFrame(results).set_index("Model")



import numpy as np
import pandas as pd

from src.knn import KNN
from src.metrics import accuracy, precision, recall, f1_score_metric, roc_auc


def evaluate_test_set(
    X_train,
    y_train,
    X_test,
    y_test,
    best_k
):
    """
    Final evaluation on test set (run once).
    """

    model = KNN(k=best_k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC-ROC"
        ],
        "Test Score": [
            accuracy(y_test, y_pred),
            precision(y_test, y_pred),
            recall(y_test, y_pred),
            f1_score_metric(y_test, y_pred),
            roc_auc(y_test, y_prob)
        ]
    }

    return pd.DataFrame(results).set_index("Metric")



import numpy as np
import matplotlib.pyplot as plt

from src.knn import KNN
from src.metrics import f1_score_metric
from src.splits import stratified_kfold


def run_cross_validation_sweep(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    n_splits: int = 5
):
    """
    Run cross-validation for different k values
    and choose the best one based on mean F1.
    """

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]

    folds = stratified_kfold(
        X_dev,
        y_dev,
        n_splits=n_splits,
        random_state=42
    )

    mean_scores = []
    std_scores = []

    for k in k_values:

        fold_scores = []

        for X_train_f, y_train_f, X_val_f, y_val_f in folds:

            model = KNN(k=k)
            model.fit(X_train_f, y_train_f)

            y_pred = model.predict(X_val_f)

            score = f1_score_metric(y_val_f, y_pred)
            fold_scores.append(score)

        mean_scores.append(np.mean(fold_scores))
        std_scores.append(np.std(fold_scores))

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)

    best_idx = np.argmax(mean_scores)
    best_k = k_values[best_idx]

    plt.figure(figsize=(8, 6))

    plt.plot(
        k_values,
        mean_scores,
        marker="o",
        label="Mean CV F1",
        color="blue"
    )

    plt.fill_between(
        k_values,
        mean_scores - std_scores,
        mean_scores + std_scores,
        color="blue",
        alpha=0.2,
        label="±1 Std"
    )

    plt.axvline(
        best_k,
        color="red",
        linestyle="--",
        label=f"best k = {best_k}"
    )

    plt.title(f"{n_splits}-Fold Cross Validation")
    plt.xlabel("k")
    plt.ylabel("F1 Score")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(
        "figures/cv_4_2.pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return best_k, mean_scores[best_idx]




import numpy as np
from sklearn.preprocessing import StandardScaler
from src.splits import stratified_kfold
from src.knn import KNN
from src.metrics import f1_score_metric

def run_scaling_experiment(X_dev: np.ndarray, y_dev: np.ndarray, best_k: int, n_splits: int = 5):
    """
    Reruns the cross-validation for the best k using standard scaling.
    Ensures no data leakage by fitting the scaler only on the training folds.
    """
    # Generate the exact same folds for a fair comparison
    folds = stratified_kfold(X_dev, y_dev, n_splits=n_splits, random_state=42)
    
    scaled_f1_scores = []
    
    for X_tr_f, y_tr_f, X_val_f, y_val_f in folds:
        # 1. Initialize a fresh scaler for this fold
        scaler = StandardScaler()
        
        # 2. Fit strictly on the training fold, and transform it
        X_tr_scaled = scaler.fit_transform(X_tr_f)
        
        # 3. Transform the validation fold using the training fold's parameters
        X_val_scaled = scaler.transform(X_val_f)
        
        # 4. Train and predict
        knn = KNN(k=best_k)
        knn.fit(X_tr_scaled, y_tr_f)
        y_val_pred = knn.predict(X_val_scaled)
        
        # 5. Record the score
        scaled_f1_scores.append(f1_score_metric(y_val_f, y_val_pred))
        
    mean_scaled_f1 = np.mean(scaled_f1_scores)
    return mean_scaled_f1