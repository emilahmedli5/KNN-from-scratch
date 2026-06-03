import numpy as np


class KNN:
    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        weights: str = "uniform"
    ) -> None:
        self.k = k
        self.metric = metric
        self.q = q
        self.task = task
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """Memorize the training data. Returns self."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized pairwise distance from every test point to every
        training point.  No Python loop over training points.

        Parameters
        ----------
        X : (n_queries, p)

        Returns
        -------
        D : (n_queries, N_train)  distance matrix
        """
        # X: (n_queries, p)   self.X_train: (N_train, p)
        # Expand dims so broadcasting produces (n_queries, N_train, p)
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]

        if self.metric == "euclidean":
            # (n_queries, N_train)
            return np.sqrt(np.sum(diff ** 2, axis=2))

        if self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)

        if self.metric == "minkowski":
            return np.sum(np.abs(diff) ** self.q, axis=2) ** (1.0 / self.q)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Return per-neighbor weights.

        Parameters
        ----------
        distances : (n_queries, k)  distances to the k nearest neighbors

        Returns
        -------
        weights : (n_queries, k)
        """
        if self.weights == "uniform":
            return np.ones_like(distances)

        # distance weighting — guard against division by zero
        return 1.0 / (distances + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (classification) or values (regression)."""
        X = np.asarray(X)

        # D: (n_queries, N_train)
        D = self._compute_distances(X)

        # nearest_idx: (n_queries, k)  — indices into training set
        nearest_idx = np.argsort(D, axis=1)[:, : self.k]

        # nearest_labels: (n_queries, k)
        nearest_labels = self.y_train[nearest_idx]

        # nearest_dist: (n_queries, k)
        nearest_dist = D[np.arange(len(X))[:, np.newaxis], nearest_idx]

        # weights: (n_queries, k)
        weights = self._get_weights(nearest_dist)

        if self.task == "classification":
            classes = np.sort(np.unique(self.y_train))

            # Accumulate weighted votes: (n_queries, n_classes)
            vote_matrix = np.zeros((len(X), len(classes)))

            for ci, c in enumerate(classes):
                # mask where neighbor label == c  →  (n_queries, k)
                mask = nearest_labels == c
                vote_matrix[:, ci] = (weights * mask).sum(axis=1)

            # argmax over classes; ties go to the smaller class (first in sorted order)
            best_class_idx = np.argmax(vote_matrix, axis=1)
            return classes[best_class_idx]

        if self.task == "regression":
            weight_sums = weights.sum(axis=1, keepdims=True)   # (n_queries, 1)
            return (weights * nearest_labels).sum(axis=1) / weight_sums.ravel()

        raise ValueError(f"Unknown task: {self.task}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-class probabilities (classification only).
        Output shape: (n_queries, n_classes).

        With k neighbors the probabilities take values in
        {0, 1/k, 2/k, ..., 1} for uniform weights.
        """
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        X = np.asarray(X)
        classes = np.sort(np.unique(self.y_train))

        # D: (n_queries, N_train)
        D = self._compute_distances(X)

        # nearest_idx: (n_queries, k)
        nearest_idx = np.argsort(D, axis=1)[:, : self.k]

        # nearest_labels: (n_queries, k)
        nearest_labels = self.y_train[nearest_idx]

        # nearest_dist: (n_queries, k)
        nearest_dist = D[np.arange(len(X))[:, np.newaxis], nearest_idx]

        # weights: (n_queries, k)
        weights = self._get_weights(nearest_dist)

        # prob_matrix: (n_queries, n_classes)
        prob_matrix = np.zeros((len(X), len(classes)))

        for ci, c in enumerate(classes):
            mask = nearest_labels == c          # (n_queries, k)
            prob_matrix[:, ci] = (weights * mask).sum(axis=1)

        # Normalize each row so probabilities sum to 1
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix /= row_sums

        return prob_matrix