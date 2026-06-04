import numpy as np


class KNN:
    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        weights: str = "uniform"
    ):
        self.k = k
        self.metric = metric
        self.q = q
        self.task = task
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]

        if self.metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=2))

        elif self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)

        elif self.metric == "minkowski":
            return np.sum(np.abs(diff) ** self.q, axis=2) ** (1 / self.q)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        if self.weights == "uniform":
            return np.ones_like(distances)

        return 1.0 / (distances + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        distances = self._compute_distances(X)

        neighbor_idx = np.argsort(distances, axis=1)[:, :self.k]

        neighbor_labels = self.y_train[neighbor_idx]

        neighbor_distances = distances[
            np.arange(len(X))[:, np.newaxis],
            neighbor_idx
        ]

        weights = self._get_weights(neighbor_distances)

        if self.task == "classification":
            classes = np.sort(np.unique(self.y_train))

            votes = np.zeros((len(X), len(classes)))

            for i, cls in enumerate(classes):
                votes[:, i] = (
                    weights * (neighbor_labels == cls)
                ).sum(axis=1)

            predicted_idx = np.argmax(votes, axis=1)
            return classes[predicted_idx]

        elif self.task == "regression":
            weighted_sum = (weights * neighbor_labels).sum(axis=1)
            total_weight = weights.sum(axis=1)

            return weighted_sum / total_weight

        raise ValueError(f"Unknown task: {self.task}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        X = np.asarray(X)

        classes = np.sort(np.unique(self.y_train))

        distances = self._compute_distances(X)

        neighbor_idx = np.argsort(distances, axis=1)[:, :self.k]

        neighbor_labels = self.y_train[neighbor_idx]

        neighbor_distances = distances[
            np.arange(len(X))[:, np.newaxis],
            neighbor_idx
        ]

        weights = self._get_weights(neighbor_distances)

        probabilities = np.zeros((len(X), len(classes)))

        for i, cls in enumerate(classes):
            probabilities[:, i] = (
                weights * (neighbor_labels == cls)
            ).sum(axis=1)

        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities