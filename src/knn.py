import numpy as np
from collections import Counter


class KNN:
    def __init__(
        self,
        k=5,
        metric="euclidean",
        q=2.0,
        task="classification",
        weights="uniform"
    ):
        self.k = k
        self.metric = metric
        self.q = q
        self.task = task
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, x):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)

        if self.metric == "minkowski":
            return np.sum(
                np.abs(self.X_train - x) ** self.q,
                axis=1
            ) ** (1 / self.q)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _get_weights(self, distances):
        if self.weights == "uniform":
            return np.ones(len(distances))

        # Avoid division by zero
        return 1 / (distances + 1e-8)

    def predict(self, X):
        predictions = []

        for sample in X:
            distances = self._compute_distances(sample)

            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_idx]
            nearest_distances = distances[nearest_idx]

            weights = self._get_weights(nearest_distances)

            if self.task == "classification":

                # Get all classes in sorted order
                classes = np.sort(np.unique(self.y_train))

                # Store vote counts
                votes = {c: 0 for c in classes}

                for label, weight in zip(nearest_labels, weights):
                    votes[label] += weight

                # If there is a tie, smaller class wins
                prediction = max(
                    classes,
                    key=lambda c: votes[c]
                )

            elif self.task == "regression":

                prediction = (
                    np.dot(weights, nearest_labels)
                    / np.sum(weights)
                )

            else:
                raise ValueError(
                    f"Unknown task: {self.task}"
                )

            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        classes = np.unique(self.y_train)
        probabilities = []

        for sample in X:
            distances = self._compute_distances(sample)

            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_idx]
            nearest_distances = distances[nearest_idx]

            weights = self._get_weights(nearest_distances)

            prob_dict = {c: 0.0 for c in classes}

            for label, weight in zip(nearest_labels, weights):
                prob_dict[label] += weight

            total_weight = sum(prob_dict.values())

            probs = [
                prob_dict[c] / total_weight
                for c in classes
            ]

            probabilities.append(probs)

        return np.array(probabilities)