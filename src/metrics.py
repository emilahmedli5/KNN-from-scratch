import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    return correct / len(y_true)


def precision(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positive = 0
    predicted_positive = 0

    for true, pred in zip(y_true, y_pred):
        if pred == positive_label:
            predicted_positive += 1

            if true == positive_label:
                true_positive += 1

    if predicted_positive == 0:
        return 0.0

    return true_positive / predicted_positive


def recall(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positive = 0
    actual_positive = 0

    for true, pred in zip(y_true, y_pred):
        if true == positive_label:
            actual_positive += 1

            if pred == positive_label:
                true_positive += 1

    if actual_positive == 0:
        return 0.0

    return true_positive / actual_positive


def f1_score(y_true, y_pred, positive_label=1):
    precision_value = precision(y_true, y_pred, positive_label)
    recall_value = recall(y_true, y_pred, positive_label)

    if precision_value + recall_value == 0:
        return 0.0

    return (
        2 * precision_value * recall_value
    ) / (precision_value + recall_value)


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = sorted(set(y_score), reverse=True)
    thresholds.insert(0, float("inf"))

    roc_points = []

    for threshold in thresholds:
        y_pred = []

        for score in y_score:
            if score >= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 0 and pred == 0:
                tn += 1
            else:
                fn += 1

        if tp + fn > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = 0

        if fp + tn > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0

        roc_points.append((fpr, tpr))

    roc_points.append((1, 1))
    roc_points = sorted(set(roc_points))

    auc = 0

    for i in range(len(roc_points) - 1):
        fpr1, tpr1 = roc_points[i]
        fpr2, tpr2 = roc_points[i + 1]

        width = fpr2 - fpr1
        average_height = (tpr1 + tpr2) / 2

        auc += width * average_height

    return auc


if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1]
    y_score = [0.90, 0.20, 0.80, 0.40, 0.30, 0.70]

    print("Accuracy :", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall   :", recall(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC AUC  :", roc_auc(y_true, y_score))