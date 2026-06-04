import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)


def precision(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_pred[i] == positive_label:
            fp += 1
            if y_true[i] == positive_label:
                tp += 1

    if fp == 0:
        return 0.0

    return tp / fp


def recall(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = 0
    fn = 0

    for i in range(len(y_true)):
        if y_true[i] == positive_label:
            if y_pred[i] == positive_label:
                tp += 1
            else:
                fn += 1

    if (tp + fn) == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred, positive_label=1):
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = sorted(set(y_score), reverse=True)
    thresholds = [float("inf")] + thresholds

    roc_points = []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = fp = tn = fn = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
            else:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0

        roc_points.append((fpr, tpr))

    roc_points = sorted(set(roc_points))
    roc_points.append((1, 1))

    auc = 0
    for i in range(len(roc_points) - 1):
        x1, y1 = roc_points[i]
        x2, y2 = roc_points[i + 1]

        auc += (x2 - x1) * (y1 + y2) / 2

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