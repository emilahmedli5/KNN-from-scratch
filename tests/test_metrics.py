import numpy as np
import pytest
from sklearn import metrics as sk_metrics
from src.metrics import accuracy, precision, recall, f1_score, roc_auc
# Assuming your functions are saved in src/metrics.py
# from src.metrics import accuracy, precision, recall, f1_score_metric, roc_auc

@pytest.fixture
def test_data():
    """Generates seeded random data for testing."""
    np.random.seed(42)  
    
    # Generate 100 random binary labels
    y_true = np.random.randint(0, 2, size=100)
    
    # Generate 100 random binary predictions
    y_pred = np.random.randint(0, 2, size=100)
    
    # Generate 100 random probability scores between 0 and 1 for AUC
    y_score = np.random.rand(100)
    
    return y_true, y_pred, y_score

def test_accuracy(test_data):
    y_true, y_pred, _ = test_data
    custom_acc = accuracy(y_true, y_pred)
    sk_acc = sk_metrics.accuracy_score(y_true, y_pred)
    
    assert custom_acc == pytest.approx(sk_acc), "Accuracy metric failed to match sklearn."

def test_precision(test_data):
    y_true, y_pred, _ = test_data
    custom_prec = precision(y_true, y_pred)
    sk_prec = sk_metrics.precision_score(y_true, y_pred, zero_division=0)
    
    assert custom_prec == pytest.approx(sk_prec), "Precision metric failed to match sklearn."

def test_recall(test_data):
    y_true, y_pred, _ = test_data
    custom_rec = recall(y_true, y_pred)
    sk_rec = sk_metrics.recall_score(y_true, y_pred, zero_division=0)
    
    assert custom_rec == pytest.approx(sk_rec), "Recall metric failed to match sklearn."

def test_f1_score(test_data):
    y_true, y_pred, _ = test_data
    custom_f1 = f1_score(y_true, y_pred)
    sk_f1 = sk_metrics.f1_score(y_true, y_pred, zero_division=0)
    
    assert custom_f1 == pytest.approx(sk_f1), "F1 score metric failed to match sklearn."

def test_roc_auc(test_data):
    y_true, _, y_score = test_data
    custom_auc = roc_auc(y_true, y_score)
    sk_auc = sk_metrics.roc_auc_score(y_true, y_score)
    
    assert custom_auc == pytest.approx(sk_auc), "ROC-AUC metric failed to match sklearn."