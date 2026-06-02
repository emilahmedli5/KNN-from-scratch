import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Make sure to import your KNN class! 
# Assuming it is in src/knn.py
from src.knn import KNN 

# ==========================================
# FIXTURES (Setting up our dummy data)
# ==========================================

@pytest.fixture
def clf_data():
    """Generates random data for classification tasks."""
    np.random.seed(42)
    # 100 samples, 5 features
    X_train = np.random.rand(100, 5) 
    # 3 classes (0, 1, 2) just to make sure multi-class works
    y_train = np.random.randint(0, 3, size=100) 
    
    # 20 test samples
    X_test = np.random.rand(20, 5)
    return X_train, y_train, X_test

@pytest.fixture
def reg_data():
    """Generates random data for regression tasks."""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100) * 100 # Continuous target variables
    
    X_test = np.random.rand(20, 5)
    return X_train, y_train, X_test


# ==========================================
# TESTS
# ==========================================

def test_knn_classification_euclidean(clf_data):
    """Checks if standard Euclidean classification matches sklearn."""
    X_train, y_train, X_test = clf_data
    
    # Run custom KNN
    my_knn = KNN(k=5, metric="euclidean", task="classification")
    my_knn.fit(X_train, y_train)
    my_preds = my_knn.predict(X_test)
    
    # Run Sklearn's KNN
    sk_knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    sk_knn.fit(X_train, y_train)
    sk_preds = sk_knn.predict(X_test)
    
    # Since these are integer class labels, they should match exactly
    np.testing.assert_array_equal(
        my_preds, sk_preds, 
        err_msg="Custom Euclidean classification doesn't match sklearn!"
    )

def test_knn_predict_proba(clf_data):
    """Checks if the probability distributions match sklearn."""
    X_train, y_train, X_test = clf_data
    
    my_knn = KNN(k=7, metric="euclidean", task="classification")
    my_knn.fit(X_train, y_train)
    my_probs = my_knn.predict_proba(X_test)
    
    sk_knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    sk_knn.fit(X_train, y_train)
    sk_probs = sk_knn.predict_proba(X_test)
    
    # Because probabilities are floats, we use assert_allclose to avoid tiny rounding errors
    np.testing.assert_allclose(
        my_probs, sk_probs, 
        err_msg="Custom predict_proba is calculating different probabilities than sklearn!"
    )

def test_knn_manhattan_distance(clf_data):
    """Checks if the Manhattan distance logic is working."""
    X_train, y_train, X_test = clf_data
    
    my_knn = KNN(k=3, metric="manhattan", task="classification")
    my_knn.fit(X_train, y_train)
    my_preds = my_knn.predict(X_test)
    
    sk_knn = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    sk_knn.fit(X_train, y_train)
    sk_preds = sk_knn.predict(X_test)
    
    np.testing.assert_array_equal(
        my_preds, sk_preds, 
        err_msg="Manhattan metric predictions are failing!"
    )

def test_knn_regression(reg_data):
    """Since you added regression support, we definitely need to test it!"""
    X_train, y_train, X_test = reg_data
    
    my_knn = KNN(k=5, task="regression")
    my_knn.fit(X_train, y_train)
    my_preds = my_knn.predict(X_test)
    
    sk_knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
    sk_knn.fit(X_train, y_train)
    sk_preds = sk_knn.predict(X_test)
    
    # Regression outputs floats, so again we use assert_allclose
    np.testing.assert_allclose(
        my_preds, sk_preds, 
        err_msg="Regression averages don't match sklearn!"
    )

def test_invalid_metric_raises_error():
    """Sanity check to make sure the code crashes safely if the user typos a metric."""
    my_knn = KNN(metric="fake_metric")
    my_knn.fit(np.array([[1, 2]]), np.array([0]))
    
    # This checks that a ValueError is actually raised
    with pytest.raises(ValueError, match="Unknown metric"):
        my_knn.predict(np.array([[1, 2]]))