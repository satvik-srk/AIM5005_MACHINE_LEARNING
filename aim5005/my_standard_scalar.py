import numpy as np

class StandardScaler:
    """
    StandardScaler mimics the sklearn API, used to standardize features 
    by removing the mean and scaling to unit variance.
    
    Attributes:
    -----------
    mean_ : ndarray of shape (n_features,)
        Mean of the features in the training set.
    
    scale_ : ndarray of shape (n_features,)
        Standard deviation of the features in the training set.
    """

    def __init__(self):
        """
        Initialize the StandardScaler. Attributes `mean_` and `scale_` are set to None initially.
        """
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and standard deviation of each feature in X.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation for scaling.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Ensure the input is a numpy array
        X = np.array(X)
        
        # Compute the mean and standard deviation of each feature
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        
        # If a feature has zero variance, avoid division by zero by setting scale to 1
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input data X by removing the mean and scaling by the standard deviation.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The data to be transformed.

        Returns:
        --------
        X_scaled : ndarray of shape (n_samples, n_features)
            The transformed data, where each feature is centered and scaled.
        """
        # Raise an error if the scaler has not been fitted
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler instance is not fitted yet. Call 'fit' before using this method.")
        
        # Center the data and scale by the standard deviation
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation, and then transform.

        Returns:
        --------
        X_scaled : ndarray of shape (n_samples, n_features)
            The transformed data, where each feature is centered and scaled.
        """
        # First fit the data and then transform it in a single step
        self.fit(X)
        return self.transform(X)

def test_standard_scaler_basic():
    """
    Test that the StandardScaler correctly scales data with various means and variances.
    """
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if the mean is approximately zero after scaling
    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    
    # Check if the standard deviation is approximately one after scaling
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    
    print("test_standard_scaler_basic passed")


def test_standard_scaler_with_different_scales():
    """
    Test the scaler's ability to handle features with very different scales.
    """
    X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if the mean is approximately zero after scaling
    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    
    # Check if the standard deviation is approximately one after scaling
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    
    print("test_standard_scaler_with_different_scales passed")


def test_standard_scaler_edge_cases():
    """
    Test the scaler's handling of constant feature values (zero variance).
    """
    X = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    scaler = StandardScaler()
    
    # No error should be raised, and output should be all zeros
    X_scaled = scaler.fit_transform(X)
    
    assert np.allclose(X_scaled, np.zeros_like(X)), "Constant feature values should result in zeroed output"
    print("test_standard_scaler_edge_cases passed")


def test_standard_scaler_not_fitted():
    """
    Ensure the StandardScaler raises a ValueError if transform is called before fit.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = StandardScaler()
    
    try:
        # This should raise an error since fit has not been called
        scaler.transform(X)
    except ValueError as e:
        print("test_standard_scaler_not_fitted passed - Caught ValueError:", e)


# Run the tests manually
test_standard_scaler_basic()
test_standard_scaler_with_different_scales()
test_standard_scaler_edge_cases()
test_standard_scaler_not_fitted()
