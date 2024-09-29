import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # Corrected bug: Apply the scaling formula in the right order
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

# StandardScaler Implementation

import pytest  

# Test 1: Basic functionality of StandardScaler
def test_standard_scaler_basic():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    print("test_standard_scaler_basic passed")

# Test 2: StandardScaler with features of different scales
def test_standard_scaler_with_different_scales():
    X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    print("test_standard_scaler_with_different_scales passed")

# Test 3: Handling constant feature values
def test_standard_scaler_edge_cases():
    X = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    scaler = StandardScaler()
    
    try:
        X_scaled = scaler.fit_transform(X)
        print("test_standard_scaler_edge_cases passed - No error for constant features")
    except ZeroDivisionError:
        print("test_standard_scaler_edge_cases failed - ZeroDivisionError")

# Run the tests manually
test_standard_scaler_basic()
test_standard_scaler_with_different_scales()
test_standard_scaler_edge_cases()
