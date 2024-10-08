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

import numpy as np

# Test 1: Basic functionality of StandardScaler
def test_standard_scaler_basic():
    """
    Test that StandardScaler correctly scales data to have a mean of 0 and a standard deviation of 1.
    """
    # Input array with different values in each feature
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    # Initialize the StandardScaler and fit-transform the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check that the mean of the scaled data is close to 0 for each feature
    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    
    # Check that the standard deviation of the scaled data is close to 1 for each feature
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    
    print("test_standard_scaler_basic passed")

# Test 2: StandardScaler with features of different scales
def test_standard_scaler_with_different_scales():
    """
    Test that StandardScaler handles features with significantly different scales.
    """
    # Input array with features on different scales (1-3 vs. 1000-3000)
    X = np.array([[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]])
    
    # Initialize the StandardScaler and fit-transform the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check that the mean of the scaled data is close to 0 for each feature
    assert np.allclose(np.mean(X_scaled, axis=0), 0), "Mean of the scaled data is not 0"
    
    # Check that the standard deviation of the scaled data is close to 1 for each feature
    assert np.allclose(np.std(X_scaled, axis=0), 1), "Standard deviation of the scaled data is not 1"
    
    print("test_standard_scaler_with_different_scales passed")

# Test 3: Handling constant feature values
def test_standard_scaler_edge_cases():
    """
    Test that StandardScaler can handle constant feature values without throwing an error.
    """
    # Input array with constant values across all features
    X = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    try:
        # Try to fit-transform the data with constant values
        X_scaled = scaler.fit_transform(X)
        print("test_standard_scaler_edge_cases passed - No error for constant features")
    except ZeroDivisionError:
        # Catch ZeroDivisionError (if any) due to division by zero in scaling constant features
        print("test_standard_scaler_edge_cases failed - ZeroDivisionError")

# Run the tests manually
test_standard_scaler_basic()
test_standard_scaler_with_different_scales()
test_standard_scaler_edge_cases()

# Explanation of the Test Cases: test_standard_scaler_basic:

# This test checks the basic functionality of the StandardScaler. 
# It verifies that the transformed data has a mean close to 0 and a standard deviation close to 1 for each feature after scaling. 
# The np.allclose function ensures that the values are very close to the expected mean and standard deviation within a numerical tolerance. test_standard_scaler_with_different_scales:

# This test verifies that StandardScaler works correctly when the input features have significantly different scales. 
# It ensures that the scaling process correctly transforms both small-scale and large-scale features to have a mean of 0 and a standard deviation of 1. test_standard_scaler_edge_cases:

# This test handles the edge case where all the features have constant values. 
# Since scaling constant values can lead to division by zero, this test ensures that StandardScaler handles this scenario without raising errors like ZeroDivisionError. 
# The test is wrapped in a try-except block to catch any potential errors.
