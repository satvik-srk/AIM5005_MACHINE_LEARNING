import numpy as np

class LabelEncoder:
    """
    LabelEncoder mimics the sklearn API to encode target labels with values between 0 and n_classes-1.

    Attributes:
    -----------
    classes_ : ndarray of shape (n_classes,)
        Holds the unique classes found during fitting.
    """

    def __init__(self):
        """
        Initialize the LabelEncoder. The `classes_` attribute is initially set to None.
        """
        self.classes_ = None

    def fit(self, y: list) -> 'LabelEncoder':
        """
        Fit the label encoder by finding the unique classes in the input labels.

        Parameters:
        -----------
        y : list
            The list of labels to fit.

        Returns:
        --------
        self : object
            Returns the instance of the LabelEncoder.
        """
        # Convert the input list to a numpy array
        y = np.array(y)
        
        # Store the unique values from y as the classes
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: list) -> np.ndarray:
        """
        Transform labels to normalized encoding (integer labels).

        Parameters:
        -----------
        y : list
            The list of labels to transform.

        Returns:
        --------
        encoded_labels : ndarray of shape (n_samples,)
            Array of encoded labels, where each label is replaced with its corresponding integer.

        Raises:
        -------
        ValueError:
            If the label encoder has not been fitted before calling transform.
        """
        # Convert the input list to a numpy array
        y = np.array(y)
        
        # Raise an error if the encoder is not yet fitted
        if self.classes_ is None:
            raise ValueError("LabelEncoder instance is not fitted yet. Call 'fit' before using this method.")

        # Encode each label by finding its index in the classes_ attribute
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y: list) -> np.ndarray:
        """
        Fit the label encoder and transform labels to normalized encoding in a single step.

        Parameters:
        -----------
        y : list
            The list of labels to fit and transform.

        Returns:
        --------
        encoded_labels : ndarray of shape (n_samples,)
            Array of encoded labels, where each label is replaced with its corresponding integer.
        """
        # Fit the labels and then transform them in one step
        self.fit(y)
        return self.transform(y)


def test_label_encoder_basic():
    """
    Test that the LabelEncoder correctly encodes basic categorical labels.
    """
    # Input labels (categorical)
    labels = ['cat', 'dog', 'fish', 'cat', 'dog', 'fish']
    
    # Initialize the LabelEncoder and fit-transform the labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Check if the unique classes are correctly identified
    assert encoder.classes_.tolist() == ['cat', 'dog', 'fish'], "Classes are incorrect"
    
    # Check if the encoded labels match the expected output
    assert (encoded_labels == [0, 1, 2, 0, 1, 2]).all(), "Encoding is incorrect"
    print("test_label_encoder_basic passed")

def test_label_encoder_with_numbers():
    """
    Test that the LabelEncoder correctly encodes numeric labels.
    """
    # Input labels (numbers)
    labels = [10, 20, 30, 10, 20, 30]
    
    # Initialize the LabelEncoder and fit-transform the numeric labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Check if the unique classes are correctly identified
    assert encoder.classes_.tolist() == [10, 20, 30], "Classes are incorrect"
    
    # Check if the encoded labels match the expected output
    assert (encoded_labels == [0, 1, 2, 0, 1, 2]).all(), "Encoding is incorrect"
    print("test_label_encoder_with_numbers passed")

def test_label_encoder_not_fitted():
    """
    Test that the LabelEncoder raises a ValueError if transform is called before fitting.
    """
    encoder = LabelEncoder()
    
    try:
        # Attempt to transform without fitting, which should raise an error
        encoder.transform(['cat', 'dog'])
    except ValueError as e:
        # Catch the ValueError and print the success message
        print("test_label_encoder_not_fitted passed - Caught ValueError:", e)

# Run the tests manually
test_label_encoder_basic()
test_label_encoder_with_numbers()
test_label_encoder_not_fitted()
