import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y: list) -> 'LabelEncoder':
        """
        Fit label encoder by finding the unique classes in the input labels.
        """
        y = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: list) -> np.ndarray:
        """
        Transform labels to normalized encoding (integer labels).
        """
        y = np.array(y)
        if self.classes_ is None:
            raise ValueError("LabelEncoder instance is not fitted yet. Call 'fit' before using this method.")

        # Find the index of each label in the classes_
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y: list) -> np.ndarray:
        """
        Fit the label encoder and return encoded labels in a single step.
        """
        self.fit(y)
        return self.transform(y)
def test_label_encoder_basic():
    labels = ['cat', 'dog', 'fish', 'cat', 'dog', 'fish']
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    assert encoder.classes_.tolist() == ['cat', 'dog', 'fish'], "Classes are incorrect"
    assert (encoded_labels == [0, 1, 2, 0, 1, 2]).all(), "Encoding is incorrect"
    print("test_label_encoder_basic passed")

def test_label_encoder_with_numbers():
    labels = [10, 20, 30, 10, 20, 30]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    assert encoder.classes_.tolist() == [10, 20, 30], "Classes are incorrect"
    assert (encoded_labels == [0, 1, 2, 0, 1, 2]).all(), "Encoding is incorrect"
    print("test_label_encoder_with_numbers passed")

def test_label_encoder_not_fitted():
    encoder = LabelEncoder()
    try:
        encoder.transform(['cat', 'dog'])
    except ValueError as e:
        print("test_label_encoder_not_fitted passed - Caught ValueError:", e)

# Run the tests manually
test_label_encoder_basic()
test_label_encoder_with_numbers()
test_label_encoder_not_fitted()

# Explanation of the Tests:
# Basic Test (test_label_encoder_basic): Tests encoding of categorical labels (cat, dog, fish) and checks if the classes_ are stored correctly.
# Test with Numbers (test_label_encoder_with_numbers): Tests encoding of numeric labels and checks if the encoder handles non-string labels correctly.
# Not Fitted Test (test_label_encoder_not_fitted): Tests that an error is raised if the transform method is called before fitting the encoder.
