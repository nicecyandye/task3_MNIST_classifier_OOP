import numpy as np
from abc import ABC, abstractmethod


class DigitClassificationInterface(ABC):

    @abstractmethod
    def predict(self, image):
        pass


class CNNModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image):
        # Custom transformation
        raise NotImplementedError("CNN model prediction not implemented")


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image):
        # Custom transformation
        raise NotImplementedError("Random Forest model prediction not implemented")


class RandomModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image):
        # Custom transformation
        raise NotImplementedError("Random model prediction not implemented")


class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Invalid algorithm. Supported algorithms are 'cnn', 'rf', and 'rand'.")

    def predict(self, image):
        return self.model.predict(image)


# Example usage:
if __name__ == "__main__":
    example_image = np.random.rand(28, 28, 1)
    classifier = DigitClassifier(algorithm='cnn')
    prediction = classifier.predict(example_image)
    print(f"Prediction: {prediction}")
