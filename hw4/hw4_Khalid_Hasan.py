"""
Name: Khalid Hasan
BearPass: M03543550

How to run:
1. Place the source file "hw4_Khalid_Hasan.py", "training_data.txt", and "testing_data.txt" in the same directory.
2. Open a terminal (or, command-prompt) in this directory and run the following command:
    python hw4_Khalid_Hasan.py

**Make sure, you have python and python's numpy and pandas library installed to run this command.

Summary:
1. This file contains the "Perceptron" class for the implementation of the perceptron algorithm.
2. The function named "fit" is to train the model and update weights and bias parameters
    so that the model gives the best possible accuracy.
3. The function named "predict" predicts the class labels of given test data using the parameters trained by fit method.
4. Stopping criteria during the model training: Certain number of iterations. Here, 100 epochs have been used.
5. The main function calls the Perceptron class to predict the test data
    and prints some evaluation statistics along with actual and predicted labels.
6. Accuracy of the model: This implementation has received 100% accuracy for given test data from "testing_data.txt".
"""
import random

import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self, learning_rate=0.1, threshold=0):
        # Initialize parameters and hyperparameters
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0
        self.threshold = threshold
        self.epoch = 100

    def linear_prediction(self, features):
        # Y = X.W + b
        return np.dot(features, self.weights) + self.bias

    def threshold_activation(self, accumulated_value: np.ndarray):
        """
        Apply threshold/step activation on a list of "accumulated_value".
        The activation function: 1 if value > threshold else -1

        :param accumulated_value: an array of value
        :return: Applied threshold activation result
        """

        return np.where(accumulated_value > self.threshold, 1, -1)

    def fit(self, features, classes):
        """
        Train the model for given training data.

        :param features: Training features
        :param classes: Training class labels
        :return: The trained model
        """

        # initializing weights based on given number of attributes
        self.weights = np.zeros(features.shape[1])

        for _ in range(self.epoch):
            # Get predictions for given features and current parameters
            _predictions = self.threshold_activation(self.linear_prediction(features))

            # Calculate gradient descent, dw and db
            # dw: error w.r.t weights
            # db: error w.r.t bias
            dw = (1 / len(features)) * np.dot(features.T, (_predictions - classes))
            db = (1 / len(features)) * np.sum(_predictions - classes)

            # Update parameters based on gradient descent
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

        return self

    def predict(self, features):
        """
        Get predicted class labels for given test features.

        :param features: Test features
        :return: Predicted class labels
        """

        _predictions = self.threshold_activation(self.linear_prediction(features))

        return _predictions

    @staticmethod
    def accuracy_score(expected, predicted):
        # return accuracy measure
        return np.sum(expected == predicted) / len(expected)


if __name__ == "__main__":
    # Read training and test data
    training_df = pd.read_csv("./training_data.txt", sep=' ', header=None)
    testing_df = pd.read_csv("./testing_data.txt", sep=' ', header=None)

    # Separate features and class labels for both training and test data
    training_features, training_classes = training_df.iloc[:, :2], training_df.iloc[:, 2]
    testing_features, testing_classes = testing_df.iloc[:, :2], testing_df.iloc[:, 2]

    # Train the model for training data
    perceptron = Perceptron().fit(training_features, training_classes)
    predictions = []

    # Seed the random number generator so that we get the same results each time
    random.seed(0)

    # Iterate through test features to get prediction for each instance
    for i in range(len(testing_features)):
        # Get the instance of testing features and classes at index i
        testing_feature, testing_class = testing_features.iloc[i], testing_classes.iloc[i]

        # Get prediction for the instance at index i
        prediction = perceptron.predict([testing_feature])
        print(f"{list(testing_feature)} Actual label: {testing_class} Predicted label: {prediction[0]}")
        predictions.append(prediction[0])

    # Calculate the prediction accuracy
    accuracy = perceptron.accuracy_score(testing_classes, predictions)

    print(f"Accuracy rate: {round(accuracy, 4) * 100}%")
    print(f"Learned weights are: {list(perceptron.weights)}")
    print(f"Learned bias: {perceptron.bias}")
