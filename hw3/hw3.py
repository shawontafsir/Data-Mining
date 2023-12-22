"""
Name: Khalid Hasan
BearPass: M03543550

How to run:
1. Place the source file "hw3.py", "MNIST_test.csv", and "MNIST_train.csv" in the same directory.
2. Open a terminal (or, command-prompt) in this directory and run the following command:
    python hw3.py

**Make sure, you have python and python's numpy and pandas library installed to run this command.

Summary:
1. This file contains the "KNeighborsClassifier" class for the K-Nearest Neighbors (KNN) algorithm implementation
    using the brute-force technique.
2. The function named "fit" stores the training attributes and target.
3. The function named "predict" predicts the class labels of given test data using KNN algorithm which uses Euclidean
    distance for the distance metric.
4. The Evaluation.get_best_K_value() is used for getting the best K value with the best accuracy for validation samples.
    To calculate the best K:
    (a) The training data was split into training and validation set.
    (b) Using dun-ham's rule of thumb, for a range (1, sqrt(number of training samples)) of K,
        the KNN model was predicted for the split validation samples.
    (c) Retrieve the K with the best accuracy and print the K value with the accuracy.
4. The main function calls the KNeighborsClassifier class to predict the test data
    and prints some evaluation statistics along with desired and computed classes.
"""

import numpy as np
import pandas as pd


class KNeighborsClassifier:

    def __init__(self, n_neighbors):
        self.X_train = None
        self.y_train = None
        self.K = n_neighbors

    @staticmethod
    def euclidean_distance(_training_df: pd.DataFrame, _test_point: pd.Series) -> list[set[float, int]]:
        """
        Measure Euclidean distance from a point (Series) to a set of data points (DataFrame).
        distance(p, q) = sqrt(sum(p-q)***2)

        :param _training_df: A dataframe consisting a set of points: (samples length * features)
        :param _test_point: A series of features: (features,)
        :return: A list of tuple(Euclidean distance, index) order by distance: (samples length * 2)
        """
        distances = np.sqrt(np.sum((_training_df - _test_point) ** 2, axis=1))

        return sorted(zip(distances, _training_df.index))

    @staticmethod
    def weight_factor(distance: float) -> float:
        """
        Measure Weight Factor: 1 / (distance)**2

        :param distance: A float value
        :return: the weight factor of the distance
        """
        return 1 / (distance ** 2)

    def fit(self, _X_train: pd.DataFrame, _y_train: pd.Series):
        """
        Save the training data

        :param _X_train: training attribute samples
        :param _y_train: training target samples
        :return: the own class instance
        """
        self.X_train, self.y_train = _X_train, _y_train

        return self

    def predict(self, new_observations: pd.DataFrame):
        """
        The K Nearest Neighbors algorithm to classify instances

        :param new_observations: A dataframe of test features
        :return: The class labels of the test samples
        """
        # A list to store predictions order by given new_observations
        _predictions = []

        for _, new_observation in new_observations.iterrows():
            # Get Euclidean distance for each training samples in ascending order
            distances = self.euclidean_distance(self.X_train, new_observation)

            # Get the first K-smallest neighbors
            neighbors = distances[:self.K]

            # Mapping for weights of classes from neighbors
            weighted_classes = dict()

            # Iterate through neighbors to calculate the weights of classes
            for neighbor in neighbors:
                # The distance and index of neighbor
                distance, index = neighbor[0], neighbor[1]

                # The class label of neighbor accessing with direct index
                label = self.y_train[index]

                # Update the weights of class labels by using weight factor
                weighted_classes[label] = weighted_classes.get(label, 0) + self.weight_factor(distance)

            # Get the predicted class label from the measured weights of classes
            max_weighted_class, max_weight = None, 0
            for class_label, weight in weighted_classes.items():
                if weight > max_weight:
                    max_weighted_class, max_weight = class_label, weight

            # Store the predicted class
            _predictions.append(max_weighted_class)

        return _predictions


class Evaluation:
    """
    This class includes utils and evaluation measures
    """

    @staticmethod
    def feature_target_split(df: pd.DataFrame):
        # Return feature and target separately
        return df.iloc[:, 1:], df.iloc[:, 0]

    @staticmethod
    def accuracy_score(expected, predicted):
        # return accuracy measure
        return np.sum(expected == predicted) / len(expected)

    @classmethod
    def train_test_split(cls, df: pd.DataFrame, test_size=0.2) -> object:
        """
        Split into train and test samples considering random measures

        :param df: The data-set to be split
        :param test_size: The fraction of given data-set to be testing samples
        :return: a set of (train feature, train target, test feature, test target)
        """
        # Creating random samples with test_size fraction of original dataframe
        _test_df = df.sample(frac=test_size)

        # Creating dataframe with rest of the samples
        _train_df = df.drop(_test_df.index)

        # get split feature and target for training and test data
        _train_feature, _train_target = cls.feature_target_split(_train_df)
        _test_feature, _test_target = cls.feature_target_split(_test_df)

        # return split train and test dataframes in feature and target form
        return _train_feature, _train_target, _test_feature, _test_target

    @classmethod
    def get_best_K_value(cls, _training_data: pd.DataFrame):
        # Initialize training and validation data and target
        _X_train, _y_train, _X_validation, _y_validation = cls.train_test_split(_training_data)

        # Taking K values in consideration based on Dun-ham's rule of thumb
        K_values = [k for k in range(1, int(np.sqrt(len(_X_train))))]
        _best_K, _best_accuracy = None, 0

        for k in K_values:
            print(f"Checking accuracy for K: {k}")

            # Fit with training data with k neighbors
            _knn = KNeighborsClassifier(n_neighbors=k).fit(_X_train, _y_train)

            # Predict and get accuracy with validation samples
            _y_pred = _knn.predict(_X_validation)
            _accuracy = cls.accuracy_score(_y_validation, _y_pred)

            # Get the best k with the best accuracy
            if _accuracy > _best_accuracy:
                _best_K, _best_accuracy = k, _accuracy

        print(f"After validation, found K={_best_K} with the best accuracy {round(_best_accuracy, 4) * 100}%")

        return _best_K


if __name__ == "__main__":
    # Read training and test data
    training_df = pd.read_csv("./MNIST_train.csv")
    testing_df = pd.read_csv("./MNIST_test.csv")

    # Evaluate the best K value
    best_K = Evaluation.get_best_K_value(training_df)

    # Initialize training data and target
    X_train, y_train = Evaluation.feature_target_split(training_df)
    # Initialize test data and target
    X_test, y_test = Evaluation.feature_target_split(testing_df)

    # Fit the model
    knn = KNeighborsClassifier(n_neighbors=best_K).fit(X_train, y_train)
    # Get predictions
    y_pred = knn.predict(X_test)

    print(f"K = {knn.K}")

    # Get correctly classified test samples
    correctly_classified = 0
    for i in range(len(y_pred)):
        print(f"Desired class: {y_test[i]} Computed class: {y_pred[i]}")

        if y_test[i] == y_pred[i]:
            correctly_classified += 1

    misclassified = len(y_test) - correctly_classified
    accuracy = correctly_classified / len(y_pred)

    print(f"Accuracy rate: {round(accuracy, 4) * 100}%")
    print(f"Number of misclassified test samples: {misclassified}")
    print(f"Total number of test samples: {len(X_test)}")
