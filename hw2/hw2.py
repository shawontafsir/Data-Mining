"""
Name: Khalid Hasan
BearPass: M03543550

How to run:
1. Place the source file "hw2.py" and "input.csv" in a directory.
2. Open a terminal (or, command-prompt) in this directory and run the following command:
    python hw2.py

**Make sure, you have python and python's pandas library installed to run this command.

Summary:
This file contains a part of ID3 algorithm implementation: a function named "get_best_test_attribute"
for getting the best test attribute and information gain at the root of the decision tree
from a given dataset named "input.csv".
"""


import math
import pandas as pd


class ID3:

    @staticmethod
    def get_entropy(dataframe: pd.DataFrame):
        attributes = dataframe.columns
        entropy = 0  # Initialize resultant entropy

        # Group data set by class labels located at the last column
        for class_label, group in dataframe.groupby(attributes[-1]):
            # Measure probability of a class
            probability = group.size / dataframe.size

            # Measure entropy for a class
            class_entropy = probability * math.log2(probability)

            # Update the resultant entropy
            entropy -= class_entropy

        return entropy

    @classmethod
    def get_best_test_attribute(cls, dataframe: pd.DataFrame):
        attributes = dataframe.columns

        # Initialize min entropy and the best attribute
        min_entropy = float("inf")
        best_attribute = None

        # Iterate over data set attributes
        for attribute in attributes[:-1]:
            attribute_entropy = 0

            # Group data set by attribute values
            for key, group in dataframe.groupby(attribute):
                # Update attribute entropy based on the entropy for an attribute value
                attribute_entropy += group.size / dataframe.size * cls.get_entropy(group)

            # If minimum entropy is found for an attribute,
            # update the global min entropy and best attribute
            if attribute_entropy < min_entropy:
                min_entropy = min(min_entropy, attribute_entropy)
                best_attribute = attribute

        # Get information gain
        information_gain = cls.get_entropy(dataframe) - min_entropy

        return best_attribute, round(information_gain, 3)


if __name__ == "__main__":
    df = pd.read_csv("./input.csv")
    best_attribute, information_gain = ID3.get_best_test_attribute(df)

    print(f"Best test attribute is {best_attribute}")
    print(f"Information gained by splitting on {best_attribute} is {information_gain}")
