##############################################################################
# Name:           generator.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:    Randomly generates data for Classification and Regression as per
#                   the formulas described in the assignment details
# Date:           6 March 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 2
##############################################################################
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

## init constants

np.random.seed(0)
number_samples = 2500
number_dimensions = 10

training_size = 2000
batch_size = 64


def gen_features() -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
    """
    Generates data sets for regression and classification to a Gaussian Distribution of features
    :return: Training and Testing DataLoaders
    """
    print("Generating data...")

    # Generate Features
    X = torch.randn((number_samples, number_dimensions))            #output

    # Generate Classification Target
    w1 = torch.randn(number_dimensions)
    w2 = torch.randn(number_dimensions)

    epsilon = torch.normal(mean=0, std=0.1, size=(number_samples,))

    class_calc = torch.sin(X @ w1) + 0.5 * (X @ w2) + epsilon   # given formula

    classifications = (class_calc > 0).int()                        #output

    # Generate Regression Target
    w = torch.randn(number_dimensions)

    epsilon = torch.normal(mean=0, std=0.1, size=(number_samples,))
                                                                # given formula
    targets = torch.sin(X @ w) + 0.1 * (X.norm(dim=1)**2) + epsilon #output

    # divide data sets
    training_set = TensorDataset(
        (X[:training_size] - X[:training_size].mean(dim=0)) / X[:training_size].std(dim=0),     # normalize input features (0 mean, unit variance)
        classifications[:training_size],
        targets[:training_size]
    )

    test_set = TensorDataset(
        X[training_size:],
        classifications[training_size:],
        targets[training_size:]
    )

    training_set = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return training_set, test_set


if __name__ == "__main__":
    gen_features()
