"""In this assignment you will train and test a one layer network with multiple
outputs to classify images from the CIFAR-10 dataset. You will train the network
using mini-batch gradient descent applied to a cost function that computes the
cross-entropy loss of the classifier applied to the labelled training data and
an L2 regularization term on the weight matrix."""


import pickle
import matplotlib.pyplot as plt
import numpy as np
import unittest
import statistics
import re
from functions import *


__author__ = "Henrik Holm"


def main():
    datasets_folder = "Datasets/cifar-10-batches-py/"

    labels = unpickle(datasets_folder + "batches.meta")[b'label_names']
    print(labels)

    # Training, validation and testing respectively.
    train_dict = load_batch(datasets_folder + "data_batch_1")
    val_dict = load_batch(datasets_folder + "data_batch_2")
    test_dict = load_batch(datasets_folder + "test_batch")

    X_train, Y_train, y_train = split_batch(train_dict)
    X_val, Y_val, y_val = split_batch(val_dict)
    X_test, Y_test, y_test = split_batch(test_dict)

    print(y_train)
    print(X_train)
    print(Y_train)




if __name__ == '__main__':
    main()
