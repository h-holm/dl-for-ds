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


class SingleLayerNetwork():
    """ Single-layer network classifier based on mini-batch gradient descent """

    def __init__(self, labels, data):
        """ W: weight matrix of size K x d
            b: bias matrix of size K x 1 """
        self.labels = labels
        self.K = len(self.labels)

        self.data = data
        self.d = self.data['train_set']['X'].shape[0]
        self.n = self.data['train_set']['X'].shape[1]

        # Initialize as Gaussian random values with 0 mean and 0.01 stdev.
        self.W = np.random.normal(0, 0.01, (self.K, self.d))
        self.b = np.random.normal(0, 0.01, (self.K, 1))

    def evaluate_classifier(self, X):
        """ Implement SoftMax using equations 1 and 2
            Each column of X corresponds to an image and it has size d x n """
        # Element-wise multiplication: @
        s = (self.W @ X) + self.b
        # p has size K x n, where n is n of the input X.
        p = self.soft_max(s)
        return p

    def soft_max(self, s):
    	""" Standard definition of the softmax function """
    	return np.exp(s) / np.sum(np.exp(s), axis=0)


def main():
    np.random.seed(12345)

    print()
    print("------------------------ Loading dataset ------------------------")
    datasets_folder = "Datasets/cifar-10-batches-py/"

    labels = unpickle(datasets_folder + "batches.meta")[b'label_names']

    # Training with 1, validation with 2 and testing with test.
    train_set = load_dataset(datasets_folder, "data_batch_1", num_of_labels=len(labels))
    test_set = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
    val_set = load_dataset(datasets_folder, "test_batch", num_of_labels=len(labels))

    datasets = {'train_set': train_set, 'test_set': test_set, 'val_set': val_set}
    print()
    print("------------------------ Preparing dataset ------------------------")
    for dataset_name, dataset in datasets.items():
        dataset['X'] = preprocess_dataset(dataset['X'])

    clf = SingleLayerNetwork(labels, datasets)
    clf.evaluate_classifier(datasets['train_set']['X'][:, :100])

    print()

    return


if __name__ == '__main__':
    main()
