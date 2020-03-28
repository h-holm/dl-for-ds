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
        """ Implement SoftMax using equations 1 and 2.
            Each column of X corresponds to an image and it has size d x n. """
        # Element-wise multiplication: @
        s = (self.W @ X) + self.b
        # p has size K x n, where n is n of the input X.
        p = self.soft_max(s)
        return p

    def soft_max(self, s):
    	""" Standard definition of the softmax function """
        # Maybe change to identity matrix
    	return np.exp(s) / np.sum(np.exp(s), axis=0)

    def compute_cost(self, X, Y, our_lambda):
        """ Compute cost using the cross-entropy loss.
            - each column of X corresponds to an image and X has size d x N.
            - Y corresponds to the one-hot ground truth label matrix.
            - our_lambda is the regularization term ("lambda" is reserved).
            Returns the cost, which is a scalar. """
        N = X.shape[1]
        # print(X.shape)
        # print(N)
        # print(Y.shape)
        # print(our_lambda)
        p = self.evaluate_classifier(X)
        # If label is encoded as one-hot repr., then cross entropy is -log(yTp).
        cost = ((1 / N) * -np.sum(Y * np.log(p))) + (our_lambda * np.sum(self.W**2))
        return cost

    def compute_accuracy(self, X, y):
        """ Compute classification accuracy
            - each column of X corresponds to an image and X has size d x N.
            - y is a vector pf ground truth labels of length N
            Returns the accuracy. which is a scalar. """
        print(X.shape)
        # print(y)
        print(y.shape)

        print(np.asarray(y))
        N = X.shape[1]
        highest_P = np.argmax(self.evaluate_classifier(X), axis=0)
        count = highest_P.T[highest_P == np.asarray(y)].shape[0]

        return count / N

    def compute_gradients(self, X_batch, Y_batch, our_lambda):
        """ Compute gradients of the weight and bias.
            - X_batch is a D x N matrix
            - Y_batch is a C x N one-hot-encoding vector
            - our_lambda is the regularization term ("lambda" is reserved).
            Returns the gradients of the weight and bias. """
        self.W = self.W[:, :10]

        N = X_batch.shape[1]
        C = Y_batch.shape[0]

        P_batch = self.evaluate_classifier(X_batch)

        # As per the last slide of lecture 3.
        G_batch = - (Y_batch - P_batch)

        grad_W = (1 / N) * (G_batch @ X_batch.T) + (2 * our_lambda * self.W)

        # No regularization term necessary.
        grad_b = np.reshape((1 / N) * (G_batch @ np.ones(N)), (C, 1))

        return grad_W, grad_b

    def compute_gradients_num(self, X_batch, Y_batch, our_lambda=0, h=1e-6):
        """ Compute gradients of the weight and bias numerically.
            - X_batch is a D x N matrix.
            - Y_batch is a C x N one-hot-encoding vector.
            - our_lambda is the regularization term ("lambda" is reserved).
            - h is a marginal offset.
            Returns the gradients of the weight and bias. """

        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)

        b_try = np.copy(self.b)
        W_try = np.copy(self.W)

        for i in range(len(self.b)):
            self.b = b_try
            self.b[i] -= h
            c1 = self.compute_cost(X_batch, Y_batch, our_lambda)
            self.b[i] += (2 * h)
            c2 = self.compute_cost(X_batch, Y_batch, our_lambda)
            grad_b[i] = (c2 - c1) / (2 * h)

        # Given the shape of an array, an ndindex instance iterates over the
        # N-dimensional index of the array. At each iteration a tuple of indices
        # is returned, the last dimension is iterated over first.
        for i in np.ndindex(self.W.shape):
            self.W = W_try
            self.W[i] -= h
            c1 = self.compute_cost(X_batch, Y_batch, our_lambda)
            self.W[i] += (2 * h)
            c2 = self.compute_cost(X_batch, Y_batch, our_lambda)
            grad_W[i] = (c2 - c1) / (2 * h)

        return grad_W, grad_b


def main():
    np.random.seed(12345)
    test_numerically = False

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
    p = clf.evaluate_classifier(datasets['train_set']['X'][:, :100])
    # print(p)

    if test_numerically:
        grad_W, grad_b = clf.compute_gradients(datasets['train_set']['X'][:10, :2],
                                               datasets['train_set']['Y'][:, :2],
                                               our_lambda=0)
        grad_W_num, grad_b_num = clf.compute_gradients_num(datasets['train_set']['X'][:10, :2],
                                                           datasets['train_set']['Y'][:, :2],
                                                           our_lambda=0)

        # From the assignment PDF: "If all these absolutes difference are small
        # (<1e-6), then they have produced the same result.
        # np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
        print(np.allclose(grad_W, grad_W_num, atol=1e-07))
        print(grad_W - grad_W_num)

    print()

    return


if __name__ == '__main__':
    main()
