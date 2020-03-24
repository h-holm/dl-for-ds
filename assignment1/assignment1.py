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

    # data = {
    #     'X_train': X_train,
    #     'Y_train': Y_train,
    #     'y_train': y_train,
    #     'X_val': X_val,
    #     'Y_val': Y_val,
    #     'y_val': y_val,
    #     'X_test': X_test,
    #     'Y_test': Y_test,
    #     'y_test': y_test
    # }
    #
    # clf = Classifier(data, labels)
    #
    # lambdas = [0, 0, .1, 1]
    # etas = [.1, .01, .01, .01]
    #
    # for i in range(4):
    #     acc_train_set = []
    #     acc_val_set = []
    #     acc_test_set = []
    #     for j in range(10):
    #         acc_train, acc_val, acc_test = clf.mini_batch_gd(
    #                 X_train,
    #                 Y_train,
    #                 labda=lambdas[i],
    #                 eta=etas[i],
    #                 verbose=False)
    #
    #         acc_train_set.append(acc_train)
    #         acc_val_set.append(acc_val)
    #         acc_test_set.append(acc_test)
    #     print("Settting " + str(i) + ":\n")
    #     print("Train mean acc:" + str(statistics.mean(acc_train_set)))
    #     print("Val mean acc:" + str(statistics.mean(acc_val_set)))
    #     print("Test mean acc:" + str(statistics.mean(acc_test_set)))
    #     print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
    #     print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
    #     print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))
    #
    #     np.random.seed(0)
    #
    #     # Param settings 1
    #     clf.mini_batch_gd(X_train, Y_train, title=str(i) + "_cost_plot_test",
    #             labda=lambdas[i], eta=etas[i], plot_performance=True)
    #     clf.plot_learned_images(save=True, num=i)
    #
    # # Unit testing
    # unittest.main()
    print()

    return


if __name__ == '__main__':
    main()
