"""In this assignment you will train and test a one layer network with multiple
outputs to classify images from the CIFAR-10 dataset. You will train the network
using mini-batch gradient descent applied to a cost function that computes the
cross-entropy loss of the classifier applied to the labelled training data and
an L2 regularization term on the weight matrix."""


import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Henrik Holm"


def load_batch(filepath):
	""" Copied from the dataset website """
	with open(filepath, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def split_batch(batch, num_of_labels):
	""" Split the input batch into its labels and its data
	X: image pixel data, size d x n, entries between 0 and 1. n is number of
	   images (10000) and d dimensionality of each image (3072 = 32 * 32 * 3).
	Y: K x n (with K = # of labels = 10) and contains the one-hot
	   representation of the label for each image (0 to 9).
	y: vector of length n containing label for each image, 0 to 9. """

	X = (batch[b'data'] / 255).T # Normalize by dividing over 255.
	y = np.asarray(batch[b'labels'])
	Y = (np.eye(num_of_labels)[y]).T

	return X, Y, y


def load_dataset(folder, filename, num_of_labels):
	""" Load a batch and split it before returning it as a dictionary
	folder: folder where data is located.
	filename: name of file.
	num_of_labels: number of labels in data. """
	dataset_dict = load_batch(folder + filename)
	X, Y, y = split_batch(dataset_dict, num_of_labels=num_of_labels)
	dataset = {'X': X, 'Y': Y, 'y': y}

	return dataset


def normalize_dataset(data, verbose=False):
	""" Pre-process data by normalizing """
	# "Both mean_X and std_X have size d x 1".
	mean_X = np.mean(data, axis=1)
	std_X = np.std(data, axis=1)

	data = data - np.array([mean_X]).T
	data = data / np.array([std_X]).T

	if verbose:
		print()
		print(f'Mean after normalizing:\t{np.mean(data)}')
		print(f'Std after normalizing:\t{np.std(data)}')

	return data


def unpickle(filename):
	""" Unpickle a file """
	with open(filename, 'rb') as f:
		file_dict = pickle.load(f, encoding='bytes')

	return file_dict


def montage(W, title, labels):
	""" Display the image for each label in W """
	fig, ax = plt.subplots(2, 5)
	for i in range(2):
		for j in range(5):
			im  = W[((i * 5) + j), :].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title(str(labels[((5 * i) + j)]), fontsize=10)
			ax[i][j].axis('off')

	plt.savefig(f'plots/weights_{title}.png', bbox_inches="tight")
	plt.show()

	return


def plot_lines(line_A, line_B, label_A, label_B, xlabel, ylabel, title, show=False):
   """ Plots performance curves """
   assert(line_A.shape == line_B.shape)

   fig, ax = plt.subplots(figsize=(10, 8))
   ax.plot(range(len(line_A)), line_A, label=label_A)
   ax.plot(range(len(line_B)), line_B, label=label_B)
   ax.legend()

   ax.set(xlabel=xlabel, ylabel=ylabel)
   ax.grid()

   plt.savefig(f'plots/{title}.png', bbox_inches="tight")

   if show:
	   plt.show()

   return plt


def plot_three_subplots(costs, losses, accuracies, title, show=False):
	fig, (ax_costs, ax_losses, ax_accuracies) = plt.subplots(1, 3, figsize=(16, 6))
	fig.suptitle(title)

	xlabel = 'epoch'
	label_A = 'training'
	label_B = 'validation'

	# ax_costs.plot(costs[0], costs[1])
	ax_costs.plot(range(len(costs[0])), costs[0], label=label_A + ' cost')
	ax_costs.plot(range(len(costs[1])), costs[1], label=label_B + ' cost')
	ax_costs.legend()
	ax_costs.set(xlabel=xlabel, ylabel='cost')
	ax_costs.grid()

	# ax_losses.plot(losses[0], losses[1])
	ax_losses.plot(range(len(losses[0])), losses[0], label=label_A + ' loss')
	ax_losses.plot(range(len(losses[1])), losses[1], label=label_B + ' loss')
	ax_losses.legend()
	ax_losses.set(xlabel=xlabel, ylabel='loss')
	ax_losses.grid()

	# ax_accuracies.plot(accuracies[0], accuracies[1])
	ax_accuracies.plot(range(len(accuracies[0])), accuracies[0], label=label_A + ' accuracy')
	ax_accuracies.plot(range(len(accuracies[1])), accuracies[1], label=label_B + ' accuracy')
	ax_accuracies.legend()
	ax_accuracies.set(xlabel=xlabel, ylabel='accuracy')
	ax_accuracies.grid()

	plt.savefig(f'plots/{title}.png', bbox_inches="tight")

	if show:
		plt.show()

	return


class KLayerNetwork():
	""" K-layer network classifier based on mini-batch gradient descent """

	def __init__(self, labels, data, alpha=0.9, batch_norm=False, verbose=0):
		""" W: weight matrix of size K x d
			b: bias matrix of size K x 1 """
		self.alpha = alpha
		self.batch_norm = batch_norm
		self.verbose = verbose

		self.labels = labels
		K = len(self.labels)

		self.data = data
		d = self.data['train_set']['X'].shape[0]

		self.layers = layers
		self.k = len(layers) - 1

		# Initialize as Gaussian random values with 0 mean and 1/sqrt(d) stdev.
		self.W1 = np.random.normal(0, 1 / np.sqrt(d), (m, d))	# (m, d)

		# Initialize as Gaussian random values with 0 mean and 1/sqrt(m) stdev.
		self.W2 = np.random.normal(0, 1 / np.sqrt(m), (K, m))	# (K, m)

		# "Set biases equal to zero"
		self.b1 = np.zeros((m, 1))	# (m, 1)
		self.b2 = np.zeros((K, 1))	# (K, 1)

		if self.verbose:
			print()
			print(f'Shape of W1:\t\t{self.W1.shape}')
			print(f'Shape of W2:\t\t{self.W2.shape}')
			print(f'Shape of b1:\t\t{self.b1.shape}')
			print(f'Shape of b2:\t\t{self.b2.shape}')

	def __soft_max(self, s):
		""" Standard definition of the softmax function """
		# return np.exp(s) / np.sum(np.exp(s), axis=0)
		return np.exp(s - np.max(s, axis=0)) / np.exp(s - np.max(s, axis=0)).sum(axis=0)

	def __relu(self, s):
		""" Standard definition of the softmax function """
		return np.maximum(s, 0)

	def __evaluate_classifier(self, X):
		""" Implement SoftMax using equations 1 and 2.
			Each column of X corresponds to an image and it has size d x n. """
		s1 = np.dot(self.W1, X) + self.b1
		h = self.__relu(s1)

		s2 = np.dot(self.W2, h) + self.b2
		p = self.__soft_max(s2)

		if self.verbose > 1:
			print()
			print(f'Shape of s1:\t\t{s1.shape}')
			print(f'Shape of h:\t\t{h.shape}')
			print(f'Shape of s2:\t\t{s2.shape}')
			print(f'Shape of p:\t\t{p.shape}')

		return h, p

	def __compute_loss_and_cost(self, X, Y, our_lambda):
		""" Compute cost using the cross-entropy loss.
			- each column of X corresponds to an image and X has size d x N.
			- Y corresponds to the one-hot ground truth label matrix.
			- our_lambda is the regularization term ("lambda" is reserved).
			Returns the cost, which is a scalar. """
		N = X.shape[1]
		_, p = self.__evaluate_classifier(X)

		# If label is encoded as one-hot repr., then cross entropy is -log(yTp).
		loss = (1 / N) * (-np.sum(Y * np.log(p)))
		reg = our_lambda * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
		cost = loss + reg

		return loss, cost

	def __compute_accuracy(self, X, y):
		""" Compute classification accuracy
			- each column of X corresponds to an image and X has size d x N.
			- y is a vector pf ground truth labels of length N
			Returns the accuracy. which is a scalar. """
		N = X.shape[1]
		highest_P = np.argmax(self.__evaluate_classifier(X)[1], axis=0)
		count = highest_P.T[highest_P == np.asarray(y)].shape[0]

		return count / N

	def compute_gradients(self, X_batch, Y_batch, our_lambda):
		N = X_batch.shape[1]
		K = Y_batch.shape[0]

		# 1) evalutate the network (the forward pass)
		H_batch, P_batch = self.__evaluate_classifier(X_batch)

		# 2) compute the gradients (the backward pass)
		# Page 49 in Lecture4.pdf
		G_batch = -(Y_batch - P_batch)

		grad_W2 = (1 / N) * np.dot(G_batch, H_batch.T) + (2 * our_lambda * self.W2)
		grad_b2 = np.reshape((1 / N) * np.dot(G_batch, np.ones(N)), (K, 1))

		# G_batch = self.W2.T@G_batch
		G_batch = np.dot(self.W2.T, G_batch)
		H_batch = np.maximum(H_batch, 0)
		# H_batch[H_batch <= 0] = 0

		# Indicator function on H_batch to yield only values larger than 0.
		G_batch = np.multiply(G_batch, H_batch > 0)

		# No need to multiply by 2
		grad_W1 = (1 / N) * np.dot(G_batch, X_batch.T) + (our_lambda * self.W1)
		# grad_W1 = (1 / N) * np.dot(G_batch, X_batch.T) + (2 * our_lambda * self.W1)
		grad_b1 = np.reshape((1 / N) * np.dot(G_batch, np.ones(N)), (self.m, 1))

		if self.verbose > 1:
			print()
			print(f'shape of grad_W1:\t{grad_W1.shape}')
			print(f'shape of grad_W2:\t{grad_W2.shape}')
			print(f'shape of grad_b1:\t{grad_b1.shape}')
			print(f'shape of grad_b2:\t{grad_b2.shape}')

		return grad_W1, grad_b1, grad_W2, grad_b2

	def compute_gradients_num(self, X_batch, Y_batch, our_lambda=0, h=1e-5):
		""" Compute gradients of the weight and bias numerically.
			- X_batch is a D x N matrix.
			- Y_batch is a C x N one-hot-encoding vector.
			- our_lambda is the regularization term ("lambda" is reserved).
			- h is a marginal offset.
			Returns the gradients of the weight and bias. """

		bs, Ws = dict(), dict()

		for i in range(1, 3):
			b_string = 'b' + str(i)
			W_string = 'W' + str(i)

			b = getattr(self, b_string)
			W = getattr(self, W_string)

			bs[b_string] = np.zeros(b.shape)
			Ws[W_string] = np.zeros(W.shape)

			b_try = np.copy(b)
			W_try = np.copy(W)

			for j in range(len(b)):
				b = b_try[:]
				b[j] -= h
				# b[j] += h
				_, c1 = self.__compute_loss_and_cost(X_batch, Y_batch, our_lambda)
				getattr(self, b_string)[j] += (2 * h)
				# getattr(self, b_string)[j] -= (2 * h)
				_, c2 = self.__compute_loss_and_cost(X_batch, Y_batch, our_lambda)
				bs[b_string][j] = (c2 - c1) / (2 * h)

			# Given the shape of an array, an ndindex instance iterates over the
			# N-dimensional index of the array. At each iteration a tuple of indices
			# is returned, the last dimension is iterated over first.
			for j in np.ndindex(W.shape):
				self.W = W_try[:, :]
				self.W[j] -= h
				# self.W[j] += h
				_, c1 = self.__compute_loss_and_cost(X_batch, Y_batch, our_lambda)
				getattr(self, W_string)[j] += (2 * h)
				# getattr(self, W_string)[j] -= (2 * h)
				_, c2 = self.__compute_loss_and_cost(X_batch, Y_batch, our_lambda)
				Ws[W_string][j] = (c2 - c1) / (2 * h)

		return Ws['W1'], bs['b1'], Ws['W2'], bs['b2']

	def mini_batch_gradient_descent(self, X, Y, our_lambda=0, n_batch=100, eta_min=1e-5, eta_max=1e-1, n_s=500, n_epochs=20):
		""" Learn the model by performing mini-batch gradient descent
			our_lambda 	- regularization term
			n_batch 	- number of examples per mini-batch
			eta_min		- minimum learning rate
			eta_max		- maximum learning rate
			n_s			- step size
			n_epochs 	- number of runs through the whole training set """

		accuracies = dict()
		accuracies['train'] = np.zeros(n_epochs)
		accuracies['val'] = np.zeros(n_epochs)
		accuracies['test'] = np.zeros(n_epochs)

		list_settings = list()

		if self.verbose:
			print()

			print(f'Accuracy training:\t{round(self.__compute_accuracy(self.data["train_set"]["X"], self.data["train_set"]["y"]), 4)}')
			print(f'Accuracy validation:\t{round(self.__compute_accuracy(self.data["val_set"]["X"], self.data["val_set"]["y"]), 4)}')
			print(f'Accuracy testing:\t{round(self.__compute_accuracy(self.data["test_set"]["X"], self.data["test_set"]["y"]), 4)}')

		losses, costs = dict(), dict()
		losses['train'], costs['train'] = np.zeros(n_epochs), np.zeros(n_epochs)
		losses['val'], costs['val'] = np.zeros(n_epochs), np.zeros(n_epochs)

		eta = eta_min

		t = 0
		for n in range(n_epochs):
			for i in range(n_batch):
				N = int(X.shape[1] / n_batch)
				i_start = (i) * N
				i_end = (i + 1) * N

				X_batch = X[:, i_start:i_end]
				Y_batch = Y[:, i_start:i_end]

				grad_W1, grad_b1, grad_W2, grad_b2 = \
				self.compute_gradients(X_batch, Y_batch, our_lambda)

				self.W1 -= eta * grad_W1
				self.b1 -= eta * grad_b1
				self.W2 -= eta * grad_W2
				self.b2 -= eta * grad_b2

				# Equations (14) and (15).
				if t <= n_s:
					eta = eta_min + ((t / n_s) * (eta_max - eta_min))
				elif t <= (2 * n_s):
					eta = eta_max - (((t - n_s) / n_s) * (eta_max - eta_min))
				t = (t + 1) % (2 * n_s)

				# if t == (n_s + 1) or t == (2 * n_s - 1):
				if t == (2 * n_s - 1):
					tr_acc = self.__compute_accuracy(self.data['train_set']['X'],
													 self.data['train_set']['y'])
					v_acc = self.__compute_accuracy(self.data['val_set']['X'],
													self.data['val_set']['y'])
					te_acc = self.__compute_accuracy(self.data['test_set']['X'],
													 self.data['test_set']['y'])

					settings = {'t': t, 'our_lambda': our_lambda, 'eta': eta,
								'eta_min': eta_min, 'eta_max': eta_max,
								'n_batch': n_batch, 'n_s': n_s, 'n_epochs': n_epochs,
								'tr-acc': tr_acc, 'v-acc': v_acc, 'te-acc': te_acc}

					list_settings.append(settings)

			losses['train'][n], costs['train'][n] = \
			self.__compute_loss_and_cost(X, Y, our_lambda)

			losses['val'][n], costs['val'][n] = \
			self.__compute_loss_and_cost(self.data['val_set']['X'],
										 self.data['val_set']['Y'],
										 our_lambda)

			accuracies['train'][n] = self.__compute_accuracy(self.data['train_set']['X'],
															 self.data['train_set']['y'])
			accuracies['val'][n] = self.__compute_accuracy(self.data['val_set']['X'],
														   self.data['val_set']['y'])
			accuracies['test'][n] = self.__compute_accuracy(self.data['test_set']['X'],
															self.data['test_set']['y'])

			if self.verbose:
				print()
				print(f'Loss training:\t\t{round(losses["train"][n], 4)}')
				print(f'Cost training:\t\t{round(costs["train"][n], 4)}')
				print(f'Loss validation:\t{round(losses["val"][n], 4)}')
				print(f'Cost validation:\t{round(costs["val"][n], 4)}')
				print(f'Accuracy training:\t{round(accuracies["train"][n], 4)}')
				print(f'Accuracy validation:\t{round(accuracies["val"][n], 4)}')
				# print(f'Accuracy testing:\t{accuracies["test"][n]}')

			# print(f'Current learning rate: {eta}')

		settings = {'t': t, 'our_lambda': our_lambda, 'eta': eta,
					'eta_min': eta_min, 'eta_max': eta_max, 'n_batch': n_batch,
					'n_s': n_s, 'n_epochs': n_epochs,
					'tr-acc': accuracies["train"][-1],
					'v-acc': accuracies["val"][-1],
					'te-acc': accuracies["test"][-1]}
		list_settings.append(settings)


		return accuracies, costs, losses, list_settings


def main():
	seed = 12345
	np.random.seed(seed)
	test_numerically = False
	sanity_check = False # Deprecated
	fig_3 = False
	fig_4 = False
	search = False
	best = False
	find_end_of_cycle_settings = False
	assignment_3 = True

	if test_numerically or sanity_check or fig_3 or fig_4:
		search, best, find_end_of_cycle_settings = False, False, False

	print()
	print("------------------------ Loading dataset ------------------------")
	datasets_folder = "Datasets/cifar-10-batches-py/"
	labels = unpickle(datasets_folder + "batches.meta")[b'label_names']

	if search or best or find_end_of_cycle_settings or assignment_3:
		if search:
			num_val = 5000
		else:
			num_val = 1000

		# Use all available data for training. Reduce validation to num_val.
		train_set_1 = load_dataset(datasets_folder, "data_batch_1", num_of_labels=len(labels))
		train_set_2 = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
		train_set_3 = load_dataset(datasets_folder, "data_batch_3", num_of_labels=len(labels))
		train_set_4 = load_dataset(datasets_folder, "data_batch_4", num_of_labels=len(labels))
		train_set_5 = load_dataset(datasets_folder, "data_batch_5", num_of_labels=len(labels))

		train_set = dict()
		train_set['X'] = np.concatenate((train_set_1['X'], train_set_2['X'], train_set_3['X'], train_set_4['X'], train_set_5['X']), axis=1)
		train_set['Y'] = np.concatenate((train_set_1['Y'], train_set_2['Y'], train_set_3['Y'], train_set_4['Y'], train_set_5['Y']), axis=1)
		train_set['y'] = np.concatenate((train_set_1['y'], train_set_2['y'], train_set_3['y'], train_set_4['y'], train_set_5['y']))

		# Use last num_val for validation ...
		val_set = dict()
		val_set['X'] = train_set['X'][:, -num_val:]
		val_set['Y'] = train_set['Y'][:, -num_val:]
		val_set['y'] = train_set['y'][-num_val:]

		# ... and subsequently remove them from the training data.
		train_set['X'] = train_set['X'][:, :-num_val]
		train_set['Y'] = train_set['Y'][:, :-num_val]
		train_set['y'] = train_set['y'][:-num_val]

		test_set = load_dataset(datasets_folder, "test_batch", num_of_labels=len(labels))

		datasets = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
	else:
		train_set = load_dataset(datasets_folder, "data_batch_1", num_of_labels=len(labels))
		val_set = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
		test_set = load_dataset(datasets_folder, "test_batch", num_of_labels=len(labels))

		datasets = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}

	print()
	print("---------------------- Normalizing dataset ----------------------")
	for dataset_name, dataset in datasets.items():
		dataset['X'] = normalize_dataset(dataset['X'], verbose=1)

	if test_numerically:
		print()
		print("-------------------- Running gradient tests ---------------------")
		our_lambda = 0.01
		num_nodes = 50 # Number of nodes in the hidden layer

		num_pixels = 20
		num_images = 10
		atol = 1e-05

		train_set['X'] = train_set['X'][:num_pixels, :num_images]
		train_set['Y'] = train_set['Y'][:num_pixels, :num_images]

		X_batch = train_set['X']
		Y_batch = train_set['Y']

		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=1)

		grad_W1, grad_b1, grad_W2, grad_b2 = clf.compute_gradients(X_batch,
																   Y_batch,
																   our_lambda=0)

		grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = \
		clf.compute_gradients_num(X_batch, Y_batch, our_lambda=0)

		# From the assignment PDF: "If all these absolutes difference are small
		# (<1e-6), then they have produced the same result.
		# np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
		print()
		print('grad_W1')
		print(grad_W1)
		print('grad_W1_num')
		print(grad_W1_num)
		print()
		print('grad_W2')
		print(grad_W2[:5, :20])
		print('grad_W2_num')
		print(grad_W2_num[:5, :20])
		print()
		print(f'All close: {np.allclose(grad_W1, grad_W1_num, atol=atol)}')
		print(f'All close: {np.allclose(grad_W2, grad_W2_num, atol=atol)}')
		# WRITE: With atol 1e-04 we get all to be close.
		quit()

	print()
	print("-------------------- Instantiating classifier -------------------")

	print()
	print("---------------------- Learning classifier ----------------------")
	# Has been deprecated. Was used to check if model could overfit. It could.
	if sanity_check:
		print()
		print("------------------------ Sanity check ------------------------")
		our_lambda = 0.01
		n_epochs = 60
		n_batch = 100
		eta = 0.001
		n_s = 500
		decay_factor = 1.0
		num_nodes = 50 # Number of nodes in the hidden layer
		test_numerically = False
		sanity_check = False
		fig_3 = False
		fig_4 = True
		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=1)
		# See if we can overfit, i.e. achieve a very small loss on the training
		# data by training on the following 100 examples.
		num_pixels = 3072
		num_images = 100
		train_set['X'] = train_set['X'][:num_pixels, :num_images]
		train_set['Y'] = train_set['Y'][:num_pixels, :num_images]

	if fig_3:
		print()
		print("------------------------- Figure 3 -------------------------")
		our_lambda = 0.01
		n_epochs = 10
		n_batch = 100
		eta_min = 1e-5
		eta_max = 1e-1
		n_s = 500
		num_nodes = 50 # Number of nodes in the hidden layer

		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=1)

		accuracies, costs, losses, _ = \
		clf.mini_batch_gradient_descent(datasets['train_set']['X'],
										datasets['train_set']['Y'],
										our_lambda=our_lambda,
										n_batch=n_batch,
										eta_min=eta_min,
										eta_max=eta_max,
										n_s=n_s,
										n_epochs=n_epochs)

		tracc = round(accuracies["train"][-1], 4)
		vacc = round(accuracies["val"][-1], 4)
		teacc = round(accuracies["test"][-1], 4)

		print()
		print(f'Final training data accuracy:\t\t{tracc}')
		print(f'Final validation data accuracy:\t\t{vacc}')
		print(f'Final test data accuracy:\t\t{teacc}')

		title = f'lambda{our_lambda}_n-batch{n_batch}_n-epochs{n_epochs}_n-s{n_s}_m{num_nodes}_eta-min{eta_min}_eta-max{eta_max}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'

		plot_three_subplots(costs=(costs['train'], costs['val']),
							losses=(losses['train'], losses['val']),
							accuracies=(accuracies['train'], accuracies['val']),
							title='fig3_' + title)

	if fig_4:
		print()
		print("------------------------- Figure 4 -------------------------")
		our_lambda = 0.01
		n_epochs = 50
		n_batch = 100
		eta_min = 1e-5
		eta_max = 1e-1
		n_s = 800
		num_nodes = 50 # Number of nodes in the hidden layer

		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=1)

		accuracies, costs, losses, _ = \
		clf.mini_batch_gradient_descent(datasets['train_set']['X'],
										datasets['train_set']['Y'],
										our_lambda=our_lambda,
										n_batch=n_batch,
										eta_min=eta_min,
										eta_max=eta_max,
										n_s=n_s,
										n_epochs=n_epochs)

		tracc = round(accuracies["train"][-1], 4)
		vacc = round(accuracies["val"][-1], 4)
		teacc = round(accuracies["test"][-1], 4)

		print()
		print(f'Final training data accuracy:\t\t{tracc}')
		print(f'Final validation data accuracy:\t\t{vacc}')
		print(f'Final test data accuracy:\t\t{teacc}')

		title = f'lambda{our_lambda}_n-batch{n_batch}_n-epochs{n_epochs}_n-s{n_s}_m{num_nodes}_eta-min{eta_min}_eta-max{eta_max}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'

		plot_three_subplots(costs=(costs['train'], costs['val']),
							losses=(losses['train'], losses['val']),
							accuracies=(accuracies['train'], accuracies['val']),
							title='fig4_' + title)

	if search:
		print()
		print("------------------------- Exercise 4 -------------------------")
		coarse = False
		results_file = 'results/results.csv'
		# If file not exists, create it with its headers.
		if not os.path.exists(results_file):
			headers = ['top_5_vacc', 'tracc', 'vacc', 'teacc', 'lambda', 'n_batch',
					   'eta_min', 'eta_max', 'm', 'n_s', 'n_epochs', 'seed']
			with open(results_file, 'w+') as f:
				writer = csv.writer(f, dialect='excel', delimiter=';')
				writer.writerow(headers)

		n_batch = 100
		eta_min = 1e-5
		eta_max = 1e-1
		num_nodes = 50 # Number of nodes in the hidden layer

		# As per Assignment PDF.
		n_s = 2 * int(np.floor(datasets['train_set']['X'].shape[1] / n_batch))

		# Number of epochs set to equal two cycles.
		n_epochs = int(4 * (n_s / n_batch))

		if coarse:
			# Coarse lambda search.
			lambda_min = 1e-5
			lambda_max = 1e-1
			lambdas = np.linspace(lambda_min, lambda_max, 8)
		else:
			# Fine lambda search.
			lambda_min = 1e-5
			lambda_max = 0.0285785714285714 # Pasted in from coarse results.
			lambdas = np.linspace(lambda_min, lambda_max, 8)

		for our_lambda in lambdas:
			our_lambda = round(our_lambda, 4)
			clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=0)

			accuracies, costs, losses, _ = \
			clf.mini_batch_gradient_descent(datasets['train_set']['X'],
											datasets['train_set']['Y'],
											our_lambda=our_lambda,
											n_batch=n_batch,
											eta_min=eta_min,
											eta_max=eta_max,
											n_s=n_s,
											n_epochs=n_epochs)

			tracc = round(accuracies["train"][-1], 4)
			vacc = round(accuracies["val"][-1], 4)
			teacc = round(accuracies["test"][-1], 4)
			top_5_mean = np.sum(sorted(accuracies['val'][:], reverse=True)[:5]) / 5
			top_5_mean = round(top_5_mean, 4)

			print()
			print(f'Final training data accuracy:\t\t{tracc}')
			print(f'Final validation data accuracy:\t\t{vacc}')
			print(f'Final test data accuracy:\t\t{teacc}')
			print(f'Final top 5 mean:\t\t\t{top_5_mean}')

			title = f'lambda{our_lambda}_n-batch{n_batch}_n-epochs{n_epochs}_n-s{n_s}_m{num_nodes}_eta-min{eta_min}_eta-max{eta_max}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'
			plot_three_subplots(costs=(costs['train'], costs['val']),
								losses=(losses['train'], losses['val']),
								accuracies=(accuracies['train'], accuracies['val']),
								title='search_' + title, show=False)

			results = [top_5_mean, tracc, vacc, teacc, our_lambda, n_batch,
					   eta_min, eta_max, num_nodes, n_s, n_epochs, seed]
			with open(results_file, 'a') as f:
				writer = csv.writer(f, dialect='excel', delimiter=';')
				writer.writerow(results)

	if best:
		print()
		print("------------------- Training best classifier -------------------")
		results_file = 'results/results_best.csv'
		# If file not exists, create it with its headers.
		if not os.path.exists(results_file):
			headers = ['top_5_vacc', 'tracc', 'vacc', 'teacc', 'lambda', 'n_batch',
					   'eta_min', 'eta_max', 'm', 'n_s', 'n_epochs', 'seed']
			with open(results_file, 'w+') as f:
				writer = csv.writer(f, dialect='excel', delimiter=';')
				writer.writerow(headers)

		n_batch = 100
		eta_min = 1e-5
		eta_max = 1e-1
		num_nodes = 50 # Number of nodes in the hidden layer

		n_s = 4 * int(np.floor(datasets['train_set']['X'].shape[1] / n_batch))

		# Number of epochs set to equal four cycles.
		n_epochs = int(8 * (n_s / n_batch))

		our_lambda = 0.00821

		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=0)

		accuracies, costs, losses, _ = \
		clf.mini_batch_gradient_descent(datasets['train_set']['X'],
										datasets['train_set']['Y'],
										our_lambda=our_lambda,
										n_batch=n_batch,
										eta_min=eta_min,
										eta_max=eta_max,
										n_s=n_s,
										n_epochs=n_epochs)

		tracc = round(accuracies["train"][-1], 4)
		vacc = round(accuracies["val"][-1], 4)
		teacc = round(accuracies["test"][-1], 4)
		top_5_mean = np.sum(sorted(accuracies['val'][:], reverse=True)[:5]) / 5
		top_5_mean = round(top_5_mean, 4)

		print()
		print(f'Final training data accuracy:\t\t{tracc}')
		print(f'Final validation data accuracy:\t\t{vacc}')
		print(f'Final test data accuracy:\t\t{teacc}')
		print(f'Final top 5 mean:\t\t\t{top_5_mean}')
		print()
		print(sorted(accuracies['val'][:10], reverse=True))

		title = f'lambda{our_lambda}_n-batch{n_batch}_n-epochs{n_epochs}_n-s{n_s}_m{num_nodes}_eta-min{eta_min}_eta-max{eta_max}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'
		plot_three_subplots(costs=(costs['train'], costs['val']),
							losses=(losses['train'], losses['val']),
							accuracies=(accuracies['train'], accuracies['val']),
							title='best_' + title, show=True)

		results = [top_5_mean, tracc, vacc, teacc, our_lambda, n_batch,
				   eta_min, eta_max, num_nodes, n_s, n_epochs, seed]
		with open(results_file, 'a') as f:
			writer = csv.writer(f, dialect='excel', delimiter=';')
			writer.writerow(results)
			writer.writerow(list(accuracies['val']))

	if find_end_of_cycle_settings:
		print()
		print("-------------------- Training best classifier -------------------")
		results_file = 'results/results_bonus.csv'
		# If file not exists, create it with its headers.
		if not os.path.exists(results_file):
			headers = ['top_5_vacc', 'tracc', 'vacc', 'teacc', 'lambda', 'n_batch',
					   'eta_min', 'eta_max', 'm', 'n_s', 'n_epochs', 'seed']
			with open(results_file, 'w+') as f:
				writer = csv.writer(f, dialect='excel', delimiter=';')
				writer.writerow(headers)

		n_batch = 100
		eta_min = 1e-5
		eta_max = 1e-1
		num_nodes = 50 # Number of nodes in the hidden layer

		n_s = 4 * int(np.floor(datasets['train_set']['X'].shape[1] / n_batch))

		# Number of epochs set to equal four cycles.
		n_epochs = int(8 * (n_s / n_batch))

		our_lambda = 0.00821

		clf = KLayerNetwork(labels, datasets, m=num_nodes, verbose=0)

		accuracies, costs, losses, settings = \
		clf.mini_batch_gradient_descent(datasets['train_set']['X'],
										datasets['train_set']['Y'],
										our_lambda=our_lambda,
										n_batch=n_batch,
										eta_min=eta_min,
										eta_max=eta_max,
										n_s=n_s,
										n_epochs=n_epochs)

		for setting in settings:
			print()
			print(setting)

		tracc = round(accuracies["train"][-1], 4)
		vacc = round(accuracies["val"][-1], 4)
		teacc = round(accuracies["test"][-1], 4)
		top_5_mean = np.sum(sorted(accuracies['val'][:], reverse=True)[:5]) / 5
		top_5_mean = round(top_5_mean, 4)

		print()
		print(f'Final training data accuracy:\t\t{tracc}')
		print(f'Final validation data accuracy:\t\t{vacc}')
		print(f'Final test data accuracy:\t\t{teacc}')
		print(f'Final top 5 mean:\t\t\t{top_5_mean}')
		print()
		print(sorted(accuracies['val'], reverse=True)[:10])

		title = f'lambda{our_lambda}_n-batch{n_batch}_n-epochs{n_epochs}_n-s{n_s}_m{num_nodes}_eta-min{eta_min}_eta-max{eta_max}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'
		plot_three_subplots(costs=(costs['train'], costs['val']),
							losses=(losses['train'], losses['val']),
							accuracies=(accuracies['train'], accuracies['val']),
							title='best_' + title, show=True)

		results = [top_5_mean, tracc, vacc, teacc, our_lambda, n_batch,
				   eta_min, eta_max, num_nodes, n_s, n_epochs, seed]
		with open(results_file, 'a') as f:
			writer = csv.writer(f, dialect='excel', delimiter=';')
			writer.writerow(results)
			writer.writerow(list(accuracies['val']))


	print()

	return


if __name__ == '__main__':
	main()
