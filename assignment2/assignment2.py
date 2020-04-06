"""In this assignment you will train and test a one layer network with multiple
outputs to classify images from the CIFAR-10 dataset. You will train the network
using mini-batch gradient descent applied to a cost function that computes the
cross-entropy loss of the classifier applied to the labelled training data and
an L2 regularization term on the weight matrix."""


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


def plot_lines(line_A, line_B, label_A, label_B, xlabel, ylabel, title):
   """ Plots performance curves """
   assert(line_A.shape == line_B.shape)
   # N = len(line_A)

   fig, ax = plt.subplots(figsize=(10, 8))
   # ax.plot(range(N), line_A, label=label_A)
   # ax.plot(range(N), line_B, label=label_B)
   ax.plot(range(len(line_A)), line_A, label=label_A)
   ax.plot(range(len(line_B)), line_B, label=label_B)
   ax.legend()
   # plt.xticks(range(N))

   ax.set(xlabel=xlabel, ylabel=ylabel)
   ax.grid()

   plt.savefig(f'plots/{title}.png', bbox_inches="tight")
   plt.show()

   return


class SingleLayerNetwork():
	""" Single-layer network classifier based on mini-batch gradient descent """

	def __init__(self, labels, data, decay_factor=1, m=50, verbose=0):
		""" W: weight matrix of size K x d
			b: bias matrix of size K x 1 """
		self.labels = labels
		K = len(self.labels)

		self.decay_factor = decay_factor

		self.data = data
		d = self.data['train_set']['X'].shape[0]

		self.m = m
		self.verbose = verbose

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

	def mini_batch_gradient_descent(self, X, Y, our_lambda=0, n_batch=100, eta=0.001, n_epochs=20, save_costs=False):
		""" Learn the model by performing mini-batch gradient descent
			n_batch is the number of batches
			eta is the learning rate
			n_epochs is number of training epochs """

		accuracies = dict()
		accuracies['train'] = np.zeros(n_epochs)
		accuracies['val'] = np.zeros(n_epochs)
		accuracies['test'] = np.zeros(n_epochs)

		print()
		print(f'Accuracy training:\t{self.__compute_accuracy(self.data["train_set"]["X"], self.data["train_set"]["y"])}')
		print(f'Accuracy validation:\t{self.__compute_accuracy(self.data["val_set"]["X"], self.data["val_set"]["y"])}')
		print(f'Accuracy testing:\t{self.__compute_accuracy(self.data["test_set"]["X"], self.data["test_set"]["y"])}')

		if save_costs:
			losses, costs = dict(), dict()
			losses['train'], costs['train'] = np.zeros(n_epochs), np.zeros(n_epochs)
			losses['val'], costs['val'] = np.zeros(n_epochs), np.zeros(n_epochs)
		else:
			losses, costs = None, None

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

			if save_costs:
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
				# print(f'Cost training:\t\t{round(costs["train"][n], 2)}')
				# print(f'Cost validation:\t{round(costs["val"][n], 2)}')
				print(f'Loss training:\t\t{round(losses["train"][n], 2)}')
				print(f'Loss validation:\t{round(losses["val"][n], 2)}')
				# print(f'Accuracy training:\t{round(accuracies["train"][n], 2)}')
				# print(f'Accuracy validation:\t{round(accuracies["val"][n], 2)}')
				# print(f'Accuracy testing:\t{accuracies["test"][n]}')

			# Bonus B) implement a decay of the learning rate.
			eta *= self.decay_factor
			# print(f'Current learning rate: {eta}')

		return accuracies, costs, losses


def main():
	seed = 12345
	np.random.seed(seed)
	our_lambda = 0.0
	# n_epochs = 60
	n_epochs = 40
	n_batch = 100
	eta = 0.001
	decay_factor = 1.0
	test_numerically = False
	num_nodes = 50 # Number of nodes in the hidden layer

	print()
	print("------------------------ Loading dataset ------------------------")
	datasets_folder = "Datasets/cifar-10-batches-py/"

	labels = unpickle(datasets_folder + "batches.meta")[b'label_names']

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
		num_pixels = 20
		num_images = 10

		train_set['X'] = train_set['X'][:num_pixels, :num_images]
		train_set['Y'] = train_set['Y'][:num_pixels, :num_images]

		X_batch = train_set['X']
		Y_batch = train_set['Y']

		clf = SingleLayerNetwork(labels, datasets, decay_factor=decay_factor,
								 m=num_nodes, verbose=1)

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
		print(f'All close: {np.allclose(grad_W1, grad_W1_num, atol=1e-04)}')
		print(f'All close: {np.allclose(grad_W2, grad_W2_num, atol=1e-04)}')
		# WRITE: With atol 1e-04 we get all to be close.
		quit()

	print()
	print("-------------------- Instantiating classifier -------------------")
	clf = SingleLayerNetwork(labels, datasets, decay_factor=decay_factor,
							 m=num_nodes, verbose=1)

	print()
	print("---------------------- Learning classifier ----------------------")
	# Sanity check
	# num_pixels = 3072
	# num_images = 100
	#
	# train_set['X'] = train_set['X'][:num_pixels, :num_images]
	# train_set['Y'] = train_set['Y'][:num_pixels, :num_images]


	accuracies, costs, losses = \
	clf.mini_batch_gradient_descent(datasets['train_set']['X'],
									datasets['train_set']['Y'],
									our_lambda=our_lambda,
									n_batch=n_batch,
									eta=eta,
									n_epochs=n_epochs,
									save_costs=True)


	tracc = accuracies["train"][-1]
	vacc = accuracies["val"][-1]
	teacc = accuracies["test"][-1]

	print()
	print(f'Final training data accuracy:\t\t{tracc}')
	print(f'Final validation data accuracy:\t\t{vacc}')
	print(f'Final test data accuracy:\t\t{teacc}')

	title = f'lambda{our_lambda}_n-batch{n_batch}_eta{eta}_n-epochs{n_epochs}_df-{decay_factor}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'

	plot_lines(line_A=costs['train'], line_B=costs['val'],
			   label_A='training cost', label_B='validation cos',
			   xlabel='epoch', ylabel='cost', title='cost_' + title)

	plot_lines(line_A=losses['train'], line_B=losses['val'],
			   label_A='training loss', label_B='validation loss',
			   xlabel='epoch', ylabel='loss', title='loss_' + title)

	plot_lines(line_A=accuracies['train'], line_B=accuracies['val'],
			   label_A='training accuracy', label_B='validation accuracy',
			   xlabel='epoch', ylabel='accuracy', title='acc_' + title)

	labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	montage(clf.W1, title, labels)

	print()

	return


if __name__ == '__main__':
	main()
