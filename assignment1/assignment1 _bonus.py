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


def preprocess_dataset(training_data):
	""" Pre-process data by normalizing """
	# "Both mean_X and std_X have size d x 1".
	mean_X = np.mean(training_data, axis=1)
	std_X = np.std(training_data, axis=1)

	training_data = training_data - np.array([mean_X]).T
	training_data = training_data / np.array([std_X]).T

	return training_data


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

   N = len(line_A)

   fig, ax = plt.subplots(figsize=(10, 8))
   ax.plot(range(N), line_A, label=label_A)
   ax.plot(range(N), line_B, label=label_B)
   ax.legend()

   plt.xticks(range(N))

   ax.set(xlabel=xlabel, ylabel=ylabel)
   ax.grid()

   plt.savefig(f'plots/{title}.png', bbox_inches="tight")
   plt.show()

   return


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
		s = np.dot(self.W, X) + self.b
		# p has size K x n, where n is n of the input X.
		p = self.soft_max(s)
		return p

	def soft_max(self, s):
		""" Standard definition of the softmax function """
		return np.exp(s) / np.sum(np.exp(s), axis=0)

	def compute_cost(self, X, Y, our_lambda):
		""" Compute cost using the cross-entropy loss.
			- each column of X corresponds to an image and X has size d x N.
			- Y corresponds to the one-hot ground truth label matrix.
			- our_lambda is the regularization term ("lambda" is reserved).
			Returns the cost, which is a scalar. """
		N = X.shape[1]
		p = self.evaluate_classifier(X)
		# If label is encoded as one-hot repr., then cross entropy is -log(yTp).
		cost = ((1 / N) * -np.sum(Y * np.log(p))) + (our_lambda * np.sum(self.W**2))
		return cost

	def compute_accuracy(self, X, y):
		""" Compute classification accuracy
			- each column of X corresponds to an image and X has size d x N.
			- y is a vector pf ground truth labels of length N
			Returns the accuracy. which is a scalar. """
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

	def mini_batch_gradient_descent(self, X, Y, our_lambda=0, n_batch=100,
									eta=0.001, n_epochs=20, save_costs=False):
		""" Learn the model by performing mini-batch gradient descent
			n_batch is the number of batches
			eta is the learning rate
			n_epochs is number of training epochs """

		accuracies = dict()

		if save_costs:
			costs = dict()
			costs['train'] = np.zeros(n_epochs)
			costs['val'] = np.zeros(n_epochs)
		else:
			costs = None

		for n in range(n_epochs):
			for j in range(n_batch):
				N = int(X.shape[1] / n_batch)
				j_start = (j) * N
				j_end = (j+1) * N

				X_batch = X[:, j_start:j_end]
				Y_batch = Y[:, j_start:j_end]

				grad_W, grad_b = self.compute_gradients(X_batch, Y_batch, our_lambda)
				self.W -= eta * grad_W
				self.b -= eta * grad_b

			if save_costs:
				costs['train'][n] = self.compute_cost(X, Y, our_lambda)
				costs['val'][n] = self.compute_cost(self.data['val_set']['X'],
													self.data['val_set']['Y'],
													our_lambda)

		accuracies['train'] = self.compute_accuracy(self.data['train_set']['X'],
													self.data['train_set']['y'])
		accuracies['val'] = self.compute_accuracy(self.data['val_set']['X'],
												  self.data['val_set']['y'])
		accuracies['test'] = self.compute_accuracy(self.data['test_set']['X'],
												   self.data['test_set']['y'])


		return accuracies, costs


def main():
	seed = 12345
	np.random.seed(seed)
	test_numerically = False
	our_lambda = 1.0
	n_batch = 100
	eta = 0.001
	n_epochs = 40

	print()
	print("------------------------ Loading dataset ------------------------")
	datasets_folder = "Datasets/cifar-10-batches-py/"

	labels = unpickle(datasets_folder + "batches.meta")[b'label_names']

	# Bonus a) use all available training data. Decrease valiation set to 1000.
	X_train1, Y_train1, y_train1 = \
	  load_batch("datasets/cifar-10-batches-py/data_batch_1")
	X_train2, Y_train2, y_train2 = \
	  load_batch("datasets/cifar-10-batches-py/data_batch_2")
	X_train3, Y_train3, y_train3 = \
	  load_batch("datasets/cifar-10-batches-py/data_batch_3")
	X_train4, Y_train4, y_train4 = \
	  load_batch("datasets/cifar-10-batches-py/data_batch_4")
	X_train5, Y_train5, y_train5 = \
	  load_batch("datasets/cifar-10-batches-py/data_batch_5")

	X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5),
	      axis=1)
	Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5),
	      axis=1)
	y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
	X_val = X_train[:, -1000:]
	Y_val = Y_train[:, -1000:]
	y_val = y_train[-1000:]
	X_train = X_train[:, :-1000]
	Y_train = Y_train[:, :-1000]
	y_train = y_train[:-1000]


	# Training with 1, validation with 2 and testing with test.
	train_set_1 = load_dataset(datasets_folder, "data_batch_1", num_of_labels=len(labels))
	train_set_2 = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
	train_set_3 = load_dataset(datasets_folder, "data_batch_3", num_of_labels=len(labels))
	train_set_4 = load_dataset(datasets_folder, "data_batch_4", num_of_labels=len(labels))
	train_set_5 = load_dataset(datasets_folder, "data_batch_5", num_of_labels=len(labels))

	test_set = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
	val_set = load_dataset(datasets_folder, "test_batch", num_of_labels=len(labels))

	# Training with 1, validation with 2 and testing with test.
	train_set = load_dataset(datasets_folder, "data_batch_1", num_of_labels=len(labels))
	test_set = load_dataset(datasets_folder, "data_batch_2", num_of_labels=len(labels))
	val_set = load_dataset(datasets_folder, "test_batch", num_of_labels=len(labels))


	datasets = {'train_set': train_set, 'test_set': test_set, 'val_set': val_set}
	print()
	print("----------------------- Preparing dataset -----------------------")
	for dataset_name, dataset in datasets.items():
		dataset['X'] = preprocess_dataset(dataset['X'])

	if test_numerically:
		print()
		print("-------------------- Running gradient tests ---------------------")
		num_images = 100
		num_pixels = 3072
		test_train, test_val, test_test = dict(), dict(), dict()

		test_train['X'] = train_set['X'][:num_images, :num_pixels]
		test_val['X'] = val_set['X'][:num_images, :num_pixels]
		test_test['X'] = test_set['X'][:num_images, :num_pixels]

		test_train['Y'] = train_set['Y'][:, :num_pixels]
		test_val['Y'] = val_set['Y'][:, :num_pixels]
		test_test['Y'] = test_set['Y'][:, :num_pixels]

		test_train['y'] = train_set['y'][:num_pixels]
		test_val['y'] = val_set['y'][:num_pixels]
		test_test['y'] = test_set['y'][:num_pixels]

		test_datasets = {'train_set': test_train, 'test_set': test_test, 'val_set': test_val}
		clf = SingleLayerNetwork(labels, test_datasets)

		grad_W, grad_b = clf.compute_gradients(test_datasets['train_set']['X'],
											   test_datasets['train_set']['Y'],
											   our_lambda=0)
		grad_W_num, grad_b_num = clf.compute_gradients_num(test_datasets['train_set']['X'],
														   test_datasets['train_set']['Y'],
														   our_lambda=0)

		# From the assignment PDF: "If all these absolutes difference are small
		# (<1e-6), then they have produced the same result.
		# np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
		print()
		print(f'All close: {np.allclose(grad_W, grad_W_num, atol=1e-05)}')

	print()
	print("---------------------- Learning classifier ----------------------")
	clf = SingleLayerNetwork(labels, datasets)
	accuracies, costs = clf.mini_batch_gradient_descent(datasets['train_set']['X'],
														datasets['train_set']['Y'],
														our_lambda=our_lambda,
														n_batch=n_batch,
														eta=eta,
														n_epochs=n_epochs,
														save_costs=True)
	print()
	print(f'Training data accuracy:\t\t{accuracies["train"]}')
	print(f'Validation data accuracy:\t{accuracies["val"]}')
	print(f'Test data accuracy:\t\t{accuracies["test"]}')

	tracc = accuracies["train"]
	vacc = accuracies["val"]
	teacc = accuracies["test"]
	title = f'lambda{our_lambda}_n-batch{n_batch}_eta{eta}_n-epochs{n_epochs}_tr-acc{tracc}_v-acc{vacc}_te-acc{teacc}_seed{seed}'
	plot_lines(line_A=costs['train'], line_B=costs['val'],
			   label_A='training loss', label_B='validation loss',
			   xlabel='epoch', ylabel='loss', title=title)

	labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	montage(clf.W, title, labels)

	print()

	return


if __name__ == '__main__':
	main()
