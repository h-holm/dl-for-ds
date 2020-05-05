"""In this assignment you will train an RNN to synthesize English text character
by character. You will train a vanilla RNN with outputs, as described in lecture
9. This is done by using the provided text from the bo_arraysk Harry Potter and The
Goblet of Fire by J.K. Rowling. The variation of SGD you will use for the
optimization will be AdaGrad."""


import os
import json
import re
import emoji
import pickle
import numpy as np


__author__ = "Henrik Holm"


def read_input_files(list_of_filepaths):
	def remove_emoji(text):
	    return emoji.get_emoji_regexp().sub(u'', text)

	output = dict()
	contents = ''
	all_tweets = list()

	for filepath in list_of_filepaths:
		with open(filepath, 'r') as f:
			tweets = json.load(f)
			for tweet in tweets:
				# Remove non-latin characters such as Chinese characters.
				tweet = re.sub(r'[^\u0020-\u024F]', '', tweet['text'])
				# Remove links.
				tweet = re.sub(r'https?://\S+', '', tweet)
				# Remove emojis.
				tweet = remove_emoji(tweet)

				if len(tweet) < 4:
					continue

				all_tweets.append(tweet)

				# Here, ¶ is used a stop character at the end of each tweet.
				contents += tweet + '¶ '

	output['all_tweets'] = all_tweets
	output['contents'] = contents
	output['contents_length'] = len(contents)

	unique_characters = list(set(contents))
	output['unique_characters'] = unique_characters
	output['vocab_length'] = len(unique_characters)

	# Set up map_arraysing functions. Each char to unique idx, each idx to char.
	idx_to_char = dict()
	for idx, char in enumerate(unique_characters):
		idx_to_char[idx] = char
	output['idx_to_char'] = idx_to_char

	char_to_idx = dict()
	for idx, char in enumerate(unique_characters):
		char_to_idx[char] = idx
	output['char_to_idx'] = char_to_idx

	return output


class RecurrentNeuralNetwork():
	""" K-layer network classifier based on mini-batch gradient descent """

	def __init__(self, data, m=100, eta=0.1, sigma=0.01, verbose=0):
		""" W: weight matrix of size K x d
			b: bias matrix of size K x 1 """
		self.data = data
		self.K = data['vocab_length']
		self.m, self.eta = m, eta
		self.verbose = verbose
		self.__initialize_parameters(sigma)
		self.params = {'b': self.b, 'c': self.c, 'U': self.U, 'W': self.W, 'V': self.V}

	@staticmethod
	def __tanh(x):
		return np.tanh(x)

	@staticmethod
	def __softmax(s):
		""" Standard definition of the softmax function """
		return np.exp(s - np.max(s, axis=0)) / np.exp(s - np.max(s, axis=0)).sum(axis=0)

	def __initialize_parameters(self, sigma=0.01):
		""" Initializes bias vectors b and c and weight matrices U, W and V """
		self.b = np.zeros((self.m, 1))
		self.c = np.zeros((self.K, 1))
		self.U = np.random.normal(0, sigma, size=(self.m, self.K))
		self.W = np.random.normal(0, sigma, size=(self.m, self.m))
		self.V = np.random.normal(0, sigma, size=(self.K, self.m))
		return

	def __evaluate_classifier(self, h, x):
		""" Equations 1-4 in Assignment4.pdf """
		a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
		h = self.__tanh(a)
		o = np.dot(self.V, h) + self.c
		p = self.__softmax(o)
		return a, h, o, p

	def __forward_pass(self, inputs, labels, h_prev):
		""" Compute output of network given input """
		a_arrays, x_arrays, h_arrays, o_arrays, p_arrays = \
			dict(), dict(), dict(), dict(), dict()
		h_arrays[-1] = np.copy(h_prev)
		loss = 0
		for t in range(len(inputs)):
			x_arrays[t] = np.zeros((self.data['vocab_length'], 1))
			x_arrays[t][inputs[t]] = 1
			a_arrays[t], h_arrays[t], o_arrays[t], p_arrays[t] = \
				self.__evaluate_classifier(h_arrays[t-1], x_arrays[t])
			loss -= np.log(p_arrays[t][labels[t]][0])

		return loss, a_arrays, x_arrays, h_arrays, o_arrays, p_arrays

	def __compute_gradients(self, inputs, labels, h_prev):
		""" Computes gradients of the weights and biases analytically """
		loss, a_arrays, x_arrays, h_arrays, _, p_arrays = \
			self.__forward_pass(inputs, labels, h_prev)

		gradients = dict()
		for param_name, param_matrix in self.params.items():
			gradients[param_name] = np.zeros_like(param_matrix)
		gradients['o'] = np.zeros_like(p_arrays[0])
		gradients['h'] = np.zeros_like(h_arrays[0])
		gradients['h_next'] = np.zeros_like(h_arrays[0])
		gradients['a'] = np.zeros_like(a_arrays[0])

		# Backpropagation using equations from Lecture9.pdf
		for t in reversed(range(len(inputs))):
			gradients['o'] = np.copy(p_arrays[t])
			gradients['o'][labels[t]] -= 1
			gradients['V'] += np.dot(gradients['o'], h_arrays[t].T)
			gradients['c'] += gradients['o']
			gradients['h'] = np.dot(self.V.T, gradients['o']) + gradients['h_next']
			gradients['a'] = np.multiply(gradients['h'], (1 - np.square(h_arrays[t])))
			gradients['U'] += np.dot(gradients['a'], x_arrays[t].T)
			gradients['W'] += np.dot(gradients['a'], h_arrays[t-1].T)
			gradients['b'] += gradients['a']
			gradients['h_next'] = np.dot(self.W.T, gradients['a'])

		gradients = {k: gradients[k] for k in gradients if k not in ['o', 'h', 'h_next', 'a']}

		# Clip gradients to avoid the exploding gradient problem.
		for param in gradients.keys():
			gradients[param] = np.clip(gradients[param], -5, 5)
		h = h_arrays[len(inputs) - 1]

		return gradients, loss, h

	def __compute_gradients_num(self, inputs, labels, h_prev, h=1e-4, num_comps=20):
		""" Compute gradients of the weights and biases numerically """
		gradients = dict()
		for param_name, param_matrix in self.params.items():
			gradients[param_name] = np.zeros_like(param_matrix)
			for i in range(num_comps):
				old_value = self.params[param_name].flat[i]
				self.params[param_name].flat[i] = old_value + h
				loss1, _, _, _, _, _ = self.__forward_pass(inputs, labels, h_prev)
				self.params[param_name].flat[i] = old_value - h
				loss2, _, _, _, _, _ = self.__forward_pass(inputs, labels, h_prev)
				self.params[param_name].flat[i] = old_value
				gradients[param_name].flat[i] = (loss1 - loss2) / (2 * h)

		return gradients

	def __check_gradient_similarity(self, gradients_ana, gradients_num, num_comps=20):
		""" Calculate maximum relative error of analytical/numerical gradients """
		print()
		for param in self.params:
			numerator = abs(gradients_ana[param].flat[:num_comps] - \
							gradients_num[param].flat[:num_comps])
			denominator = np.asarray([max(abs(a), abs(b)) + 1e-10 for a, b in \
									  zip(gradients_ana[param].flat[:num_comps],
										  gradients_num[param].flat[:num_comps])])
			max_relative_error = max(numerator / denominator)
			print(f'\t{param} maximum relative error:\t{max_relative_error}')

		return

	def run_gradient_check(self, inputs, labels, h_prev, num_comps=20):
		""" Run functions to compute gradients and to compare the results """
		gradients_ana, _, _ = self.__compute_gradients(inputs, labels, h_prev)
		gradients_num = self.__compute_gradients_num(inputs, labels, h_prev, num_comps=num_comps)
		self.__check_gradient_similarity(gradients_ana, gradients_num, num_comps=num_comps)
		return

	def synthesize_text(self, h, idx, text_length=140):
		""" Generates text snippet given input hidden state sequence """
		x_next = np.zeros((self.K, 1))
		x_next[idx] = 1
		text = ''
		for _ in range(text_length):
			_, h, _, p = self.__evaluate_classifier(h, x_next)
			idx = np.random.choice(range(self.K), p=p.flat)
			x_next = np.zeros((self.K, 1))
			x_next[idx] = 1
			text += self.data['idx_to_char'][idx]

		return text

	def adagrad(self, seq_length, n_epochs):
		""" AdaGrad algorithm as per page 2 in Assignment4.pdf """
		# e: position tracker; n: update step
		n, epoch = 0, 0
		smooth_losses = list()
		h_prev = np.zeros((self.m, 1))

		m_params = dict()
		for param_name, param_matrix in self.params.items():
			m_params[param_name] = np.zeros(param_matrix.shape)

		print()
		while epoch < n_epochs:
			for tweet in self.data['all_tweets']:
				h_prev = np.zeros((self.m, 1))
				X = [self.data['char_to_idx'][char] for char in tweet]
				Y = [self.data['char_to_idx'][char] for char in tweet[1:] + ' ']

				gradients, loss, h_prev = self.__compute_gradients(X, Y, h_prev)

				if n == 0 and epoch == 0:
					smooth_loss = loss
				smooth_loss = (0.999 * smooth_loss) + (0.001 * loss)

				if n % 100 == 0:
					print(f'Smooth loss after {n} tweets:  \t{smooth_loss}')

				if n % 500 == 0:
					text = self.synthesize_text(h_prev, X, text_length=140)
					print(f'\nSynthesized text after {n} tweets:\n{text}\n')

				# AdaGrad update step
				for param_name, param_matrix in self.params.items():
					m_params[param_name] += gradients[param_name] * gradients[param_name]
					param_matrix -= self.eta / np.sqrt(m_params[param_name] + \
									np.finfo(np.float64).eps) * gradients[param_name]

				n += 1

			print(f'\nEpoch {epoch} finished\n')
			epoch += 1

		return smooth_losses


def main():
	seed = 12345
	np.random.seed(seed)

	print("\n------------------------ Loading dataset ------------------------")

	pickle_file = 'trump_tweets.pkl'
	if os.path.isfile(pickle_file):
		with open(pickle_file, 'rb') as pickle_in:
			input_data = pickle.load(pickle_in)
	else:
		input_files = list()
		years_with_data = ['2009', '2010', '2011', '2012', '2013', '2014', '2015',
						   '2016', '2017', '2018']
		file_prefix = '/Users/henrikholm/Github/dl-for-ds/assignment4/input/condensed_'
		file_suffix = '.json'
		for year in years_with_data:
			input_files.append(file_prefix + year + file_suffix)

		input_data = read_input_files(input_files)

		with open(pickle_file, 'wb') as pickle_out:
			pickle.dump(input_data, pickle_out)

	print("\n---------------------- Learning classifier ----------------------")

	m = 100
	eta = 0.1
	seq_length = 25
	sigma = 0.01
	RNN = RecurrentNeuralNetwork(input_data, m, eta, sigma)

	n_epochs = 30
	RNN.adagrad(seq_length, n_epochs)

	print()

	return


if __name__ == '__main__':
	main()
