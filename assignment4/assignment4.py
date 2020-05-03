"""In this assignment you will train an RNN to synthesize English text character
by character. You will train a vanilla RNN with outputs, as described in lecture
9. This is done by using the provided text from the bo_arraysk Harry Potter and The
Goblet of Fire by J.K. Rowling. The variation of SGD you will use for the
optimization will be AdaGrad."""


import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Henrik Holm"


def prepare_data(filepath):
	""" Copied from the dataset website """
	output = dict()

	with open(filepath, 'r') as f:
		contents = f.read()
	output['contents'] = contents

	unique_characters = list(set(contents))
	output['unique_characters'] = unique_characters
	output['vocab_length'] = len(unique_characters)

	# Set up map_arraysing functions. Each char to unique idx, each idx to char.
	idx_to_char = dict()
	for idx, char in enumerate(unique_characters):
		# idx_to_char.ap_arraysend((idx, char))
		idx_to_char[idx] = char
	output['idx_to_char'] = idx_to_char

	char_to_idx = dict()
	for idx, char in enumerate(unique_characters):
		# char_to_idx.ap_arraysend((char, idx))
		char_to_idx[char] = idx
	output['char_to_idx'] = char_to_idx

	return output


class RecurrentNeuralNetwork():
	""" K-layer network classifier based on mini-batch gradient descent """

	def __init__(self, data, m=100, seq_length=25, eta=0.1, sigma=0.01, verbose=0):
		""" W: weight matrix of size K x d
			b: bias matrix of size K x 1 """
		self.data = data
		self.K = data['vocab_length']
		self.m, self.N, self.eta = m, seq_length, eta
		self.verbose = verbose
		self.__initialize_parameters(sigma)
		self.params = {'b': self.b, 'c': self.c, 'U': self.U, 'W': self.W, 'V': self.V}

	@staticmethod
	def __tanh(x):
		return np.tanh(x)

	@staticmethod
	def __softmax(s):
		""" Standard definition of the softmax function """
		# return np.exp(s) / np.sum(np.exp(s), axis=0)
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

	def __forward_pass(self, inputs, targets, h_prev):
		a_arrays, x_arrays, h_arrays, o_arrays, p_arrays = \
			dict(), dict(), dict(), dict(), dict()
		h_arrays[-1] = np.copy(h_prev)
		loss = 0
		for t in range(len(inputs)):
			x_arrays[t] = np.zeros((self.data['vocab_length'], 1))
			x_arrays[t][inputs[t]] = 1
			a_arrays[t], h_arrays[t], o_arrays[t], p_arrays[t] = \
				self.__evaluate_classifier(h_arrays[t-1], x_arrays[t])
			loss -= np.log(p_arrays[t][targets[t]][0])

		return loss, a_arrays, x_arrays, h_arrays, o_arrays, p_arrays

	def synthesize_text(self, h, idx, n):
		""" Generetes text snip_arrayset given input hidden state sequence """
		x_next = np.zeros(self.K, 1)
		x_next[idx] = 1
		text = ''
		for t in range(n):
			_, h, _, p = self.__evaluate_classifier(h, x_next)
			idx = np.random.choice(range(self.K), p=p.flat)
			x_next = np.zeros(self.K, 1)
			x_next[idx] = 1
			text += self.data['idx_to_char'][idx]

		print(text)
		return text

	def compute_gradients(self, inputs, targets, h_prev):
		""" Computes gradients of the weights and biases analytically """
		loss, a_arrays, x_arrays, h_arrays, o_arrays, p_arrays = \
			self.__forward_pass(inputs, targets, h_prev)

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
			gradients['o'][targets[t]] -= 1
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
		# for grad in gradients:
		# 	gradients[grad] = np.clip(gradients[grad], -5, 5)
		h = h_arrays[len(inputs) - 1]

		return gradients, loss, h

	def compute_gradients_num(self, inputs, targets, h_prev, h=1e-4, num_comps=20):
		""" Compute gradients of the weights and biases numerically """
		gradients = dict()
		for param_name, param_matrix in self.params.items():
			gradients[param_name] = np.zeros_like(param_matrix)
			for i in range(num_comps):
				old_value = self.params[param_name].flat[i]
				self.params[param_name].flat[i] = old_value + h
				loss1, _, _, _, _, _ = self.__forward_pass(inputs, targets, h_prev)
				self.params[param_name].flat[i] = old_value - h
				loss2, _, _, _, _, _ = self.__forward_pass(inputs, targets, h_prev)
				self.params[param_name].flat[i] = old_value
				gradients[param_name].flat[i] = (loss1 - loss2) / (2 * h)

		return gradients

	def check_gradient_similarity(self, gradients_ana, gradients_num, num_comps=20):
		for param in self.params:
			numerator = abs(gradients_ana[param].flat[:num_comps] - \
							gradients_num[param].flat[:num_comps])
			denominator = np.asarray([max(abs(a), abs(b)) + 1e-10 for a, b in \
									  zip(gradients_ana[param].flat[:num_comps],
										  gradients_num[param].flat[:num_comps])])
			max_relative_error = max(numerator / denominator)
			print(f'\t{param} maximum relative error:\t{max_relative_error}')

		return


def main():
	seed = 12345
	np.random.seed(seed)

	gradients = True
	exercise_1 = False

	print("\n------------------------ Loading dataset ------------------------")
	datasets_folder = '/Users/henrikholm/Github/dl-for-ds/assignment4/input/goblet_book.txt'
	input_data = prepare_data(datasets_folder)

	print("\n-------------------- Instantiating classifier -------------------")

	print("\n---------------------- Learning classifier ----------------------")
	if gradients:
		print()
		print("--------------------------- Gradients ---------------------------")
		m = 5
		eta = 0.1
		seq_length = 25
		sigma = 0.01
		num_comps = 5

		RNN = RecurrentNeuralNetwork(input_data, m, seq_length, eta, sigma)

		h_prev = np.zeros((RNN.m, 1))
		inputs = [RNN.data['char_to_idx'][char] for char in RNN.data['contents'][: RNN.N]]
		targets = [RNN.data['char_to_idx'][char] for char in RNN.data['contents'][1: RNN.N + 1]]

		gradients_ana, _, _ = RNN.compute_gradients(inputs, targets, h_prev)
		gradients_num = RNN.compute_gradients_num(inputs, targets, h_prev, num_comps=num_comps)
		RNN.check_gradient_similarity(gradients_ana, gradients_num, num_comps=num_comps)

	if exercise_1:
		print()
		print("-------------------------- Exercise 1 --------------------------")
		m = 100
		eta = 0.1
		seq_length = 25
		sigma = 0.01
		RNN = RecurrentNeuralNetwork(input_data, m, seq_length, eta, sigma)

	print()

	return


if __name__ == '__main__':
	main()
