__author__ = 'christof'

"""
network.py
--------------


This work is based on the book "Neural Network and Deep Learning" by Michael Nielsen.
It is awsome and I can totaly recomand to read it!!!


This module implements a stochasitc gradient descent learning algorithm for a feedforward neural network.
The gradients are computed using backpropagation.

"""

import random
import numpy as np

class Network(object):
    def __init__(self,sizes):
        """
        :param sizes:   list representing the numers of neurons per layer. [3,4,4,2] represents 3 input, 4 hidden, 4 hidden and 2 output nerons
        :return:
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # input neurons will get no biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # weights are all possible combinations between nerons from layer n to layer n+1
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        """
        :param a:   output of the network for input vector a
        :return:
        """

        for b,w in zip (self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
            # input vector a is multiplied with the first weight matrix giving the vector of the second layer.
            # Adding the bias gives the vector that is multiplied witht he weight matrix 2... resulting in the output

        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        """

        :param training_data:       list of tuples (x,y) with x - training input data and y - output
        :param epochs:              training epochs
        :param mini_batch_size:     size of the training batch
        :param eta:                 learning rate
        :param test_data:           network will be tested against this data after n epochs
        :return:
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            # devide the trainingdata into equal sized mini_batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:


    def update_mini_batch(self,mini_batch,eta):
        """

        :param mini_batch:
        :param eta:
        :return:
        """

    def backprop(self,x,y):
        """

        :param x:
        :param y:
        :return:
        """

    def evaluate(self,test_data):
        """

        :param test_data:
        :return:
        """

    def cost_derivative(self,output_activations,y):
        """

        :param output_activations:
        :param y:
        :return:
        """








def sigmoid(z):
    # returns the sigmoid function of z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid
    return sigmoid(z)*(1-sigmoid(z))
































