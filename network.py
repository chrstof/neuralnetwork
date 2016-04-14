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

            # update mini batch for all mini batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            # Output testing results
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j,self.evaluate(test_data), n_test))
            else:
                print("Epoch {0}/{1}".format(j,n_test))




    def update_mini_batch(self,mini_batch,eta):
        """
        Updates the weights and biases using backpropagation based on  a single mini batch.

        :param mini_batch:      list of tuples (x,y): x - Input, y - Output --> training data
        :param eta:             learning rate
        :return:
        """

        #initialize nabla bias and nabla wieghts for all:
        nabla_b = [np.zero(b.shape) for b in self.biases]
        nabla_w = [np.zero(w.shape) for w in self.weights]

        for x,y in mini_batch:
            # using backpropagation to compute delta nabla
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)

            # new nabla = old nabla + delta nabla:
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

        # update the weights and biases: new weights = old weights - eta*nabla weight / n
        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases  = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.weights,nabla_b)]

    def backprop(self,x,y):
        """
        Computes the gradient of the cost function for all biases and weights of the network.
        THIS FUNCTION IS THE HEARD OF THE NETWORK!

        :param x:       Input
        :param y:       Output
        :return:        Nabla weights and nabla biases
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        # initial activations
        activation = x

        # save all activations layer by layer in a list
        activations = [x]

        # z-vectors layer by layer
        zs = []

        for b,w in zip(self.biases,self.weights):
            # compute the z vector for all layers and store in zs
            z = np.dot(w,activation) + b
            zs.append(z)

            # compute the sigmoid of z and store it as activation
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass, starting from the last layer:
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        # now that the biases of the last and the weights between the second last and the last layer are computed, we
        # can continue to compute biases and weights step by step towards the first layer.
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return (nabla_b, nabla_w)


    def evaluate(self,test_data):
        """
        Evaluates the network upon the test data and returns the number of correct results

        :param test_data:       the test data
        :return:                number of correct estimates
        """

        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)


    def cost_derivative(self,output_activations,y):
        """
        Computes the partial derivatives of the of the cost function C_x

        :param output_activations:
        :param y:
        :return:
        """

        return (output_activations - y)








def sigmoid(z):
    # returns the sigmoid function of z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid
    return sigmoid(z)*(1-sigmoid(z))
































