__author__ = 'christof'

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network



training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)


net = network.Network([784,30,10])

net.SGD(training_data,30,10,3.0,test_data=test_data)




