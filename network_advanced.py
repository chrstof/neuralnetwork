
import random
import numpy as np



################################################################################
#--------------------Define the Cost functions---------------------------------#
################################################################################

class CrossEntropyCost(object):
    """
    This class defines everything about the cost function, that is
    1) the cross entropy cost function CrossEntropyCost.fn computing how well the networks output a matches the desired output y
    2) network output error for the back propagation algorithm CrossEntropyCost.delta
    """

    # staticmethod --> method does not depend on the object, thus self is not passed

    @staticmethod
    def fn(a,y):
        """
        Computes the cost associated with the modeloutput a and the desired output y based on the Cross Entropy

        :param a:   Model Output
        :param y:   Desired Output
        :return:
        """

        # np.nan_to_nan ensures correct output for log (almost zero) i.e. a=y=1 --> returns 0.0 and not nan
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)np.log(1-a)))

    # back propagarion output error:
    @staticmethod
    def delta(z,a,y):
        """
        returns the error of the output vector

        :param z:   not used, but included for consistency with respect to other cost functions
        :param a:   Model Output
        :param y:   Desired Output (training output)
        :return:
        """
        return (a-y)

class QuadraticCost(object):
    """
    For comparisment to the corss entropy cost function here the standard Quadratic cost
    """

    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        return (a-y) * sigmoid_prime(z)


def sigmoid(z):
    # returns the sigmoid function of z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid
    return sigmoid(z)*(1-sigmoid(z))

################################################################################
#--------------------Define the Network---------------------------------#
################################################################################

class Network(object):
    def __init__(self,sizes,cost = CrossEntropyCost):
        """
        :param sizes:   list representing the numers of neurons per layer. [3,4,4,2] represents 3 input, 4 hidden, 4 hidden and 2 output nerons
        :return:
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        # variable weight initializer and cost function:
        self.default_weight_initializer()
        self.cost=cost


    def default_weight_initializer(self):
        # biases are initialized the same
        # weights are normalized by the square root of their number of the associated layer!
        # Imagine 100 input neurons with 100 weights leading to the first neron of the first hidden layer... when all
        # weights have std of 1 then the std of the sum of all is (100 x 1² + 1²)^0.5 --> one extra for the bias
        # Thus it is likely, that the activation |z| is very large, meaning a saturated sigmoid neuron, requireing
        # way more iteration steps!

        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]


    def large_weight_initializer(self):
        # input neurons will get no biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # weights are all possible combinations between nerons from layer n to layer n+1
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]





