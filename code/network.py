"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# json is used for saving networks
import json 

# Third-party libraries
import numpy as np




#### Define the quadratic and cross-entropy cost functions

class QuadraticCost():

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output a and desired output y."""
        """ the norm of a the a-y column matrix is simply its absolute value if treated as a vector."""
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        """This is the partial derivative of the cost function with respect
        to each weighted input to each node in the last layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost():

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output a and desired output y.
        Note that np.nan_to_num is used to ensure numerical stability.
        In particular, if both a and y have a 1.0 in the same slot,
        then the expression (1-y)*np.log(1-a) returns nan.
        The np.nan_to_num ensures that that is converted to the correct value (0.0)."""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter z is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes."""
        return (a-y)





class Network():

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly using default_weight_initializer."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers."""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self):
        """Initialize each weight and bias using Guassian distribution with mean 0
        and standard deviation 1. Will lead to saturated hidden neurons and therefore to slow learning.
        Only use it for comparison purposes."""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda,
            test_data=None, 
            monitor_test_cost=False,
            monitor_test_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        counter = 0
        if test_data: n_test = len(test_data)
        n = len(training_data)
        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
#                print("Progress: ", counter/n)
                counter += mini_batch_size
            print("Epoch {0} complete".format(j))
            if test_data and monitor_test_accuracy:
                accuracy = self.evaluate(test_data)
                test_accuracy.append(accuracy)
                print("\tTest data accuracy: {0} / {1}".format(accuracy, n_test))
            if test_data and monitor_test_cost:
                cost = self.total_cost(test_data, lmbda)
                test_cost.append(cost)
                print("\tTest data cost: {0}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data)
                training_accuracy.append(accuracy)
                print("\tTraining data accuracy: {0} / {1}".format(accuracy, n))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                test_cost.append(cost)
                print("\tTraining data cost: {0}".format(cost))

        return_val = {"test_cost": test_cost, "test_accuracy": test_accuracy, "training_cost": training_cost, "training_accuracy": training_accuracy, "n_test": n_test, "n_training": n}
        return return_val


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights_1 = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        biases_1 = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        """

        # compute matrices of training samples. each column is a distinct training sample.
        x_matrix = np.column_stack([training_sample[0] for training_sample in mini_batch])
        y_matrix = np.column_stack([training_sample[1] for training_sample in mini_batch])
        nabla_b, nabla_w = self.matrixbackprop(x_matrix, y_matrix)
        weights_2 = [(1.0 - lmbda*eta/n)*w - eta*nw for w, nw in zip(self.weights, nabla_w)]
        biases_2 = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]

        """
        for w1, w2 in zip(weights_1, weights_2):
            assert np.allclose(w1, w2)
        for b1, b2 in zip(biases_1, biases_2):
            assert np.allclose(b1, b2)
        """

        self.biases = biases_2
        self.weights = weights_2

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            #print(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def matrixbackprop(self, x, y):
        """Same as backprop, with the exception that x is a matrix of dimension (input_nodes, training_samples)
        and y is a matrix of dimension (output_nodes, training_samples).
        That is, matrixbackprop computes nabla_b and nabla_w for multiple training samples at a time."""
        """The computed nabla_b and nabla_w are averages."""
        num_samples = x.shape[1]
        assert num_samples == y.shape[1]

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x

        # list to store all the activations, layer by layer
        # the activitations for a layer consists of a matrix where each column represents the activiation for a particular training sample
        activations = [x] 
        # list to store all the z vectors, layer by layer
        # as activitations, each column corresponds to a particular training sample
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            # we neeed to add the biases as a matrix so that it is done for each training sample
            # this means that we want to make the now-column matrix b into a 2d matrix with repeated columns (as many as the training samples)
            assert activation.shape[1] == num_samples
            z = np.dot(w, activation) + np.repeat(b, num_samples, axis=1)
            zs.append(z)
            #print(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = col_avg(delta)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) / num_samples
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = col_avg(delta)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) / num_samples
        return (nabla_b, nabla_w)
    


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
#        for (x, y) in test_data:
#            print(np.argmax(self.feedforward(x)))

        test_results = [(maxarg(self.feedforward(x)), y)
                        for (x, y) in test_data]
#        print(test_results)
        type1 = 0
        type2 = 0
        correct_type1 = 0
        correct_type2 = 0
        for (x, y) in test_results:
            if y[0][0] == 1:
                type1 += 1
                correct_type1 += int(np.array_equal(x, y))
            else:
                type2 += 1
                correct_type2 += int(np.array_equal(x, y))

        print("Correct type1: " + str(correct_type1) + " / " + str(type1) + "    Correct type2: " + str(correct_type2) + " / " + str(type2))

        return sum(int(np.array_equal(x, y)) for (x, y) in test_results)

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


    def save(self, filename):
        """Save the neural network to the filename."""
        """Save it in json format, so as to be able to update the network structure later."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        with open(filename, "w") as f:
            json.dump(data, f)

    # load from file
    @classmethod
    def load(cls, filename):
        """initialize the network from the file"""
        data = ""
        with open(filename, "r") as f:
            data = json.load(f)
        assert data != ""
        cost = getattr(sys.modules[__name__], data["cost"])
        network = cls(data["sizes"])
        network.weights = [np.array(w) for w in data["weights"]]
        network.biases = [np.array(b) for b in data["biases"]]
        return network




#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# compute the average of the columns into a column matrix
def col_avg(x):
    return np.sum(x,axis=1,keepdims=True) / x.shape[1]

def maxarg(x):
    """instead of numpy's argmax, since numpy sucks"""
    if(x[0][0] >= x[1][0]):
        return np.array([[1], [0]])
    else:
        return np.array([[0], [1]])



