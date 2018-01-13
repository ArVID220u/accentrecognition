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

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda,
            test_data=None):
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
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
#                print("Progress: ", counter/n)
                counter += mini_batch_size
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

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
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
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
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
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

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


    def save(self, filename):
        """Save the neural network to the filename."""
        """Save it in json format, so as to be able to update the network structure later."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
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
