import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import train_test_split
from sklearn import datasets


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        self.layers = layers

        self.weights = []

    def forward_propagate(self, inputs, sigmoid=True):

        # the input layer activation is just the input itself
        activations = inputs
        # iterate through the network layers
        # print(self.weights)
        for w in self.weights:
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self.activationFunc(net_inputs, sigmoid)

        # return output layer activation
        return activations

    def activationFunc(self, x, sigmoid):
        if sigmoid:
            y = 1.0 / (1 + np.exp(-x))
        else:
            # tanh
            y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return y


def accuracy(y_true, y_pred):
    """
    uses the predicted classifications and
    expected classifications to determine the
    accuracy of the perception in classification.
    """
    y_pred = y_pred.flatten()
    accuracyVal = np.sum(y_true == y_pred) / len(y_true)
    return accuracyVal


def test_solution(best_solution):
    # If needed import different dataset here
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    new_weights1 = best_solution[:10]
    new_weights2 = best_solution[10:]
    new_weights1 = new_weights1.reshape((2, 5))
    new_weights2 = new_weights2.reshape((5, 1))
    best_weights = []
    best_weights.append(new_weights1)
    best_weights.append(new_weights2)
    mlp = MLP(2, [5], 1)
    mlp.weights = best_weights
    outputs = mlp.forward_propagate(X_test)
    print(accuracy(y_test, np.round(outputs)))


def f(z):
    # If needed import different dataset here
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    mlp = MLP(2, [5], 1)
    new_weights1 = z[:10]
    new_weights2 = z[10:]
    new_weights1 = new_weights1.reshape((2, 5))
    new_weights2 = new_weights2.reshape((5, 1))
    new_weights = []
    new_weights.append(new_weights1)
    new_weights.append(new_weights2)
    mlp.weights = new_weights
    sum_error = 0
    for input, target in zip(X_train, y_train):
        output = mlp.forward_propagate(input)
        error = target - output
        sum_error += np.average(error**2)
    total_error = sum_error/len(X_train)
    return total_error


if __name__ == "__main__":


    varbound=np.array([[-1, 1]]*15)

    algorithm_param = {'max_num_iteration': 100, \
                       'population_size': 100, \
                       'mutation_probability': 0.20, \
                       'elit_ratio': 0.08, \
                       'crossover_probability': 0.4, \
                       'parents_portion': 0.2, \
                       'crossover_type': 'uniform', \
                       'max_iteration_without_improv': 60}

    model = ga(function=f, dimension=15, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()




