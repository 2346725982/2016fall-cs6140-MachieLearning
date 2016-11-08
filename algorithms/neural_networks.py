import numpy as np


class NeuralNetworks(object):
    def __init__(self, hidden_layers=(1, 100)):
        self.inputs = None
        self.outputs = None
        self.labels = {}
        self.hidden_layers = hidden_layers

    def fit(self, X, y):
        sample_size, feature_size = X.shape
        labels = np.unique(y)

        self.inputs = feature_size
        self.outputs = len(labels)

        new_y = np.zeros((sample_size, self.outputs))
        for i in xrange(y.shape[0]):
            for index, label in enumerate(labels):
                self.labels[index] = label
                if y[i][0] == label:
                    new_y[i][index] = 1

        self.back_propogation(X, new_y)

    def back_propogation(self, X, y):
        sample_size, feature_size = X.shape

        nn_input_dim = feature_size
        nn_hdim = self.hidden_layers[1]
        nn_output_dim = self.outputs

        num_passes = 1000
        alpha = 0.01 # learning rate for gradient descent
        reg_lambda = 0.01 # regularization strength

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        # This is what we return at the end

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):

            # Forward propagation
            z2 = X.dot(W1) + b1
            a2 = self.sigmoid_array(z2)
            z3 = a2.dot(W2) + b2
            a3 = self.sigmoid_array(z3)

            # Backpropagation
            delta3 = (a3 - y) * ((1 - a3) * a3)
            dW2 = (a2.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * ((1 - a2) * a2)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            W1 = W1 - alpha * (dW1 / sample_size + reg_lambda * W1)
            W2 = W2 - alpha * (dW2 / sample_size + reg_lambda * W2)
            b1 += -alpha * db1
            b2 += -alpha * db2

        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    def predict(self, X):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        z1 = X.dot(W1) + b1
        a1 = self.sigmoid_array(z1)
        z2 = a1.dot(W2) + b2
        a2 = self.sigmoid_array(z2)

        y = np.ndarray((a2.shape[0], 1))
        for i in xrange(a2.shape[0]):
            max_prob = -1
            max_index = -1
            for index, prob in enumerate(a2[i]):
                if prob > max_prob:
                    max_prob = prob
                    max_index = index
            y[i][0] = self.labels[max_index]

        return y

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))
