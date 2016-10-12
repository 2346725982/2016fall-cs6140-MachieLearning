import numpy as np
import scipy


class LogisticRegression(object):
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.model = None

    def fit(self, X, y):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((X, intercept), axis=1)

        X = self.feature_mapping(X)

        labels = np.unique(y)

        self.coefficients = np.ones((len(labels), X.shape[1] - 1))
        self.intercept = np.ones((1, len(labels)))

        X = np.asmatrix(X)
        y = np.asmatrix(y)

        if len(labels) > 2:
            for i in xrange(len(labels)):
                self.gradient_descent(X, y, i, labels[i])

    def gradient_descent(self, X, y, i, label, alpha=1, epsilon=1e-5, max_iter=10000):
        sample_size, feature_size = X.shape

        w = np.ones((feature_size, 1))
        coverge = False
        iteration = 0
        while not coverge and iteration < max_iter:
            loss = self.loss_function(X, y, label, w)
            gradient = np.dot(X.T, loss) / sample_size # + 0.01 * w
            #  gradient = -X.T.dot(y - scipy.special.expit(X.dot(w)))
            w = w + alpha * gradient

            coverge = (np.linalg.norm(gradient) <= epsilon)
            iteration += 1

        self.coefficients[i] = w.A1[:-1]
        self.intercept[0][i] = w.A1[-1]

    def loss_function(self, X, y, label, w):
        sample_size, feature_size = y.shape
        loss = np.ones((sample_size, 1))
        for i in xrange(sample_size):
            if y[i] != label:
                loss[i][0] = 0
            loss[i][0] = loss[i][0] - self.sigmoid(np.dot(X[i], w))
        return loss

    def predict(self, X):
        samle_size = X.shape[0]
        result = np.empty((X.shape[0], 1))
        label = 0
        max_probability = 0
        for i in xrange(samle_size):
            for j in xrange(len(self.coefficients)):
                probability = self.sigmoid(np.dot(X[i], self.coefficients[j]) + self.intercept[0][j])
                #  print "X[", i,"]", X[i]
                #  print "coefficients[", j, "]", self.coefficients[j]
                #  print "probability", probability
                if probability > max_probability:
                    max_probability = probability
                    label = j
                result[i][0] = label
        return result

    def feature_mapping(self, X):
        return X

    @staticmethod
    def sigmoid(t):
        return 1. / (1. + np.exp(-t))
