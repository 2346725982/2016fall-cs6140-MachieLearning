import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.coefficients = list()
        self.intercept = 0
        self.model = None

    def fit(self, X, Y, fit_intercept=True, solver="BGD"):
        if fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((X, intercept), axis=1)

        X = np.asmatrix(X)
        Y = np.asmatrix(Y)

        if solver == "matrix":
            self.fit_by_linear_algebra(X, Y)
        elif solver == "BGD":
            self.gradient_descent(X, Y)

    def fit_by_linear_algebra(self, X, Y):
        tmp1 = np.dot(X.T, X)
        tmp2 = np.dot(tmp1.I, X.T)
        arguments = np.dot(tmp2, Y)

        self.coefficients = np.asarray(arguments[:-1])
        self.intercept = np.asarray(arguments[-1])

    def gradient_descent(self, X, Y, alpha=0.1, epsilon=1e-5, max_iter=10000):
        sample_size, feature_size = X.shape

        theta = np.ones((feature_size, 1))
        coverge = False
        iteration = 0
        while not coverge and iteration < max_iter:
            hypothesis = np.dot(X, theta)
            loss = hypothesis - Y
            gradient = 2 * np.dot(X.T, loss) / sample_size
            theta = theta - alpha * gradient

            if np.linalg.norm(gradient) <= epsilon:
                coverge = True
            iteration += 1

        self.coefficients = np.asarray(theta[:-1])
        self.intercept = np.asarray(theta[-1])

    def predict(self, X):
        result = np.empty((X.shape[0], 1))
        for i in range(len(result)):
            result[i] = np.dot(self.coefficients.T, X[i]) + self.intercept
        return result
