import collections
import numpy as np
import scipy
from report import Report


class RidgeRegression(object):
    def __init__(self, k_folds=None):
        self.intercept = 0
        self.coefficients = None
        self.alpha = None
        self.training_method = None
        self.k_folds = k_folds

    def fit(self, X, y, alpha=10, fit_intercept=True, solver="matrix"):
        if fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((X, intercept), axis=1)

        if solver == "matrix":
            self.training_method = self.linear_algebra
        elif solver == "BGD":
            self.training_method = self.gradient_descent

        if self.k_folds:
            parameters = self.cross_validation(X, y)
        else:
            parameters = self.training_method(X, y, alpha)

        self.coefficients = np.asarray(parameters[:-1])
        self.intercept = np.asarray(parameters[-1])

    def cross_validation(self, X, y):
        result = collections.defaultdict(dict)
        for alpha in np.arange(0, 100, 0.1):
            split_X = np.split(X, self.k_folds)
            split_y = np.split(y, self.k_folds)
            for i in xrange(self.k_folds):
                test_data = split_X[i]
                test_result = split_y[i]
                np.delete(split_X, i)
                np.delete(split_y, i)
                training_data = np.concatenate(split_X)
                training_result = np.concatenate(split_y)

                parameters = self.training_method(training_data,
                                                  training_result,
                                                  alpha)

                self.coefficients = np.asarray(parameters[:-1])
                self.intercept = np.asarray(parameters[-1])

                hold_out_error = \
                    Report.report_error(test_result,
                                        self.predict(test_data,
                                                     with_intercept=True)),

                result[alpha]["parameters"] = parameters

                if "error" not in result[alpha]:
                    result[alpha]["error"] = 0
                result[alpha]["error"] += hold_out_error[0]

        optimal = min(result, key=lambda r: result[r]["error"])
        self.alpha = optimal
        return result[optimal]["parameters"]

    def linear_algebra(self, X, Y, alpha):
        Tikhonov_matrix = alpha * np.eye(X.shape[1])
        feature_multiplicatioin = np.dot(X.T, X)
        tmp3 = np.add(feature_multiplicatioin, Tikhonov_matrix)
        tmp4 = np.dot(scipy.linalg.inv(tmp3), X.T)
        arguments = np.dot(tmp4, Y)

        return arguments

    def gradient_descent(self, X, Y, alpha, epsilon=0.001, max_iter=1000):
        sample_size, feature_size = X.shape

        theta = np.ones((feature_size, 1))
        coverge = False
        iteration = 0
        while not coverge and iteration < max_iter:
            hypothesis = np.dot(X, theta)
            loss = hypothesis - Y
            gradient = (2 * np.dot(X.T, loss) / sample_size + 0.01 * theta)
            theta = theta - 0.01 * gradient

            if np.linalg.norm(gradient) <= epsilon:
                coverge = True
            iteration += 1

        return theta

    def predict(self, X, with_intercept=False):
        result = np.empty((X.shape[0], 1))
        for i in range(len(result)):
            if with_intercept:
                result[i] = np.dot(X[i], np.append(self.coefficients, self.intercept))
            else:
                result[i] = np.dot(X[i], self.coefficients) + self.intercept
        return result
