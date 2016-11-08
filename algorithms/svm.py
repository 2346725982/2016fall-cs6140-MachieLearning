import random
import numpy as np


class SVM(object):
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, X, y):
        sample_size, feature_size = y.shape

        self.X = X

        self.y = {}
        self.a = {}
        self.b = {}

        labels = np.unique(y)
        for index, label in enumerate(labels, start=1):
            new_y = [1 if y[i][0] == label else -1 for i in xrange(y.shape[0])]
            a, b = self.smo(X, new_y)

            #  y = np.hstack((y, new_y))
            #  a, b = self.smo(X, y[:,index])
            self.y[label] = new_y
            self.a[label] = a
            self.b[label] = b
            #  print b
            #  break

    def smo(self, X, y, C=0.05, tol=0.001, max_passes=100):
        sample_size, feature_size = X.shape
        alpha = np.array([0.0] * sample_size)
        bias = 0

        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in xrange(sample_size):
                ei = np.sum(alpha * y * X.dot(X[i])) + bias - y[i]

                if (y[i] * ei < -tol and alpha[i] < C) or \
                   (y[i] * ei > tol and alpha[i] > 0):
                    j = random.randrange(0, sample_size-1)
                    j = sample_size - 1 if j == i else j
                    ej = np.sum(alpha * y * X.dot(X[j])) + bias - y[j]

                    ai_old = alpha[i]
                    aj_old = alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C+alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - C)
                        H = min(C, alpha[i] + alpha[j])
                    #  print y[i], y[j]
                    #  print L, H
                    #  print ""

                    if L == H:
                        continue

                    yita = 2 * X[i].dot(X[j]) - X[i].dot(X[i]) - X[j].dot(X[j])
                    if yita >= 0:
                        continue

                    alpha[j] = alpha[j] - y[j] * (ei - ej) / yita

                    if alpha[j] > H:
                        alpha[j] = H
                    elif alpha[j] < L:
                        alpha[j] = L

                    if abs(alpha[j] - aj_old) < 10 ** -5:
                        continue

                    alpha[i] = alpha[i] + y[i] * y[j] * (aj_old - alpha[j])
                    b1 = bias - ei - y[i] * (alpha[i] - ai_old) * X[i].dot(X[i]) - \
                         y[j] * (alpha[j] - aj_old) * X[i].dot(X[j])
                    b2 = bias - ej - y[i] * (alpha[i] - ai_old) * X[i].dot(X[j]) - \
                         y[j] * (alpha[j] - aj_old) * X[j].dot(X[j])

                    if 0 < alpha[i] < C:
                        bias = b1
                    elif 0 < alpha[j] < C:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2.0
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        return alpha, bias

    def predict(self, X):
        result = np.zeros((X.shape[0], 1))
        for i in xrange(X.shape[0]):
            max_prob = -9999999
            max_label = None
            for label in self.a:
                prob = np.sum(self.a[label] * self.y[label] * self.X.dot(X[i])) + self.b[label]
                if prob > max_prob:
                    max_prob = prob
                    max_label = label
            result[i][0] = max_label
        return result

    def feature_mapping(self, X):
        return X
