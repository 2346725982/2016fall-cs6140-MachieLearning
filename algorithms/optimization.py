
class Optimization(object):
    @classmethod
    def batch_gradient_descent(self, X, y, i, label, alpha=1, epsilon=1e-5, max_iter=10000):
    #  def gradient_descent(self, X, y, i, label, alpha=1, epsilon=1e-5, max_iter=10000):
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
