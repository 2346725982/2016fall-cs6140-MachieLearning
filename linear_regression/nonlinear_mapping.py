import numpy as np


class NonlinearMapping(object):
    def __init__(self):
        pass

    def convert(self, data, degree=2):
        sample_size, feature_size = data.shape
        new_data = np.ones((sample_size, 0))
        for j in xrange(feature_size):
            for d in xrange(1, degree + 1):
                new_values = [[data[i][j] ** d] for i in xrange(sample_size)]
                new_column = self.normalize(np.array(new_values))
                new_data = np.hstack((new_data, new_column))
        return new_data

    def normalize(self, column):
        mean = np.mean(column)
        std_deviation = np.std(column)
        for i in xrange(column.shape[0]):
            column[i] = (column[i] - mean) / std_deviation
        return column
