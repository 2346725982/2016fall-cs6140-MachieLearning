import numpy as np


class NonlinearMapping(object):
    def __init__(self):
        pass

    def convert(self, training_data, test_data, degree=2):
        training_sample_size, feature_size = training_data.shape
        test_sample_size, feature_size = test_data.shape
        new_training_data = np.ones((training_sample_size, 0))
        new_test_data = np.ones((test_sample_size, 0))
        for j in range(feature_size):
            for d in range(1, degree + 1):
                new_training_values = [[training_data[i][j] ** d] for i in xrange(training_sample_size)]
                new_test_values = [[test_data[i][j] ** d] for i in xrange(test_sample_size)]

                new_training_column = np.array(new_training_values)
                new_test_column = np.array(new_test_values)

                mean = np.mean(new_training_column)
                std_deviation = np.std(new_training_column)

                for i in xrange(new_training_column.shape[0]):
                    new_training_column[i] = (new_training_column[i] - mean) / std_deviation

                for i in xrange(new_test_column.shape[0]):
                    new_test_column[i] = (new_test_column[i] - mean) / std_deviation

                new_training_data = np.hstack((new_training_data, new_training_column))
                new_test_data = np.hstack((new_test_data, new_test_column))

        return new_training_data, new_test_data

    def normalize(self, column):
        mean = np.mean(column)
        std_deviation = np.std(column)
        for i in xrange(column.shape[0]):
            column[i] = (column[i] - mean) / std_deviation
        return column
