import numpy


class RegressionReport(object):
    @classmethod
    def report_error(cls, original, predicted):
        error = numpy.subtract(original, predicted)
        return numpy.linalg.norm(error)


class ClassificationReport(object):
    @classmethod
    def report_error(cls, original, predicted):
        sample_size = original.shape[0]
        error = 0
        for i in xrange(sample_size):
            if original[i] != predicted[i]:
                error += 1
        return float(error) / sample_size

    @classmethod
    def report_confusion_matrix(cls, original, predicted):
        pass
