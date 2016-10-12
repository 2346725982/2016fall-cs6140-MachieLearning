import numpy


class Report(object):
    @classmethod
    def report_error(cls, original, predicted):
        error = numpy.subtract(original, predicted)
        return numpy.linalg.norm(error)

    @classmethod
    def report_classification_error(cls, original, predicted):
        sample_size = original.shape[0]
        error = 0
        for i in xrange(sample_size):
            if original[i] != predicted[i]:
                error += 1
        return float(error) / sample_size
