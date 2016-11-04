import numpy


class RegressionReport(object):
    @classmethod
    def report_error(cls, original, predicted):
        error = numpy.subtract(original, predicted)
        return numpy.linalg.norm(error)
