import numpy as np


class ClassificationReport(object):
    @classmethod
    def report_error(cls, original, predicted):
        error = np.mean(original.flat != predicted.flat)
        return error

    @classmethod
    def report_confusion_matrix(cls, original, predicted):
        pass
