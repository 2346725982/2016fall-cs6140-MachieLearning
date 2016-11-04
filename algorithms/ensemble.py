#! /usr/bin/python


class Ensemble(object):
    """
    Implement Ensemble using template pattern
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def boosting(self, classifiers, X, y):
        final_classifier = list()
        for classifier in classifiers:
            training_sample = self.get_training_sample()
            classifier(training_sample)
            weight = self.calculate_weight()

            final_classifier.append((weight, classifier))

        return final_classifier

    def get_training_sample(self):
        if self.algorithm == "bagging":
            pass
        elif self.algorithm == "ada":
            pass
        pass

    def calculate_weight(self):
        if self.algorithm == "bagging":
            pass
        elif self.algorithm == "ada":
            pass
        pass
