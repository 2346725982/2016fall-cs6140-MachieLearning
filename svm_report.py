import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from algorithms.svm import SVM
from model_evaluation import ClassificationReport as Report


if __name__ == '__main__':
    mat_contents = sio.loadmat("data/data.mat")
    #  print mat_contents

    training_data = mat_contents['X_trn']
    training_result = mat_contents['Y_trn']

    test_data = mat_contents['X_tst']
    test_result = mat_contents['Y_tst']

    model = SVC()
    model.fit(training_data, training_result)
    predict = model.predict(test_data)
    print model.get_params
    print predict
    print Report.report_error(test_result, predict)

    model = SVM()
    model.fit(training_data, training_result)
    print Report.report_error(test_result, model.predict(test_data))
