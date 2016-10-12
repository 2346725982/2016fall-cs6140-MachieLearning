import scipy.io as sio
from sklearn import linear_model
from logistic_regression.logistic_regression import LogisticRegression
from linear_regression import (LinearRegression, NonlinearMapping)
from sklearn.metrics import classification_report
from report import Report

if __name__ == '__main__':
    mat_contents = sio.loadmat("data/logistic_regression.mat")
    #  mapping = NonlinearMapping()
    #  training_data = mapping.convert(mat_contents['X_trn'], degree=1)
    training_data = mat_contents['X_trn']
    test_data = mat_contents['X_tst']
    training_result = mat_contents['Y_trn']
    test_result = mat_contents['Y_tst']

    for model in (linear_model.LogisticRegression(), ):
        model.fit(training_data, training_result)
        print model.intercept_
        print model.coef_
        print "###############################"
        print "training error\t",
        print Report.report_classification_error(training_result,
                                  model.predict(training_data)),
        print "\t",
        print "test error\t",
        print Report.report_classification_error(test_result,
                                  model.predict(test_data)),
        print ""
        print "###############################"

    #  for model in (linear_model.LogisticRegression(), LogisticRegression()):
    for model in (LogisticRegression(),):
        model.fit(training_data, training_result)
        #  print model.predict(test_data)
        print model.intercept
        print model.coefficients
        print "###############################"
        print "training error\t",
        print Report.report_classification_error(training_result,
                                  model.predict(training_data)),
        print "\t",
        print "test error\t",
        print Report.report_classification_error(test_result,
                                  model.predict(test_data)),
        print ""
        print "###############################"
