import scipy.io as sio
from sklearn import linear_model
from linear_regression import (RidgeRegression, NonlinearMapping)
from report import Report


if __name__ == '__main__':
    mat_contents = sio.loadmat("data/linear_regression.mat")
    mapping = NonlinearMapping()

    for n in (1, 2, 5, 10, 20):
        training_data = mapping.convert(mat_contents['X_trn'], degree=n)
        test_data = mapping.convert(mat_contents['X_tst'], degree=n)
        training_result = mat_contents['Y_trn']
        test_result = mat_contents['Y_tst']

        print "###############################"
        model = linear_model.Ridge()
        model.fit(training_data, training_result)

        print "n =\t", n,
        print "\ttraining error\t",
        print Report.report_error(training_result,
                                  model.predict(training_data)),
        print "\ttesting error\t",
        print Report.report_error(test_result,
                                  model.predict(test_data)),
        print ""

        for k_folds in (2, 5, 10, training_data.shape[0]):
            model = RidgeRegression(k_folds=k_folds)
            model.fit(training_data, training_result)

            print "n =\t", n,
            print "\t",
            print "folds =\t", k_folds,
            print "\t",
            print "optimal lambda\t",
            print model.alpha,
            print "\t",
            print "training error\t",
            print Report.report_error(training_result,
                                      model.predict(training_data)),
            print "\ttesting error\t",
            print Report.report_error(test_result,
                                      model.predict(test_data)),
            print ""
        print "###############################"
