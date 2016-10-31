import scipy.io as sio
from sklearn import linear_model
from regression.linear_regression import LinearRegression
from feature_processing.nonlinear_mapping import NonlinearMapping
from model_evaluation.model_evaluation import RegressionReport as Report


if __name__ == '__main__':
    mat_contents = sio.loadmat("data/linear_regression.mat")
    mapping = NonlinearMapping()

    for n in (1, 2, 5, 10, 20):
        training_data, test_data = mapping.convert(mat_contents['X_trn'],
                                                   mat_contents['X_tst'],
                                                   degree=n)
        training_result = mat_contents['Y_trn']
        test_result = mat_contents['Y_tst']

        print "###############################"
        for model in (linear_model.LinearRegression(), LinearRegression()):
            model.fit(training_data, training_result)

            print "n =\t", n,
            print "\ttraining error\t",
            print Report.report_error(training_result,
                                      model.predict(training_data)),
            print "\ttesting error\t",
            print Report.report_error(test_result,
                                      model.predict(test_data)),
            print ""
        print "###############################"
