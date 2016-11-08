import scipy.io as sio
from sklearn.neural_network import MLPClassifier
from algorithms.neural_networks import NeuralNetworks
from model_evaluation import ClassificationReport as Report


if __name__ == '__main__':
    mat_contents = sio.loadmat("data/data.mat")
    #  print mat_contents

    training_data = mat_contents['X_trn']
    training_result = mat_contents['Y_trn']

    test_data = mat_contents['X_tst']
    test_result = mat_contents['Y_tst']

    model = MLPClassifier()
    model.fit(training_data, training_result)
    predict = model.predict(test_data)
    print model.get_params
    print Report.report_error(test_result, predict)
    predict_result = model.predict(test_data)
    print Report.report_error(test_result, predict_result)

    model = NeuralNetworks()
    model.fit(training_data, training_result)
    predict_result = model.predict(training_data)
    #  print predict_result
    print Report.report_error(training_result, predict_result)
    predict_result = model.predict(test_data)
    #  print predict_result
    print Report.report_error(test_result, predict_result)
