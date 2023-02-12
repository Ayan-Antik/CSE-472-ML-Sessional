from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator: base estimator
        :param n_estimator: number of estimators
        :return: None
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []


    def fit(self, X, y):
        """
        :param X: training data
        :param y: training labels
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        for _ in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))
        # print(f'self.estimators: {self.estimators}')
        return self
        

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X: feature matrix of shape (total datapoints, number of features)
        :return: predicted labels of shape (total datapoints, 1)
        """
        # todo: implement

        y_pred = np.zeros((X.shape[0], 1))
        for estimator in self.estimators:   
            y_pred += estimator.predict(X)
            # print(y_pred.T)
        # print(y_pred.T)
        y_pred = np.where(y_pred > self.n_estimator / 2, 1, 0)
        # print(y_pred.T)
        
        return y_pred
