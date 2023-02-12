import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, num_iter = 1000):
        """
        figure out necessary params to take as input
        :param params: dictionary of parameters
        """
        # todo: implement
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.w = []
        self.b = 0
           
    def fit(self, X, y, print_cost = False):
        """
        :param X: feature matrix of shape (total datapoints(1097), # of features(4))
        :param y: labels of shape (total datapoints(1372), 1)
        :param print_cost: boolean to print cost after every 100 iterations
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        # * initialize weights and bias
        # w = np.random.rand(X.shape[1], 1) # w is (features(4), 1)
        w = np.zeros((X.shape[1], 1)) # w is (features(4), 1)
        b = 0
        # * initialize cost
        cost = 0
        # * initialize gradient
        dw = 0
        db = 0
        X = X.T # X is now (number of features(4), number of datapoints(1097))
        # initialize m
        m = X.shape[1] # m is number of datapoints

        for i in range(self.num_iter):
            # * forward propagation
            z = np.dot(w.T, X) + b # z is (1, datapoints(1097))
            H = 1 / (1 + np.exp(-z)) # H is (1, datapoints(1097))
            cost = -np.sum(y * np.log(H) + (1 - y) * np.log(1 - H)) / m # cost is (1, 1)
            # * backward propagation
            dw = np.dot(X, (H - y).T) / m # X is (features(4), datapoints(1097)) and H - y is (1, datapoints(1097)) so dw is (features(4),1)
            db = np.sum(H - y) / m # db is (1, 1)
            # * update weights and bias
            w = w - (self.learning_rate * dw)
            b = b - (self.learning_rate * db)
            # * print cost every 100 iterations
            if i % 100 == 0 and print_cost:
                print(f'Cost after iteration {i}: {cost}')
        self.w = w
        self.b = b
        # print(f'w: {self.w} --- b: {self.b}')
        return self



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X: feature matrix of shape (total datapoints, number of features)
        :return: predicted labels of shape (total datapoints, 1)
        """
        # todo: implement
        X = X.T # X is now (number of features(4), total datapoints(275))
        # initialize m
        m = X.shape[1] # m is total datapoints
        # initialize y_pred
        y_pred = np.zeros((m, 1)) # y_pred is (total datapoints(275), 1)
        # forward propagation
        z = np.dot(self.w.T, X) + self.b # z is (1, total datapoints(275))
        H = 1 / (1 + np.exp(-z)) # H is (1, total datapoints(275))
        # convert probabilities to 0 or 1
        # print(H)
        for i in range(H.shape[1]):
            if H[0, i] >= 0.5:
                y_pred[i, 0] = 1
            else:
                y_pred[i, 0] = 0
        # print(y_pred.T)
        return y_pred
        
