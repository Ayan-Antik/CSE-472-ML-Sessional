import pandas as pd
import numpy as np

def load_dataset(csv_path = 'data/data_banknote_authentication.csv'):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return: X, y
    """
    # todo: implement
    df = pd.read_csv(csv_path)
    # normalized df
    print(df.info())
    print(df.describe())
    print(df['status'].value_counts())
    # print(df.dtypes)
    
    df = df.drop('name', axis=1)
    X = df.drop('status', axis = 1).values
    X = (X - X.min()) / (X.max() - X.min())
    # X = (X - X.mean()) / X.std()
    y = df['status'].values
    print(f'X.shape: {X.shape}')
    print(f'y.shape: {y.shape}')
    return X, y


def split_dataset(X, y, test_size = 0.2, shuffle = True):
    """
    function for spliting dataset into train and test
    :param X: feature matrix
    :param y: labels
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return: X_train, y_train, X_test, y_test
    """
    # todo: implement.

    if shuffle:
        np.random.seed(42)
        shuffler = np.random.permutation(X.shape[0])
        X = X[shuffler]
        y = y[shuffler]
    # print(f'{y[-100:]}')
    X_train, y_train, X_test, y_test = X[:int(X.shape[0] * (1 - test_size))], y[:int(X.shape[0] * (1 - test_size))], X[int(X.shape[0] * (1 - test_size)):], y[int(X.shape[0] * (1 - test_size)):]
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_test.shape: {y_test.shape}')
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X: feature matrix
    :param y: labels
    :return: X_sample, y_sample
    """
    # todo: implement
    # percent = 0.4
    # X_sample = np.zeros((int(X.shape[0]*percent), X.shape[1]))
    # y_sample = np.zeros(int(y.shape[0]*percent))
    X_sample = np.zeros(X.shape)
    y_sample = np.zeros(y.shape)
    
    # rand_checker = set()
    np.random.choice(X, replace=True, size = X.shape[0])
    # for i in range(X.shape[0]):
    #     rand_i = np.random.randint(0, X.shape[0])
    #     # rand_checker.add(rand_i)
    #     # print(f'{rand_i}')
    #     X_sample[i] = X[rand_i]
    #     y_sample[i] = y[rand_i]

    # print(f'len(rand_checker): {len(rand_checker)}')
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
