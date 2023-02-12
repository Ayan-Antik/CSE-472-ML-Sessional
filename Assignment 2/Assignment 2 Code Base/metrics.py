"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


def accuracy(y_true, y_pred):
    """

    :param y_true: true labels of shape (total datapoints, 1)
    :param y_pred: predicted labels of shape (total datapoints, 1)
    :return: accuracy score
    """
    # todo: implement
    true = 0
    total = len(y_true)
    # print(y_true.T)
    # print(y_pred.T)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            true += 1
    return (true / total)
    

def precision_score(y_true, y_pred):
    """

    :param y_true: true labels of shape (total datapoints, 1)
    :param y_pred: predicted labels of shape (total datapoints, 1)
    :return: precision score
    """
    # todo: implement
    tp = 0
    fp = 0
    total = len(y_true)
    # print(y_pred.T)
    for i in range(total):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """

    :param y_true: true labels of shape (total datapoints, 1)
    :param y_pred: predicted labels of shape (total datapoints, 1)
    :return: recall score
    """
    # todo: implement
    tp = 0
    fn = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """

    :param y_true: true labels of shape (total datapoints, 1)
    :param y_pred: predicted labels of shape (total datapoints, 1)
    :return: f1 score
    """
    # todo: implement
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r)
    
