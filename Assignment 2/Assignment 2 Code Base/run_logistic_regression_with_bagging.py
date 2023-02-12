from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score, accuracy

if __name__ == '__main__':
    # data load
    X, y = load_dataset('data/parkinsons.csv')
    

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    # params = dict()
    base_estimator = LogisticRegression()
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print(f'Accuracy: {round(accuracy(y_true=y_test, y_pred=y_pred)*100, 2)}%')
    print(f'Precision score:  {round(precision_score(y_true=y_test, y_pred=y_pred)*100, 2)}%')
    print(f'Recall score: {round(recall_score(y_true=y_test, y_pred=y_pred)*100, 2)}%')
    print(f'F1 score: {round(f1_score(y_true=y_test, y_pred=y_pred)*100, 2)}%')
