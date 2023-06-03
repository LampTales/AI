import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_split(test_size, data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


def merge_data_labels(data, labels):
    return pd.concat([data, labels], axis=1)


def sklearn_pretreat(data, labels, print_info=False):
    # nunique = data.nunique()
    types = data.dtypes
    categorical_columns = []
    categorical_dims = {}
    for col in data.columns:
        if types[col] == 'object':
            l_enc = LabelEncoder()
            data[col] = data[col].fillna("VV_likely")
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        # else:
        #     data.fillna(data.loc[train_indices, col].mean(), inplace=True)
    if print_info:
        print(categorical_columns)
        print()
        print(categorical_dims)
        print()
        print(data.dtypes)

    return data, labels


def self_pretreat(data, labels, print_info=False):
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(data.dtypes)
    # print(data.nunique())
    object_list = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for obj in object_list:
        df_encoded = pd.get_dummies(data[obj], prefix=obj)
        data = pd.concat([data, df_encoded], axis=1)
        data.drop(obj, axis=1, inplace=True)

    if print_info:
        print(data.dtypes)

    return data, labels


# tree part
def do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False, do_evaluation=True):
    X = X_train
    y = y_train
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)

    # show the tree
    if print_tree:
        text_representation = tree.export_text(clf)
        print(text_representation)

    if do_evaluation:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

    return clf

def do_forest(X_train, X_test, y_train, y_test, depth=None, estimators=100, do_evaluation=True):
    X = X_train
    y = y_train
    clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators)
    clf.fit(X, y.values.ravel())

    if do_evaluation:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

    return clf


def do_xgboost(X_train, X_test, y_train, y_test, depth=6, estimators=100, learning_rate=0.3, do_evaluation=True):
    X = X_train
    y = y_train
    clf = XGBClassifier(max_depth=depth, n_estimators=estimators, learning_rate=learning_rate)
    clf.fit(X, y)

    if do_evaluation:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

    return clf


def do_net(X_train, X_test, y_train, y_test, do_evaluation=True):
    return None


def main():
    data = pd.read_csv('data/traindata.csv')
    labels = pd.read_csv('data/trainlabel.txt')
    # data, labels = sklearn_pretreat(data, labels)
    data, labels = self_pretreat(data, labels)

    X_train, X_test, y_train, y_test = data_split(0.1, data, labels)
    # clf = do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False)
    # clf = do_forest(X_train, X_test, y_train, y_test)
    clf = do_xgboost(X_train, X_test, y_train, y_test)

    # for i in range(20):
    #     X_train, X_test, y_train, y_test = data_split(0.1, data, labels)
    #     # clf = do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False)
    #     clf = do_forest(X_train, X_test, y_train, y_test, depth=None, estimators=100)




if __name__ == "__main__":
    main()
