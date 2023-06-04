import pandas as pd
import torch
import warnings

# import tools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

# import regression models
from sklearn.linear_model import LogisticRegression

# import svm models
from sklearn import svm

# import tree models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree

# import nn models
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def write_data_frame_to_file(data, file_name="test.txt"):
    data.to_csv('data/' + file_name, index=False)


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
    object_list = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                   'native.country']
    for obj in object_list:
        df_encoded = pd.get_dummies(data[obj], prefix=obj)
        data = pd.concat([data, df_encoded], axis=1)
        data.drop(obj, axis=1, inplace=True)

    if print_info:
        print(data.dtypes)

    return data, labels


def evaluate(y_test, y_pred, do_evaluation):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    if do_evaluation:
        print("accuracy: ", accuracy, "\tprecision: ", precision)
    return accuracy


# regression part
def do_regression(X_train, X_test, y_train, y_test, do_evaluation=True):
    warnings.filterwarnings("ignore")
    clf = LogisticRegression()
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


# svm part
def do_svm(X_train, X_test, y_train, y_test, C=1, do_evaluation=True):
    clf = svm.SVC(C=C)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


# tree part
def do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False, do_evaluation=True):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    # show the tree
    if print_tree:
        text_representation = tree.export_text(clf)
        print(text_representation)

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


def do_forest(X_train, X_test, y_train, y_test, depth=None, estimators=100, do_evaluation=True):
    clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


def do_xgboost(X_train, X_test, y_train, y_test, depth=6, estimators=100, learning_rate=0.3, do_evaluation=True):
    clf = XGBClassifier(max_depth=depth, n_estimators=estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


def do_sk_net(X_train, X_test, y_train, y_test, print_net_info=False, do_evaluation=True):
    clf = MLPClassifier(solver='adam', activation='relu', tol=1e-5, hidden_layer_sizes=(100, ), max_iter=2000, verbose=print_net_info)
    clf.fit(X_train, y_train.values.ravel())

    if print_net_info:
        print(clf.n_layers_)
        print(clf.n_iter_)
        print(clf.loss_)
        print(clf.out_activation_)

    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_test, y_pred, do_evaluation)

    return clf, accuracy


class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(features, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.sigmoid(x)


def do_sf_net(X_train, X_test, y_train, y_test, print_net_info=False, do_evaluation=True):
    net = Net(X_train.shape[1])
    # print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    return None



def main():
    use_self_encode = False
    use_imblearn = False
    use_smote = False

    data = pd.read_csv('data/traindata.csv')
    labels = pd.read_csv('data/trainlabel.txt')

    # select data split
    if use_self_encode:
        data, labels = self_pretreat(data, labels)
    else:
        data, labels = sklearn_pretreat(data, labels)
    X_train, X_test, y_train, y_test = data_split(0.1, data, labels)

    # deal data imbalance
    if use_imblearn:
        if use_smote:
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        else:
            X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

    # regression part
    # clf, acc = do_regression(X_train, X_test, y_train, y_test)

    # svm part
    # clf, acc = do_svm(X_train, X_test, y_train, y_test, C=10)

    # tree part
    # clf, acc = do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False)
    # clf, acc = do_forest(X_train, X_test, y_train, y_test)
    # clf, acc = do_xgboost(X_train, X_test, y_train, y_test)

    # nn part
    # clf, acc = do_sk_net(X_train, X_test, y_train, y_test, print_net_info=True)
    do_sf_net(X_train, X_test, y_train, y_test)

    # multiple test
    # avg_acc = 0
    # for i in range(30):
    #     X_train, X_test, y_train, y_test = data_split(0.1, data, labels)
    #     if use_imblearn:
    #         if use_smote:
    #             X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    #         else:
    #             X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    #
    #     # clf, acc = do_regression(X_train, X_test, y_train, y_test)
    #
    #     # clf, acc = do_tree(X_train, X_test, y_train, y_test, depth=7, print_tree=False)
    #     # clf, acc = do_forest(X_train, X_test, y_train, y_test, depth=None, estimators=100)
    #     # clf, acc = do_xgboost(X_train, X_test, y_train, y_test)
    #     clf, acc = do_svm(X_train, X_test, y_train, y_test)
    #
    #     avg_acc += acc
    # print("avg_acc: ", avg_acc / 30)


if __name__ == "__main__":
    main()
