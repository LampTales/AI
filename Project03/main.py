import pandas as pd
import torch

data = pd.read_csv('data/traindata.csv')
labels = pd.read_csv('data/trainlabel.txt')

# merge data and labels
# data = pd.concat([data, labels], axis=1)
# print(data.dtypes)

# type transition
data['workclass'] = data['workclass'].astype('category')
# print(data.dtypes)


# tree part
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X = data
y = labels
tree_clf = DecisionTreeClassifier(max_depth=5)
tree_clf.fit(X, y)

# show the tree
text_representation = tree.export_text(tree_clf)
print(text_representation)