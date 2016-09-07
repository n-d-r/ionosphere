#===============================================================================
# Setup
#===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cross_validation import KFold, train_test_split

#===============================================================================
# Loading and preparing the data
#===============================================================================

data = pd.read_csv('ionosphere_processed.csv', index_col=0)
data_mat = data.as_matrix()
n_rows, n_cols = data_mat.shape
X = data_mat[:, :n_cols-1]
y = data_mat[:, n_cols-1]

#===============================================================================
# Function definitions
#===============================================================================

# nothing here yet

#===============================================================================
# Prediction, single logistic regression, entire dataset
#===============================================================================

logit_classifier = LogisticRegression(solver='liblinear')
logit_classifier.fit(X, y)
yhat = logit_classifier.predict(X)
probabilities = logit_classifier.predict_proba(X)

fpr, tpr, thresholds = roc_curve(y_true=y, 
                                 y_score=probabilities[:, 1], 
                                 pos_label='g')
auc = auc(fpr, tpr)
plt.figure(figsize=(7, 4))
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, '--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
plt.close('all')

#===============================================================================
# Prediction, single logistic regression, cross-validation
#===============================================================================

list_of_auc = []
kf = KFold(n=data_mat.shape[0], n_folds=5, shuffle=True)

for train_index, test_index in kf:
  classifier = LogisticRegression(solver='liblinear').fit(X[train_index, :],
                                                          y[train_index])
  probabilities = classifier.predict_proba(X[test_index, :])
  fpr, tpr, thresholds = roc_curve(y_true=y[test_index],
                                   y_score=probabilities[:, 1],
                                   pos_label='g')
  list_of_auc.append(auc(fpr, tpr))

#===============================================================================
# Prediction, single decision tree, entire dataset
#===============================================================================

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X, y)
yhat = tree_classifier.predict(X)
conf_mat = confusion_matrix(y_true=y, y_pred=yhat)

#===============================================================================
# Prediction, single decision tree, train_test_split
#===============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
                                                    random_state=0)

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
yhat = tree_classifier.predict(X_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=yhat)

#===============================================================================
# Prediction, single decision tree, cross-validation
#===============================================================================

list_of_conf_mat = []
kf = KFold(n=data_mat.shape[0], n_folds=5, shuffle=True)

for train_index, test_index in kf:
  classifier = DecisionTreeClassifier()
  classifier.fit(X[train_index, :], y[train_index])
  yhat = classifier.predict(X[test_index, :])
  list_of_conf_mat.append(confusion_matrix(y_true=y[test_index],
                                           y_pred=yhat))