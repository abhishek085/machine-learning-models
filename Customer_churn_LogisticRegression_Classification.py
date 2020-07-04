# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:59:24 2020

@author: abhis
"""


# =============================================================================
# LOGISTIC REGRESSION_ CUSTOMER_CHURN
# # ===========================================================================
#  create a model for a telecommunication company, to predict when its customers 
#  will leave for a competitor, so that they can take some action to retain the customers.

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

churn_df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv")
churn_df.head()

# =============================================================================
# Lets select some features for the modeling. Also we change the target data 
#type to be integer, as it is a requirement by the skitlearn algorithm
# =============================================================================

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

print(churn_df.shape)
churn_df.columns

# Lets define X, and y for our dataset:
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(churn_df['churn'])
y [0:5]

# we normalize the dataset:
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

 # we split our dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=3)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# =============================================================================
# Build the Model
#and optimize
# The version of Logistic Regression in Scikit-learn, support regularization.
#  Regularization is a technique used to solve the overfitting problem in machine 
#  learning models. C parameter indicates inverse of regularization strength which 
#  must be a positive float. Smaller values specify stronger regularization. 
#  Now lets fit our model with train set:
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

# =============================================================================
# # C : float, default=1.0
#         Inverse of regularization strength; must be a positive float.
#         Like in support vector machines, smaller values specify stronger
#         regularization.

 # solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
 #            default='lbfgs'

 #        Algorithm to use in the optimization problem.

 #        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
 #          'saga' are faster for large ones.
 #        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
 #          handle multinomial loss; 
           # -'liblinear' is limited to one-versus-rest
 #          schemes.
 #        - 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
 #        - 'liblinear' and 'saga' also handle L1 penalty
 #        - 'saga' also supports 'elasticnet' penalty
 #        - 'liblinear' does not support setting ``penalty='none'``
# =============================================================================

yhat = LR.predict(X_test)
yhat

# =============================================================================
#predict_proba returns estimates for all classes, ordered by the label of classes.
 # So, the first column is the probability of class 1, P(Y=1|X), and second column
 # is probability of class 0, P(Y=0|X) 
# =============================================================================

yhat_prob = LR.predict_proba(X_test)
yhat_prob

# =============================================================================
# EVALUATION
  
 #    jaccard index
 # Lets try jaccard index for accuracy evaluation. we can define jaccard as the 
 # of the intersection divided by the size of the union of two label sets. If the
 # entire set of predicted labels for a sample strictly match with the true set of
 # labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# =============================================================================

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

# =============================================================================
# Confusion matrix

  # Another way of looking at accuracy of classifier is to look at confusion matrix.
# =============================================================================
# =============================================================================
# 
from sklearn.metrics import classification_report, confusion_matrix
import itertools
#Function starts
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
###Function Ends    
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  
                      title='Confusion matrix')
# =============================================================================
# =============================================================================
# Look at first row. The firsr row is for customers whose actual churn value
 # in test set is 1. As you can calculate, out of 40 customers, the churn value 
 # of 15 of them is 1. And out of these 15, the classifier correctly predicted
 # 6 of them as 1, and 9 of them as 0.

# It means, for 6 customers, the actual churn value were 1 in test set, and
#  classifier also correctly predicted those as 1. However, while the actual label 
#  of 9 customers were 1, the classifier predicted those as 0, which is not
#  very good. We can consider it as error of the model for first row.

# What about the customers with churn value 0? Lets look at the second row.
#  It looks like there were 25 customers whom their churn value were 0.

# The classifier correctly predicted 24 of them as 0, and one of them wrongly as
#  1. So, it has done a good job in predicting the customers with churn value 0.
#  A good thing about confusion matrix is that shows the model’s ability to 
#  correctly predict or separate the classes. In specific case of binary 
#  classifier, such as this example, we can interpret these numbers as the 
#  count of true positives, false positives, true negatives, and false negatives.
# =============================================================================

print (classification_report(y_test, yhat))

# =============================================================================
# Based on the count of each section, we can calculate precision and recall of 
# each label:

# Precision is a measure of the accuracy provided that a class label has been
#  predicted. It is defined by: precision = TP / (TP + FP)

# Recall is true positive rate. It is defined as: Recall = TP / (TP + FN)

# So, we can calculate precision and recall of each class.

# F1 score: Now we are in the position to calculate the F1 scores for each label
#  based on the precision and recall of that label.

# The F1score is the harmonic average of the precision and recall, where an
#  F1 score reaches its best value at 1 (perfect precision and recall) and 
#  worst at 0. It is a good way to show that a classifer has a good value for 
#  both recall and precision.

# And finally, we can tell the average accuracy for this classifier is the
#  average of the f1-score for both labels, which is 0.72 in our case.
# =============================================================================

# =============================================================================
# log loss
#  Now, lets try log loss for evaluation. In logistic regression, the output can
#  be the probability of customer churn is yes (or equals to 1). This probability
#  is a value between 0 and 1. Log loss( Logarithmic loss) measures the
#  performance of a classifier where the predicted output is a probability
#  value between 0 and 1.
# =============================================================================

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)