# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:19:07 2020

@author: abhis
"""
# =============================================================================
# KNN-Classifcation
# =============================================================================

# K-Nearest Neighbors is an algorithm for supervised learning. Where the data 
# is 'trained' with data points corresponding to their classification. Once a
#  point is to be predicted, it takes into account the 'K' nearest points to it
#  to determine it's classification.

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
# =============================================================================
# 
# 
# Imagine a telecommunications provider has segmented its customer base by 
# service usage patterns, categorizing the customers into four groups. If 
# demographic data can be used to predict group membership, the company can 
# customize offers for individual prospective customers. It is a classification 
# problem. That is, given the dataset, with predefined labels, we need to build 
# a model to be used to predict class of a new or unknown case.
# 
# The example focuses on using demographic data, such as region, age, and 
# marital, to predict usage patterns.
# 
# The target field, called custcat, has four possible values that correspond
#  to the four customer groups, as follows: 1- Basic Service 2- E-Service
#  3- Plus Service 4- Total Service
# 
# Our objective is to build a classifier, to predict the class of unknown cases.
# =============================================================================
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv')
df.head()

df['custcat'].value_counts()#types of target values

df.hist(column='income', bins=50)#distribution of income

df.columns #column names

# Convert pandas to numpy array to use skit learn library
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

#What are our lables?
y = df['custcat'].values
y[0:5]

# Normalize the data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# splitting data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
#If int, random_state is the seed used by the random number generator
# Test_size is used to define the size of split of test and train data
# It can be float and int ,float 0.2 means 20% of the data

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# =============================================================================
# KNN
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# =============================================================================
# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
#         Algorithm used to compute the nearest neighbors:
# 
#         - 'ball_tree' will use :class:`BallTree`
#         - 'kd_tree' will use :class:`KDTree`
#         - 'brute' will use a brute-force search.
#         - 'auto' will attempt to decide the most appropriate 
# algorithm
#           based on the values passed to :meth:`fit` method.

# Leaf Size :This can affect the
        # speed of the construction and query, as well as the memory
        # required to store the tree.

#  p=2 means minkowski using eucledian distance
# =============================================================================

#predict
yhat = neigh.predict(X_test)
yhat[0:5]

# In multilabel classification, accuracy classification score function
#  computes subset accuracy. This function is equal to the 
#  jaccard_similarity_score function. Essentially, it calculates how 
#  match the actual labels and predicted labels are in the test set.

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#############
# Now for k = 6
# write your code here

k = 6
#Train Model and Predict  
neigh_1 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh_1
yhat_1 = neigh_1.predict(X_test)
yhat_1[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh_1.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat_1))

#so we can see as we increased the k the train and test accuracy both
#decreased.

#We can calucalte the accuracy of KNN for different Ks
Ks = 10
mean_acc = np.zeros((Ks-1))
#created array of size 1 less than Ks with values as zeroes
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
#created an empty list ConfusionMx
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#the greater the accuracy the better is classifier

#plot the accuracy with sd
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
