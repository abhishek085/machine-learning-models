# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:43:41 2020

@author: abhis
"""
# =============================================================================
# Support Vector Machines
# =============================================================================

# We will use SVM (Support Vector Machines) to build and train a model using 
#  human cell records, and classify cells to whether the samples are benign or
#  malignant.

# SVM works by mapping data to a high-dimensional feature space so that data 
# points can be categorized, even when the data are not otherwise linearly separable.
#  A separator between the categories is found, then the data are transformed in 
#  such a way that the separator could be drawn as a hyperplane.

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load Data
cell_df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv")
cell_df.head()

# # The ID field contains the patient identifiers. The characteristics of the
#  cell samples from each patient are contained in fields Clump to Mit. The 
#  values are graded from 1 to 10, with 1 being the closest to benign.

# The Class field contains the diagnosis, as confirmed by separate medical 
# procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).


#plot clump size vs uniformity of cell 
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

cell_df.dtypes

# It looks like the BareNuc column includes some values that are not numerical. 
# We can drop those rows:
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

# We want the model to predict the value of Class (that is, benign (=2) or 
# malignant (=4)). As this field can have one of only two possible values, 
# we need to change its measurement level to reflect this

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]

# Train test split data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# The SVM algorithm offers a choice of kernel functions for performing its 
# processing. Basically, mapping data into a higher dimensional space is called
#  kernelling. The mathematical function used for the transformation is known 
#  as the kernel function, and can be of different types, such as:

# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

# Predict
yhat = clf.predict(X_test)
yhat [0:5]

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import itertools

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
    
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix') 


#F1 Score Calculation
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')    

#Jaccard Index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

