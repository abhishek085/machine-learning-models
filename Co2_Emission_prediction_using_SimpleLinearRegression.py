# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:51:33 2020

@author: abhis
"""
# =============================================================================
# CognitiveCLass Machine leanring
# =============================================================================
#Simple Linear Regression
# =============================================================================
#  Use scikit-learn to implement simple linear regression
#We download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. 
# Then, we split our data into training and test sets,
 # create a model using training set, Evaluate your model using test set, and finally
 # use model to predict unknown value   
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

url="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
df=pd.read_csv(url)

# take a look at the dataset
df.head()

# summarize the data
df.describe() 

# Lets select some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# Now, lets plot each of these features vs the Emission, to see how linear is their relation:

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# =============================================================================
# =============================================================================
# # Train/Test Split involves splitting the dataset into training
#  and testing sets respectively, which are mutually exclusive. 
#  After which, you train with the training set and test with the
#  testing set. This will provide a more accurate evaluation on 
#  out-of-sample accuracy because the testing dataset is not part 
#  of the dataset that have been used to train the data. It is 
#  more realistic for real world problems.
# 
# This means that we know the outcome of each data point in this 
# dataset, making it great to test with! And since this data has 
# not been used to train the model, the model has no knowledge of 
# the outcome of these data points. So, in essence, it is truly an
#  out-of-sample testing.
# =============================================================================
# =============================================================================

msk = np.random.rand(len(df)) < 0.8
#here it is trying to create a random number of the size of dataframe
#and compare those number with a Value
#to have a  TRUE or FALSE array of length equal to Dataframe

print(msk)
train = cdf[msk]
#setting True values to Train data set
test = cdf[~msk]
#setting false values to Test data set.

# So there is no percentage formula used here its purely random train and test

# Simple Regression Model
# Linear Regression fits a linear model with coefficients 
# B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the independent
#  x in the dataset, and the dependent y by the linear approximation

# Train data distribution

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Modeling
# Using sklearn package to model data.

#OLS(ordinary least square) linear Regression
# =============================================================================
# LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
# to minimize the residual sum of squares between the observed targets in
# the dataset, and the targets predicted by the linear approximation.
# =============================================================================
from sklearn import linear_model
regr = linear_model.LinearRegression()
#using y=m*x+c linear line and assigning x and y values from training data set below

train_x = np.asanyarray(train[['ENGINESIZE']])#array of float
train_y = np.asanyarray(train[['CO2EMISSIONS']])#array of float

#uses fit method to input the value to the model
#fit method takes x and y values array
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_) #value of m that is slope
print ('Intercept: ',regr.intercept_) #value of C 

# =============================================================================
# # As mentioned before, __Coefficient__ and __Intercept__ in the simple linear 
# # regression, are the parameters of the fit line. 
# # Given that it is a simple linear regression, with only 2 parameters, and
# #  knowing that the parameters are the intercept and slope of the line,
# #  sklearn can estimate them directly from our data. 
# # Notice that all of the data must be available to traverse and calculate
# #  the parameters.
# 
# =============================================================================


# we can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
#plotting just a line of x & y i.e(EngineSize(x) and y=(mx+c)) 
#r is for color red

plt.xlabel("Engine size")
plt.ylabel("Emission")
# =============================================================================
# 
# Evaluation
# we compare the actual values and predicted values to calculate the accuracy of a 
# regression model. Evaluation metrics provide a key role in the development of a 
# model, as it provides insight to areas that require improvement.
# 
# There are different model evaluation metrics, lets use MSE here to calculate the 
# accuracy of our model based on the test set:
# 
# - Mean absolute error: It is the mean of the absolute value of the errors. This is 
# the easiest of the metrics to understand since it’s just average error.
# - Mean Squared Error (MSE) is the mean of the squared error.
#  It’s more popular than Mean absolute error because the focus is geared more 
#  towards large errors. This is due to the squared term exponentially increasing 
#  larger errors in comparison to smaller ones.
# - Root Mean Squared Error (RMSE).
# - R-squared is not error, but is a popular metric for accuracy of your model.
#  It represents how close the data are to the fitted regression line.
#  The higher the R-squared, the better the model fits your data.
#  Best possible score is 1.0 and it can be negative (because the model can be 
#  arbitrarily worse).
# =============================================================================

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
