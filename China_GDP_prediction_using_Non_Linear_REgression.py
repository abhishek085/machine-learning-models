# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:32:57 2020

@author: abhis
"""
# Non -Linear Regression  China GDP database

import numpy as np
import matplotlib.pyplot as plt

# Creating different types of functions and plotiing them!

#LINEAR plot
x = np.arange(-5.0, 5.0, 0.1)#return values between -5 and 5 with 0.1 interval
##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size) #random sample with normal distribution
#and size equal to size of x array size.
ydata = y + y_noise
plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
#here 'bo' means c='b' color blue and marker = 'o' for scatter plot
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# =============================================================================
# Non-linear regressions are a relationship between independent variables  𝑥 
 # and a dependent variable  𝑦  which result in a non-linear function modeled
 # data. Essentially any relationship that is not linear can be termed as
 # non-linear, and is usually represented by the polynomial of  𝑘  degrees 
 # (maximum power of  𝑥 ).

#  𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑 
 
# Non-linear functions can have elements like exponentials, logarithms, 
# fractions, and others. For example:
# 𝑦=log(𝑥)
 
# Or even, more complicated such as :
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)


x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
# =============================================================================
# =============================================================================
# Quadratic function
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
# =============================================================================
# =============================================================================
# Exponential

X = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

Y= np.exp(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
# =============================================================================
# =============================================================================
# Logarithmic
X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
# =============================================================================
# =============================================================================
# SIGMOID/Logistic function
X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
# =============================================================================
# =============================================================================
# Non - Linear Regression Analysis
# =============================================================================

# We download a dataset with two columns, the first, a year between 1960 and 
# 2014, the second, China's corresponding annual gross domestic income in US 
# dollars for that year.

import numpy as np
import pandas as pd

#downloading dataset

    
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv")
df.head(10)

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#we can see that its a non linear relation
# Choosing a model based on the above observation- we can use exponential or logistic graphs

#logistic model
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# =============================================================================
# The formula for the logistic function is the following:

# $$ \hat{Y} = \frac1{1+e^{\beta_1(X-\beta_2)}}$$

# $\beta_1$: Controls the curve's steepness,

# $\beta_2$: Slides the curve on the x-axis.
# =============================================================================

# Now, let's build our regression model and initialize its parameters. 
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
 
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')    


# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# we can use curve_fit which uses non-linear least squares to fit our sigmoid 
# function, to data. Optimal values for the parameters so that the sum of the 
# squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

# popt are our optimized parameters.

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata) #non linear least square method
# to fit function sigmoid or any other function
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# =============================================================================
# popt : array
#         Optimal values for the parameters so that the sum of the squared
#         residuals of ``f(xdata, *popt) - ydata`` is minimized
#     pcov : 2d array
#         The estimated covariance of popt. The diagonals provide the variance
#         of the parameter estimate. To compute one standard deviation errors
#         on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
# =============================================================================

# Plot resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#let train the model as curve is fitted and 
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

#r2 between y target values and estimated values as inputs.
#bigger then r2 better is the estimation or if 0 that means predict same values of y
# irrespective of inputs