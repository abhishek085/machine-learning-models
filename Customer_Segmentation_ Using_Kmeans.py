# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:21:23 2020

@author: abhis
"""

# =============================================================================
# k-Means on a randomly generated dataset
# =============================================================================
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs


#random data generation
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')

#initialize K means

# =============================================================================
# The KMeans class has many parameters that can be used, but we will be using these three:

# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart 
# way to speed up convergence.
# n\_clusters: The number of clusters to form as well as the number of 
# centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n\_init: Number of time the k-means algorithm will be run with different
 # centroid seeds. The final results will be the best output of n\_init 
 # consecutive runs in terms of inertia.
# Value will be: 12
# =============================================================================
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

#fit the model
k_means.fit(X)

#grab the labels created by unlabelled data
k_means_labels = k_means.labels_
k_means_labels

k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# Now let plot the data with centroid for each clusters

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


######################################
# Try to cluster the above dataset into 3 clusters.
# Notice: do not generate data again, use the same dataset as above

k_means_t = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

#fit the model
k_means_t.fit(X)

#grab the labels created by unlabelled data
k_means_labels_t = k_means_t.labels_
k_means_labels_t

k_means_cluster_centers_t = k_means_t.cluster_centers_
k_means_cluster_centers_t

# Now let plot the data with centroid for each clusters

# Initialize the plot with the specified dimensions.
fig_1 = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels_t))))

# Create a plot
ax_1 = fig_1.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
#zip function is used to join two tuples eg: a= ('a','b') and b=('c','d') then zip(a,b)=(('a','c'),('b','d'))
for j, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels_t == j)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers_t[j]
    
    # Plots the datapoints with color col.
    ax_1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax_1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='m', markersize=6)
#markedgecolor can be m=magenta,w=white,k=black,none
# Title of the plot
ax_1.set_title('KMeans')

# Remove x-axis ticks
ax_1.set_xticks(())

# Remove y-axis ticks
ax_1.set_yticks(())

# Show the plot
plt.show()


# =============================================================================
# ######CUSTOMER SEGMENTATION USING K MEANS
# =============================================================================
import pandas as pd
cust_df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv")
cust_df.head()

# #Pre-Processing
# As you can see, Address in this dataset is a categorical variable. k-means algorithm 
# isn't directly applicable to categorical variables because Euclidean distance function 
# isn't really meaningful for discrete variables. So, lets drop this feature and run clustering.

df=cust_df.drop('Address',axis=1)
df.head()

# =============================================================================
# # Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place?
#  Normalization is a statistical method that helps mathematical-based algorithms to 
#  interpret features with different magnitudes and distributions equally. We use 
#  StandardScaler() to normalize our dataset.
# =============================================================================


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]#all rows and for column 1 i.e Age till end .values gives an array
X = np.nan_to_num(X)#replace nan to zero or infinity to large numbers
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

# =============================================================================
# Modeling
# In our example (if we didn't have access to the k-means algorithm), it would be the
#                 same as guessing that each customer group would have certain age,
#                 income, education, etc, with multiple tests and experiments. However,
#                 using the K-means clustering we can do all this process much easier.

# Lets apply k-means on our dataset, and take look at cluster labels
# =============================================================================

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#assign labels to each row of dataframe
df["Clus_km"] = labels
df.head(5)

#get centroid value by averaging each value of column group by clus_km
df.groupby('Clus_km').mean()

#plot distribution of customer based on Age and Income
area = np.pi * ( X[:, 1])**2  #area of circle varies with education level
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

########plot 3D chart for more insights about clusters
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))