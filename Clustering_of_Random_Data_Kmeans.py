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

