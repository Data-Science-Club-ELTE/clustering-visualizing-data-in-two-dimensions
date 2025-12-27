# Representation Learning

## Overview
Representation/feature learning is a broader term for techniques that allow an ML system to automatically discover representations needed for further tasks such as feature detection and classification. These tasks require inputs that are mathematically easy to represent and convenient to process and do operations with, however real world data is often the opposite (images, videos, sensor data, etc...). These representations are usually continuous vectors.

## Supervised vs. Unsupervised vs. Self-supervised
In supervised feature learning, the input data is labelled, meaning data given to the model includes input-label pairs. This resorts in a representation with high label prediction accuracy. Examples include supervised neural networks, multilayer perceptrons, and dictionary learning.

Unsupervised feature learning uses unlabelled input data, it learns features by analyzing the relationship between data points. Examples include Clustering (K-means), ICA and matrix factorization.

In self-supervised learning features are learned using unlabelled data like unsupervised learning, however input-label pairs are constructed from each data point, enabling learning the structure of the data through supervised methods. Examples include word embeddings and autoencoders.

## Autoencoders

An autoencoder is a type of neural network architecture that is having three core components: the encoder, the decoder, and the latent-space representation. The encoder compresses the input to a lower latent-space representation via weight matrixes biases and a nonlinear activation function and then the decoder reconstructs it.

## Prototype Learning
Prototype learning is a family of methods where a model represents each class, cluster, or concept using prototypes—vectors in a learned feature space that act as representative examples. A model then makes predictions by comparing new samples to these prototypes. These prototypes provide an abstract representation for many natural categories and concepts.

## K-means Clustering Introduction

K-means clustering is prototype learning algorithm used to categorize n (usually vectors of unlabelled data) items into k number of clusters. It is useful for identifying natural groupings of data and structuring raw data. Determining what k should be is important for the segmentation to be meaningful and there are multiple methods for determining the optimal value for k. K-means has a variety of uses due to its simplicity and effectiveness, including data segmentation, image compression and anomaly detection.

## Working of K-means

The algorithm will categorize the inputs into k clusters based on similarity. Measuring of similarity is done with the square of the Euclidean distance of the data vectors. Summarized, the algorithm works as follows:

1. Initialization: Randomly select k number of centerpoints for the clusters, called centroids.
2. Assignment: Each data point is assigned to a cluster based on which centroid is the nearest to it (using Euclidean distance).
3. Update: We recalculate the position of each centroid based on the average of the data points in each cluster.
4. Repeat: We repeat this process until the position of each centroid is unchanged or we reach a pre-defined iteration limit