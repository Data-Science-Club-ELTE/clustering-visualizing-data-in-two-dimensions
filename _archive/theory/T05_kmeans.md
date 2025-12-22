## K-means Clustering
---

K-means clustering is prototype learning algorithm used to categorize n (usually vectors of unlabelled data) items into k number of clusters. It is useful for identifying natural groupings of data and structuring raw data. Determining what k should be is important for the segmentation to be meaningful and there are multiple methods for determining the optimal value for k. K-means has a variety of uses due to its simplicity and effectiveness, including data segmentation, image compression and anomaly detection.

### Algorithm

The algorithm will categorize the inputs into k clusters based on similarity. Measuring of similarity is done with the square of the Euclidean distance of the data vectors. Summarized, the algorithm works as follows:

1. Initialization: Randomly select k number of centerpoints for the clusters, called centroids.
2. Assignment: Each data point is assigned to a cluster based on which centroid is the nearest to it (using Euclidean distance).
3. Update: We recalculate the position of each centroid based on the average of the data points in each cluster.
4. Repeat: We repeat this process until the position of each centroid is unchanged or we reach a pre-defined iteration limit
