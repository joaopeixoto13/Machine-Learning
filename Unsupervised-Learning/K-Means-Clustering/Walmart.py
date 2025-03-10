# PROBLEM:
# Walmart wants to open a chain of stores across Florida and wants to find out optimal store
# locations to maximize revenue

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()    # For plot styling
import numpy as np

from sklearn.datasets._samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()


from sklearn.cluster import KMeans

# Create KMeans with 4 clusters
kmeans = KMeans(n_clusters=4)

# Train the model passing the data
kmeans.fit(X)

# Store into y_means the values predicted (not take account because is our training data)
y_kmeans = kmeans.predict(X)


# IMPLEMENTATION OF KMEANS CLUSTERING

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()