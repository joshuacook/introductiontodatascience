from scipy.spatial import distance
from pandas import DataFrame
from sklearn.metrics import euclidean_distances
import numpy as np

def BIC(KMeans_model, X):
    if type(X) == DataFrame: X = X.values
    clusters = np.array([X[KMeans_model.labels_ == i] for i in range(KMeans_model.n_clusters)])
    centroids = KMeans_model.cluster_centers_
    num_points = sum(len(cluster) for cluster in clusters)
    num_dims = clusters[0][0].shape[0]
    num_clusters = len(clusters)
    
    likelihood = _calculate_likelihood(num_points, num_dims,
                                           clusters, centroids)
    
    complexity = _calculate_complexity(num_clusters, num_points, num_dims)

    return complexity - likelihood

def _calculate_complexity(num_clusters, num_points, num_dims):
    num_params = _free_params(num_clusters, num_dims)
    return num_params * np.log(num_points)

def _calculate_likelihood(num_points, num_dims, clusters, centroids):
    ll = 0
    for cluster in clusters:
        fRn = len(cluster)
        t1 = fRn * np.log(fRn)
        t2 = fRn * np.log(num_points)
        variance = max(
            _cluster_variance(num_points, clusters, centroids),
            np.nextafter(0, 1))
        t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)
        t4 = num_dims * (fRn - 1.0) / 2.0
        ll += t1 - t2 - t3 - t4
    return 2*ll

def _free_params(num_clusters, num_dims):
    return num_clusters * (num_dims + 1)

def _cluster_variance(num_points, clusters, centroids):
    s = 0
    num_dims = clusters[0][0].shape[0]
    denom = float(num_points - len(centroids)) * num_dims
    for cluster, centroid in zip(clusters, centroids):
        distances = euclidean_distances(cluster, centroid.reshape(-1,1).T)
        s += (distances*distances).sum()
    return s / denom