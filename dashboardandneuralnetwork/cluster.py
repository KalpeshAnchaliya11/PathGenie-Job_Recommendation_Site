from sklearn.cluster import KMeans

def get_cluster(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    return clusters
