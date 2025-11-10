import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=100, tol=0.0001):
        """
        KMeans Clustering initialization
        Args:
            k (int): The number of clusters
            max_iters (int): Maximum iterations
            tol (float): tolerance
        """
        self.k = k 
        self.max_iters = max_iters 
        self.tol = tol # tolerance that determines the algorithm's convergence condition, if the change of center point of the cluster is smaller than this, convergence.
        self.centroids = None  # Variable that stores the center points
        self.labels = None     # Variable that stores the cluster label of the data point

    def _initialize_centroids(self, X):
        """
        Cluster Centroids Initialization
        One of the easiest way is to select random k data points
        Args:
            X (np.ndarray): Input data (n_samples, n_features)
        Returns:
            np.ndarray: k initialized centroids (k, n_features)
        """
        n_samples = X.shape[0] # Total sample number of X, (The number of rows)

        random_indices = np.random.choice(np.arange(n_samples), size = self.k, replace = False)

        initial_centroids = X[random_indices]

        return initial_centroids

    def _assign_clusters(self, X):
        """
        Assign data points to the closest centroid (Assignment Step).
        Find the closest centroid using Euclid distance
        Args:
            X (np.ndarray): Input data(n_samples, n_features)
        Returns:
            np.ndarray: Cluster lables assigned to the data points
        """
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)

        for i in range(n_samples):
            point = X[i]
            distances_to_centroids = np.empty(self.k)

            for centroid_idx in range(self.k):
                distance = np.linalg.norm(point - self.centroids[centroid_idx])
                distances_to_centroids[centroid_idx] = distance

            cluster_label = np.argmin(distances_to_centroids)
            labels[i] = cluster_label

        return labels

    def _update_centroids(self, X, labels):
        """
        Calculate new centroids for each clusters (Update Step).
        New centroids are a mean value of all of the data points in the cluster.
        Args:
            X (np.ndarray): Input data (n_samples, n_features)
            labels (np.ndarray): Cluster lable of the data point (n_samples,)
        Returns:
            np.ndarray: k updated centroids (k, n_features)
        """
        new_centroids = np.copy(self.centroids) # Copy for handling empty clusters, can also initialize with zeros

        for j in range(self.k):
            points_in_cluster = X[labels == j]

            if points_in_cluster.shape[0] > 0: # If points_in_cluster is not empty
                new_centroids[j] = np.mean(points_in_cluster, axis = 0) # Calculate mean for each column
        
        return new_centroids

    def fit(self, X):
        """
        Fit K-means algorithm to the data X.
        Args:
            X (np.ndarray): Input data (n_samples, n_features)
        """
        # 1. Initialize centroids
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iters):
            old_centroids = np.copy(self.centroids)

            # 2. Assign clusters (Assignment Step)
            self.labels = self._assign_clusters(X)

            # 3. Update centroids (Update Step)
            self.centroids = self._update_centroids(X, self.labels)

            # 4. Check convergence condition
            difference = np.sum((self.centroids - old_centroids) ** 2)

            if difference < self.tol:
                print(f"Converged at iteration {i + 1} with difference = {difference:.6f}")
                break
            
            if i == self.max_iters - 1:
                print(f"Reached maximum iterations ({self.max_iters}). Last difference: {difference:.6f}")

        print("K-Means fitting finished.")

    def predict(self, X):
        """
        Predict cluster labels for new data points using the trained centroids.
        Args:
            X (np.ndarray): New input data (n_samples, n_features)
        Returns:
            np.ndarray: Cluster labels assigned to each data point (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype = int)

        for i in range(n_samples):
            point = X[i]
            distances_to_centroids = np.empty(self.k)

            for centroid_idx in range(self.k):
                distance = np.linalg.norm(point - self.centroids[centroid_idx])
                distances_to_centroids[centroid_idx] = distance

            labels[i] = np.argmin(distances_to_centroids)

        return labels

    def plot_clusters(self, X, title="K-Means Clustering"):
        """
        Visualize the clustering results.
        Args:
            X (np.ndarray): Original data
            title (str): Graph title
        """
        if self.labels is None or self.centroids is None:
            print("Model not fitted yet or no labels assigned. Call fit() first.")
            return

        plt.figure(figsize=(10, 7))
        # Using 'tab10' colormap (10 distinct colors)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=self.labels, s=50, cmap='tab10', alpha=0.7)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=100, marker='X', c='black', edgecolor='white', linewidth=1.0, label='Centroids')
        plt.title(title)
        plt.xlabel('Weight (kg)')
        plt.ylabel('Length (cm)')

        plt.grid(True, alpha=0.3)
        plt.show()