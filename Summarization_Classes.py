from os import close
from typing import Union

from sympy import Number
from Grand_Parent import GrandParent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
<<<<<<< HEAD
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans, AgglomerativeClustering

=======
from Embedding_Classes import EmbeddingResNet
>>>>>>> 7b88f6a74bbc587a8ef7214218a6dd1c6581240b

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


<<<<<<< HEAD
# K = 2 # number of clusters
# N = 2 # number of components
# SEED = 42 #random seed


# class SummerizationPCAKmeans(SummarizationParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "PCA_Kmeans"

#     def load_dummy_data(self):
#         from sklearn.datasets import load_breast_cancer
#         data = load_breast_cancer()
#         return data

#     def create_dataframe(self, data):
#         df1 = pd.DataFrame(data['data'], columns=data['feature_names'])
#         return df1

#     def scale_data(self, df1):
#         scaling = StandardScaler()
#         scaling.fit(df1)
#         Scaled_data = scaling.transform(df1)
#         return Scaled_data

#     def apply_pca(self, Scaled_data):
#         pca = PCA(n_components=N)
#         pca.fit(Scaled_data)
#         x = pca.transform(Scaled_data)
#         return x
        
#     def visualize_data(self, x, data):
#         plt.figure(figsize=(10,10))
#         plt.scatter(x[data['target'] == 0, 0], x[data['target'] == 0, 1], label=data['target_names'][1])
#         plt.scatter(x[data['target'] == 1, 0], x[data['target'] == 1, 1], label=data['target_names'][0])
#         plt.xlabel('pc1')
#         plt.ylabel('pc2')
#         plt.legend(title='Classes')
#         plt.show()
    
#     def apply_kmeans(self, data, n_clusters):
#         self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
#         self.kmeans.fit(data)
#         return self.kmeans.labels_

#     def visualize_clusters(self, x, labels):
#         plt.figure(figsize=(10,10))
#         unique_labels = np.unique(labels)
#         for label in unique_labels:
#             plt.scatter(x[labels == label, 0], x[labels == label, 1], label=f'Cluster {label}', cmap='viridis')
#         plt.xlabel('pc1')
#         plt.ylabel('pc2')
#         plt.legend()
#         plt.show()
    
#     def get_closest_points_to_centroids(self, pca_data, cluster_labels):
#         centroids = self.kmeans.cluster_centers_

#         closest_points = []
#         for i, c in enumerate(centroids):
#             # Get the points in this cluster
#             points_in_cluster = pca_data[cluster_labels == i]
            
#             # Calculate the distance between the centroid and the points in the cluster
#             distances = distance.cdist([c], points_in_cluster)[0]
            
#             # Get the index of the smallest distance
#             closest_point_idx = np.argmin(distances)
            
#             # Get the closest point and add it to the list
#             closest_points.append(points_in_cluster[closest_point_idx])

#         return closest_points
        
#     def run(self, **kwargs):
#         visualize = kwargs['visualize']
#         # df1 = kwargs['data']
#         data = self.load_dummy_data()
#         # print(f'label names = {data["target_names"]}')
#         # print(f'feature names = {data["feature_names"]}')
#         df1 = self.create_dataframe(data)
#         # print(f"data before PCA\n{df1.head()}")
#         Scaled_data = self.scale_data(df1)
#         pca_data = self.apply_pca(Scaled_data)
#         # print(f"data after PCA\n{pd.DataFrame(pca_data).head()}")
#         cluster_labels = self.apply_kmeans(pca_data, K)
#         if visualize:
#             # self.visualize_data(pca_data, df1)
#             self.visualize_clusters(pca_data, cluster_labels)
#         return pca_data, cluster_labels
       
# if __name__ == "__main__":
#     pca = SummerizationPCAKmeans()
#     x, y = pca.run(visualize=True)
#     print(f'{x.shape} {y.shape}')
#     df = pd.DataFrame(x, columns=['pc1', 'pc2'])
#     df['Cluster'] = y
#     print(df)
#     cluster_counts = df['Cluster'].value_counts()
#     print(cluster_counts)
#     closest_points = pca.get_closest_points_to_centroids(x, y)
#     print(closest_points)


SEED = 42
N = 2  # Number of PCA components
K = 2  # Number of clusters for k-means

class SummerizationHiericalClustering:
=======
class SummerizationPCAKmeans(SummarizationParent):
>>>>>>> 7b88f6a74bbc587a8ef7214218a6dd1c6581240b
    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "Hierarchical_and_KMeans_clustering"

<<<<<<< HEAD
    def load_dummy_data(self):
        data = load_breast_cancer()
        return data

    def create_dataframe(self, data):
        df1 = pd.DataFrame(data['data'], columns=data['feature_names'])
        return df1

=======
>>>>>>> 7b88f6a74bbc587a8ef7214218a6dd1c6581240b
    def scale_data(self, df1):
        scaling = StandardScaler()
        scaling.fit(df1)
        Scaled_data = scaling.transform(df1)
        return Scaled_data

    def apply_pca(self, Scaled_data, N):
        pca = PCA(n_components=N)
        pca.fit(Scaled_data)
        x = pca.transform(Scaled_data)
        return x
<<<<<<< HEAD

    def visualize_data(self, x, data):
        plt.figure(figsize=(10, 10))
        plt.scatter(x[data['target'] == 0, 0], x[data['target'] == 0, 1], label=data['target_names'][1])
        plt.scatter(x[data['target'] == 1, 0], x[data['target'] == 1, 1], label=data['target_names'][0])
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend(title='Classes')
        plt.show()
=======
    
    def apply_kmeans(self, data, n_clusters, seed):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        self.kmeans.fit(data)
        return self.kmeans.labels_
>>>>>>> 7b88f6a74bbc587a8ef7214218a6dd1c6581240b

    def apply_kmeans(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
        kmeans.fit(data)
        return kmeans.labels_

    def apply_hierarchical(self, data, n_clusters):
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(data)
        return labels

    def visualize_clusters(self, x, labels, title):
        plt.figure(figsize=(10, 10))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(x[labels == label, 0], x[labels == label, 1], label=f'Cluster {label}', cmap='viridis')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend(title=f'Clusters - {title}')
        plt.title(title)
        plt.show()

    def get_closest_points_to_centroids(self, pca_data, cluster_labels):
        centroids = np.array([np.mean(pca_data[cluster_labels == i], axis=0) for i in np.unique(cluster_labels)])

        closest_points = []
        for i, c in enumerate(centroids):
            points_in_cluster = pca_data[cluster_labels == i]
            distances = distance.cdist([c], points_in_cluster)[0]
            closest_point_idx = np.argmin(distances)
            closest_points.append(points_in_cluster[closest_point_idx])

        return closest_points

    def run(self, **kwargs):
        data = kwargs['data']
        K = kwargs['K']
        N = kwargs['N']
        visualize = kwargs['visualize']
<<<<<<< HEAD
        data = self.load_dummy_data()
        df1 = self.create_dataframe(data)
        Scaled_data = self.scale_data(df1)
        pca_data = self.apply_pca(Scaled_data)
        kmeans_labels = self.apply_kmeans(pca_data, K)
        hierarchical_labels = self.apply_hierarchical(pca_data, K)
        
        if visualize:
            self.visualize_clusters(pca_data, kmeans_labels, title='K-Means')
            self.visualize_clusters(pca_data, hierarchical_labels, title='Hierarchical')

        return pca_data, kmeans_labels, hierarchical_labels


if __name__ == "__main__":
    pca = SummerizationHiericalClustering()
    x, y_kmeans, y_hierarchical = pca.run(visualize=True)
    print(f'{x.shape} {y_kmeans.shape} {y_hierarchical.shape}')  # Corrected variable name 'y' to 'y_kmeans'
    df = pd.DataFrame(x, columns=['pc1', 'pc2'])
    df['Cluster_KMeans'] = y_kmeans  # Updated column name to reflect KMeans clusters
    df['Cluster_Hierarchical'] = y_hierarchical  # Added column for Hierarchical clusters
    print(df)
    
    cluster_counts_kmeans = df['Cluster_KMeans'].value_counts()
    cluster_counts_hierarchical = df['Cluster_Hierarchical'].value_counts()
    
    print("KMeans Cluster Counts:")
    print(cluster_counts_kmeans)
    
    print("Hierarchical Cluster Counts:")
    print(cluster_counts_hierarchical)
    
    closest_points_kmeans = pca.get_closest_points_to_centroids(x, y_kmeans)
    closest_points_hierarchical = pca.get_closest_points_to_centroids(x, y_hierarchical)
    
    print("Closest Points to KMeans Centroids:")
    print(closest_points_kmeans)
    
    print("Closest Points to Hierarchical Centroids:")
    print(closest_points_hierarchical)


=======
        seed = kwargs['seed']
        data = kwargs['data']
        Scaled_data = self.scale_data(data)
        # print(pd.DataFrame(Scaled_data))
        pca_data = self.apply_pca(Scaled_data, N)
        cluster_labels = self.apply_kmeans(pca_data, K, seed)
        closest_points = self.get_closest_points_to_centroids(pca_data, cluster_labels)
        if visualize:
            self.visualize_clusters(pca_data, cluster_labels)
        df = pd.DataFrame(pca_data, columns=['pc1', 'pc2'])
        df['Cluster'] = cluster_labels
        return df, closest_points
       
if __name__ == "__main__":
    pca = SummerizationPCAKmeans()
    data = pd.read_pickle('embeddings_test.pkl')
    print(data)
    df, z = pca.run(data=data, K=3, N=5, visualize=True, seed=42)
    print(df)
    print(z)
>>>>>>> 7b88f6a74bbc587a8ef7214218a6dd1c6581240b
