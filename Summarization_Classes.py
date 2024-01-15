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
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans, AgglomerativeClustering

from Embedding_Classes import EmbeddingResNet

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class SummerizationPCAKmeans(SummarizationParent):
    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "Hierarchical_and_KMeans_clustering"

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
    
    def apply_kmeans(self, data, n_clusters, seed):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        self.kmeans.fit(data)
        return self.kmeans.labels_

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
