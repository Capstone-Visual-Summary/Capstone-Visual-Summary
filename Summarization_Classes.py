from typing import Union
from Parent import GrandParent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


K = 2 # number of clusters
N = 2 # number of components
SEED = 42 #random seed


class SummerizationPCAKmeans(SummarizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PCA"

    def load_dummy_data(self):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data

    def create_dataframe(self, data):
        df1 = pd.DataFrame(data['data'], columns=data['feature_names'])
        return df1

    def scale_data(self, df1):
        scaling = StandardScaler()
        scaling.fit(df1)
        Scaled_data = scaling.transform(df1)
        return Scaled_data

    def apply_pca(self, Scaled_data):
        pca = PCA(n_components=N)
        pca.fit(Scaled_data)
        x = pca.transform(Scaled_data)
        return x
        
    def visualize_data(self, x, data):
        plt.figure(figsize=(10,10))
        plt.scatter(x[data['target'] == 0, 0], x[data['target'] == 0, 1], label=data['target_names'][1])
        plt.scatter(x[data['target'] == 1, 0], x[data['target'] == 1, 1], label=data['target_names'][0])
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend(title='Classes')
        plt.show()
    
    def apply_kmeans(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
        kmeans.fit(data)
        return kmeans.labels_

    def visualize_clusters(self, x, labels):
        plt.figure(figsize=(10,10))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(x[labels == label, 0], x[labels == label, 1], label=f'Cluster {label}', cmap='viridis')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend()
        plt.show()
        
    def run(self, visualize = True):
        data = self.load_dummy_data()
        # print(f'label names = {data["target_names"]}')
        # print(f'feature names = {data["feature_names"]}')
        df1 = self.create_dataframe(data)
        # print(f"data before PCA\n{df1.head()}")
        Scaled_data = self.scale_data(df1)
        pca_data = self.apply_pca(Scaled_data)
        # print(f"data after PCA\n{pd.DataFrame(pca_data).head()}")
        cluster_labels = self.apply_kmeans(pca_data, K)
        if visualize:
            self.visualize_data(pca_data, data)
            self.visualize_clusters(pca_data, cluster_labels)
        return pca_data, cluster_labels
       
pca = SummerizationPCAKmeans()
pca.run(visualize=True)