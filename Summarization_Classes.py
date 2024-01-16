from os import close
from typing import Union, List, Dict
from sympy import Number
from Grand_Parent import GrandParent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import distance
from Embedding_Classes import EmbeddingResNet
import pickle
from torch import tensor
import torch
import os


# I don't know why, but otherwise I'm getting an error
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

data = torch.load("summarization_data.pth")

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)
    
class Summerization(SummarizationParent):
    '''
    This class first does pca and uses the pca data to cluster it using
    the following different clustering techniques: KMeans and Hierical Clustering

    '''
    '''
    This class performs Principal Component Analysis (PCA) on input data and applies
    two clustering techniques: KMeans and Hierarchical Clustering.

    Attributes:
        version (float | str): The version of the Summerization class.
        name (str): The name of the clustering technique followed by PCA (e.g., "PCA_Kmeans_Hierical").
    '''
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PCA_Kmeans_Hierical"
        
    def apply_pca(self, data: dict[int, tensor], N: int) -> dict[int, list[float]]:
        '''
        Applies Principal Component Analysis (PCA) on the input data.

        Parameters:
            data (dict[int, tensor]): A dictionary where keys are IDs, and values are tensors.
            N (int): The number of principal components to retain.

        Returns:
            dict[int, list[float]]: A dictionary where keys are IDs, and values are lists
            of transformed data after PCA.
        '''
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()

        pca = PCA(n_components=N)
        pca.fit(numerical_data)
        transformed_data = pca.transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict
    
    
    def apply_kmeans(self, data: Dict[int, List[float]], n_clusters: int, seed: int) -> Dict[str, List[int]]:
        '''
        Applies KMeans clustering on the input data.

        Parameters:
            data (dict[int, list[float]]): A dictionary where keys are IDs, and values are lists
            of transformed data after PCA.
            n_clusters (int): The number of clusters to form.
            seed (int): The random seed for reproducibility.

        Returns:
            Dict[str, List[int]]: A dictionary where keys are cluster names (e.g., "Cluster 1"),
            and values are lists of corresponding IDs assigned to each cluster.
        '''
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        kmeans.fit(list(data.values()))
        id_label_dict = dict(zip(data.keys(), kmeans.labels_))

        label_id_dict = self._create_label_id_dict(id_label_dict)

        return label_id_dict

    def apply_hierarchical(self, data: Dict[int, List[float]], n_clusters: int) -> Dict[str, List[int]]:
        '''
        Applies Hierarchical Clustering on the input data. Same parameters return as KMeans
        '''
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(list(data.values()))

        id_label_dict = dict(zip(data.keys(), labels))
        label_id_dict = self._create_label_id_dict(id_label_dict)

        return label_id_dict

    def _create_label_id_dict(self, id_label_dict: Dict[int, int]) -> Dict[str, List[int]]:
        '''
        Creates a dictionary where keys are cluster names in the format "Cluster X" and 
        values are lists of corresponding IDs assigned to each cluster.

        Parameters:
            id_label_dict (Dict[int, int]): A dictionary where keys are IDs and values are
            corresponding cluster labels.

        Returns:
            Dict[str, List[int]]: A dictionary where keys are cluster names and values are
            lists of IDs assigned to each cluster.
        '''
        label_id_dict = {}

        for id_, cluster in id_label_dict.items():
            cluster_name = f"Cluster {cluster + 1}"
            if cluster_name not in label_id_dict:
                label_id_dict[cluster_name] = []
            label_id_dict[cluster_name].append(id_)

        return label_id_dict

    def run(self, **kwargs):
        N_pca_d = 5
        n_clusters = 3
        pca_data = self.apply_pca(data, N_pca_d)
        kmeans = self.apply_kmeans(pca_data, n_clusters=n_clusters, seed = 42)
        hierarchical = self.apply_hierarchical(pca_data, n_clusters)
        print(kmeans, hierarchical)
        return kmeans
    
if __name__ == "__main__":
    summarization = Summerization()
    summarization.run()