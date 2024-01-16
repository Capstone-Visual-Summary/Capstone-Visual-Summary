from os import close
from typing import Union, List, Dict
from sympy import Number
from Grand_Parent import GrandParent
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import distance
from Embedding_Classes import EmbeddingResNet
import pickle
from torch import tensor
import torch
import os
from scipy.spatial import distance



# I don't know why, but otherwise I'm getting an error
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['summarization_version'] if 'summarization_version' in kwargs else -1
        
        return super().run(version, **kwargs)
    
class Summerization(SummarizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PCA_Kmeans"
        
    def apply_pca(self, **kwargs ) -> dict[int, list[float]]:
        data: dict[int, tensor] = kwargs['data']
        N: int = kwargs['N_dimensions']
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()

        pca = PCA(n_components=N)
        pca.fit(numerical_data)
        transformed_data = pca.transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict
    
    def apply_kmeans(self, **kwargs) -> List[int]:
        data: dict[int, list[float]] = kwargs['data']
        n_clusters: int = kwargs['n_clusters']
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
        
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        self.kmeans.fit(list(data.values()))
        id_label_dict = dict(zip(data.keys(), self.kmeans.labels_))

        label_id_dict = {}

        for id_, cluster in id_label_dict.items():
            cluster_name = f"Cluster {cluster}"
            if cluster_name not in label_id_dict:
                label_id_dict[cluster_name] = []
            label_id_dict[cluster_name].append(id_)

        return label_id_dict
    
    def get_cluster_centers(self, **kwargs) -> dict[int, str]:
        data: dict[int, list[float]] = kwargs['data']
        center_images = {}
        for i in range(len(self.kmeans.cluster_centers_)):
            center = self.kmeans.cluster_centers_[i]
            min_distance = float('inf')
            center_image_id = None
            for image_id, image_data in data.items():
                dist = distance.euclidean(center, image_data)
                if dist < min_distance:
                    min_distance = dist
                    center_image_id = image_id
            center_images[f'Centroid Cluster {i}'] = center_image_id
        return center_images

    def run(self, **kwargs):
        data = kwargs['data']
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        n_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        kmeans = self.apply_kmeans(data=pca_data, n_clusters=n_clusters, seed=42)
        centers = self.get_cluster_centers(data=pca_data)
        return kmeans, centers
   
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
        self.version: float | str = '1.1WIP'
        self.name: str = "PCA_Kmeans_Hierical"
        
    def apply_pca(self, **kwargs ) -> dict[int, list[float]]:
        '''
        Applies Principal Component Analysis (PCA) on the input data.

        Parameters:
            data (dict[int, tensor]): A dictionary where keys are IDs, and values are tensors.
            N (int): The number of principal components to retain.

        Returns:
            dict[int, list[float]]: A dictionary where keys are IDs, and values are lists
            of transformed data after PCA.
        '''
        data: dict[int, tensor] = kwargs['data']
        N: int = kwargs['N_dimensions']
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()

        pca = PCA(n_components=N)
        pca.fit(numerical_data)
        transformed_data = pca.transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict
    
    
    def apply_kmeans(self, **kwargs) -> Dict[str, List[int]]:
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
        data: dict[int, list[float]] = kwargs['data']
        n_clusters: int = kwargs['n_clusters']
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
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
    
        id_label_dict = dict(zip(data.keys(), self.kmeans.labels_))

        label_id_dict = {}

        for id_, cluster in id_label_dict.items():
            cluster_name = f"Cluster {cluster + 1}"
            if cluster_name not in label_id_dict:
                label_id_dict[cluster_name] = []
            label_id_dict[cluster_name].append(id_)

        return label_id_dict
    
    def get_cluster_centers(self, **kwargs) -> dict[int, str]:
        data: dict[int, list[float]] = kwargs['data']
        center_images = {}
        for i in range(len(self.kmeans.cluster_centers_)):
            center = self.kmeans.cluster_centers_[i]
            min_distance = float('inf')
            center_image_id = None
            for image_id, image_data in data.items():
                dist = distance.euclidean(center, image_data)
                if dist < min_distance:
                    min_distance = dist
                    center_image_id = image_id
            center_images[f'Centroid Cluster {i}'] = center_image_id
        return center_images

    def run(self, **kwargs):
        data = kwargs['data']
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        n_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        kmeans = self.apply_kmeans(data=pca_data, n_clusters=n_clusters, seed=42)
        hierarchical = self.apply_hierarchical(pca_data, n_clusters)
        centers = self.get_cluster_centers(data=pca_data)
        return kmeans, centers


if __name__ == "__main__":
    data = torch.load("summarization_data.pth")
    summarization = SummarizationParent()
    kmeans, centers = summarization.run(
        summarization_version=1.0,
        data=data,
        N_dimensions=2,
        N_clusters=4
    )
    
    #Pretty printing the output
    for key in sorted(kmeans.keys()):
        print(f"{key}: {kmeans[key]}")
    for key in sorted(centers.keys()):
        print(f"{key}: {centers[key]}")

#Example Output
#{'Cluster 0': ['53568', '53570', '53572', '53574'], 'Cluster 1': ['53569', '53573'], 'Cluster 2': ['53571'], 'Cluster 3': ['53575']}
#{'Centroid Cluster 0': '53570', 'Centroid Cluster 1': '53569', 'Centroid Cluster 2': '53571', 'Centroid Cluster 3': '53575'}