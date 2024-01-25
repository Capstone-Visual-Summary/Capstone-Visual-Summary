import ast
import torch
from typing import Union, List, Dict
from Grand_Parent import GrandParent
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from scipy.spatial import distance
from torch import norm, normal, tensor
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# I don't know why, but otherwise I'm getting an error
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, SummarizationParent]]] = dict()
        self.children_names: set[int] = set()
    
    
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
    
    def generate_output_dict(self, clusters, centers):
        output = {}
        for k, v in clusters.items():
            clusters = k.split(' ')[1]  # Extract the cluster number from the key
            output[k] = {
                'selected': centers['Centroid Cluster ' + str(int(clusters) - 1)],  # Get the corresponding center
                'cluster': v
            }
        return output

    def run(self, **kwargs):
        version = kwargs['summarization_version'] if 'summarization_version' in kwargs else -1        
        return super().run(version, **kwargs)

class DimensionalityReducer:
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
        # Check if the number of dimensions is not larger than the number of data points
        key = list(data.keys())[0]
        max_dimensions = min(len(data[key]), len(data))
        N: int = kwargs['N_dimensions'] if kwargs['N_dimensions'] < max_dimensions else max_dimensions
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()
        pca = PCA(n_components=N)
        pca.fit(numerical_data)
        transformed_data = pca.transform(numerical_data)
        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict
    
class Clusterer:
    def create_dendrogram_plot(self, **kwargs) -> None:
        '''
        Creates a dendrogram plot of all the images using hierarchical clustering.

        Parameters:
            data (dict[int, list[float]]): A dictionary where keys are IDs, and values are lists of transformed data after PCA.
            n_clusters (int): The number of clusters to form in the hierarchical clustering.
        '''
        data: dict[int, list[float]] = kwargs['data']
        n_clusters: int = kwargs['n_clusters']

        # Generate linkage matrix for dendrogram
        linkage_matrix = linkage(list(data.values()), method='ward')

        # Set color_threshold based on the number of clusters
        color_threshold = linkage_matrix[-(n_clusters - 1), 2]  # Adjust for 0-based indexing

        # Create dendrogram
        dendrogram(linkage_matrix, color_threshold=color_threshold)

        # Add a horizontal line to mark the cut based on the number of clusters
        cutting_line_1 = linkage_matrix[-n_clusters + 1, 2]  # Adjust for 0-based indexing
        cutting_line_2 = linkage_matrix[-n_clusters, 2]  # Adjust for 0-based indexing
        middle_cutting_line = (cutting_line_1 + cutting_line_2) / 2

        plt.axhline(y=middle_cutting_line, color='r', linestyle='--', label=f'Cutting Line for {n_clusters} Clusters')

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Image IDs')
        plt.ylabel('Distance')
        plt.legend()
        plt.show()
    
class SummerizationPCAHirarchyDendogram(SummarizationParent, DimensionalityReducer, Clusterer):
    '''
        Creates a dendogram of all the images using hirarical clustering

        Parameters:
            data (dict[int, list[float]]): A dictionary where keys are IDs, and values are lists
            of transformed data after PCA.


        Returns:
            A plot with the image id's on the x-axis
        '''
    def __init__(self) -> None:
        super().__init__()
        self.version: float | str = 5.0
        self.name: str = "PCA_Hirarchy_Dendogram"
        self.hierarchical = None  # Initialize the hierarchical attribute

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2

        # Apply PCA
        pca_data = self.apply_pca(data=data, N_dimensions=N_dimensions)
        print(pca_data)

        # Apply hierarchical clustering with dendrogram creation
        # hierarchical_labels = self.apply_hierarchical(data=pca_data, n_clusters=7)
        self.create_dendrogram_plot(data=pca_data, n_clusters=7)

if __name__ == "__main__":
    
    test_data = pd.read_csv('Embedding Files\data_for_time_comparison.csv', delimiter=',')
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index('image_id')['tensor'].to_dict().items()}
        
    summarization = SummarizationParent()
    output = summarization.run(
        summarization_version= 5.0,
        data=data,
        N_dimensions=10,
        N_clusters=4,
        min_samples=6
    )
    
    print('DONE')