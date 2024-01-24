import ast
import os
import time
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from sympy import plot
import torch
from scipy.spatial import distance
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import tensor
from tqdm import tqdm
from umap import UMAP
import matplotlib.pyplot as plt

from Grand_Parent import GrandParent



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
    
    def apply_TSNE(self, **kwargs) -> dict[int, list[float]]:
        data: dict[int, tensor] = kwargs['data']
        N: int = min(kwargs['N_dimensions'], 3)
        perplexity: int = kwargs['perplexity'] if 'perplexity' in kwargs else 30
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()
        # Adjust perprexity if there are to few samples
        perplexity = perplexity if len(numerical_data) > perplexity else len(numerical_data) - 1

        tsne = TSNE(n_components=N, random_state=42, perplexity=perplexity)
        transformed_data = tsne.fit_transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict
    
    def apply_UMAP(self, **kwargs) -> Dict[str, List[float]]:
        data: dict[int, tensor] = kwargs['data']
        N: int = kwargs['N_dimensions']
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
        # Adjust n_neighbors and n_components if there are to few samples
        n_neighbors = 15 if len(data) > 15 else len(data) - 1
        N = N if N < n_neighbors else n_neighbors - 1
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()
        
        umap_reducer = UMAP(n_components=N, random_state=seed, n_jobs=1, n_neighbors=n_neighbors)
        transformed_data = umap_reducer.fit_transform(numerical_data)
        
        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
        
        return result_dict

class Clusterer:
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
        n_clusters: int = kwargs['N_clusters']
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        kmeans.fit(list(data.values()))
        self.kmeans = kmeans  # Set the kmeans attribute
        id_label_dict = dict(zip(data.keys(), kmeans.labels_))

        label_id_dict = self._create_label_id_dict(id_label_dict)

        return label_id_dict
    
    def apply_hierarchical(self, **kwargs) -> Dict[str, List[int]]:
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
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.hierarchical.fit_predict(list(data.values()))
        # self.hierarchical = hierarchical  # Set the kmeans attribute
        id_label_dict = dict(zip(data.keys(), self.hierarchical.labels_))
        label_id_dict = self._create_label_id_dict(id_label_dict)

        return label_id_dict
    
    def apply_density(self, **kwargs) -> Dict[str, List[int]]:
        '''
        Applies Density Clustering on the input data.

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
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5
        
        self.density = OPTICS(min_samples=min_samples)
        self.density.fit_predict(list(data.values()))

        id_label_dict = dict(zip(data.keys(), self.density.labels_))
        label_id_dict = self._create_label_id_dict(id_label_dict)

        return label_id_dict

class ClusterFinder:
    def find_clusters(self, **kwargs) -> int:
        data = kwargs['data']
        
        max_cluster = 0
        samples = []
        clusters = []
        for i in tqdm(range(2, 50), desc='Finding Clusters'):
            density = self.apply_density(data=data, min_samples=i)
            samples.append(i)
            clusters.append(len(density))
            if len(density) == 1:
                break
            if len(density) > max_cluster and 2 < len(density) < 8:
                max_cluster = len(density)
                print(f'found {max_cluster} clusters at {i} min_samples')
        
        # Plotting
        # plt.scatter(samples, clusters)
        # plt.xlabel('Samples')
        # plt.ylabel('Clusters')
        # plt.title('Relationship between min_samples and number of clusters')
        # plt.show() 
        
        return max_cluster

class CentreFinder:
    def get_kmeans_centre(self, **kwargs) -> dict[int, str]:
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
    
    def get_hierarchical_centre(self, **kwargs) -> dict[int, str]:
        '''
        Calculates representative cluster centers for each cluster in the hierarchical clustering results.

        Parameters:
            **kwargs: Additional keyword arguments.
                - data (dict[int, list[float]]): A dictionary where keys are image IDs, and values are lists
                representing the transformed data after PCA.
        
        Returns:
            dict[int, str]: A dictionary where keys are cluster names (e.g., "Mean Center Cluster 1"),
            and values are image IDs representing the closest data point to the mean center of each cluster.
        '''
        data: dict[int, list[float]] = kwargs['data']
        hierarchical_labels = self.hierarchical.labels_

        cluster_centers = {}
        for cluster_id in np.unique(hierarchical_labels):
            cluster_points = [data[image_id] for image_id, label in zip(data.keys(), hierarchical_labels) if label == cluster_id]
            mean_center = np.mean(cluster_points, axis=0)
            
            min_distance = float('inf')
            center_image_id = None
            for image_id, image_data in data.items():
                dist = distance.euclidean(mean_center, image_data)
                if dist < min_distance:
                    min_distance = dist
                    center_image_id = image_id

            cluster_centers[f'Centroid Cluster {cluster_id}'] = center_image_id

        return cluster_centers
    
    def get_density_centre(self, **kwargs) -> dict[int, str]:
        '''
        Calculates representative cluster centers for each cluster in the density clustering results.

        Parameters:
            **kwargs: Additional keyword arguments.
                - data (dict[int, list[float]]): A dictionary where keys are image IDs, and values are lists
                representing the transformed data after PCA.
        
        Returns:
            dict[int, str]: A dictionary where keys are cluster names (e.g., "Mean Center Cluster 1"),
            and values are image IDs representing the closest data point to the mean center of each cluster.
        '''
        data: dict[int, list[float]] = kwargs['data']
        density_labels = self.density.labels_

        cluster_centers = {}
        for cluster_id in np.unique(density_labels):
            cluster_points = [data[image_id] for image_id, label in zip(data.keys(), density_labels) if label == cluster_id]
            mean_center = np.mean(cluster_points, axis=0)
            
            min_distance = float('inf')
            center_image_id = None
            for image_id, image_data in data.items():
                dist = distance.euclidean(mean_center, image_data)
                if dist < min_distance:
                    min_distance = dist
                    center_image_id = image_id

            cluster_centers[f'Centroid Cluster {cluster_id}'] = center_image_id

        return cluster_centers

########################################## 1.0 ##########################################################

class SummerizationPCAKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PCA_Kmeans"
    
    def run(self, **kwargs):
        data = kwargs['data']
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        n_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        kmeans = self.apply_kmeans(data=pca_data, N_clusters=n_clusters, seed=42)
        centers = self.get_kmeans_centre(data=pca_data)
        return self.generate_output_dict(kmeans, centers)
    
########################################## 2.0 ##########################################################  

class SummerizationPCAHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    '''
    This class performs Principal Component Analysis (PCA) on input data and applies
     clustering Hierarchical Clustering.

    Attributes:
        version (float | str): The version of the Summerization class.
        name (str): The name of the clustering technique followed by PCA (e.g., "PCA_Kmeans_Hierical").
    '''
    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "PCA_Hierical"   

    def run(self, **kwargs):
        data = kwargs['data']
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        N_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        hierarchical = self.apply_hierarchical(data=pca_data, n_clusters=N_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=pca_data)
        return self.generate_output_dict(hierarchical, centers)
    
########################################## 2.1 ########################################################## 

class SummerizationPCADensity(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    '''
    This class performs Principal Component Analysis (PCA) on input data and applies
     clustering Density Clustering.

    Attributes:
        version (float | str): The version of the Summerization class.
        name (str): The name of the clustering technique followed by PCA (e.g., "PCA_Kmeans_Hierical").
    '''
    def __init__(self) -> None:
        self.version: float | str = 2.1
        self.name: str = "PCA_Density"

    def run(self, **kwargs):
        data = kwargs['data']
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5
        
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        density = self.apply_density(data=pca_data, min_samples=min_samples)
        centers = self.get_density_centre(data=pca_data)
        return self.generate_output_dict(density, centers)

########################################## 2.2 ##########################################################

class SummerizationTSNEKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.2
        self.name: str = "TSNE_Kmeans"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        N_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        TSNE_data = self.apply_TSNE(data=data, N_dimensions=N_dimensions)
        kmeans = self.apply_kmeans(data=TSNE_data, N_clusters=N_clusters, seed=42)
        centers = self.get_kmeans_centre(data=TSNE_data)
        return self.generate_output_dict(kmeans, centers)

########################################## 2.3 ##########################################################

class SummerizationTSNEHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.3
        self.name: str = "TSNE_Hierical"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        N_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        TSNE_data = self.apply_TSNE(data=data, N_dimensions=N_dimensions)
        hierarchical = self.apply_hierarchical(data=TSNE_data, n_clusters=N_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=TSNE_data)
        return self.generate_output_dict(hierarchical, centers)

########################################## 2.4 ##########################################################

class SummerizationTSNEDensity(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.4
        self.name: str = "TSNE_Density"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5
        
        TSNE_data = self.apply_TSNE(data=data, N_dimensions=N_dimensions)
        density = self.apply_density(data=TSNE_data, min_samples=min_samples)
        centers = self.get_density_centre(data=TSNE_data)
        return self.generate_output_dict(density, centers)

########################################## 2.5 ##########################################################

class SummerizationUMAPKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.5
        self.name: str = "UMAP_Kmeans"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        N_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        UMAP_data = self.apply_UMAP(data=data, N_dimensions=N_dimensions, seed=42)
        kmeans = self.apply_kmeans(data=UMAP_data, N_clusters=N_clusters, seed=42)
        centers = self.get_kmeans_centre(data=UMAP_data)
        return self.generate_output_dict(kmeans, centers)

########################################## 2.6 ##########################################################

class SummerizationUMAPHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.6
        self.name: str = "UMAP_Hierical"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        N_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        UMAP_data = self.apply_UMAP(data=data, N_dimensions=N_dimensions, seed=42)
        hierarchical = self.apply_hierarchical(data=UMAP_data, n_clusters=N_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=UMAP_data)
        return self.generate_output_dict(hierarchical, centers)

########################################## 2.7 ##########################################################

class SummerizationUMAPDensity(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.7
        self.name: str = "UMAP_Density"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5
        
        UMAP_data = self.apply_UMAP(data=data, N_dimensions=N_dimensions, seed=42)
        density = self.apply_density(data=UMAP_data, min_samples=min_samples)
        centers = self.get_density_centre(data=UMAP_data)
        return self.generate_output_dict(density, centers)

########################################## 3.0 ##########################################################

class SummerizationPCADensityKmeans(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 3.0
        self.name: str = "PCA_Density_Kmeans"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        
        pca_data = self.apply_pca(data=data, N_dimensions=N_dimensions)
        N_clusters = self.find_clusters(data=pca_data)
        if N_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            N_clusters = 5
        kmeans = self.apply_kmeans(data=pca_data, N_clusters=N_clusters, seed=42)
        centers = self.get_kmeans_centre(data=pca_data)
        return self.generate_output_dict(kmeans, centers)

########################################## 3.1 ##########################################################

class SummerizationPCADensityDensity(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 3.1
        self.name: str = "PCA_Density_Density"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        
        pca_data = self.apply_pca(data=data, N_dimensions=N_dimensions)
        N_clusters = self.find_clusters(data=pca_data)
        if N_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            N_clusters = 5
        density = self.apply_density(data=pca_data, min_samples=3, seed=42)
        centers = self.get_density_centre(data=pca_data)
        return self.generate_output_dict(density, centers)

########################################## 3.2 ##########################################################

class SummerizationPCADensityHirarchy(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 3.2
        self.name: str = "PCA_Density_Hirarchy"

    def run(self, **kwargs):
        data = kwargs['data']
        N_dimensions = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        
        pca_data = self.apply_pca(data=data, N_dimensions=N_dimensions)
        N_clusters = self.find_clusters(data=pca_data)
        if N_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            N_clusters = 5
        hierarchical = self.apply_hierarchical(data=pca_data, n_clusters=N_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=pca_data)
        return self.generate_output_dict(hierarchical, centers)

#########################################################################################################
def time_version(**kwargs) -> float:
    '''
    Returns the time taken to run the summarization algorithm with the given version.
    '''
    
    test_data = pd.read_csv('Embedding Files\Embeddings_v1_0_2392_4783.csv', delimiter=';')    
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index('image_id')['tensor'].to_dict().items()}
    
    summarization = SummarizationParent()
    repeats = kwargs['repeats']
    data = kwargs['data']
    version = kwargs['summarization_version']
    
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        summarization.run(
            summarization_version=version,
            data=data,
            N_dimensions=10,
            N_clusters=5
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    standard_error = np.std(times) / np.sqrt(repeats)
    return np.mean(times), standard_error

'''
def accuracy_version(**kwargs) -> float:
    
    test_data = pd.read_csv('Embedding Files\Embeddings_v1_0_2392_4783.csv', delimiter=';')
    repeats = 5
    
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index('image_id')['tensor'].to_dict().items()}
    
    summarization = SummarizationParent()
    data = kwargs['data']
    version = kwargs['summarization_version']
    
    accuracies = []
    for _ in range(repeats):
        # some way to measure accuracy
        accuracies.append(sum([len(v['cluster'].intersection(v['selected'])) for v in output.values()]) / len(data))
    
    standard_error = np.std(accuracies) / np.sqrt(repeats)
    return np.mean(accuracies), standard_error
'''

def compare_versions() -> pd.DataFrame:
    '''	
    returns a dataframe with the time taken to run each version of the summarization algorithm
    '''	
    repeats = 5
    versions = {
        1.0 : 'PCA  Kmeans',
        2.0 : 'PCA  Hierical',
        2.1 : 'PCA  Density',
        2.2 : 'TSNE Kmeans',
        2.3 : 'TSNE Hierical',
        2.4 : 'TSNE Density',
        2.5 : 'UMAP Kmeans',
        2.6 : 'UMAP Hierical',
        2.7 : 'UMAP Density',
    }
    
    times = {}
    for version in tqdm(versions.keys(), desc='Comparing Times'):
        mean, standard_error = time_version(data=data, summarization_version=version, repeats=repeats)
        time = f"{mean:.3f} ± {standard_error:.3f}"
        times[versions[version]] = time
    
    '''
    accuracy = {}
        for version in tqdm(versions.keys(), desc='Comparing Accuracy'):
        mean, standard_error = accuracy_version(data=data, summarization_version=version)
        acc = f"{mean:.3f} ± {standard_error:.3f}"
        accuracy[versions[version]] = acc
    '''
    
    times_df = pd.DataFrame.from_dict(times, orient='index', columns=['Mean Time (s)'])
    # accuracy df = pd.DataFrame.from_dict(accuracy, orient='index', columns=['Mean Accuracy (%)'])
    # merged_df = pd.merge(times_df, accuracy_df, left_index=True, right_index=True)
    # return merged_df
    
    return times_df

def pretty_print(output):
    for key in sorted(output.keys(), key=lambda x:int(x.split(' ')[1])):
        print(f"{key}: {output[key]}")

if __name__ == "__main__":
    
    test_data = pd.read_csv('Embedding Files\data_for_time_comparison.csv', delimiter=',')
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index('image_id')['tensor'].to_dict().items()}

    print(compare_versions())
        
    # summarization = SummarizationParent()
    # output = summarization.run(
    #     summarization_version= 3.0,
    #     data=data,
    #     N_dimensions=10,
    #     N_clusters=4,
    #     min_samples=6
    # )
    
    # # print(output)
    # pretty_print(output)
    
    # print('DONE')
    
#Example Output
#{
# 'Cluster 4': {'selected': 4130, 'cluster': {4096, 4362, 4366, 4370, 4372, 4374, 4378, 4380, 4126, 4382, 4384, 4130, 4387, 4388, 4389, 4134, 4392, 4393, 4138, 4396, 4142, 4399, 4400, 4401, 4404, 4405, 4408, 4411, 4412, 4413, 4419, 4338, 4342, 4350}},
# 'Cluster 1': {'selected': 4120, 'cluster': {4097, 4098, 4099, 4353, 4358, 4359, 4367, 4120, 4121, 4122, 4123, 4124, 4125, 4127, 4128, 4129, 4383, 4131, 4132, 4133, 4136, 4137, 4139, 4140, 4141, 4143, 4416}},
# 'Cluster 3': {'selected': 4386, 'cluster': {4385, 4386, 4390, 4391, 4135, 4394, 4395, 4397, 4398, 4402, 4403, 4406, 4407, 4409, 4410, 4414, 4415, 4417, 4418}}
#}