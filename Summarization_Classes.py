import ast
import time
from typing import Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import tensor
from tqdm import tqdm
from umap import UMAP
import csv
import ast

from Grand_Parent import GrandParent

# Below are the building blocks that make up the different versions of the summarization algorithm


class SummarizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str,
                                      Union[str, SummarizationParent]]] = dict()
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
            # Extract the cluster number from the key
            clusters = k.split(' ')[1]
            output[k] = {
                # Get the corresponding center
                'selected': centers['Centroid Cluster ' + str(int(clusters) - 1)],
                'cluster': v
            }
        return output
    
    def get_summarization_file_name(self, **kwargs):
        database_version = str(kwargs['database_version']).split('.')
        embedder_version = str(kwargs['embedder_version']).split('.')
        summarization_version = str(kwargs['summarization_version']).split('.')

        file_name = f'Summaries/summary_D{database_version[0]}_{database_version[1]}_E{embedder_version[0]}_{embedder_version[1]}_S{summarization_version[0]}_{summarization_version[1]}.csv'

        return file_name
    
    def get_summary_from_file(self, file_name_summary, **kwargs):
        try:
            with open(file_name_summary, mode='r', newline='', encoding='utf-8') as csvfile:
                temp = csv.DictReader(csvfile, delimiter=';')

                for row in temp:
                    if row['neighbourhood_id'] == kwargs['neighbourhood_id']:
                        return ast.literal_eval(row['summary_dict'])

            return False
        except:
            return False

    def save_summary(self, file_name_summary, summary, **kwargs):
        with open(file_name_summary, mode='a+', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['neighbourhood_id', 'summary_dict'], delimiter=';')

            if csvfile.tell() == 0:
                csv_writer.writeheader()

            csv_writer.writerow({'neighbourhood_id': kwargs['neighbourhood_id'], 'summary_dict': summary})

    def run(self, **kwargs):
        version = kwargs['summarization_version'] if 'summarization_version' in kwargs else -1
        return super().run(version, **kwargs)


class Plotter:
    def plot_clusters(self, x, y):
        plt.scatter(x, y)
        plt.xlabel('min_samples')
        plt.ylabel('number of clusters K')
        for i in range(len(x)):
            plt.text(x[i], y[i], f'  k={y[i]}')
        plt.show()

    def create_dendrogram_plot(self, **kwargs) -> None:
        '''
        Creates a dendrogram plot of all the images using hierarchical clustering.

        Parameters:
            data (dict[int, list[float]]): A dictionary where keys are IDs, and values are lists of transformed data after PCA.
            n_clusters (int): The number of clusters to form in the hierarchical clustering.
        '''
        data: dict[int, list[float]] = kwargs['data']
        n_clusters: int = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5

        # Generate linkage matrix for dendrogram
        linkage_matrix = linkage(list(data.values()), method='ward')

        # Set color_threshold based on the number of clusters
        # Adjust for 0-based indexing
        color_threshold = linkage_matrix[-(n_clusters - 1), 2]

        # Create dendrogram
        dendrogram(linkage_matrix, color_threshold=color_threshold)

        # Add a horizontal line to mark the cut based on the number of clusters
        # Adjust for 0-based indexing
        cutting_line_1 = linkage_matrix[-n_clusters + 1, 2]
        # Adjust for 0-based indexing
        cutting_line_2 = linkage_matrix[-n_clusters, 2]
        middle_cutting_line = (cutting_line_1 + cutting_line_2) / 2

        plt.axhline(y=middle_cutting_line, color='r', linestyle='--',
                    label=f'Cutting Line for {n_clusters} Clusters')

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Image IDs')
        plt.ylabel('Distance')
        plt.legend()
        plt.show()


class DimensionalityReducer:
    def apply_pca(self, **kwargs) -> dict[int, list[float]]:
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
        n_dimensions: int = kwargs['n_dimensions'] if kwargs['n_dimensions'] < max_dimensions else max_dimensions
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()
        pca = PCA(n_components=n_dimensions)
        pca.fit(numerical_data)
        transformed_data = pca.transform(numerical_data)
        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist(
        ) for i, id_ in enumerate(data.keys())}

        return result_dict

    def apply_TSNE(self, **kwargs) -> dict[int, list[float]]:
        '''	
        Applies t-distributed Stochastic Neighbor Embedding (t-SNE) on the input data.

        Parameters:
            data (dict[int, tensor]): A dictionary where keys are IDs, and values are tensors.
            n_dimensions (int): The number of dimensions to reduce to.
            perplexity (int): The perplexity value for t-SNE.
            seed (int): The random seed for reproducibility.

        Returns:
            dict[int, list[float]]: A dictionary where keys are IDs, and values are lists

        '''
        data: dict[int, tensor] = kwargs['data']
        n_dimensions: int = min(kwargs['n_dimensions'], 3)
        perplexity: int = kwargs['perplexity'] if 'perplexity' in kwargs else 30
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()
        # Adjust perprexity if there are to few samples
        perplexity = perplexity if len(
            numerical_data) > perplexity else len(numerical_data) - 1

        tsne = TSNE(n_components=n_dimensions,
                    random_state=seed, perplexity=perplexity)
        transformed_data = tsne.fit_transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist(
        ) for i, id_ in enumerate(data.keys())}

        return result_dict

    def apply_UMAP(self, **kwargs) -> Dict[str, List[float]]:
        '''	
        Applies Uniform Manifold Approximation and Projection (UMAP) on the input data.

        Parameters:
            data (dict[int, tensor]): A dictionary where keys are IDs, and values are tensors.
            n_dimensions (int): The number of dimensions to reduce to.
            seed (int): The random seed for reproducibility.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are IDs, and values are lists
        '''
        data: dict[int, tensor] = kwargs['data']
        n_dimensions: int = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        seed: int = kwargs['seed'] if 'seed' in kwargs else 42
        # Adjust n_neighbors and n_components if there are to few samples
        n_neighbors = 15 if len(data) > 15 else len(data) - 1
        n_dimensions = n_dimensions if n_dimensions < n_neighbors else n_neighbors - 1
        # Extract numerical values from tensors
        numerical_data = torch.stack(list(data.values())).numpy()

        umap_reducer = UMAP(n_components=n_dimensions, random_state=seed,
                            n_jobs=1, n_neighbors=n_neighbors)
        transformed_data = umap_reducer.fit_transform(numerical_data)

        # Create a dictionary with IDs as keys and transformed data as values for overview
        result_dict = {id_: transformed_data[i].tolist(
        ) for i, id_ in enumerate(data.keys())}

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
        n_clusters: int = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
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
        n_clusters: int = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
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


class ClusterFinder(Plotter):
    def find_clusters(self, **kwargs) -> int:
        '''	
        Detects a reasonable number of clusters in the data using Density Clustering.

        Parameters:
            data (dict[int, list[float]]): A dictionary where keys are IDs, and values are lists
            of transformed data after PCA.
            plot (bool): Whether to plot the number of clusters found for each min_samples value.

        Returns:
            int: The number of clusters found.
        '''
        data = kwargs['data']
        plot = kwargs['plot'] if 'plot' in kwargs else False

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
                # print(f'found {max_cluster} clusters at {i} min_samples')

        if plot:
            self.plot_clusters(samples, clusters)

        return max_cluster


class CentreFinder:
    def get_kmeans_centre(self, **kwargs) -> dict[int, str]:
        '''	
        Finds the closest data point to each cluster center in the KMeans clustering results.

        Parameters:
            **kwargs: Additional keyword arguments.
                - data (dict[int, list[float]]): A dictionary where keys are image IDs, and values are lists
                representing the transformed data

        Returns:
            dict[int, str]: A dictionary where keys are cluster names (e.g., "Centroid Cluster 1"),
            and values are image IDs representing the closest data point to each cluster center.

        '''
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
            cluster_points = [data[image_id] for image_id, label in zip(
                data.keys(), hierarchical_labels) if label == cluster_id]
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
            cluster_points = [data[image_id] for image_id, label in zip(
                data.keys(), density_labels) if label == cluster_id]
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

# Below are the different versions of the summarization algorithm


class SummerizationPCAKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    '''	
    This class performs Principal Component Analysis (PCA) on input data and applies
        clustering KMeans.

    Attributes:
        version (float | str): The version of the Summerization class.
        name (str): The name of the clustering technique followed by PCA (e.g., "PCA_Kmeans").

    '''

    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PCA_Kmeans"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary

        data = kwargs['data']
        n_pca_d = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        pca_data = self.apply_pca(data=data, n_dimensions=n_pca_d)
        kmeans = self.apply_kmeans(
            data=pca_data, n_clusters=n_clusters, seed=42)
        centers = self.get_kmeans_centre(data=pca_data)

        summary = self.generate_output_dict(kmeans, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.0 ##########################################################


class SummerizationPCAHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    '''
    This class performs Principal Component Analysis (PCA) on input data and applies
        clustering Hierarchical Clustering.

    Attributes:
        version (float | str): The version of the Summerization class.
        name (str): The name of the clustering technique followed by PCA (e.g., "PCA_Hierical").
    '''

    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "PCA_Hierical"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)
        
        if summary != False:
            return summary
        
        data = kwargs['data']
        n_pca_d = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        pca_data = self.apply_pca(data=data, n_dimensions=n_pca_d)
        hierarchical = self.apply_hierarchical(
            data=pca_data, n_clusters=n_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=pca_data)

        summary = self.generate_output_dict(hierarchical, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

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
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_pca_d = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5

        pca_data = self.apply_pca(data=data, n_dimensions=n_pca_d)
        density = self.apply_density(data=pca_data, min_samples=min_samples)
        centers = self.get_density_centre(data=pca_data)

        summary = self.generate_output_dict(density, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.2 ##########################################################


class SummerizationTSNEKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.2
        self.name: str = "TSNE_Kmeans"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        TSNE_data = self.apply_TSNE(data=data, n_dimensions=n_dimensions)
        kmeans = self.apply_kmeans(
            data=TSNE_data, n_clusters=n_clusters, seed=42)
        centers = self.get_kmeans_centre(data=TSNE_data)
        
        summary = self.generate_output_dict(kmeans, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.3 ##########################################################


class SummerizationTSNEHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.3
        self.name: str = "TSNE_Hierical"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        TSNE_data = self.apply_TSNE(data=data, n_dimensions=n_dimensions)
        hierarchical = self.apply_hierarchical(
            data=TSNE_data, n_clusters=n_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=TSNE_data)

        summary = self.generate_output_dict(hierarchical, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.4 ##########################################################


class SummerizationTSNEDensity(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.4
        self.name: str = "TSNE_Density"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5

        TSNE_data = self.apply_TSNE(data=data, n_dimensions=n_dimensions)
        density = self.apply_density(data=TSNE_data, min_samples=min_samples)
        centers = self.get_density_centre(data=TSNE_data)

        summary = self.generate_output_dict(density, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.5 ##########################################################


class SummerizationUMAPKmeans(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.5
        self.name: str = "UMAP_Kmeans"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        UMAP_data = self.apply_UMAP(
            data=data, n_dimensions=n_dimensions, seed=42)
        kmeans = self.apply_kmeans(
            data=UMAP_data, n_clusters=n_clusters, seed=42)
        centers = self.get_kmeans_centre(data=UMAP_data)

        summary = self.generate_output_dict(kmeans, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.6 ##########################################################


class SummerizationUMAPHierarchy(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.6
        self.name: str = "UMAP_Hierical"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        n_clusters = kwargs['n_clusters'] if 'n_clusters' in kwargs else 5
        UMAP_data = self.apply_UMAP(
            data=data, n_dimensions=n_dimensions, seed=42)
        hierarchical = self.apply_hierarchical(
            data=UMAP_data, n_clusters=n_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=UMAP_data)

        summary = self.generate_output_dict(hierarchical, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 2.7 ##########################################################


class SummerizationUMAPDensity(SummarizationParent, DimensionalityReducer, Clusterer, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 2.7
        self.name: str = "UMAP_Density"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25
        min_samples: int = kwargs['min_samples'] if 'min_samples' in kwargs else 5

        UMAP_data = self.apply_UMAP(
            data=data, n_dimensions=n_dimensions, seed=42)
        density = self.apply_density(data=UMAP_data, min_samples=min_samples)
        centers = self.get_density_centre(data=UMAP_data)

        summary = self.generate_output_dict(density, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 3.0 ##########################################################


class SummerizationPCADensityKmeans(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 3.0
        self.name: str = "PCA_Density_Kmeans"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25

        pca_data = self.apply_pca(data=data, n_dimensions=n_dimensions)
        n_clusters = self.find_clusters(data=pca_data, plot=False)
        if n_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            n_clusters = 5
        kmeans = self.apply_kmeans(
            data=pca_data, n_clusters=n_clusters, seed=42)
        centers = self.get_kmeans_centre(data=pca_data)

        summary = self.generate_output_dict(kmeans, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 3.1 ##########################################################


class SummerizationPCADensityDensity(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder):
    def __init__(self) -> None:
        self.version: float | str = 3.1
        self.name: str = "PCA_Density_Density"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25

        pca_data = self.apply_pca(data=data, n_dimensions=n_dimensions)
        n_clusters = self.find_clusters(data=pca_data, plot=False)
        if n_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            n_clusters = 5
        density = self.apply_density(data=pca_data, min_samples=3, seed=42)
        centers = self.get_density_centre(data=pca_data)
        
        summary = self.generate_output_dict(density, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

########################################## 3.2 ##########################################################


class SummerizationPCADensityHirarchy(SummarizationParent, DimensionalityReducer, Clusterer, ClusterFinder, CentreFinder, Plotter):
    def __init__(self) -> None:
        self.version: float | str = 3.2
        self.name: str = "PCA_Density_Hirarchy"

    def run(self, **kwargs):
        file_name_summary = self.get_summarization_file_name(**kwargs)

        summary = self.get_summary_from_file(file_name_summary, **kwargs)

        if summary != False:
            return summary
        
        data = kwargs['data']
        n_dimensions = kwargs['n_dimensions'] if 'n_dimensions' in kwargs else 25

        pca_data = self.apply_pca(data=data, n_dimensions=n_dimensions)
        n_clusters = self.find_clusters(data=pca_data, plot=False)
        if n_clusters == 0:
            print('No reasonable number of clusters found, using default value of 5')
            n_clusters = 5
        # self.create_dendrogram_plot(data=pca_data, n_clusters=n_clusters)
        hierarchical = self.apply_hierarchical(
            data=pca_data, n_clusters=n_clusters, seed=42)
        centers = self.get_hierarchical_centre(data=pca_data)

        summary = self.generate_output_dict(hierarchical, centers)
        self.save_summary(file_name_summary, summary, **kwargs)

        return summary

# Below functions are used to compare the different versions of the summarization algorithm


def time_version(**kwargs) -> float:
    '''
    Returns the time taken to run the summarization algorithm with the given version.
    '''

    test_data = pd.read_csv(
        'Embedding Files\Embeddings_v1_0_2392_4783.csv', delimiter=';')
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index(
        'image_id')['tensor'].to_dict().items()}

    summarization = SummarizationParent()
    repeats = kwargs['repeats'] if 'repeats' in kwargs else 3
    version = kwargs['summarization_version']

    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        summarization.run(
            summarization_version=version,
            data=data,
            n_dimensions=10,
            n_clusters=5
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    standard_error = np.std(times) / np.sqrt(repeats)
    return np.mean(times), standard_error


def get_distance(**kwargs):
    summary = kwargs['summary']
    data = kwargs['data']

    summary_embeddings = []
    for cluster in summary:
        id = (summary[cluster]['selected'])
        if id in data:
            summary_embeddings.append(data[id])
        else:
            raise ValueError('id not in neighbourhood embeddings')

    neighbourhood_embeddings = [data[key] for key in data]

    avg_summary = sum(summary_embeddings) / len(summary_embeddings)
    avg_neighbourhood = sum(neighbourhood_embeddings) / \
        len(neighbourhood_embeddings)

    difference = torch.norm(avg_summary - avg_neighbourhood).item()

    return (difference)


def distance_version(**kwargs) -> float:

    distances = []

    for path in kwargs['paths']:
        test_data = pd.read_csv(path, delimiter=';')

        data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index(
            'image_id')['tensor'].to_dict().items()}
        summarization = SummarizationParent()
        version = kwargs['summarization_version']

        output = summarization.run(
            summarization_version=version,
            data=data,
            n_dimensions=10,
            n_clusters=5
        )
        percentage = get_distance(data=data, summary=output)
        distances.append(percentage)

    standard_error = np.std(distances) / np.sqrt(len(distances))
    return np.mean(distances), standard_error


def compare_versions() -> pd.DataFrame:
    '''	
    returns a dataframe with the time taken to run each version of the summarization algorithm
    '''
    repeats = 3
    paths = [
        'Embedding Files\Embeddings_v1_0_2392_4783.csv',
        'Embedding Files\Embeddings_v1_0_4784_7175.csv',
        'Embedding Files\Embeddings_v1_0_7176_9567.csv',
    ]
    versions = {
        1.0: 'PCA  Kmeans',
        2.0: 'PCA  Hierical',
        2.1: 'PCA  Density',
        2.2: 'TSNE Kmeans',
        2.3: 'TSNE Hierical',
        2.4: 'TSNE Density',
        2.5: 'UMAP Kmeans',
        2.6: 'UMAP Hierical',
        2.7: 'UMAP Density',
    }

    times = {}
    for version in tqdm(versions.keys(), desc='Comparing Times'):
        mean, standard_error = time_version(
            data=data, summarization_version=version, repeats=repeats)
        time = f"{mean:.3f} ± {standard_error:.3f}"
        times[versions[version]] = time

    distance = {}
    for version in tqdm(versions.keys(), desc='Comparing Distance'):
        mean, standard_error = distance_version(
            paths=paths, summarization_version=version)
        acc = f"{mean:.3f} ± {standard_error:.3f}"
        distance[versions[version]] = acc

    times_df = pd.DataFrame.from_dict(
        times, orient='index', columns=['Mean Time (s)'])
    distance_df = pd.DataFrame.from_dict(
        distance, orient='index', columns=['Mean Distance (-)'])
    merged_df = pd.merge(times_df, distance_df,
                         left_index=True, right_index=True)

    print(merged_df)


def compare_dimensions(**kwargs) -> None:
    dimensions_to_test = list(range(1, 80, 5))
    version = kwargs['version']
    data = kwargs['data']

    distances = []

    for dimension in tqdm(dimensions_to_test, desc='Comparing Dimensions'):
        summarization = SummarizationParent()
        output = summarization.run(
            summarization_version=version,
            data=data,
            n_dimensions=dimension,
            n_clusters=4,
            min_samples=6
        )

        distances.append(get_distance(data=data, summary=output))

    plt.scatter(dimensions_to_test, distances)
    plt.xlabel('n_dimensions (K)')
    plt.ylabel('distance')
    plt.ylim(bottom=0)
    for i in range(len(dimensions_to_test)):
        plt.text(dimensions_to_test[i], distances[i],
                 f'  k={dimensions_to_test[i]}')
    plt.show()


def pretty_print(output):
    for key in sorted(output.keys(), key=lambda x: int(x.split(' ')[1])):
        print(f"{key}: {output[key]}")


if __name__ == "__main__":

    print('START')

    # loading in the embeddings to test on
    test_data = pd.read_csv(
        'Embedding Files\Embeddings_v1_0_14352_16743.csv', delimiter=';')
    data = {key: torch.tensor(ast.literal_eval(value)) for key, value in test_data.set_index(
        'image_id')['tensor'].to_dict().items()}

    # uncomment to run the comparisons
    # compare_versions()
    # compare_dimensions(version=1.0, data=data)

    summarization = SummarizationParent()
    output = summarization.run(
        summarization_version=3.2,
        data=data,
        n_dimensions=25,
        n_clusters=5,
        min_samples=6
    )

    # uncomment to print the output
    pretty_print(output)

    print('DONE')

# Example Output
# {
# 'Cluster 4': {'selected': 4130, 'cluster': {4096, 4362, 4366, 4370, 4372, 4374, 4378, 4380, 4126, 4382, 4384, 4130, 4387, 4388, 4389, 4134, 4392, 4393, 4138, 4396, 4142, 4399, 4400, 4401, 4404, 4405, 4408, 4411, 4412, 4413, 4419, 4338, 4342, 4350}},
# 'Cluster 1': {'selected': 4120, 'cluster': {4097, 4098, 4099, 4353, 4358, 4359, 4367, 4120, 4121, 4122, 4123, 4124, 4125, 4127, 4128, 4129, 4383, 4131, 4132, 4133, 4136, 4137, 4139, 4140, 4141, 4143, 4416}},
# 'Cluster 3': {'selected': 4386, 'cluster': {4385, 4386, 4390, 4391, 4135, 4394, 4395, 4397, 4398, 4402, 4403, 4406, 4407, 4409, 4410, 4414, 4415, 4417, 4418}}
# }
