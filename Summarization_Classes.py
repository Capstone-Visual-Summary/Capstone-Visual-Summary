from typing import Union, List
from Grand_Parent import GrandParent
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

    def run(self, **kwargs):
        N_pca_d = kwargs['N_dimensions'] if 'N_dimensions' in kwargs else 2
        n_clusters = kwargs['N_clusters'] if 'N_clusters' in kwargs else 3
        pca_data = self.apply_pca(data=data, N_dimensions=N_pca_d)
        kmeans = self.apply_kmeans(data=pca_data, n_clusters=n_clusters, seed=42)
        return kmeans
    
if __name__ == "__main__":
    summarization = Summerization()
    Kmeans = summarization.run(N_dimensions=2, N_clusters=4)
    for key in sorted(Kmeans.keys()):
        print(f"{key}: {Kmeans[key]}")