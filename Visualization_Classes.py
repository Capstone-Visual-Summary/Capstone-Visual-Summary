from typing import Union
from Grand_Parent import GrandParent
from KeplerGL_Config import generate_config

from tqdm import tqdm
import math as math
import matplotlib.pyplot as plt
from PIL import Image
import torch as torch
from matplotlib import gridspec
import numpy as np
from keplergl import KeplerGl
import webbrowser
import os
from sklearn.decomposition import PCA


class VisualizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Visulization"
        self.children: dict[str, dict[str, Union[str, VisualizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['visualization_version'] if 'visualization_version' in kwargs else -1
        return super().run(version, **kwargs)
    
    def min_max_colors_opacity(self, unprocessed_data) -> dict[str, torch.Tensor]:
        global_min_color_red: float | torch.Tensor = float('inf')
        global_max_color_red: float | torch.Tensor = -float('inf')
        global_min_color_green: float | torch.Tensor = float('inf')
        global_max_color_green: float | torch.Tensor = -float('inf')
        global_min_color_blue: float | torch.Tensor = float('inf')
        global_max_color_blue: float | torch.Tensor = -float('inf')

        for neighbourhood_id, color_tensor in unprocessed_data.items():
            global_min_color_red = color_tensor[0] if color_tensor[0] < global_min_color_red else global_min_color_red
            global_max_color_red = color_tensor[0] if color_tensor[0] > global_max_color_red else global_max_color_red
            global_min_color_green = color_tensor[1] if color_tensor[1] < global_min_color_green else global_min_color_green
            global_max_color_green = color_tensor[1] if color_tensor[1] > global_max_color_green else global_max_color_green
            global_min_color_blue = color_tensor[2] if color_tensor[2] < global_min_color_blue else global_min_color_blue
            global_max_color_blue = color_tensor[2] if color_tensor[2] > global_max_color_blue else global_max_color_blue

        normalized_data: dict[str, torch.Tensor] = dict()

        for neighbourhood_id, color_tensor in unprocessed_data.items():
            normalized_red = (unprocessed_data[neighbourhood_id][0] - global_min_color_red) / (global_max_color_red - global_min_color_red) * 255
            normalized_green = (unprocessed_data[neighbourhood_id][1] - global_min_color_green) / (global_max_color_green - global_min_color_green) * 255
            normalized_blue = (unprocessed_data[neighbourhood_id][2] - global_min_color_blue) / (global_max_color_blue - global_min_color_blue) * 255
            normalized_tensor = torch.tensor([normalized_red, normalized_green, normalized_blue])
            
            normalized_data[neighbourhood_id] = normalized_tensor
        
        return normalized_data
    
    def process_visualization_data(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], dict[str, str], list[int]]:
        summaries = kwargs['summaries']
        neighbourhoods = kwargs['neighbourhoods']

        color_getter = VisualizationVerification()
        summary_image_getter = VisualizationPLT()

        visualization_data: dict[str, torch.Tensor] = dict()
        summary_images: dict[str, str] = dict()
        summary_images_ids: list[int] = []

        for neighbourhood_id in tqdm(summaries, total=len(summaries)):
            for cluster_num, cluster in summaries[neighbourhood_id].items():
                summary_images_ids.append(int(cluster['selected']))

            temp_storage_color = color_getter.run(summary=summaries[neighbourhood_id], neighbourhood_id=neighbourhood_id, **kwargs)
            temp_storage_image = summary_image_getter.run(summary=summaries[neighbourhood_id], neighbourhood_id=neighbourhood_id, only_summary=True, **kwargs)

            visualization_data[neighbourhood_id] = torch.Tensor(temp_storage_color[0])
            summary_images[neighbourhood_id] = '.' + temp_storage_image

        normalized_data = self.min_max_colors_opacity(visualization_data)
        
        map_neighbourhoods_vis_data = dict()

        if 'BU_NAAM' in neighbourhoods:
            for neighbourhood_id in visualization_data:
                bu_naam = neighbourhoods.loc[(neighbourhoods['neighbourhood_id'] == int(neighbourhood_id)), 'BU_NAAM'].iloc[0]
                map_neighbourhoods_vis_data[bu_naam] = normalized_data[neighbourhood_id]
        else:
            map_neighbourhoods_vis_data = normalized_data

        colors = {key: value.tolist() for key, value in map_neighbourhoods_vis_data.items()}

        return map_neighbourhoods_vis_data, colors, summary_images, summary_images_ids
    
    def generate_map(self, config, summary_neighbourhoods, map_neighbourhoods_vis_data, summary_images_ids, **kwargs) -> None:
        database_version = kwargs['database_version']
        embedder_version = kwargs['embedder_version']
        summarization_version = kwargs['summarization_version']

        file_path = f"Interactive Maps/neighbourhoods_{kwargs['start_hood']}_{kwargs['stop_hood'] - 1}_D{str(database_version).split('.')[0]}{str(database_version).split('.')[1]}E{str(embedder_version).split('.')[0]}{str(embedder_version).split('.')[1]}S{str(summarization_version).split('.')[0]}{str(summarization_version).split('.')[1]}.html"
        
        if os.path.exists(file_path):
            webbrowser.open('file://' + os.path.abspath(file_path), new=2)
        else:
            images = kwargs['images']
            images = images.loc[images['img_id_com'].isin(summary_images_ids)]

            visual_map = KeplerGl(height=400, config=config)
            visual_map.add_data(images, name='Summary Images')

            for neighbourhood_id in map_neighbourhoods_vis_data:
                if 'BU_NAAM' in summary_neighbourhoods:
                    visual_map.add_data(summary_neighbourhoods.loc[summary_neighbourhoods['BU_NAAM'] == neighbourhood_id], name=neighbourhood_id)
                else:
                    visual_map.add_data(summary_neighbourhoods.loc[summary_neighbourhoods['neighbourhood_id'] == int(neighbourhood_id)], name=neighbourhood_id)

            visual_map.save_to_html(file_name=file_path)

            webbrowser.open('file://' + os.path.abspath(file_path), new=2)

    def generate_complete_images(self, summary, images, neighbourhood_id, database_version, embedder_version, summarization_version):
        n_clusters = len(summary)

        max_cluster = 0
        for cluster in summary:
            max_cluster = len(summary[cluster]['cluster']) if len(summary[cluster]['cluster']) > max_cluster else max_cluster

        col_height = max_cluster
        col_width = 1
        while col_height > col_width * n_clusters :
            col_width += 1
            col_height = math.ceil(max_cluster/col_width)

        save_path_complete = f"./Visual_Summaries/neighbourhood_{neighbourhood_id}_D{str(database_version).split('.')[0]}{str(database_version).split('.')[1]}E{str(embedder_version).split('.')[0]}{str(embedder_version).split('.')[1]}S{str(summarization_version).split('.')[0]}{str(summarization_version).split('.')[1]}_complete.png"
        
        try:
            image_opener = Image.open(save_path_complete)
        except:
            fig = plt.figure(figsize=(col_width + col_height, n_clusters * col_width), facecolor='#404040')
            fig.suptitle(f'Neighbourhood {neighbourhood_id}', color='white', fontsize=16)

            # Define the main layout
            outer_grid = gridspec.GridSpec(2, n_clusters, wspace=0.1, hspace=0.1)

            # Iterate over the main columns
            for i, cluster in tqdm(enumerate(summary), total=len(summary), desc='Complete'):
                # Top subplot for the single image with a title and bottom title
                path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
                img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
                #img = Image.open('TEST.png')
                ax_top = fig.add_subplot(outer_grid[0, i])
                ax_top.imshow(img)
                ax_top.set_title(path, color='white', fontsize=16)
                ax_top.axis('off')

                # Bottom subplot for the grid of images
                ax_bottom = fig.add_subplot(outer_grid[1, i])
                ax_bottom.axis('off')

                # Create a nested grid within the bottom subplot
                nested_grid = gridspec.GridSpecFromSubplotSpec(col_height, col_width, subplot_spec=outer_grid[1, i], wspace=0.05, hspace=0.05)

                # Populate the nested grid with images
                for j in range(col_height * col_width):
                    ax_nested = plt.Subplot(fig, nested_grid[j])
                    if j < len(summary[cluster]['cluster']):
                            path = images.loc[(images['img_id_com'] == int(summary[cluster]['cluster'][j])), 'path'].iloc[0]
                            # print(path)
                            img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
                            #img = Image.open('TEST.png')
                            ax_nested.imshow(img)
                    ax_nested.axis('off')
                    fig.add_subplot(ax_nested)

            # Adjust layout
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            
            fig_width_in_inches = fig.get_size_inches()[0]  # Get the width of the figure in inches
            desired_pixel_width = 1920  # Maximum pixel width
            dpi = desired_pixel_width / fig_width_in_inches  # Calculate the DPI
            
            fig.savefig(save_path_complete, dpi=dpi)
            plt.close(fig)

        return save_path_complete

    def generate_summary_images(self, summary, images, neighbourhood_id, database_version, embedder_version, summarization_version):
        n_clusters = len(summary)
        
        save_path_summary = f"./Visual_Summaries/neighbourhood_{neighbourhood_id}_D{str(database_version).split('.')[0]}{str(database_version).split('.')[1]}E{str(embedder_version).split('.')[0]}{str(embedder_version).split('.')[1]}S{str(summarization_version).split('.')[0]}{str(summarization_version).split('.')[1]}_summary.png"

        try:
            image_opener = Image.open(save_path_summary)
        except:
            #create a second plot only containing the summary
            fig = plt.figure(figsize=(n_clusters * 5, 5), facecolor='#404040')
            fig.suptitle(f'Neighbourhood {neighbourhood_id}', color='white', fontsize=16)

            # Define the main layout
            outer_grid = gridspec.GridSpec(1, n_clusters, wspace=0.1, hspace=0.1)

            # Iterate over the main columns
            for i, cluster in tqdm(enumerate(summary), total=len(summary), desc='Summary'):
                # Top subplot for the single image with a title and bottom title
                path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
                img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
                #img = Image.open('TEST.png')
                ax_top = fig.add_subplot(outer_grid[0, i])
                ax_top.imshow(img)
                ax_top.set_title(path, color='white', fontsize=16)
                ax_top.axis('off')

            # Adjust layout
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            
            fig_width_in_inches = fig.get_size_inches()[0]  # Get the width of the figure in inches
            desired_pixel_width = 400  # Maximum pixel width
            dpi = desired_pixel_width / fig_width_in_inches  # Calculate the DPI

            fig.savefig(save_path_summary, dpi=dpi)
            plt.close(fig)
        
        return save_path_summary


class VisualizationPLT(VisualizationParent):
    """
    VisualizationPLT

    A class that extends VisualizationParent and provides methods for generating and saving visual summaries
    of clusters in a neighborhood using the PLT library.

    Attributes:
    - version (float | str): The version number of the visualization tool.
    - name (str): The name of the visualization tool.

    Methods:
    - run(**kwargs): Generates and saves visual summaries of clusters in a neighborhood.

    Parameters:
    - summary (dict): A dictionary containing cluster information.
    - images (DataFrame): A DataFrame containing image information.
    - neighbourhood_id (int): The identifier for the neighborhood.
    - embedder_version (float): The version number of the embedder used.
    - summarization_version (float): The version number of the summarization algorithm used.

    Returns:
    - Tuple[str, str]: A tuple of file paths for the complete and summary visualizations.
    """
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PLT"

    def run(self, **kwargs):
        """
        Generates and saves visual summaries of clusters in a neighborhood using the PLT library.

        Parameters:
        - summary (dict): A dictionary containing cluster information.
        - images (DataFrame): A DataFrame containing image information.
        - neighbourhood_id (int): The identifier for the neighborhood.
        - embedder_version (float): The version number of the embedder used.
        - summarization_version (float): The version number of the summarization algorithm used.

        Returns:
        - Tuple[str, str]: A tuple of file paths for the complete and summary visualizations.
        """
        summary = kwargs['summary']
        images = kwargs['images']
        neighbourhood_id = kwargs['neighbourhood_id']
        database_version = kwargs['database_version']
        embedder_version = kwargs['embedder_version']
        summarization_version = kwargs['summarization_version']

        if 'only_summary' in kwargs and kwargs['only_summary']:
            return self.generate_summary_images(summary, images, neighbourhood_id, database_version, embedder_version, summarization_version)
        
        save_path_complete = self.generate_complete_images(summary, images, neighbourhood_id, database_version, embedder_version, summarization_version)
        save_path_summary = self.generate_summary_images(summary, images, neighbourhood_id, database_version, embedder_version, summarization_version)

        return save_path_complete, save_path_summary

    
class VisualizationVerification(VisualizationParent):
    """
    VisualizationVerification

    A class that extends VisualizationParent and provides methods for visual verification of embeddings
    using PCA and average percentage difference.

    Attributes:
    - version (float | str): The version number of the visualization tool.
    - name (str): The name of the visualization tool.

    Methods:
    - run(**kwargs): Generates and returns color and average percentage difference based on embeddings.
    - tensor_average_percentage_difference(group1: list[torch.Tensor], group2: list[torch.Tensor]): 
      Calculates the average percentage difference between two groups of tensors.
    - train_pca_on_data(dict_of_dicts): Trains a PCA model on the provided data.
    - transform_tuples_with_pca(pca, tuple_list): Transforms a list of tuples using a trained PCA model.
    - average_of_transformed(dict_of_dicts, pca): Calculates the average of transformed data for each key in a dictionary.
    """

    def __init__(self) -> None:
        self.version: float | str = '2.0'
        self.name: str = "Verification"

    def run(self, **kwargs) -> tuple[torch.Tensor, float]: #type: ignore
        """
        Generates and returns color and difference between all embeddings and the summary embeddings.

        Parameters:
        - summary (dict): A dictionary containing cluster information.
        - embeddings (temp[int, tensor]): A dictionary containing embeddings.
        - neighbourhood_id (int): The identifier for the neighborhood.

        Returns:
        - Tuple[np.ndarray, float]: A tuple containing color (average of PCA-transformed embeddings) and
          and difference between all embeddings and the summary embeddings.
        """
        summary = kwargs['summary']
        data: dict[str, torch.Tensor] = kwargs['embeddings']
        neighbourhood_id = kwargs['neighbourhood_id']

        summary_embeddings = []
        
        for cluster in summary:
            id = (summary[cluster]['selected'])
            if id in data[str(neighbourhood_id)]:
                summary_embeddings.append(data[str(neighbourhood_id)][id])
            else:
                raise ValueError('id not in neighbourhood embeddings')
        
        neighbourhood_embeddings = [data[str(neighbourhood_id)][key] for key in data[str(neighbourhood_id)]]

        difference = self.tensor_average_percentage_difference(neighbourhood_embeddings, summary_embeddings)

        #add some functionality to only train pca once and store it as an attribute
        if not hasattr(self, 'pca'):
            self.pca = self.train_pca_on_data(data)

        result = self.transform_tuples_with_pca(self.pca, neighbourhood_embeddings)
        color = np.mean(result, axis=0)

        return(color, difference)
        
    def tensor_average_percentage_difference(self, group1: list[torch.Tensor], group2: list[torch.Tensor]):
        # Check if groups are not empty
        if not group1 or not group2:
            raise ValueError("Input tensor groups should not be empty")

        # Calculate the average tensor for each group
        avg_tensor1 = sum(group1) / len(group1)
        avg_tensor2 = sum(group2) / len(group2)
        
        # Calculate the norm of the difference and of group 1
        diffference = torch.norm(avg_tensor1 - avg_tensor2).item()

        return diffference
    
    def train_pca_on_data(self, dict_of_dicts):
        # Flatten the tuples into a list
        data = [tup for subdict in dict_of_dicts.values() for tup in subdict.values()]

        # Train PCA model
        pca = PCA(n_components=3)
        pca.fit(data)
        return pca
    
    def transform_tuples_with_pca(self, pca, tuple_list):
        # Transform the list of tuples using the trained PCA
        transformed_data = pca.transform(tuple_list)
        return transformed_data
    
    def average_of_transformed(self, dict_of_dicts, pca):
        averages = {}
        for key, subdict in dict_of_dicts.items():
            transformed_data = self.transform_tuples_with_pca(pca, list(subdict.values()))
            averages[key] = np.mean(transformed_data, axis=0)
        return averages


class VisualizationCOLOR(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = '1.2 WIP'
        self.name: str = "COLOR"

    def run(self, **kwargs) -> tuple[torch.Tensor, float]: # type: ignore
        summary = kwargs['summary']
        images = kwargs['images']
        data: dict[str, torch.Tensor] = kwargs['embeddings']
        neighbourhood_id = kwargs['neighbourhood_id']

        summary_embeddings = []
        for cluster in summary:
            id = (summary[cluster]['selected'])
            if id in data[str(neighbourhood_id)]:
                summary_embeddings.append(data[str(neighbourhood_id)][id])
            else:
                raise ValueError('id not in neighbourhood embeddings')

        neighbourhood_embeddings = [data[str(neighbourhood_id)][key] for key in data[str(neighbourhood_id)]]

        percentage = self.tensor_average_percentage_difference(neighbourhood_embeddings, summary_embeddings)

        #add some functionality to only train pca once and store it as an attribute
        if not hasattr(self, 'pca'):
            self.pca = self.train_pca_on_data(data)
        
        result = self.transform_tuples_with_pca(self.pca, neighbourhood_embeddings)
        color = np.mean(result, axis=0)

        return(color, percentage)

    def tensor_average_percentage_difference(self, group1: list[torch.Tensor], group2: list[torch.Tensor]):
        # Check if groups are not empty
        if not group1 or not group2:
            raise ValueError("Input tensor groups should not be empty")

        # Calculate the average tensor for each group
        avg_tensor1 = sum(group1) / len(group1)
        avg_tensor2 = sum(group2) / len(group2)

        # Calculate the norm of the difference and of group 1
        norm1 = torch.norm(avg_tensor1).item()
        norm2 = torch.norm(avg_tensor1 - avg_tensor2).item()

        if norm1 != 0:
            size = norm2/norm1
        else:
            size = 1
        return size

    def train_pca_on_data(self, dict_of_dicts):
        # Flatten the tuples into a list
        data = [tup for subdict in dict_of_dicts.values() for tup in subdict.values()]

        # Train PCA model
        pca = PCA(n_components=3)
        pca.fit(data)
        return pca

    def transform_tuples_with_pca(self, pca, tuple_list):
        # Transform the list of tuples using the trained PCA
        transformed_data = pca.transform(tuple_list)
        return transformed_data

    def average_of_transformed(self, dict_of_dicts, pca):
        averages = {}
        for key, subdict in dict_of_dicts.items():
            transformed_data = transform_tuples_with_pca(pca, list(subdict.values()))
            averages[key] = np.mean(transformed_data, axis=0)
        return averages


class VisualizationInteractiveMap(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 3.0
        self.name: str = 'Interactive Map'

    def run(self, **kwargs) -> None: # type: ignore
        """
        Based on the summaries, creates an interactive map of the neighbourhoods, with colour similarity,
        and upon hovering, show the image summary of the neighbourhood and where the sumary images are located in the neighbourhood
        Saves and opens an html file
        """

        summary_neighbourhoods = kwargs['neighbourhoods']
        
        map_neighbourhoods_vis_data, colors, summary_images, summary_images_ids = self.process_visualization_data(**kwargs)

        for neighbourhood_id, summary_image in summary_images.items():
            summary_neighbourhoods.loc[summary_neighbourhoods['neighbourhood_id'] == int(neighbourhood_id), '<img>-summary'] = summary_image
        
        config = generate_config(colors)
        self.generate_map(config, summary_neighbourhoods, map_neighbourhoods_vis_data, summary_images_ids, **kwargs)
