from typing import Union
from Grand_Parent import GrandParent
from KeplerGL_Config import generate_config

import math as math
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch as torch
from torch import tensor
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import gridspec
import numpy as np
import csv
import ast
import geopandas as gpd
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
    
    def min_max_colors_opacity(self, unprocessed_data) -> dict[str, tuple[torch.Tensor, float]]:
        global_min_color: float | torch.Tensor = float('inf')
        global_max_color: float | torch.Tensor = -float('inf')
        global_min_opacity = float('inf')
        global_max_opacity = -float('inf')

        for neighbourhood_id, color_tensor in unprocessed_data.items():
            global_min_color = torch.min(color_tensor[0]) if torch.min(color_tensor[0]) < global_min_color else global_min_color
            global_max_color = torch.max(color_tensor[0]) if torch.max(color_tensor[0]) > global_max_color else global_max_color
            global_min_opacity = color_tensor[1] if color_tensor[1] < global_min_opacity else global_min_opacity
            global_max_opacity = color_tensor[1] if color_tensor[1] > global_max_opacity else global_max_opacity

        normalized_data = dict()

        for neighbourhood_id, color_tensor in unprocessed_data.items():
            normalized_tensor = (unprocessed_data[neighbourhood_id][0] - global_min_color) / (global_max_color - global_min_color) * 255

            if global_min_opacity < 0.2:
                normalized_opacity = (unprocessed_data[neighbourhood_id][1] * (1 - 0.2) + 0.2)
            else:
                normalized_opacity = max((unprocessed_data[neighbourhood_id][1] - global_min_opacity) / (global_max_opacity - global_min_opacity), 0.2)
            
            normalized_data[neighbourhood_id] = (normalized_tensor, normalized_opacity)
        
        return normalized_data

        
class VisualizationPLT(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "PLT"

    def run(self, **kwargs):
        summary = kwargs['summary']
        images = kwargs['images']
        embeddings = kwargs['embeddings']

        width = len(summary)

        height = 0
        for cluster in summary:
            height = len(summary[cluster]['cluster']) if len(summary[cluster]['cluster']) > height else height
        
        fig, axs = plt.subplots(height+1, width, figsize=(50, 50))

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

        for col, cluster in enumerate(summary):
            color = colors[col]
            path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
            img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
            img = ImageOps.expand(img, border=30, fill=  color)
            axs[0][col].imshow(img)
            axs[0][col].axis('off')
            for row in range(0, height):
                if row < len(summary[cluster]['cluster']):
                    path = images.loc[(images['img_id_com'] == int(summary[cluster]['cluster'][row])), 'path'].iloc[0]
                    img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
                    img = ImageOps.expand(img, border=30, fill=  color)
                    axs[row+1][col].imshow(img)
                axs[row+1][col].axis('off')

        plt.show()
  

class VisualizationPLT2(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.1
        self.name: str = "PLT2"

    def run(self, **kwargs):
        summary = kwargs['summary']
        images = kwargs['images']
        embeddings = kwargs['embeddings']

        n_clusters = len(summary)

        max_cluster = 0
        for cluster in summary:
            max_cluster = len(summary[cluster]['cluster']) if len(summary[cluster]['cluster']) > max_cluster else max_cluster

        col_height = max_cluster
        col_width = 1
        while col_height > col_width * n_clusters :
            col_width += 1
            col_height = math.ceil(max_cluster/col_width)
        
        fig = plt.figure(figsize=(col_width + col_height, n_clusters * col_width), facecolor='#404040')

        # Define the main layout
        outer_grid = gridspec.GridSpec(2, n_clusters, wspace=0.1, hspace=0.1)

        # Iterate over the main columns
        for i, cluster in enumerate(summary):
            # Top subplot for the single image with a title and bottom title
            path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
            img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
            #img = Image.open('TEST.png')
            ax_top = fig.add_subplot(outer_grid[0, i])
            ax_top.imshow(img)
            ax_top.set_title(path, color='white')
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
        plt.show()

        #create a second plot only containing the summary
        fig = plt.figure(figsize=(n_clusters * 5, 5), facecolor='#404040')

         # Define the main layout
        outer_grid = gridspec.GridSpec(1, n_clusters, wspace=0.1, hspace=0.1)

        # Iterate over the main columns
        for i, cluster in enumerate(summary):
            # Top subplot for the single image with a title and bottom title
            path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
            img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
            #img = Image.open('TEST.png')
            ax_top = fig.add_subplot(outer_grid[0, i])
            ax_top.imshow(img)
            ax_top.set_title(path, color='white')
            ax_top.axis('off')

        # Adjust layout
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.show()


class VisualizationInteractiveMapBorder(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = '3.0WIP'
        self.name: str = 'Interactive Map Accurarcy Border'

    def run(self, **kwargs):
        # TODO ADD PICTURE SUMMARY PER NEIGHBOURHOOD
        # neighbourhoods = kwargs['neighbourhoods']
        neighbourhoods = gpd.read_file('Geo-JSON Files/neighbourhood_info_v3_0.geojson')
        neighbourhoods['<img>-summary'] = 'Figure_1.png'

        visualization_data = {
            '7': (torch.Tensor([0.45, 0.9, 0.1]), 0.55),
            '8': (torch.Tensor([0.88, 0.4, 0.2]), 0.67),
            '9': (torch.Tensor([0, 1.2, 0.5]), 0.67)
        } # TODO CORRECT THIS
        normalized_data = self.min_max_colors_opacity(visualization_data)

        map_neighbourhoods_vis_data = dict()

        if 'BU_NAAM' in neighbourhoods:
            for neighbourhood_id in visualization_data:
                bu_naam = neighbourhoods.loc[(neighbourhoods['neighbourhood_id'] == int(neighbourhood_id)), 'BU_NAAM'].iloc[0]
                map_neighbourhoods_vis_data[bu_naam] = normalized_data[neighbourhood_id]
        else:
            map_neighbourhoods_vis_data = normalized_data
        
        colors = {key: value[0].tolist() for key, value in map_neighbourhoods_vis_data.items()}
        opacity = {key: value[1] for key, value in map_neighbourhoods_vis_data.items()}

        config = generate_config(colors, opacity, border_opacity=True)

        visual_map = KeplerGl(height=400, config=config)

        for neighbourhood_id in map_neighbourhoods_vis_data:
            if 'BU_NAAM' in neighbourhoods:
                visual_map.add_data(neighbourhoods.loc[neighbourhoods['BU_NAAM'] == neighbourhood_id], name=neighbourhood_id)
            else:
                visual_map.add_data(neighbourhoods.loc[neighbourhoods['neighbourhood_id'] == int(neighbourhood_id)], name=neighbourhood_id)
        
        visual_map.save_to_html(file_name='Output.html')

        webbrowser.open('file://' + os.path.abspath('Output.html'), new=2)


if __name__ == '__main__':
    # summary = {'7': {'Cluster 0': {'selected': '52616', 'cluster': ['52616', '52617', '52620', '52621', '52624', '52625', '52628', '52629', '52630', '52631', '52632', '52634', '52635', '52268', '52269', '52270', '52271', '52272', '52273', '52274', '52276', '52278', '52280', '52281', '52282', '52284', '52285', '52286']}, 'Cluster 1': {'selected': '52277', 'cluster': ['52618', '52619', '52622', '52623', '52626', '52627', '52633', '52275', '52277', '52279', '52283', '52287']}}}
    
    # images = gpd.read_file('Hardcoded_Images.geojson')
    
    # embeddings = dict()
    # with open('Hardcoded_Embeddings.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    #     temp = csv.DictReader(csvfile, delimiter=';')
    #     for row in temp:
    #         embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
    
    # image_embeddings = dict()
    # image_embeddings['7'] = embeddings

    # Test = VisualizationCOLOR()
    # Test.run(summary = summary['7'], embeddings=image_embeddings, neighbourhood_id='7', images = images)

    Test = VisualizationInteractiveMapBorder()
    Test.run()
    

# def train_pca_on_data(dict_of_dicts):
#     # Flatten the tuples into a list
#     data = [tup for subdict in dict_of_dicts.values() for tup in subdict.values()]

#     # Train PCA model
#     pca = PCA(n_components=3)
#     pca.fit(data)
#     return pca

# def transform_tuples_with_pca(pca, tuple_list):
#     # Transform the list of tuples using the trained PCA
#     transformed_data = pca.transform(tuple_list)
#     return transformed_data

# def average_of_transformed(dict_of_dicts, pca):
#     averages = {}
#     for key, subdict in dict_of_dicts.items():
#         transformed_data = transform_tuples_with_pca(pca, list(subdict.values()))
#         averages[key] = np.mean(transformed_data, axis=0)
#     return averages

# Example usage
# dict_of_dicts = {'category1': {'item1': (1,2,3), 'item2': (4,5,6)}, 'category2': {'item3': (7,8,9)}}
# pca_model = train_pca_on_dict(dict_of_dicts)
# transformed_averages = average_of_transformed(dict_of_dicts, pca_model)


# ----------------
    

        # # Check if the number of dimensions is not larger than the number of data points
    # key = list(data.keys())[0]
    # max_dimensions = min(len(data[key]), len(data))
    # N: int = kwargs['N_dimensions'] if kwargs['N_dimensions'] < max_dimensions else max_dimensions
    # # Extract numerical values from tensors
    # numerical_data = torch.stack(list(data.values())).numpy()
    # pca = PCA(n_components=N)
    # pca.fit(numerical_data)
    # transformed_data = pca.transform(numerical_data)
    # # Create a dictionary with IDs as keys and transformed data as values for overview
    # result_dict = {id_: transformed_data[i].tolist() for i, id_ in enumerate(data.keys())}
    #return result_dict

    # Create a figure
    # fig = plt.figure(figsize=(10, 10), facecolor='black')

    # # Define the main layout
    # outer_grid = gridspec.GridSpec(2, n_clusters, wspace=0.1, hspace=0.1)

    # # Iterate over the main columns
    # for i, cluster in enumerate(summary):
    #     # Top subplot for the single image with a title and bottom title
    #     path = images.loc[(images['img_id_com'] == int(summary[cluster]['selected'])), 'path'].iloc[0]
    #     img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
    #     ax_top = fig.add_subplot(outer_grid[0, i])
    #     ax_top.imshow(img)
    #     ax_top.set_title(path, color='white')
    #     ax_top.axis('off')

    #     # Bottom subplot for the grid of images
    #     ax_bottom = fig.add_subplot(outer_grid[1, i])
    #     ax_bottom.axis('off')

    #     # Create a nested grid within the bottom subplot
    #     nested_grid = gridspec.GridSpecFromSubplotSpec(grid_rows, grid_cols, subplot_spec=outer_grid[1, i], wspace=0.05, hspace=0.05)

    #     # Populate the nested grid with images
    #     for j in range(grid_rows * grid_cols):
    #         ax_nested = plt.Subplot(fig, nested_grid[j])
    #         if j < len(summary[cluster]['cluster']):
    #                 path = images.loc[(images['img_id_com'] == int(summary[cluster]['cluster'][j])), 'path'].iloc[0]
    #                 # print(path)
    #                 img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + path)
    #                 ax_nested.imshow(img)
    #         ax_nested.axis('off')
    #         fig.add_subplot(ax_nested)

    # # Adjust layout
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()
