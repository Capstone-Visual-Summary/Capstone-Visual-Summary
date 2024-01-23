from typing import Union
from Grand_Parent import GrandParent

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



class VisualizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Visulization"
        self.children: temp[str, temp[str, Union[str, VisualizationParent]]] = temp()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['visualization_version'] if 'visualization_version' in kwargs else -1
        return super().run(version, **kwargs)

        
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



class VisualizationCOLOR(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = '1.1 WIP'
        self.name: str = "PLT"

    def run(self, **kwargs):
        summary = kwargs['summary']
        images = kwargs['images']

        data: temp[int, tensor] = kwargs['embeddings']
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
    

        

class VisualizationPLT2(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.2
        self.name: str = "PLT"

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


if __name__ == '__main__':
    summary = {'7': {'Cluster 0': {'selected': '52616', 'cluster': ['52616', '52617', '52620', '52621', '52624', '52625', '52628', '52629', '52630', '52631', '52632', '52634', '52635', '52268', '52269', '52270', '52271', '52272', '52273', '52274', '52276', '52278', '52280', '52281', '52282', '52284', '52285', '52286']}, 'Cluster 1': {'selected': '52277', 'cluster': ['52618', '52619', '52622', '52623', '52626', '52627', '52633', '52275', '52277', '52279', '52283', '52287']}}}
    
    images = gpd.read_file('Hardcoded_Images.geojson')
    
    embeddings = dict()
    with open('Hardcoded_Embeddings.csv', mode='r', newline='', encoding='utf-8') as csvfile:
        temp = csv.DictReader(csvfile, delimiter=';')
        for row in temp:
            embeddings[row['image_id']] = torch.Tensor(ast.literal_eval(row['tensor']))
    
    image_embeddings = dict()
    image_embeddings['7'] = embeddings

    Test = VisualizationPLT2()
    Test.run(summary = summary['7'], embeddings=image_embeddings, neighbourhood_id='7', images = images)


# ----------------
    

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
