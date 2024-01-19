from typing import Union
from Grand_Parent import GrandParent

import math as math
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class VisualizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Visulization"
        self.children: dict[str, dict[str, Union[str, VisualizationParent]]] = dict()
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
            
        



# class VisualizationADDMETHODNAME(VisualizationParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "ADD METHOD NAME"

#     def run(self, **kwargs):
#         pass


# class VisualizationShow(VisualizationParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "export to png"

#     def run(self, **kwargs):
#         summary = kwargs['summary']
#         images = kwargs['images']

#         width = len(summary[1])

#         total = width
#         for cluster in summary[0]:
#             total += len(summary[0][cluster])
#         print(total)

#         height = math.ceil(total/width)
#         print(1)
#         fig, axs = plt.subplots(height, width, figsize=(50, 50))
#         print(2)
#         index = 0

#         colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
#         color_dict = {}
#         print(3)
#         for x, i in enumerate(summary[1]):
#             img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + images.loc[(images['img_id_com'] == int(summary[1][i])), 'path'].iloc[0])
#             color_dict[i] = colors[x[9:]]
#             img = ImageOps.expand(img, border=50, fill=colors[x])
#             axs[math.floor(index/width)][index % width].imshow(img)
#             axs[math.floor(index/width)][index % width].axis('off')
#             index +=1
#         print(4)
#         for cluster in summary[0]:
#             for image_id in summary[0][cluster]:
#                 img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + images.loc[(images['img_id_com'] == int(image_id)), 'path'].iloc[0])
#                 img = ImageOps.expand(img, border=50, fill=color_dict[cluster])
#                 axs[math.floor(index/width)][index % width].imshow(img)
#                 axs[math.floor(index/width)][index % width].axis('off')
#                 index +=1
#         print(5)
#         plt.show()
#         print(6)
