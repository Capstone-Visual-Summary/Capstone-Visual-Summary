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


class VisualizationShow(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "export to png"

    def run(self, **kwargs):
        summary = kwargs['summary']
        images = kwargs['images']

        width = len(summary[1])

        total = width
        for cluster in summary[0]:
            total += len(summary[0][cluster])
        print(total)

        height = math.ceil(total/width)

        fig, axs = plt.subplots(height, width, figsize=(30, 30))
        
        index = 0

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

        for x, i in enumerate(summary[1]):
            img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + images.loc[(images['img_id_com'] == int(summary[1][i])), 'path'].iloc[0])
            img = ImageOps.expand(img, border=50, fill=colors[x])
            axs[math.floor(index/width)][index % width].imshow(img)
            axs[math.floor(index/width)][index % width].axis('off')
            index +=1
        
        for x, cluster in enumerate(summary[0]):
            for image_id in summary[0][cluster]:
                img = Image.open('U:/staff-umbrella/imagesummary/data/Delft_NL/imagedb/' + images.loc[(images['img_id_com'] == int(image_id)), 'path'].iloc[0])
                img = ImageOps.expand(img, border=50, fill=colors[x])
                axs[math.floor(index/width)][index % width].imshow(img)
                axs[math.floor(index/width)][index % width].axis('off')
                index +=1
        
        plt.show()
        
                
        



# class VisualizationADDMETHODNAME(VisualizationParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "ADD METHOD NAME"

#     def run(self, **kwargs):
#         pass
