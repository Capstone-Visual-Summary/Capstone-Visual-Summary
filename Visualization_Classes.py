from typing import Union
from Grand_Parent import GrandParent

import math as math
import numpy as np
import matplotlib


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

        bottom = []
        for cluster in summary[0]:
            for image_id in summary[0][cluster]:
                bottom.append(str(image_id))

        width = math.floor(math.sqrt(len(bottom)))
        bottom = [bottom[i:i + width] for i in range(0, len(bottom), width)]        

        top = []
        for center in summary[1]:
            top.append(summary[1][center])
        
        pattern = [top, bottom]
        print(pattern)
        



# class VisualizationADDMETHODNAME(VisualizationParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "ADD METHOD NAME"

#     def run(self, **kwargs):
#         pass
