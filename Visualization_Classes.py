from typing import Union
from Grand_Parent import GrandParent


class VisualizationParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Visulization"
        self.children: dict[str, dict[str, Union[str, VisualizationParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class VisualizationADDMETHODNAME(VisualizationParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def run(self):
        pass
