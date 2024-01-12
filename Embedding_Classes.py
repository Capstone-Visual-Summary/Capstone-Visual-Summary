from typing import Union
from Parent import GrandParent


class EmbeddingParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, Embedding_Parent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class EmbeddingADDMETHODNAME(EmbeddingParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def run(self):
        pass
