from typing import Union
from Parent import Parent


class Embedding_Parent(Parent):
    def __init__(self) -> None:
        self.type = "Embedding"
        self.children: dict[str, dict[str, Union[str, Embedding_Parent]]] = dict()
        self.children_names: set[int] = set()

    def import_data(self):
        pass

    def export_data(self):
        pass


class Embedding_ADD_METHOD_NAME(Embedding_Parent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def run(self):
        pass
