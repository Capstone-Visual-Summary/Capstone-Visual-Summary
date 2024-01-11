from typing import Union
from Parent import Parent


class Summerization_Parent(Parent):
    def __init__(self) -> None:
        self.type = "Summerization"
        self.children: dict[str, dict[str, Union[str, Summerization_Parent]]] = dict()
        self.children_names: set[int] = set()

    def import_data(self):
        pass

    def export_data(self):
        pass


class Summerization_ADD_METHOD_NAME(Summerization_Parent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def run(self):
        pass
