from typing import Union
from Parent import Parent


class Database_Parent(Parent):
    def __init__(self) -> None:
        self.type = "Database"
        self.children: dict[str, dict[str, Union[str, Database_Parent]]] = dict()
        self.children_names: set[int] = set()

    def import_data(self):
        pass

    def export_data(self):
        pass


class Database_ADD_METHOD_NAME(Database_Parent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "ADD METHOD NAME"

    def run(self):
        pass
