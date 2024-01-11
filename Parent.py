import warnings


class Parent:
    def __init__(self) -> None:
        pass

    def update_children(self) -> None:
        """Updates the list of children to the parent class, making sure that they are no duplicates"""

        for subclass in self.__class__.__subclasses__():
            instance = subclass()

            version_num = str(instance.version)  # type: ignore

            if instance.name not in self.children_names:  # type: ignore
                self.children_names.add(instance.name)  # type: ignore

                if version_num in self.children:  # type: ignore
                    related_children = [
                        float(version)
                        for version in self.get_related_children(
                            related_version=version_num[0]
                        )
                        if "WIP" not in version
                    ]
                    version_num = str(round(max(related_children) + 0.1, 1))

                    message_warning = (
                        f"Duplicate version number found, system has automatically assigned {version_num} as the new version number. "
                        + f'Please update "{instance.name}" with default version {instance.version} with a new number to avoid this warning.' # type: ignore
                    )  

                    warnings.warn(message_warning, UserWarning)

                self.children[version_num] = {"Method Name": instance.name, "Instance Object": instance}  # type: ignore
            elif version_num not in self.children:  # type: ignore
                self.children[version_num] = {"Method Name": instance.name, "Instance Object": instance}  # type: ignore
            elif len(self.__class__.__subclasses__()) != len(self.children):  # type: ignore
                raise ValueError(f'Duplicate Name and Version number. Combination of version = {version_num} and name = "{instance.name}" is already used. Please change one or the other or both.')  # type: ignore

    def get_related_children(self, related_version=1) -> list[str]:
        """Returns the version numbers that are part of the same major version in STRING, i.e. returns all 1.X"""

        related_versions = []

        for version in self.children:  # type: ignore
            if version.startswith(str(related_version)):
                related_versions.append(version)

        return related_versions

    def run(self, version=-1.0):
        self.update_children()

        if str(version) in self.children:
            self.children[str(version)]["Instance Object"].run()
        else:
            latest_version = max(
                float(ver) for ver in self.children if "WIP" not in ver
            )
            self.children[str(latest_version)]["Instance Object"].run()

    def __str__(self) -> str:
        if len(self.children) < 1:  # type: ignore
            return f"There are no {self.type} Methods!"  # type: ignore

        output_str = f"{self.type} Methods:\n"  # type: ignore

        for key, value in self.children.items():  # type: ignore
            output_str += f"Method Name: {value['Method Name']}, Version: {key}\n"

        output_str += "------------------------------------------"

        return output_str
