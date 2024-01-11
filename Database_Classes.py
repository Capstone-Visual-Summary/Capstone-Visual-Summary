from typing import Union
from Parent import GrandParent
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame


class DatabaseParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Database"
        self.children: dict[str, dict[str, Union[str, DatabaseParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)


class DatabaseGeopandasPolygons(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "Geopandas Polygons"

    def run(self) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        images = gpd.read_file('Geo-JSON Files/image_info.geojson')
        neighbourhoods = gpd.read_file('Geo-JSON Files/neighbourhood_info.geojson')
        
        assigned_neighbourhoods: dict[str, set[int]] = dict()

        result = gpd.sjoin(images, neighbourhoods, how='left', predicate='within')

        for index, row in tqdm(result.iterrows(), total=len(result)):
            if pd.isnull(result.at[index, 'neighbourhood_id']):
                continue

            neighbourhood_id = str(round(int(row['neighbourhood_id']), 0))

            if neighbourhood_id not in assigned_neighbourhoods:
                assigned_neighbourhoods[neighbourhood_id] = set()

            assigned_neighbourhoods[neighbourhood_id].add(int(row['img_id']))

        return {key: list(value) for key, value in assigned_neighbourhoods.items()}, images, neighbourhoods
