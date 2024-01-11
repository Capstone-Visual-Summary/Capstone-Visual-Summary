from typing import Union
from Parent import GrandParent
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame


class DatabaseParent(GrandParent):
    # def __init__(self, images, neighbourhoods) -> None:
    def __init__(self) -> None:
        self.type = "Database"
        self.children: dict[str, dict[str, Union[str, DatabaseParent]]] = dict()
        self.children_names: set[int] = set()
        # self.images = images
        # self.neighbourhoods = neighbourhoods

    def run(self, version = -1, **kwargs):
        return super().run(version, **kwargs)
    # def import_data(self, images, neighbourhoods):
    def import_data(self):
        pass
        # self.images = images
        # self.neighbourhoods = neighbourhoods

    def export_data(self):
        pass
        


class DatabaseGeopandasPolygons(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "Geopandas Polygons"

    def run(self) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        images = gpd.read_file('image_info.geojson')
        neighbourhoods = gpd.read_file('neighbourhood_info.geojson')
        
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

    
# class DatabaseGeopandasPolygonsd(DatabaseParent):
#     def __init__(self) -> None:
#         super().__init__('h', 'w')
#         self.version: float | str = 1.1
#         self.name: str = "Geopandas Polygonsd"

#     def run(self) -> dict[str, list[int]]: # type: ignore
#         print(super().images)
#         return('hsdfdsfsdgf')
        # assigned_neighbourhoods: dict[str, list[int]] = dict()

        # for img_index, image in tqdm(super().images.iterrows(), total=len(super().images)):
        #     for neighbourhood_index, neighbourhood in super().neighbourhoods.iterrows():
        #         if neighbourhood not in assigned_neighbourhoods:
        #             assigned_neighbourhoods[str(neighbourhood['neighbourhood_id'])] = []

        #         if image['geometry'].within(neighbourhood['geomertry']):
        #             assigned_neighbourhoods[str(neighbourhood['neighbourhood_id'])].append(image['img_id'])
                    
        #             break         

        # return assigned_neighbourhoods

        # return super().export_data()
