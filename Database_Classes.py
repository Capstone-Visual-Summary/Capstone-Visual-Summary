from typing import Union
from Grand_Parent import GrandParent
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

import json


class DatabaseParent(GrandParent):
    def __init__(self) -> None:
        self.type = "Database"
        self.children: dict[str, dict[str, Union[str, DatabaseParent]]] = dict()
        self.children_names: set[int] = set()

    def run(self, **kwargs):
        version = kwargs['database_version'] if 'database_version' in kwargs else -1
        
        return super().run(version, **kwargs)

# class DatabaseADDMETHODNAME(DatabaseParent):
#     def __init__(self) -> None:
#         self.version: float | str = 1.0
#         self.name: str = "ADD METHOD NAME"

#     def run(self, **kwargs):
#         pass


class DatabaseGeopandasPolygons(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 1.0
        self.name: str = "Geopandas Polygons"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v1_0.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v1_0.json'
        images = gpd.read_file('Geo-JSON Files/image_info.geojson')
        neighbourhoods = gpd.read_file(neighbourhoods_info_file)
        
        assigned_neighbourhoods: dict[str, set[int]] = dict()

        try:
            with open(neighbourhood_pairs_file) as json_file:
                assigned_neighbourhoods = json.load(json_file)

            return assigned_neighbourhoods, images, neighbourhoods # type: ignore
        except:
            pass

        result = gpd.sjoin(images, neighbourhoods, how='left', predicate='within')

        for index, row in tqdm(result.iterrows(), total=len(result)):
            if pd.isnull(result.at[index, 'neighbourhood_id']):
                continue

            neighbourhood_id = str(round(int(row['neighbourhood_id']), 0))

            if neighbourhood_id not in assigned_neighbourhoods:
                assigned_neighbourhoods[neighbourhood_id] = set()

            assigned_neighbourhoods[neighbourhood_id].add(int(row['img_id_com']))
        
        assigned_neighbourhoods_list = {key: list(value) for key, value in assigned_neighbourhoods.items()}

        with open(neighbourhood_pairs_file, 'w') as json_file:
            json.dump(assigned_neighbourhoods_list, json_file)

        return assigned_neighbourhoods_list, images, neighbourhoods


class DatabaseUberHexSize7(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "Uber Hex Size 7"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_0.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_0.json'
        images = gpd.read_file('Geo-JSON Files/image_info.geojson')
        neighbourhoods = gpd.read_file(neighbourhoods_info_file)
        
        assigned_neighbourhoods: dict[str, set[int]] = dict()

        try:
            with open(neighbourhood_pairs_file) as json_file:
                assigned_neighbourhoods = json.load(json_file)

            return assigned_neighbourhoods, images, neighbourhoods # type: ignore
        except:
            pass

        result = gpd.sjoin(images, neighbourhoods, how='left', predicate='within')

        for index, row in tqdm(result.iterrows(), total=len(result)):
            if pd.isnull(result.at[index, 'neighbourhood_id']):
                continue

            neighbourhood_id = str(round(int(row['neighbourhood_id']), 0))

            if neighbourhood_id not in assigned_neighbourhoods:
                assigned_neighbourhoods[neighbourhood_id] = set()

            assigned_neighbourhoods[neighbourhood_id].add(int(row['img_id_com']))
        
        assigned_neighbourhoods_list = {key: list(value) for key, value in assigned_neighbourhoods.items()}

        with open(neighbourhood_pairs_file, 'w') as json_file:
            json.dump(assigned_neighbourhoods_list, json_file)

        return assigned_neighbourhoods_list, images, neighbourhoods
    

class DatabaseUberHexSize8(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.1
        self.name: str = "Uber Hex Size 8"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_1.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_1.json'
        images = gpd.read_file('Geo-JSON Files/image_info.geojson')
        neighbourhoods = gpd.read_file(neighbourhoods_info_file)
        
        assigned_neighbourhoods: dict[str, set[int]] = dict()

        try:
            with open(neighbourhood_pairs_file) as json_file:
                assigned_neighbourhoods = json.load(json_file)

            return assigned_neighbourhoods, images, neighbourhoods # type: ignore
        except:
            pass

        result = gpd.sjoin(images, neighbourhoods, how='left', predicate='within')

        for index, row in tqdm(result.iterrows(), total=len(result)):
            if pd.isnull(result.at[index, 'neighbourhood_id']):
                continue

            neighbourhood_id = str(round(int(row['neighbourhood_id']), 0))

            if neighbourhood_id not in assigned_neighbourhoods:
                assigned_neighbourhoods[neighbourhood_id] = set()

            assigned_neighbourhoods[neighbourhood_id].add(int(row['img_id_com']))
        
        assigned_neighbourhoods_list = {key: list(value) for key, value in assigned_neighbourhoods.items()}

        with open(neighbourhood_pairs_file, 'w') as json_file:
            json.dump(assigned_neighbourhoods_list, json_file)

        return assigned_neighbourhoods_list, images, neighbourhoods


class DatabaseUberHexSize9(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.2
        self.name: str = "Uber Hex Size 9"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_2.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_2.json'
        images = gpd.read_file('Geo-JSON Files/image_info.geojson')
        neighbourhoods = gpd.read_file(neighbourhoods_info_file)
        
        assigned_neighbourhoods: dict[str, set[int]] = dict()

        try:
            with open(neighbourhood_pairs_file) as json_file:
                assigned_neighbourhoods = json.load(json_file)

            return assigned_neighbourhoods, images, neighbourhoods # type: ignore
        except:
            pass

        result = gpd.sjoin(images, neighbourhoods, how='left', predicate='within')

        for index, row in tqdm(result.iterrows(), total=len(result)):
            if pd.isnull(result.at[index, 'neighbourhood_id']):
                continue

            neighbourhood_id = str(round(int(row['neighbourhood_id']), 0))

            if neighbourhood_id not in assigned_neighbourhoods:
                assigned_neighbourhoods[neighbourhood_id] = set()

            assigned_neighbourhoods[neighbourhood_id].add(int(row['img_id_com']))
        
        assigned_neighbourhoods_list = {key: list(value) for key, value in assigned_neighbourhoods.items()}

        with open(neighbourhood_pairs_file, 'w') as json_file:
            json.dump(assigned_neighbourhoods_list, json_file)

        return assigned_neighbourhoods_list, images, neighbourhoods