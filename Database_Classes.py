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

    def run(self, **kwargs)-> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        version = kwargs['database_version'] if 'database_version' in kwargs else -1
        
        return super().run(version, **kwargs) # type: ignore
    
    def neighbourhood_pairs(self, images_file, neighbourhoods_info_file, neighbourhood_pairs_file) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]:
        images = gpd.read_file(images_file)
        neighbourhoods = gpd.read_file(neighbourhoods_info_file)

        assigned_neighbourhoods: dict[str, set[int]] = dict()

        try:
            with open(neighbourhood_pairs_file) as json_file:
                assigned_neighbourhoods = json.load(json_file)

            return assigned_neighbourhoods, images, neighbourhoods # type: ignore
        except:
            pass
        
        missing_panos = [22607, 22626, 22630, 25076, 25996, 26001, 26018]

        filtered_images = images.loc[~images['img_id'].isin(missing_panos)]

        result = gpd.sjoin(filtered_images, neighbourhoods, how='left', predicate='within')

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
        images_file = 'Geo-JSON Files/image_info.geojson'
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v1_0.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v1_0.json'

        return self.neighbourhood_pairs(images_file, neighbourhoods_info_file, neighbourhood_pairs_file)


class DatabaseUberHexSize7(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.0
        self.name: str = "Uber Hex Size 7"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        images_file = 'Geo-JSON Files/image_info.geojson'
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_0.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_0.json'
        
        return self.neighbourhood_pairs(images_file, neighbourhoods_info_file, neighbourhood_pairs_file)
    

class DatabaseUberHexSize8(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.1
        self.name: str = "Uber Hex Size 8"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        images_file = 'Geo-JSON Files/image_info.geojson'
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_1.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_1.json'
        
        return self.neighbourhood_pairs(images_file, neighbourhoods_info_file, neighbourhood_pairs_file)


class DatabaseUberHexSize9(DatabaseParent):
    def __init__(self) -> None:
        self.version: float | str = 2.2
        self.name: str = "Uber Hex Size 9"

    def run(self, **kwargs) -> tuple[dict[str, list[int]], GeoDataFrame, GeoDataFrame]: # type: ignore
        images_file = 'Geo-JSON Files/image_info.geojson'
        neighbourhoods_info_file = 'Geo-JSON Files/neighbourhood_info_v2_2.geojson'
        neighbourhood_pairs_file = 'Geo-JSON Files/neighbourhoods_v2_2.json'
        
        return self.neighbourhood_pairs(images_file, neighbourhoods_info_file, neighbourhood_pairs_file)