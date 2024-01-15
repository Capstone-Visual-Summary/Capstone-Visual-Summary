import geopandas as gpd # type: ignore
import pandas as pd
from pathlib import Path

data_path = Path('U:/staff-umbrella/imagesummary/data/Delft_NL')
print('hg')
panoids = gpd.read_file(data_path / 'panoids/panoids.geojson')

print('g')
print(panoids.head())