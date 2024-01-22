import geopandas as gpd

wijken_en_buurten = gpd.read_file('buurten_2023_v1/buurten_2023_v1.shp')

# filter op gemeente delft
wijken_en_buurten = wijken_en_buurten[wijken_en_buurten['GM_NAAM'] == 'Delft']

# filter op de kolommen die we nodig hebben
wijken_en_buurten = wijken_en_buurten[['BU_NAAM', 'geometry']]

# verander het coordinaten systeem
wijken_en_buurten = wijken_en_buurten.to_crs(epsg=4326)

# Export the GeoDataFrame as a GeoJSON file
wijken_en_buurten.to_file("Geo-JSON Files\wijken_en_buurten.geojson", driver='GeoJSON')

print(wijken_en_buurten.head())