import os
import pandas as pd
import geopandas as gpd
from pyincore import DataService, IncoreClient, Dataset

client = IncoreClient()
datasvc = DataService(client)

# reading in Nathanael's building inventory
building_inventory_pop_id = '5d5433edb9219c0689b98344'
dataset = Dataset.from_data_service(building_inventory_pop_id, datasvc)
df = dataset.get_dataframe_from_csv()


# reading in dataframe
seaside_building_shapefile_id = '5df40388b9219c06cf8b0c80'
dataset = Dataset.from_data_service(seaside_building_shapefile_id, datasvc)
gdf = dataset.get_dataframe_from_shapefile()

# merging Nathanael's building inventory CSV with OSU's shapefile
gdf_new = pd.merge(gdf, df[['strctid', 'guid']], how='left', left_on='guid', right_on='guid')
print(len(df), len(gdf), len(gdf_new))	# checking lengths

# writing to shapefile
file_new = os.path.join(os.getcwd(), 'seaside_bldg.shp')
gdf_new.to_file(file_new)