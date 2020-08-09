import os
import ipyleaflet as ipylft
import ipywidgets as ipywgt
import pandas as pd
import geopandas as gpd
from pyincore import BaseAnalysis
import json
from branca.colormap import linear


class map_creation_backend(BaseAnalysis):
	def run(self):
		# --- reading and organzing data
		# reading in dataset from pyincore server
		bldg_data_set = self.get_input_dataset("buildings").get_inventory_reader()
		
		# converting from fiona to geopandas
		bldg_data_df = gpd.GeoDataFrame.from_features([feature for feature in bldg_data_set], crs='EPSG3857')

		# Get the order of the fields in the Fiona Collection; add geometry to the end
		columns = list(bldg_data_set.meta["schema"]["properties"]) + ["geometry"]
		
		# Re-order columns in the correct order
		bldg_data_df = bldg_data_df[columns]
		
		# removing empty buildings
		bldg_data_df = self.remove_null_buildings(bldg_data_df)

		data = pd.read_csv(self.get_parameter("result_file"), dtype=str)
		col_consider = list(data.columns)
		col_consider.remove('guid')
		data[col_consider] = data[col_consider].astype(float)

		# --- creating map
		self.m = self.create_basemap_ipylft(bldg_data_df)
		self.bldg_data_df = self.Merge_BldgPolygon_and_Data(data, bldg_data_df)
		self.bldg_data_json = json.loads(self.bldg_data_df.to_json())
		self.create_map_widgets(self.m, col_consider)

		m = self.m
		return m

	def remove_null_buildings(self, bldg_data_df):
		bldg_data_df.dropna(subset=['guid'], inplace=True)
		return bldg_data_df

	def create_basemap_ipylft(self, geo_dataframe):
		ext = geo_dataframe.total_bounds
		cen_x, cen_y = (ext[1]+ext[3])/2, (ext[0]+ext[2])/2
		m = ipylft.Map(center=(cen_x, cen_y), 
					   zoom=12, 
					   basemap=ipylft.basemaps.Stamen.Toner, 
					   # crs='EPSG3857', 
					   scroll_wheel_zoom=True)
		return m

	# def load_all_data(self, outfile, column_name):
	# 	return pd.read_csv(outfile, dtype = str)

	def Merge_BldgPolygon_and_Data(self, data, bldg_data_df):
		bldg_data_df = bldg_data_df.merge(data, on='guid')
		return bldg_data_df

	def create_map_widgets(self, m, col_consider):
		self.dropdown1 = ipywgt.Dropdown(description='Plot Option', 
										 options=col_consider, 
										 width=500)
		file_control1 = ipylft.WidgetControl(widget=self.dropdown1, 
											 position='bottomleft')

		button = ipywgt.Button(description='Generate Map', button_style = 'info')
		button.on_click(self.on_button_clicked)
		generatemap_control = ipylft.WidgetControl(widget=button, position='bottomleft')

		m.add_control(ipylft.LayersControl(position='topright', style = 'info'))
		m.add_control(ipylft.FullScreenControl(position='topright'))
		m.add_control(generatemap_control)
		# m.add_control(file_control2)
		m.add_control(file_control1)


	def on_button_clicked(self, b):
		print('Loading: ', self.dropdown1.value)
		key = self.dropdown1.value
		self.create_choropleth_layer(key)
		print('\n')

	def create_choropleth_layer(self, key):
		vmax_val = 1
		temp_id = list(range(len(self.bldg_data_df['guid'])))
		temp_id = [str(i) for i in temp_id]
		choro_data = dict(zip(temp_id, self.bldg_data_df[key]))
		layer = ipylft.Choropleth(geo_data=self.bldg_data_json, 
								  choro_data=choro_data, 
								  colormap=linear.RdBu_06, 
								  value_min=0, 
								  value_max=vmax_val, 
								  border_color='black', 
								  style={'fillOpacity': 0.8, 'weight':0.5}, 
								  name='parcels')
		self.m.add_layer(layer)
		print('done')


	def create_legend(self):
		legend=linear.YlOrRd_04.scale(0,self.vmax_val)
		self.m.colormap=legend
		out = ipywgt.Output(layout={'border': '1px solid black'})
		with out:
			display(legend)
		widget_control = ipylft.WidgetControl(widget=out, position='topright')
		m.add_control(widget_control)
		display(m)


	def get_spec(self):
		"""Get specifications of the damage analysis.

		Returns:
			obj: A JSON object of specifications of the building recovery time analysis.

		"""
		return {
			'name': 'map_creation',
			'description': 'map viewer for infrastructure damage and connectivity',
			'input_parameters': [
				{
					'id': 'result_file',
					'required': True,
					'description': 'path to output data',
					'type': str
				},
				{
					'id': 'string_to_num_dict',
					'required': True,
					'description': 'dictionary of damage ratios',
					'type': dict
				},
				{
					'id': 'column_name',
					'required': True,
					'description': 'column name in output to create choropleth on',
					'type': str
				}
	
			],
			'input_datasets': [
				{
					'id': 'buildings',
					'required': True,
					'description': 'Building Inventory',
					'type': ['ergo:buildingInventoryVer4', 'ergo:buildingInventoryVer5', 'incore:buriedPipelineTopology']
				}
		
			],
			'output_datasets': [

			]
		}
