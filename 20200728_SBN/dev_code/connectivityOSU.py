# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import sys
import pandas as pd
import numpy as np
import networkx as nx
# from itertools import repeat

from pyincore import BaseAnalysis


class GraphFuncs(BaseAnalysis):
	""" general connectivity functions:
		used in each of the below analyses"""
	def __init__(self):
		pass

	def create_graph(self, network_links, network_nodes=None, inf=None):
		"""converts network defined as shapefile to a networkx graph."""
		# node list 1
		fromnode = [int(network_links[i]['properties']['fromnode']) for 
						i in range(len(network_links))]
		
		# node list 2
		tonode = [int(network_links[i]['properties']['tonode']) for 
						i in range(len(network_links))]

		if inf == 'epn':
			# network Link_ID (for EPN)
			edge_id_list = [network_links[i]['properties']['Link_ID'] for 
						i in range(len(network_links))]	
	
		else:
			# network linknwid (for transportation/water)
			edge_id_list = [int(network_links[i]['properties']['linknwid']) for 
						i in range(len(network_links))]	
	
		guid = [str(network_links[i]['properties']['guid']) for 
						i in range(len(network_links))]		# network guid

		# zipping node lists (e.g. 1st node in "fromnode" connects to 1st node 
		#	in "tonode")
		nodelist = list(zip(fromnode, tonode))
		# setting up dictionary for that relates edge id's to node pairs
		edge_id = {}
		for i, node_set in enumerate(nodelist):
			edge_id[node_set] = guid[i]
		
		G = nx.Graph()				# creating empty graph
		G.add_edges_from(nodelist)	# creating graph from the nodelist
		nx.set_edge_attributes(G, edge_id, 'edge_guid')	# adding edge guid's to graph
		
		# relational table: edge_id, edge_guid, from node, to node
		guid_to_linknwid = pd.DataFrame()
		guid_to_linknwid['edge_id'] = edge_id_list
		guid_to_linknwid['edge_guid'] = guid
		guid_to_linknwid['fromnode'] = fromnode
		guid_to_linknwid['tonode'] = tonode

		# setting up node attributes (e.g. relating node id's with node_guid's in graph)
		if network_nodes is not None:
			nodenwid = [network_nodes[i]['properties']['nodenwid'] 
							for i in range(len(network_nodes))]
			node_guid = [network_nodes[i]['properties']['guid'] 
							for i in range(len(network_nodes))]
			node_id = {}
			for i in range(len(network_nodes)):
				node_id[nodenwid[i]] = node_guid[i]
			# assigning node guid's to graph
			nx.set_node_attributes(G, node_id, 'node_guid')
		
		return G, guid_to_linknwid


	def remove_edges(self, graph, guid_remove):
		""" removing edges from graph """
		ebunch = []
		# looping through edges
		for e in graph.edges():
			# if edge id in damaged/removed edges
			if graph[e[0]][e[1]]['edge_guid'] in guid_remove:
				ebunch.append(e)		# adding to list of edges to remove
		graph.remove_edges_from(ebunch)	# removing edges

		return graph

	def remove_nodes(self, graph, guid_remove):
		""" removing nodes from graph """
		nbunch = []
		for n in graph.nodes():
			if graph.nodes[n]['node_guid'] in guid_remove:
				nbunch.append(n)
		graph.remove_nodes_from(nbunch)
		
		return graph

	def find_in_list_of_list(self, mylist, char):
		for sub_list in mylist:
			if char in sub_list:
				return mylist.index(sub_list)

	def print_percent_complete(self, msg, i, n_i):
		i, n_i = int(i)+1, int(n_i)
		sys.stdout.write('\r')
		sys.stdout.write("{} ({:.1f}%)" .format(msg, (100/(n_i)*i)))
		sys.stdout.flush()
		if i==n_i:
			print()


class TransConnectivity(BaseAnalysis):
	def run(self):
		"""performs connectivity analysis"""
		self.prnt_msg = self.get_parameter('prnt_msg')

		# loading building data
		bldg_data_df = self.get_input_dataset('buildings')
		bldg_data_df = pd.DataFrame(bldg_data_df.get_inventory_reader())
		bldg_data_df = pd.DataFrame(list(bldg_data_df['properties']))
		bldg_data_df = bldg_data_df[['guid']]

		# table that links buildings to network
		bldg2netwrk_df = self.get_input_dataset('building_to_network')
		bldg2netwrk_df = pd.DataFrame(bldg2netwrk_df.get_inventory_reader())
		bldg2netwrk_df = pd.DataFrame(list(bldg2netwrk_df['properties']))
		bldg2netwrk_df['fromnode'] = bldg2netwrk_df['fromnode'].astype(int)
		bldg2netwrk_df['tonode'] = bldg2netwrk_df['tonode'].astype(int)

		road_dset = self.get_input_dataset('road_dataset')
		road_dset = road_dset.get_inventory_reader()

		road_dmg = self.get_input_dataset('road_dmg')
		road_dmg = road_dmg.get_dataframe_from_csv()
		road_dmg.set_index('guid', inplace=True)
		n_sims = len([i for i in road_dmg.keys() if "iter" in i])

		bridge_dset = self.get_input_dataset('bridge_dataset')
		bridge_dset = pd.DataFrame(bridge_dset.get_inventory_reader())
		bridge_dset = pd.DataFrame(list(bridge_dset['properties']))

		bridge_dmg = self.get_input_dataset('bridge_dmg')
		bridge_dmg = bridge_dmg.get_dataframe_from_csv()
		bridge_dmg.set_index('guid', inplace=True)

		remove_road_DS = self.get_parameter("remove_road_DS") # DS to remove road
		remove_bridge_DS = self.get_parameter("remove_bridge_DS") # DS to remove bridges

		road_reptime_log_med = self.get_parameter('roadway_reptime_log_med')
		road_reptime_covm = self.get_parameter('roadway_reptime_covm')
		bridge_reptime_log_med = self.get_parameter('bridge_reptime_log_med')
		bridge_reptime_covm = self.get_parameter('bridge_reptime_covm')

		# --- BEGIN ANALYSIS
		self.GF = GraphFuncs()
		graph, guid_to_linknwid = self.GF.create_graph(network_links=road_dset)

		# linking buildings to network; each building is assigned a nearby edge/road
		bldg_data_df = self.link_buildings_to_trans_network(bldg_data_df, bldg2netwrk_df)

		func, rep = self.trans_connectivity_analysis(
								graph=graph, 
								bldgs=bldg_data_df,
								road_dmg=road_dmg,
								bridge_dmg=bridge_dmg,
								bridge_df=bridge_dset,
								remove_road_DS=remove_road_DS,
								remove_bridge_DS=remove_bridge_DS,
								link_rep_logmed=road_reptime_log_med,
								link_rep_covm=road_reptime_covm,
								bridge_rep_logmed=bridge_reptime_log_med,
								bridge_rep_covm=bridge_reptime_covm,
								guid_to_linknwid=guid_to_linknwid)
		return func, rep


	def trans_connectivity_analysis(self, graph=None, bldgs=None, road_dmg=None,
						bridge_dmg=None, bridge_df=None, remove_road_DS=None, 
						remove_bridge_DS=None, link_rep_logmed=None, 
						link_rep_covm=None,	bridge_rep_logmed=None, 
						bridge_rep_covm=None, guid_to_linknwid=None):

		func_df = pd.DataFrame(index=bldgs.index)
		reptime_df = pd.DataFrame(index=bldgs.index)
		n_samples = list(road_dmg.keys())
		n_samples = len([i for i in n_samples if 'iter' in i])
		count = 0
		for _, key in enumerate(road_dmg.keys()):
			if 'iter' in key:
				self.GF.print_percent_complete(self.prnt_msg, count, n_samples)
				bldgs['connected'] = True 	# assuming everything is connected
				bldgs['time_conn'] = 0 # nan until becoming reconnected

				road_info_df = self.damage_trans_network( 
									road_dmg=road_dmg[key].astype(float), 
									bridge_dmg=bridge_dmg[key].astype(float),
									guid_to_linknwid=guid_to_linknwid,
									bridge_df=bridge_df,
									link_rep_logmed=link_rep_logmed,
									link_rep_covm=link_rep_covm,
									bridge_rep_logmed=bridge_rep_logmed,
									bridge_rep_covm=bridge_rep_covm,
									remove_road_DS=remove_road_DS,
									remove_bridge_DS=remove_bridge_DS)

				bldgs = self.trans_conn_crit_nodes(bldgs=bldgs,
									link_info_df=road_info_df,
									graph=graph.copy(),
									time_step=0)

				func_df[key] = list(bldgs['func'])

				unique_reptime = np.unique(road_info_df['reptime'])
				unique_reptime = unique_reptime[unique_reptime>0]
				for t in unique_reptime:
					if bldgs['connected'].sum()/len(bldgs) == 1.0:
						break
					road_info_df.loc[road_info_df['reptime']<=t, 'remove_TF'] = False
					bldgs = self.trans_conn_crit_nodes(bldgs=bldgs,
									link_info_df=road_info_df,
									graph=graph.copy(),
									time_step=t)
				reptime_df[key] = bldgs['time_conn']
				count += 1

		return func_df, reptime_df

	def damage_trans_network(self, road_dmg=None, bridge_dmg=None, 
					guid_to_linknwid=None, bridge_df=None,
					link_rep_logmed=None, link_rep_covm=None,
					bridge_rep_logmed=None, bridge_rep_covm=None,
					remove_road_DS=None, remove_bridge_DS=None):
		remove_df = pd.DataFrame(index=road_dmg.index)
		remove_df['fromnode'] = list(guid_to_linknwid['fromnode'])
		remove_df['tonode'] = list(guid_to_linknwid['tonode'])
		remove_df['edge_guid'] = list(guid_to_linknwid['edge_guid'])
		remove_df['edge_id'] = list(guid_to_linknwid['edge_id'])

		remove_df['bridge_guid'] = False
		remove_df['remove_TF'] = False
		remove_df['reptime'] = 0.0
		# reptime calcs for iteration
		n_roads = len(road_dmg)
		mvn = np.random.multivariate_normal(link_rep_logmed,
											link_rep_covm,
											n_roads)
		
		reptime = np.column_stack((np.zeros(n_roads),np.exp(mvn)))
		rep_roads = np.array([reptime[int(i), int(obj)] for 
							i, obj in enumerate(road_dmg)])

		remove_df['remove_TF'] = list(road_dmg >= remove_road_DS)
		remove_df['reptime'] = np.ceil(rep_roads)

		""" if bridges are present, overriding link results """
		if bridge_dmg is not None:	# e.g. bridge damage
			remove_TF = bridge_dmg >= remove_bridge_DS

			# reptime calcs for bridges
			n_bridges = len(bridge_dmg)
			mvn = np.random.multivariate_normal(bridge_rep_logmed,
												bridge_rep_covm,
												n_bridges)
			reptime = np.column_stack((np.zeros(n_bridges),np.exp(mvn)))
			rep_bridges = np.array([reptime[int(i), int(obj)] for 
								i, obj in enumerate(bridge_dmg)])
			rep_bridges = np.ceil(rep_bridges)

			for i, guid in enumerate(bridge_df['guid']):
				bridge_idx = remove_df['edge_id']==bridge_df['Link_ID'][i]
				remove_df.loc[bridge_idx, 'bridge_guid'] = guid
				remove_df.loc[bridge_idx, 'remove_TF'] = remove_TF[i]
				remove_df.loc[bridge_idx, 'reptime'] = rep_bridges[i]

		return remove_df

	def link_buildings_to_trans_network(self, bldgs, bldg_to_network):
		""" links buildings to edges in network. there are three pieces:
				-building guid
				-edge linknwid
				-edge guid

			bldg_to_network: links bldg_guid and edge_linknwid
			guid_to_linknwid: links edge_linknwid to edge_guid
		"""
		bldgs = pd.merge(bldgs, bldg_to_network, how='outer', left_on='guid', right_on='bldg_guid')
		bldgs.set_index('guid', inplace=True)
		bldgs = bldgs[['edge_id', 'edge_guid', 'fromnode', 'tonode']]
		return bldgs
	
	def trans_conn_crit_nodes(self, bldgs=None, graph=None, link_info_df=None, 
							  time_step=None):
		""" 
			- removes edges from network and performs connectivity analysis
			- the graph passed in to this function is the original 
			  graph and not the graph at t = t_2, t_3, ...
		"""
		edge_guid_remove = list(link_info_df.loc[
									link_info_df['remove_TF'] == True].index)
		n_nodes = graph.number_of_nodes()
		graph = self.GF.remove_edges(graph, edge_guid_remove)

		"""
		determining bin index for each node
		example:
			bins = [[4],[2],[0,1,3,5,6]]					
				nodes 4 and 2 are by themselves, 
				nodes 0, 1, 3, 5, 6 are connected in a cluster
		"""

		bins = list(nx.connected_components(graph))
		fromnodes = bldgs['fromnode'].values
		tonodes = bldgs['tonode'].values
		func = []
		for guid_i, guid in enumerate(bldgs.index.values.tolist()):
			node_bin = self.GF.find_in_list_of_list(bins, fromnodes[guid_i])
			func_temp = len(bins[node_bin])/n_nodes
			func.append(func_temp)
		func = np.array(func)
		conn_tf = func>=0.9

		bldgs['conn_org'] = bldgs['connected']
		bldgs['conn_new'] = conn_tf
		bldgs.loc[bldgs['conn_new']!=bldgs['conn_org'], 'time_conn'] = time_step
		bldgs['connected'] = bldgs['conn_new']
		bldgs['func'] = func


		return bldgs



	def get_spec(self):
		"""Get specifications of the damage analysis.

		Returns:
			obj: A JSON object of specifications of the building recovery time analysis.

		"""
		return {
			'name': 'connectivity-analysis',
			'description': 'performs connectivity analysis of buildings',
			'input_parameters': [
				{
					'id': 'remove_road_DS',
					'required': True,
					'description': 'Damage state above which to remove road from graph',
					'type': int
				},
				{
					'id': 'remove_bridge_DS',
					'required': True,
					'description': 'Damage state above which to remove bridge from graph',
					'type': int
				},

				{
					'id': 'roadway_reptime_log_med',
					'required': True,
					'description': 'road reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'roadway_reptime_covm',
					'required': True,
					'description': 'road reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'bridge_reptime_log_med',
					'required': True,
					'description': 'bridge reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'bridge_reptime_covm',
					'required': True,
					'description': 'bridge reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'prnt_msg',
					'required': True,
					'description': 'Message to print out',
					'type': str
				},

			],

			'input_datasets': [

				{
					'id': 'road_dataset',
					'required': True,
					'description': 'Data inventory of bridges',
					'type': ['ergo:roadLinkTopo', 'incore:roads',],
				},
				{
					'id': 'road_dmg',
					'required': True,
					'description': 'Damage inventory of bridges',
					'type': ['ergo:LinkDamageInventory'],
				},
				{
					'id': 'bridge_dataset',
					'required': True,
					'description': 'Data inventory of bridges',
					'type': ['ergo:NearEdgeDataInventory', 'ergo:bridges'],
				},
				{
					'id': 'bridge_dmg',
					'required': True,
					'description': 'Damage inventory of bridges',
					'type': ['ergo:NearEdgeDamageInventory'],
				},
				{
					'id': 'buildings',
					'required': True,
					'description': 'Building Inventory',
					'type': ['ergo:buildingInventoryVer4', 'ergo:buildingInventoryVer5'],
				},				
				{
					'id': 'building_to_network',
					'required': True,
					'description': 'Table that links buildings to network',
					'type': ['incore:buildingsToTransportationNetwork'],
				},
			],
			'output_datasets': [
				{
					'id': 'connectivity-result',
					'parent_type': 'networks',
					'description': 'CSV file of connectivity results',
					'type': 'output'
				}

			]
		}



class EPNConnectivity(BaseAnalysis):

	def run(self):
		"""performs connectivity analysis"""
		self.prnt_msg = self.get_parameter('prnt_msg')
		critical_nodes = self.get_parameter('critical_nodes')

		# loading building data
		bldg_data_df = self.get_input_dataset('buildings')
		bldg_data_df = pd.DataFrame(bldg_data_df.get_inventory_reader())
		bldg_data_df = pd.DataFrame(list(bldg_data_df['properties']))
		bldg_data_df = bldg_data_df[['guid']]

		# table that links buildings to network
		bldg2netwrk_df = self.get_input_dataset('building_to_network')
		bldg2netwrk_df = pd.DataFrame(bldg2netwrk_df.get_inventory_reader())
		bldg2netwrk_df = pd.DataFrame(list(bldg2netwrk_df['properties']))	

		pole_dset = self.get_input_dataset('pole_dataset')
		pole_dset = pole_dset.get_inventory_reader()
		pole_df = pd.DataFrame(pole_dset)
		pole_df = pd.DataFrame(list(pole_df['properties']))
		
		pole_dmg = self.get_input_dataset('pole_dmg')
		pole_dmg = pole_dmg.get_dataframe_from_csv()
		pole_dmg = pd.merge(pole_dmg, pole_df[['guid', 'Pole_Y_N']], how='outer', on='guid')
		pole_dmg.set_index('guid', inplace=True)

		n_sims = len([i for i in pole_dmg.keys() if "iter" in i])

		line_dset = self.get_input_dataset('line_dataset')
		line_dset = line_dset.get_inventory_reader()

		remove_pole_DS = self.get_parameter("remove_pole_DS") # DS to remove poles

		pole_reptime_log_med = self.get_parameter('pole_reptime_log_med')
		pole_reptime_covm = self.get_parameter('pole_reptime_covm')
		
		# --- BEGIN ANALYSIS
		self.GF = GraphFuncs()
		graph, _ = self.GF.create_graph(network_links=line_dset, inf='epn', 
										network_nodes = pole_dset)

		# linking buildings to network
		bldg_data_df = self.link_buildings_to_elec_network(bldg_data_df, bldg2netwrk_df)

		func, rep = self.elec_connectivity_analysis(
								graph=graph, 
								bldgs=bldg_data_df,
								pole_dmg=pole_dmg,
								remove_pole_DS=remove_pole_DS,
								pole_rep_logmed=pole_reptime_log_med,
								pole_rep_covm=pole_reptime_covm,
								critical_nodes=critical_nodes)
		return func, rep


	def elec_connectivity_analysis(self, graph=None, bldgs=None, pole_dmg=None,
							remove_pole_DS=None, pole_rep_logmed=None,
							pole_rep_covm=None, critical_nodes=None):
							

		func_df = pd.DataFrame(index=bldgs.index)
		reptime_df = pd.DataFrame(index=bldgs.index)
		n_samples = list(pole_dmg.keys())
		n_samples = len([i for i in n_samples if 'iter' in i])
		count = 0
		for _, key in enumerate(pole_dmg.keys()):
			if 'iter' in key:
				self.GF.print_percent_complete(self.prnt_msg, count, n_samples)
				bldgs['connected'] = True 	# assuming everything is connected
				bldgs['time_conn'] = 0 		# 0 until becoming reconnected

				pole_info_df = self.damage_elec_network( 
									pole_dmg=pole_dmg[[key, 'Pole_Y_N']].astype(float), 
									pole_rep_logmed=pole_rep_logmed,
									pole_rep_covm=pole_rep_covm,
									remove_pole_DS=remove_pole_DS,
									key=key)
				bldgs = self.elec_conn_crit_nodes(bldgs=bldgs,
									node_info_df=pole_info_df,
									graph=graph.copy(),
									critical_nodes=critical_nodes,
									time_step=0)

				func_df[key] = list(bldgs['connected'].astype(int))

				unique_reptime = np.unique(pole_info_df['reptime'])
				unique_reptime = unique_reptime[unique_reptime>0]
				for t in unique_reptime:
					if bldgs['connected'].sum()/len(bldgs) == 1.0:
						break
					pole_info_df.loc[pole_info_df['reptime']<=t, 'remove_TF'] = False
					bldgs = self.elec_conn_crit_nodes(bldgs=bldgs,
									node_info_df=pole_info_df,
									graph=graph.copy(),
									critical_nodes=critical_nodes,
									time_step=t)
				reptime_df[key] = bldgs['time_conn']
				count += 1

		return func_df, reptime_df

	def damage_elec_network(self, pole_dmg, pole_rep_logmed, pole_rep_covm, 
						remove_pole_DS, key):

		remove_df = pd.DataFrame(index=pole_dmg.index)
		remove_df['Pole_Y_N'] = pole_dmg['Pole_Y_N']
		remove_df['remove_TF'] = False
		remove_df['reptime'] = 0.0

		# reptime calcs for iteration
		n_poles = len(pole_dmg)
		mvn = np.random.multivariate_normal(pole_rep_logmed,
											pole_rep_covm,
											n_poles)
		
		reptime = np.column_stack((np.zeros(n_poles),np.exp(mvn)))
		rep_poles = np.array([reptime[int(i), int(obj)] for 
							i, obj in enumerate(pole_dmg[key])])

		remove_df['remove_TF'] = list(pole_dmg[key] >= remove_pole_DS)
		remove_df['reptime'] = np.ceil(rep_poles)
		remove_df.loc[remove_df['Pole_Y_N']==0, 'remove_TF'] = False
		remove_df.loc[remove_df['Pole_Y_N']==0, 'reptime'] = 0

		return remove_df

	def link_buildings_to_elec_network(self, bldgs, bldg_to_network):
		""" links buildings to nodes in network
		"""
		bldgs = pd.merge(bldgs, bldg_to_network, how='outer', left_on='guid', right_on='bldg_guid')
		bldgs = bldgs[['guid', 'node_id', 'node_guid', 'Pole_Y_N']]
		bldgs.set_index('guid', inplace=True)
		return bldgs

	def elec_conn_crit_nodes(self, bldgs=None, graph=None, node_info_df=None, 
						critical_nodes=None, time_step=None):
		""" 
			- removes edges from network and performs connectivity analysis
			  from each building to the critical nodes.
			- the graph passed in to this function is the original 
			  graph and not the graph at t = t_2, t_3, ...
		"""
		node_guid_remove = list(node_info_df.loc[
									node_info_df['remove_TF'] == True].index)
		graph = self.GF.remove_nodes(graph, node_guid_remove)
		"""
		determining bin index for each node
		example:
			bins = [[4],[2],[0,1,3,5,6]]					
				nodes 4 and 2 are by themselves, 
				nodes 0, 1, 3, 5, 6 are connected in a cluster
		"""
		
		bins = list(nx.connected_components(graph))
		# performing connectivity analysis
		running_results = np.ones((len(bldgs))).astype(bool)
		for crit_node_bunch in critical_nodes:
			crit_node_bunch_tf = np.zeros((len(bldgs))).astype(bool)
			for crit_node in crit_node_bunch:
				critical_node_bin = self.GF.find_in_list_of_list(bins, crit_node)
				if critical_node_bin==None:
					crit_node_tf = np.ones(len(bldgs))*False
				else:
					crit_node_tf = np.isin(bldgs['node_id'].values, list(bins[critical_node_bin]))
				crit_node_bunch_tf = np.logical_or(crit_node_bunch_tf, crit_node_tf)
			running_results = np.logical_and(running_results, crit_node_bunch_tf)

		bldgs['conn_org'] = bldgs['connected']
		bldgs['conn_new'] = running_results
		bldgs.loc[bldgs['conn_new']!=bldgs['conn_org'], 'time_conn'] = time_step
		bldgs['connected'] = bldgs['conn_new']
		
		return bldgs


	def get_spec(self):
		"""Get specifications of the damage analysis.

		Returns:
			obj: A JSON object of specifications of the building recovery time analysis.

		"""
		return {
			'name': 'connectivity-analysis',
			'description': 'performs connectivity analysis of buildings to critical nodes',
			'input_parameters': [
				{
					'id': 'remove_pole_DS',
					'required': True,
					'description': 'Damage state above which to remove poles from graph',
					'type': int
				},
				{
					'id': 'critical_nodes',
					'required': True,
					'description': 'list of critical node IDs',
					'type': list
				},
				{
					'id': 'pole_reptime_log_med',
					'required': False,
					'description': 'pole reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'pole_reptime_covm',
					'required': False,
					'description': 'pole reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'prnt_msg',
					'required': True,
					'description': 'Message to print out',
					'type': str
				},

			],

			'input_datasets': [
				{
					'id': 'buildings',
					'required': True,
					'description': 'Building Inventory',
					'type': ['ergo:buildingInventoryVer4', 'ergo:buildingInventoryVer5'],
				},				
				{
					'id': 'building_to_network',
					'required': True,
					'description': 'Table that links buildings to network',
					'type': ['incore:buildingsToEPN'],
				},
				{
					'id': 'line_dataset',
					'required': True,
					'description': 'EPN lines',
					'type': ['incore:powerLineTopo'],
				},
				{
					'id': 'pole_dataset',
					'required': True,
					'description': 'EPN pole',
					'type': ['incore:epf'],
				},
				{
					'id': 'pole_dmg',
					'required': True,
					'description': 'Damage inventory of poles',
					'type': ['ergo:NodeDamageInventory'],
				},
			],
			'output_datasets': [
				{
					'id': 'connectivity-result',
					'parent_type': 'networks',
					'description': 'CSV file of connectivity results',
					'type': 'output'
				}

			]
		}


class WterConnectivity(BaseAnalysis):
	def WterConn_run(self):
		"""performs connectivity analysis"""
		pipe_unit_len = 20	# assuming 20ft. length for all pipes in network

		self.prnt_msg = self.get_parameter('prnt_msg')
		n_workers = self.get_parameter('n_workers')
		pipe_reprate = self.get_parameter('pipe_reprate')

		wtp_reptime_log_med = self.get_parameter('wtp_rep_time_log_med')
		wtp_reptime_covm = self.get_parameter('wtp_rep_time_covm')
		wps_reptime_log_med = self.get_parameter('wps_rep_time_log_med')
		wps_reptime_covm = self.get_parameter('wps_rep_time_covm')
		

		# loading building data
		bldg_data_df = self.get_input_dataset('buildings')
		bldg_data_df = pd.DataFrame(bldg_data_df.get_inventory_reader())
		bldg_data_df = pd.DataFrame(list(bldg_data_df['properties']))
		bldg_data_df = bldg_data_df[['guid']]

		# table that links buildings to network
		bldg2netwrk_df = self.get_input_dataset('building_to_network')
		bldg2netwrk_df = pd.DataFrame(bldg2netwrk_df.get_inventory_reader())
		bldg2netwrk_df = pd.DataFrame(list(bldg2netwrk_df['properties']))
		bldg2netwrk_df['edge_id'] = bldg2netwrk_df['edge_id'].astype(int)	
		bldg2netwrk_df['fromnode'] = bldg2netwrk_df['fromnode'].astype(int)	
		bldg2netwrk_df['tonode'] = bldg2netwrk_df['tonode'].astype(int)	
		bldg2netwrk_df['water_pump_ID'] = bldg2netwrk_df['water_pump_ID'].astype(int)	

		# pipe dataset and damage
		pipe_dset = self.get_input_dataset('pipe_dataset')
		pipe_dset = pipe_dset.get_inventory_reader()
		pipe_df = pd.DataFrame(pipe_dset)
		pipe_df = pd.DataFrame(list(pipe_df['properties']))
		pipe_df['n_pipes'] = (pipe_df['length_km']/pipe_unit_len)*1000

		pipe_dmg = self.get_input_dataset('pipe_dmg')
		pipe_dmg = pipe_dmg.get_dataframe_from_csv()
		pipe_dmg = pd.merge(pipe_dmg, pipe_df[['guid', 'diameter', 'n_pipes']], how='outer', on='guid')
		pipe_dmg.set_index('guid', inplace=True)

		# water facility damage and dataset
		wterfclty_dset = self.get_input_dataset('wterfclty_dataset')
		wterfclty_dset = wterfclty_dset.get_inventory_reader()
		wterfclty_df = pd.DataFrame(wterfclty_dset)
		wterfclty_df = pd.DataFrame(list(wterfclty_df['properties']))

		# mappings between wterfclty guid (1) and edge guid (2) 
		wterfclty_edge_mapping = {
			'2048ed28-395e-4521-8fc5-44322534592e': '9f9a35a9-50d4-4724-8708-2085949dd0db',	# wtp
			'8d22fef3-71b6-4618-a565-955f4efe00bf': '3d014885-8424-4ce3-be71-9c88557417fd',	# wps1
			'd6ab5a29-1ca1-4096-a3c3-eb93b2178dfe': '34675ae7-e029-4d43-ba04-b228be76259f',	# wps2
			'cfe182a2-c39c-4734-bcd5-3d7cadab8aff': '4ade18c9-96c3-49e5-a900-1b482d58eea0'	# wps3
			}
		wterfclty_edge_mapping = pd.DataFrame.from_dict(wterfclty_edge_mapping, orient='index', columns=['edge_guid'])
		wterfclty_dmg = self.get_input_dataset('wterfclty_dmg')
		wterfclty_dmg = wterfclty_dmg.get_dataframe_from_csv()
		wterfclty_dmg = pd.merge(wterfclty_dmg, wterfclty_df[['guid','utilfcltyc', 'Pump_ID']], how='outer', on='guid')
		wterfclty_dmg.set_index('guid', inplace=True)
		wterfclty_dmg = pd.merge(wterfclty_dmg, wterfclty_edge_mapping, left_index=True, right_index=True)
		wterfclty_dmg['elec_node_guid'] = ['402388b9-f14c-4402-9ad9-8b0965de0937',
											'9b38bb0c-a2e9-4068-90e8-4bf0479b3a9e',
											'584f8368-c4ac-42bd-a39a-230517a5e13e',
											'0d268d61-1733-48ee-bab9-8126b3609522']

		# # electric dependency information
		# wter2elec_func = self.get_input_dataset('wter2elec_func')
		# wter2elec_func = wter2elec_func.get_dataframe_from_csv()
		# wter2elec_func = wter2elec_func.rename(columns={'Unnamed: 0': 'guid'})
		# wter2elec_func.set_index('guid', inplace=True)

		wter2elec_rep = self.get_input_dataset('wter2elec_rep')
		wter2elec_rep = wter2elec_rep.get_dataframe_from_csv()
		wter2elec_rep = wter2elec_rep.rename(columns={'Unnamed: 0': 'guid'})
		wter2elec_rep.set_index('guid', inplace=True)
		
		# --- BEGIN ANALYSIS
		self.GF = GraphFuncs()
		graph, guid_to_linknwid = self.GF.create_graph(network_links=pipe_dset)

		# linking buildings to network
		bldg_data_df = self.link_buildings_to_wter_network(bldg_data_df, bldg2netwrk_df)

		func, rep = self.wter_connectivity_analysis(
								graph=graph, 
								bldgs=bldg_data_df,
								guid_to_linknwid=guid_to_linknwid,
								
								pipe_dmg=pipe_dmg,
								wterfclty_dmg=wterfclty_dmg,

								wtp_reptime_log_med=wtp_reptime_log_med,
								wtp_reptime_covm=wtp_reptime_covm,
								wps_reptime_log_med=wps_reptime_log_med,
								wps_reptime_covm=wps_reptime_covm,
								n_workers=n_workers,
								pipe_reprate=pipe_reprate,

								wter2elec_rep=wter2elec_rep
								)


		return func, rep


	def wter_connectivity_analysis(self, 
								graph=None, 
								bldgs=None,
								guid_to_linknwid=None,
								pipe_dmg=None,
								wterfclty_dmg=None,
								wtp_reptime_log_med=None,
								wtp_reptime_covm=None,
								wps_reptime_log_med=None,
								wps_reptime_covm=None,
								n_workers=None,
								pipe_reprate=None,
								wter2elec_rep=None):

		func_df = pd.DataFrame(index=bldgs.index)
		reptime_df = pd.DataFrame(index=bldgs.index)
		n_samples = list(pipe_dmg.keys())
		n_samples = len([i for i in n_samples if 'iter' in i])
		count = 0

		""" setting up pipe and wterfclty information dataframes
			some confusing conversions below
			-------------------------------------

			n_pipes.....[pipes]
			24..........[hours/regular_day]
			reprate.....[pipes/(regular_day * workers)]
			n_workers...[workers]
			16..........[hours/ work_day]

			n_pipes * 24 
			------------------------
			reprate * n_workers * 16
		"""
		pipe_info = pd.DataFrame(index=pipe_dmg.index)
		pipe_info['diameter'] = pipe_dmg['diameter']
		pipe_info['n_pipes'] = pipe_dmg['n_pipes']
		pipe_info['reprate'] = np.nan
		pipe_info.loc[pipe_info['diameter']>20, 'reprate'] = pipe_reprate[0]	# num pipes/day/worker
		pipe_info.loc[pipe_info['diameter']<20, 'reprate'] = pipe_reprate[1]	# num pipes/day/worker
		pipe_info['reptime'] = ((pipe_info['n_pipes']/pipe_info['reprate'])/n_workers)*24/16	# workdays

		wterfclty_info = pd.DataFrame(index=wterfclty_dmg.index)
		wterfclty_info['utilfcltyc'] = wterfclty_dmg['utilfcltyc']
		wterfclty_info['edge_guid'] = wterfclty_dmg['edge_guid']
		wterfclty_info['elec_guid'] = wterfclty_dmg['elec_node_guid']
		wterfclty_info['water_pump_ID'] = wterfclty_dmg['Pump_ID']

		for _, key in enumerate(pipe_dmg.keys()):
			if 'iter' in key:
				self.GF.print_percent_complete(self.prnt_msg, count, n_samples)
				bldgs['connected'] = False 	# assuming everything is disconnected
				bldgs['time_conn'] = np.nan 		# 0 until becoming reconnected

				pipe_info_df, wterfclty_info_df = self.damage_wter_network( 
													pipe_dmg=pipe_dmg[key].astype(float), 
													pipe_info = pipe_info,
													
													wterfclty_dmg=wterfclty_dmg[key].astype(float),
													wterfclty_info=wterfclty_info,

													guid_to_linknwid=guid_to_linknwid,
													
													wtp_reptime_log_med=wtp_reptime_log_med,
													wtp_reptime_covm=wtp_reptime_covm,
													wps_reptime_log_med=wps_reptime_log_med,
													wps_reptime_covm=wps_reptime_covm,
													wter2elec_rep=wter2elec_rep[key].astype(float)
													)

				bldgs = self.wter_conn_crit_nodes(bldgs=bldgs,
									graph=graph.copy(),
									pipe_df=pipe_info_df,
									wterfclty_df=wterfclty_info_df,
									time_step=0)

				func_df[key] = list(bldgs['connected'])

				unique_reptime_pipes = np.unique(pipe_info_df['reptime'])
				unique_reptime_pipes = unique_reptime_pipes[unique_reptime_pipes>0]
				unique_reptime_wterfclty = np.unique(wterfclty_info_df['reptime'])
				unique_reptime_wterfclty = unique_reptime_wterfclty[unique_reptime_wterfclty>0]

				unique_reptime = np.unique(np.concatenate((unique_reptime_pipes, unique_reptime_wterfclty)))

				for t in unique_reptime:
					if bldgs['connected'].sum()/len(bldgs) == 1.0:
						break
					pipe_info_df.loc[pipe_info_df['reptime']<=t, 'remove_TF'] = False
					bldgs = self.wter_conn_crit_nodes(bldgs=bldgs,
									graph=graph.copy(),
									pipe_df=pipe_info_df,
									wterfclty_df=wterfclty_info_df,
									time_step=t)

				reptime_df[key] = bldgs['time_conn']
				count += 1

		return func_df, reptime_df

	def damage_wter_network(self, pipe_dmg=None, pipe_info=None, 
							wterfclty_dmg=None,	wterfclty_info=None,
							guid_to_linknwid=None,
							wtp_reptime_log_med=None, wtp_reptime_covm=None,
							wps_reptime_log_med=None, wps_reptime_covm=None,
							wter2elec_rep=None
							):
		# pipe calcs
		pipe_out_df = pd.DataFrame(index=pipe_dmg.index)
		pipe_out_df['fromnode'] = list(guid_to_linknwid['fromnode'])
		pipe_out_df['tonode'] = list(guid_to_linknwid['tonode'])
		pipe_out_df['edge_id'] = list(guid_to_linknwid['edge_id'])

		pipe_out_df['remove_TF'] = pipe_dmg.astype(bool)
		pipe_out_df['reptime'] = 0.0
		pipe_out_df['reptime'] = np.ceil(pipe_out_df['remove_TF']*pipe_info['reptime'])

		"""wterfclty calcs
			determining functionality and if nonfunctional, repair time 
			includes the electric dependency
				- eg is the wtp and wps connected to the substation
				- only relies on repair time
				- takes max of elec repair time and water facility repair time
		"""

		wterfclty_info = pd.merge(wterfclty_info, wterfclty_dmg, left_index=True, right_index=True)
		key = [i for i in list(wterfclty_info.columns) if 'iter' in i]	#renaming column from "iter_#" to "ds"
		wterfclty_info = wterfclty_info.rename(columns={key[0]:'ds'})
	
		n_wtp = sum(wterfclty_info['utilfcltyc']=='PWT2')
		n_wps = sum(wterfclty_info['utilfcltyc']=='PPP2')

		# wterfclty functionality calcs
		rv = np.random.uniform(low=0., high=1., size=(1,(n_wtp+n_wps)))	# random variable
		ds_functionality = [0, 0.02, 0.2, 0.6, 1.0]						# prob that fclty is nonfunctional for each ds (none, slight, moder, etc.)
		rv_comp = [ds_functionality[int(obj)] for i, obj in enumerate(wterfclty_info['ds'])]	# mapping above to fclty damage state for iteration
		func = np.array(rv<=rv_comp).T 									# making comparison/determining whether its functinoal
		wterfclty_info['remove_TF'] = func 								# saving above functionality

		# wterfclty reptime calcs
		reptime = np.zeros(len(wterfclty_info))
		count = 0
		for index, row in wterfclty_info.iterrows():
			if row['remove_TF']==True:
				if row['utilfcltyc'] == 'PWT2':
					log_med = wtp_reptime_log_med
					covm = wtp_reptime_covm
				elif row['utilfcltyc'] == 'PPP2':
					log_med = wps_reptime_log_med
					covm = wps_reptime_covm
				mvn = np.random.multivariate_normal(log_med, covm, 1)
				temp = np.column_stack((np.zeros(1), np.exp(mvn)))
				rep = temp[0,int(row['ds'])]
				reptime[count] = rep
				count += 1
		wterfclty_info['reptime'] = reptime


		""" merging wterfclty functionality with electric dependency.
			Considers whether the wterfcltys have access to electricity.
				if yes, 
					- "remove_TF" and "reptime" are saved
				if no, 
					- remove_TF is updated to True
					- reptime is taken as the max of "reptime" and "elec_reptime"
		"""
		wterfclty_info = pd.merge(wterfclty_info, wter2elec_rep, left_on='elec_guid', right_index=True)
		key = [i for i in list(wterfclty_info.columns) if 'iter' in i]	#renaming column from "iter_#" to "elec_reptime"
		wterfclty_info = wterfclty_info.rename(columns={key[0]:'elec_reptime'})
		wterfclty_info.loc[wterfclty_info['elec_reptime']>0., 'remove_TF'] = True

		# saving wterfclty calcs to new dataframe for passing back
		wterfclty_out = pd.DataFrame(index=wterfclty_info.index)
		wterfclty_out['utilfcltyc'] = wterfclty_info['utilfcltyc']
		wterfclty_out['water_pump_ID'] = wterfclty_info['water_pump_ID']
		wterfclty_out['edge_guid'] = wterfclty_info['edge_guid']
		wterfclty_out['remove_TF'] = wterfclty_info['remove_TF']
		wterfclty_out['reptime'] = wterfclty_info[["reptime", "elec_reptime"]].max(axis=1)	# elec-water dependence	

		return pipe_out_df, wterfclty_out

	def wter_conn_crit_nodes(self, bldgs=None, graph=None, pipe_df=None, 
								wterfclty_df=None, time_step=None):
		""" 
			- removes edges from network and performs connectivity analysis
			- the graph passed in to this function is the original 
			  graph and not the graph at t = t_2, t_3, ...
		"""
		edge_guid_remove = list(pipe_df.loc[pipe_df['remove_TF'] == True].index)
		n_nodes = graph.number_of_nodes()
		graph = self.GF.remove_edges(graph, edge_guid_remove)

		"""
		determining bin index for each node
		example:
			bins = [[4],[2],[0,1,3,5,6]]					
				nodes 4 and 2 are by themselves, 
				nodes 0, 1, 3, 5, 6 are connected in a cluster
		"""
		bins = list(nx.connected_components(graph))
		
		# determinig bin of wtp
		wtp_edge = wterfclty_df.loc[wterfclty_df['utilfcltyc']=='PWT2', 'edge_guid']
		wtp_fromnode = float(pipe_df.loc[wtp_edge, 'fromnode'])
		wtp_bin_idx = self.GF.find_in_list_of_list(bins, wtp_fromnode)
		wtp_reptime = float(wterfclty_df.loc[wterfclty_df['utilfcltyc']=='PWT2', 'reptime'])
		for index, row in wterfclty_df.iterrows():	# loop through water facilties
			""" determing whether wps is connected to wtp """
			if row['utilfcltyc']=='PPP2':	# considering wps in facilties
				if (time_step>=row['reptime']) and (time_step>=wtp_reptime):
					# if both the wps and wtp is repaired
					
					# determining bin of wps
					fromnode = pipe_df.loc[row['edge_guid'],'fromnode']
					pump_bin_idx = self.GF.find_in_list_of_list(bins, fromnode)
					pump_bin = list(bins[pump_bin_idx])
					# determining if wtp and wps are in the same bin
					if pump_bin_idx==wtp_bin_idx:	# if so, update remove_TF
						""" if it gets this far, then:
								1. wtp & wps has access to electricity
								2. wtp & wps are repaired from hazard damage
								3. wps is connected to wtp via pipes
						"""
						wterfclty_df.loc[index,'remove_TF'] = False	

						pump_id = row['water_pump_ID']
						bldgs_temp = bldgs.loc[(bldgs['water_pump_ID']==pump_id) & 
											   (bldgs['connected'] == False)]

						conn_tf = np.isin(bldgs_temp['fromnode'].values, pump_bin)

						bldgs.loc[bldgs_temp.index, 'connected'] = conn_tf
						bldgs.loc[bldgs_temp.index, 'time_conn'] = time_step

		return bldgs

	def link_buildings_to_wter_network(self, bldgs, bldg_to_network):
		""" links buildings to nodes in network
		"""
		bldgs = pd.merge(bldgs, bldg_to_network, how='outer', left_on='guid', right_on='bldg_guid')
		bldgs = bldgs[['guid', 'edge_id', 'edge_guid', 'water_pump_ID', 'fromnode', 'tonode']]
		bldgs.set_index('guid', inplace=True)
		return bldgs


	def get_spec(self):
		"""
		Get specifications of the damage analysis.
		Returns:
			obj: A JSON object of specifications of the building recovery time analysis.
		"""
		return {
			'name': 'connectivity-analysis',
			'description': 'performs connectivity analysis of water pumping stations to water treatment plant',
			'input_parameters': [
				{
					'id': 'critical_nodes',
					'required': True,
					'description': 'list of critical node IDs',
					'type': list
				},
				{
					'id': 'wtp_rep_time_log_med',
					'required': True,
					'description': 'wtp reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'wtp_rep_time_covm',
					'required': True,
					'description': 'wtp reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'wps_rep_time_log_med',
					'required': True,
					'description': 'wps reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'wps_rep_time_covm',
					'required': True,
					'description': 'wps reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'n_workers',
					'required': True,
					'description': 'number of workers available',
					'type': int
				},
				{
					'id': 'pipe_reprate',
					'required': True,
					'description': 'number of workers available',
					'type': list
				},	
				{
					'id': 'prnt_msg',
					'required': True,
					'description': 'Message to print out',
					'type': str
				},

			],

			'input_datasets': [	
				{
					'id': 'buildings',
					'required': True,
					'description': 'Building Inventory',
					'type': ['ergo:buildingInventoryVer4', 'ergo:buildingInventoryVer5'],
				},
				{
					'id': 'building_to_network',
					'required': True,
					'description': 'Table that links buildings to network',
					'type': ['incore:buildingsToWaterNetwork'],
				},
				{
					'id': 'pipe_dataset',
					'required': True,
					'description': 'water pipes',
					'type': ['ergo:buriedPipelineTopology'],
				},
				{
					'id': 'pipe_dmg',
					'required': True,
					'description': 'Damage inventory of pipes',
					'type': ['ergo:DamageInventory'],
				},
				{
					'id': 'wterfclty_dataset',
					'required': True,
					'description': 'water facilities',
					'type': ['ergo:waterFacilityTopo'],
				},
				{
					'id': 'wterfclty_dmg',
					'required': True,
					'description': 'Damage inventory of water facilities',
					'type': ['ergo:DamageInventory'],
				},
				{
					'id': 'wter2elec_func',
					'required': True,
					'description': 'connectivity bw water facilities and electricy',
					'type': ['ergo:DamageInventory'],
				},
				{
					'id': 'wter2elec_rep',
					'required': True,
					'description': 'reptime for water facilities and electricy connectivity',
					'type': ['ergo:DamageInventory'],
				},
			],
			'output_datasets': [
				{
					'id': 'connectivity-result',
					'parent_type': 'networks',
					'description': 'CSV file of connectivity results',
					'type': 'output'
				}

			]
		}


class Wter2ElecConnectivity(BaseAnalysis):
	def run(self):
		"""performs connectivity analysis"""
		self.prnt_msg = self.get_parameter('prnt_msg')
		from_nodes = self.get_parameter('from_nodes')
		critical_nodes = self.get_parameter('critical_nodes')

		wterfcltys = pd.DataFrame(index=from_nodes.values())	# setting index as guid
		wterfcltys['node_id'] = from_nodes.keys()	# adding node_id column

		pole_dset = self.get_input_dataset('pole_dataset')
		pole_dset = pole_dset.get_inventory_reader()
		pole_df = pd.DataFrame(pole_dset)
		pole_df = pd.DataFrame(list(pole_df['properties']))
		
		pole_dmg = self.get_input_dataset('pole_dmg')
		pole_dmg = pole_dmg.get_dataframe_from_csv()
		pole_dmg = pd.merge(pole_dmg, pole_df[['guid', 'Pole_Y_N']], how='outer', on='guid')

		pole_dmg.set_index('guid', inplace=True)

		n_sims = len([i for i in pole_dmg.keys() if "iter" in i])

		line_dset = self.get_input_dataset('line_dataset')
		line_dset = line_dset.get_inventory_reader()

		remove_pole_DS = self.get_parameter("remove_pole_DS") # DS to remove poles

		pole_reptime_log_med = self.get_parameter('pole_reptime_log_med')
		pole_reptime_covm = self.get_parameter('pole_reptime_covm')
		
		# --- BEGIN ANALYSIS
		self.GF = GraphFuncs()
		graph, _ = self.GF.create_graph(network_links=line_dset, 
										inf='epn', 
										network_nodes=pole_dset)


		func, rep = self.wter2elec_connectivity_analysis(
								graph=graph, 
								pole_dmg=pole_dmg,
								remove_pole_DS=remove_pole_DS,
								pole_rep_logmed=pole_reptime_log_med,
								pole_rep_covm=pole_reptime_covm,
								critical_nodes=critical_nodes,
								wterfcltys=wterfcltys)
		return func, rep


	def wter2elec_connectivity_analysis(self, graph=None, pole_dmg=None,
							remove_pole_DS=None, pole_rep_logmed=None,
							pole_rep_covm=None, critical_nodes=None, 
							wterfcltys=None):

		func_df = pd.DataFrame(index=wterfcltys.index)
		reptime_df = pd.DataFrame(index=wterfcltys.index)
		n_samples = list(pole_dmg.keys())
		n_samples = len([i for i in n_samples if 'iter' in i])
		count = 0
		for _, key in enumerate(pole_dmg.keys()):
			if 'iter' in key:
				self.GF.print_percent_complete(self.prnt_msg, count, n_samples)
				wterfcltys['connected'] = True 	# assuming everything is connected
				wterfcltys['time_conn'] = 0 		# 0 until becoming reconnected

				pole_info_df = self.damage_elec_network( 
									pole_dmg=pole_dmg[[key, 'Pole_Y_N']].astype(float), 
									pole_rep_logmed=pole_rep_logmed,
									pole_rep_covm=pole_rep_covm,
									remove_pole_DS=remove_pole_DS,
									key=key)
				wterfcltys = self.wter2elec_conn_crit_nodes(node_info_df=pole_info_df,
									graph=graph.copy(),
									wterfcltys=wterfcltys,
									critical_nodes=critical_nodes,
									time_step=0)
				func_df[key] = list(wterfcltys['connected'].astype(int))

				unique_reptime = np.unique(pole_info_df['reptime'])
				unique_reptime = unique_reptime[unique_reptime>0]
				for t in unique_reptime:
					if wterfcltys['connected'].sum()/len(wterfcltys) == 1.0:
						break
					pole_info_df.loc[pole_info_df['reptime']<=t, 'remove_TF'] = False
					wterfcltys = self.wter2elec_conn_crit_nodes(node_info_df=pole_info_df,
									graph=graph.copy(),
									wterfcltys=wterfcltys,
									critical_nodes=critical_nodes,
									time_step=t)
				reptime_df[key] = wterfcltys['time_conn']
				count += 1

		reptime_df.fillna(0, inplace=True)
		return func_df, reptime_df

	def damage_elec_network(self, pole_dmg, pole_rep_logmed, pole_rep_covm, 
						remove_pole_DS, key):

		remove_df = pd.DataFrame(index=pole_dmg.index)
		remove_df['Pole_Y_N'] = pole_dmg['Pole_Y_N']
		remove_df['remove_TF'] = False
		remove_df['reptime'] = 0.0

		# reptime calcs for iteration
		n_poles = len(pole_dmg)
		mvn = np.random.multivariate_normal(pole_rep_logmed,
											pole_rep_covm,
											n_poles)
		
		reptime = np.column_stack((np.zeros(n_poles),np.exp(mvn)))
		rep_poles = np.array([reptime[int(i), int(obj)] for 
							i, obj in enumerate(pole_dmg[key])])

		remove_df['remove_TF'] = list(pole_dmg[key] >= remove_pole_DS)
		remove_df['reptime'] = np.ceil(rep_poles)
		remove_df.loc[remove_df['Pole_Y_N']==0, 'remove_TF'] = False
		remove_df.loc[remove_df['Pole_Y_N']==0, 'reptime'] = 0

		return remove_df


	def wter2elec_conn_crit_nodes(self, graph=None, node_info_df=None, 
						wterfcltys=None, critical_nodes=None, time_step=None):
		""" 
			- removes edges from network and performs connectivity analysis
			  from each building to the critical nodes.
			- the graph passed in to this function is the original 
			  graph and not the graph at t = t_2, t_3, ...
		"""
		node_guid_remove = list(node_info_df.loc[node_info_df['remove_TF'] == True].index)
		graph = self.GF.remove_nodes(graph, node_guid_remove)
		"""
		determining bin index for each node
		example:
			bins = [[4],[2],[0,1,3,5,6]]					
				nodes 4 and 2 are by themselves, 
				nodes 0, 1, 3, 5, 6 are connected in a cluster
		"""
		
		bins = list(nx.connected_components(graph))
		# performing connectivity analysis
		running_results = np.ones((len(wterfcltys))).astype(bool)
		for crit_node_bunch in critical_nodes:
			crit_node_bunch_tf = np.zeros((len(wterfcltys))).astype(bool)
			for crit_node in crit_node_bunch:
				critical_node_bin = self.GF.find_in_list_of_list(bins, crit_node)
				if critical_node_bin==None:
					crit_node_tf = np.ones(len(wterfcltys))*False
				else:
					crit_node_tf = np.isin(wterfcltys['node_id'].values, list(bins[critical_node_bin]))
				crit_node_bunch_tf = np.logical_or(crit_node_bunch_tf, crit_node_tf)
			running_results = np.logical_and(running_results, crit_node_bunch_tf)

		wterfcltys['conn_org'] = wterfcltys['connected']
		wterfcltys['conn_new'] = running_results
		wterfcltys.loc[wterfcltys['conn_new']!=wterfcltys['conn_org'], 'time_conn'] = time_step
		wterfcltys['connected'] = wterfcltys['conn_new']
		
		return wterfcltys


	def get_spec(self):
		"""Get specifications of the damage analysis.

		Returns:
			obj: A JSON object of specifications of the building recovery time analysis.

		"""
		return {
			'name': 'connectivity-analysis',
			'description': 'performs connectivity analysis of buildings to critical nodes',
			'input_parameters': [
				{
					'id': 'remove_pole_DS',
					'required': True,
					'description': 'Damage state above which to remove poles from graph',
					'type': int
				},
				{
					'id': 'from_nodes',
					'required': True,
					'description': 'list of from node IDs (wter treatment plant and pumping stations)',
					'type': dict
				},
				{
					'id': 'critical_nodes',
					'required': True,
					'description': 'list of critical node IDs',
					'type': list
				},
				{
					'id': 'pole_reptime_log_med',
					'required': False,
					'description': 'pole reptime log med',
					'type': np.ndarray
				},
				{
					'id': 'pole_reptime_covm',
					'required': False,
					'description': 'pole reptime covariance matrix',
					'type': np.ndarray
				},
				{
					'id': 'prnt_msg',
					'required': True,
					'description': 'Message to print out',
					'type': str
				},

			],

			'input_datasets': [	

				{
					'id': 'line_dataset',
					'required': True,
					'description': 'EPN lines',
					'type': ['incore:powerLineTopo'],
				},
				{
					'id': 'pole_dataset',
					'required': True,
					'description': 'EPN pole',
					'type': ['incore:epf'],
				},
				{
					'id': 'pole_dmg',
					'required': True,
					'description': 'Damage inventory of poles',
					'type': ['ergo:NodeDamageInventory'],
				},
			],
			'output_datasets': [
				{
					'id': 'connectivity-result',
					'parent_type': 'networks',
					'description': 'CSV file of connectivity results',
					'type': 'output'
				}

			]
		}



