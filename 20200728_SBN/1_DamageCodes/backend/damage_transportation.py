from pyincore import IncoreClient, Dataset
from pyincore.analyses.bridgedamage import BridgeDamage
from pyincore.analyses.roaddamage import RoadDamage
from backend.connectivityOSU import TransConnectivity

import os
import sys
import numpy as np
import pandas as pd

"""
this code uses pyincore to perform a transportation damage and connectivity 
	anlaysis for Seaside, OR.

The following hazards are considered:
	- Earthquake
	- Tusnami
	- Earthquake + Tsunami (cumulative)

The following anlayses are completed:
	- Building damage: determines damage state probabiltiies (pyincore)
	- Monte-carlo building damage: samples from above probabiltiies
	- Monte-carlo performance: habitabilty for each iteration of MC
	- Monte-carlo reptime: repair time estimates for each iteration of MC

Four ex-ante (retrofit) options are considered:
	- retrofit_0: status quo
	- retrofit_1: all buildings are at least low-code
	- retrofit_2: all buildings are at least moderate-code
	- retrofit_3: all buildings are at least high-code

Four ex-post (recovery) measures are considered:
	- fast_0: status quo
	- fast_1: repair times are 3/4 of original
	- fast_2: repair times are 1/2 of original
	- fast_3: repair times are 1/4 of original

All modeling reults are written to 'output_dmg'
	- output_dmg:
		- damage_output_r0
			- pyincore damage results
			- mc_results
				- mc_DS
				- mc_performance
				- mc_reptime

"""

class transportation_damage():
	def __init__(self, output_path, retrofit_key_val):
		self.client = IncoreClient()

		self.output_path = os.path.join(output_path, 
							'retrofit{}' .format(retrofit_key_val))
		
		self.mc_path = os.path.join(self.output_path, 'mc_results')
		self.bridge_output_path = os.path.join(self.output_path,
						'bridge_damage_output_r{}' .format(retrofit_key_val))

		self.road_output_path = os.path.join(self.output_path, 
						'road_damage_output')

		self.makedir(self.output_path)
		self.makedir(self.mc_path)
		self.makedir(self.bridge_output_path)
		self.makedir(self.road_output_path)

	def makedir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)


	def run_bridge_damage(self, haz_type, retrofit_key):
		""" bridge damage using pyincore
		"""
		# Seaside bridges
		bridge_dataset_id = "5d6ede5db9219c34b56fc20b"
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]

		if haz_type == 'eq':
			hazard_type = "earthquake"
			rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
							  250: "5dfa41aab9219c934b64d4b2",
							  500: "5dfa4300b9219c934b64d4d0",
							  1000: "5dfa3e36b9219c934b64c231",
							  2500: "5dfa4417b9219c934b64d4d3", 
							  5000: "5dfbca0cb9219c101fd8a58d",
							 10000: "5dfa51bfb9219c934b68e6c2"}

			# Seaside Bridge Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d55c3a1b9219c0689f1f898"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5eb97f8dfd856300017f54ce'	# eq retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = "5d55c3a1b9219c0689f1f898"	# not retrofitted
			elif retrofit_key == 'retrofit3':
				mapping_id = '5eb97f8dfd856300017f54ce'	# eq retrofitted

		elif haz_type == 'tsu':
			hazard_type = "tsunami"
			rt_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
							  250: "5df910abb9219cd00cf5f0a5",
							  500: "5df90e07b9219cd00ce971e7",
							  1000: "5df90137b9219cd00cb774ec",
							  2500: "5df90761b9219cd00ccff258",
							  5000: "5df90871b9219cd00ccff273",
							  10000: "5d27b986b9219c3c55ad37d0"}

			# Default Bridge Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d275000b9219c3c553c7202"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5d275000b9219c3c553c7202'	# not retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = '5eb982f640a33d00013bdcb4'	# tsu retrofitted
			elif retrofit_key == 'retrofit3':
				mapping_id = '5eb982f640a33d00013bdcb4'	# tsu retrofitted

		# Create bridge damage
		brdg_dmg = BridgeDamage(self.client)

		# Load input datasets
		brdg_dmg.load_remote_input_dataset("bridges", bridge_dataset_id)

		# Set analysis parameters
		brdg_dmg.set_parameter("mapping_id", mapping_id)
		brdg_dmg.set_parameter("hazard_type", hazard_type)
		brdg_dmg.set_parameter("num_cpu", 4)

		for rt_val in rt:
			print('\tbridge_dmg: {} rt_{}' .format(haz_type, rt_val))
			result_name = os.path.join(self.bridge_output_path, 
									   'bridge_{}_{}yr_dmg_{}' 
									   .format(haz_type, rt_val, retrofit_key))
			hazard_id = rt_hazard_dict[rt_val]
			brdg_dmg.set_parameter("hazard_id", hazard_id)
			brdg_dmg.set_parameter("result_name", result_name)

			brdg_dmg.run_analysis()

	def run_cumulative_bridge_damage(self, retrofit_key):
		""" 
		multi-hazard bridge damage according to hazus calcuations
		"""
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		
		for rt_val in rt:
			print('\tbridge_dmg: cumulative rt_{}' .format(rt_val))
			# --- reading in damage results from above analysis


			eq_damage_results_csv = os.path.join(self.bridge_output_path, 
											'bridge_eq_{}yr_dmg_{}.csv' 
											.format(rt_val, retrofit_key))
			tsu_damage_results_csv = os.path.join(self.bridge_output_path, 
											'bridge_tsu_{}yr_dmg_{}.csv' 
											.format(rt_val, retrofit_key))
			eq_df = pd.read_csv(eq_damage_results_csv)
			tsu_df = pd.read_csv(tsu_damage_results_csv)

			cum_df = pd.DataFrame()
			cum_df['guid'] = eq_df['guid']
			
			cum_df['ds-complet'] = eq_df['ds-complet'] + tsu_df['ds-complet'] \
				- eq_df['ds-complet']*tsu_df['ds-complet']

			# --- prob of exceeding each damage state
			cum_df['ls-complet'] = cum_df['ds-complet']

			cum_df['ls-extensi'] = eq_df['ls-extensi'] + tsu_df['ls-extensi'] \
				- eq_df['ls-extensi']*tsu_df['ls-extensi']

			cum_df['ls-moderat'] = eq_df['ls-moderat'] + tsu_df['ls-moderat'] \
				- eq_df['ls-moderat']*tsu_df['ls-moderat']

			cum_df['ls-slight'] = eq_df['ls-slight'] + tsu_df['ls-slight'] \
				- eq_df['ls-slight']*tsu_df['ls-slight']

			# --- prob of being in each damage state
			cum_df['ds-extensi'] = cum_df['ls-extensi'] - cum_df['ds-complet']
			cum_df['ds-moderat'] = cum_df['ls-moderat'] - cum_df['ls-extensi']
			cum_df['ds-slight'] = cum_df['ls-slight'] - cum_df['ls-moderat']
			cum_df['ds-none'] = 1 - cum_df['ls-slight']
			cum_df['hazard'] = 'Earthquake+Tsunami'

			result_name = os.path.join(self.bridge_output_path, 
									   'bridge_cumulative_{}yr_dmg_{}.csv' 
									   .format(rt_val, retrofit_key))
			cum_df = cum_df[['guid', 
							 'ls-slight',
							 'ls-moderat',
							 'ls-extensi',
							 'ls-complet',
							 'ds-none', 
							 'ds-slight', 
							 'ds-moderat', 
							 'ds-extensi', 
							 'ds-complet', 
							 'hazard']]
			cum_df.to_csv(result_name, index=False)


	def run_road_damage(self, haz_type):
		""" road damage using pyincore"""
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]

		# Seaside roads
		road_dataset_id = "5d25118eb9219c0692cd7527"

		if haz_type == 'eq':
			hazard_type = "earthquake"
			rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
							  250: "5dfa41aab9219c934b64d4b2",
							  500: "5dfa4300b9219c934b64d4d0",
							  1000: "5dfa3e36b9219c934b64c231",
							  2500: "5dfa4417b9219c934b64d4d3", 
							  5000: "5dfbca0cb9219c101fd8a58d",
							 10000: "5dfa51bfb9219c934b68e6c2"}

			fragility_key = "pgd"

			# seaside road fragility mappng for EQ
			mapping_id = "5d545b0bb9219c0689f1f3f4"

		elif haz_type == 'tsu':

			hazard_type = "tsunami"
			rt_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
							  250: "5df910abb9219cd00cf5f0a5",
							  500: "5df90e07b9219cd00ce971e7",
							  1000: "5df90137b9219cd00cb774ec",
							  2500: "5df90761b9219cd00ccff258",
							  5000: "5df90871b9219cd00ccff273",
							  10000: "5d27b986b9219c3c55ad37d0"}
			fragility_key = "Non-Retrofit inundationDepth Fragility ID Code"

			# seaside road fragility mappng for EQ
			mapping_id = "5d274fd8b9219c3c553c71ff"


		# Run Seaside earthquake road damage
		road_dmg = RoadDamage(self.client)
		road_dmg.load_remote_input_dataset("roads", road_dataset_id)
		road_dmg.set_parameter("mapping_id", mapping_id)
		road_dmg.set_parameter("hazard_type", hazard_type)
		road_dmg.set_parameter("num_cpu", 1)
		road_dmg.set_parameter("fragility_key", fragility_key)

		for rt_val in rt:
			print('\troad_dmg: {} rt_{}' .format(haz_type, rt_val))
			result_name = os.path.join(self.road_output_path, 
									   'road_{}_{}yr_dmg' 
									   .format(haz_type, rt_val))
			hazard_id = rt_hazard_dict[rt_val]

			road_dmg.set_parameter("hazard_id", hazard_id)
			road_dmg.set_parameter("result_name", result_name)

			road_dmg.run_analysis()


	def run_cumulative_road_damage(self):
		""" 
		multi-hazard road damage according to hazus calcuations
		"""

		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		
		for rt_val in rt:
			print('\troad_dmg: cumulative rt_{}' .format(rt_val))
			# --- reading in damage results from above analysis
			eq_damage_results_csv = os.path.join(self.road_output_path, 
												 'road_eq_{}yr_dmg.csv' 
												 .format(rt_val))
			tsu_damage_results_csv = os.path.join(self.road_output_path, 
												  'road_tsu_{}yr_dmg.csv' 
												  .format(rt_val))
			eq_df = pd.read_csv(eq_damage_results_csv)
			tsu_df = pd.read_csv(tsu_damage_results_csv)


			""" Overriding EQ road damage from pyincore:
				- per Kameshwar's original code:
					# PGD (permanent ground deformation) from Shafiq's analysis is zero
				- road damage is determined from PGD (bridges from SA)
				- the below over rides the road damage to be 0 regardless of the
				  PGD in pyincore
				- drs is unsure why road PGD is 0...
			"""
			eq_df['ds-complet'] = 0
			eq_df['ls-extensi'] = 0
			eq_df['ls-moderat'] = 0
			eq_df['ls-slight'] = 0

			cum_df = pd.DataFrame()
			cum_df['guid'] = eq_df['guid']
			
			cum_df['ds-complet'] = eq_df['ds-complet'] + tsu_df['ds-complet'] \
				- eq_df['ds-complet']*tsu_df['ds-complet']
			
			# --- prob of exceeding each damage state
			cum_df['ls-complet'] = cum_df['ds-complet']

			cum_df['ls-extensi'] = eq_df['ls-extensi'] + tsu_df['ls-extensi'] \
				- eq_df['ls-extensi']*tsu_df['ls-extensi']

			cum_df['ls-moderat'] = eq_df['ls-moderat'] + tsu_df['ls-moderat'] \
				- eq_df['ls-moderat']*tsu_df['ls-moderat']

			cum_df['ls-slight'] = eq_df['ls-slight'] + tsu_df['ls-slight'] \
				- eq_df['ls-slight']*tsu_df['ls-slight']

			# --- prob of being in each damage state
			cum_df['ds-extensi'] = cum_df['ls-extensi'] - cum_df['ds-complet']
			cum_df['ds-moderat'] = cum_df['ls-moderat'] - cum_df['ls-extensi']
			cum_df['ds-slight'] = cum_df['ls-slight'] - cum_df['ls-moderat']
			cum_df['ds-none'] = 1 - cum_df['ls-slight']
			cum_df['hazard'] = 'Earthquake+Tsunami'

			result_name = os.path.join(self.road_output_path, 
									   'road_cumulative_{}yr_dmg.csv' 
									   .format(rt_val))
			cum_df = cum_df[['guid', 
							 'ls-slight',
							 'ls-moderat',
							 'ls-extensi',
							 'ls-complet',
							 'ds-none', 
							 'ds-slight', 
							 'ds-moderat', 
							 'ds-extensi', 
							 'ds-complet', 
							 'hazard']]
			cum_df.to_csv(result_name, index=False)
			

	def DS_mc_sample(self, hazard, retrofit_key, n_samples=100):
		""" using damage state probabiltiies output from pyincore to sample 
				via Monte-Carlo damage states of each tax-lot. 
			pyincore has a MC sampling routine, but the results from each 
				iteration are not easily accessible, so I wrote my own here.
				
			inputs:
				- hazard: 'eq', 'tsu', 'cumulative'
				- path_out: path to where MC results should be written. A new
				  folder, 'mc_results' will be created here
				- retrofit_key: 0, 1, 2, 3 corresponding to minimum retrofit
				  levels as status quo, pre-, moderate-, or high-code.
				- n_samples: number of mc samples; set to 100. 
			output:
				- a pickle is written out that contains a python dictionary with
				  the MC results for all tax-lots.
				- the dictionary keys correspond to the return periods (e.g.
				  rt_100, rt_250, etc.).
				- each key contains a pandas dataframe with rows representing 
				  each tax-lot and columns representing each iteration of the MC
				  sampling. 
		"""
		np.random.seed(1338)

		rts = [100, 250, 500, 1000, 2500, 5000, 10000]
		road_files = ['road_{}_{}yr_dmg.csv' 
						.format(hazard, i) for i in rts]
		bridge_files = ['bridge_{}_{}yr_dmg_{}.csv' 
						.format(hazard, i, retrofit_key) for i in rts]
		
		road_data = {}
		bridge_data = {}
		for i, _ in enumerate(road_files):
			road_data[rts[i]] = pd.read_csv(os.path.join(self.road_output_path, 
														 road_files[i]))            
			bridge_data[rts[i]] = pd.read_csv(os.path.join(self.bridge_output_path, 
														 bridge_files[i]))

		road_guids = list(road_data[rts[0]]['guid'])
		bridge_guids = list(bridge_data[rts[0]]['guid'])

		column_keys = ['iter_{}' .format(i) for i in range(n_samples)]



		# --- mc for bridges 
		for rt_i, rt in enumerate(rts):
			prnt_msg = '\tmc_bridge_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)

			ds_results = np.zeros((len(bridge_guids), n_samples))
			temp_data = bridge_data[rt]

			for guid_i, guid in enumerate(bridge_guids):
				self.print_percent_complete(prnt_msg, guid_i, len(bridge_guids))
				row = bridge_data[rt].loc[bridge_data[rt]['guid']==guid].T.squeeze()
				bins = np.array([row['ls-slight'], 
								 row['ls-moderat'], 
								 row['ls-extensi'], 
								 row['ls-complet'],
								 0])
				bins[0] += 1e-9
				rv = np.random.uniform(low=0., high=1., size=(n_samples))
				ds_results[guid_i] = np.digitize(rv, bins, right=True)
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = bridge_guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_path,
										'bridge_DS_{}_{}yr_{}.csv'
										.format(hazard, rt, retrofit_key))

			d_out.to_csv(csv_filename)

		# --- mc for roads 
		for rt_i, rt in enumerate(rts):
			prnt_msg = '\tmc_road_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)

			ds_results = np.zeros((len(road_guids), n_samples))
			temp_data = road_data[rt]
			if hazard == 'eq':
				"""
				overriding eq road damage
				see comments in run_cumulative_road_damage() on roads and pgd                
				"""
				road_data[rt]['ls-slight'] = 1e-9
				road_data[rt]['ls-moderat'] = 0
				road_data[rt]['ls-extensi'] = 0
				road_data[rt]['ls-complet'] = 0

			for guid_i, guid in enumerate(road_guids):
				self.print_percent_complete(prnt_msg, guid_i, len(road_guids))

				row = road_data[rt].loc[road_data[rt]['guid']==guid].T.squeeze()
				bins = np.array([row['ls-slight'], 
								 row['ls-moderat'], 
								 row['ls-extensi'], 
								 row['ls-complet'],
								 0])
				bins[0] += 1e-9

				rv = np.random.uniform(low=0., high=1., size=(n_samples))
				ds_results[guid_i] = np.digitize(rv, bins, right=True)
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = road_guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_path,
										'road_DS_{}_{}yr.csv'
										.format(hazard, rt))

			d_out.to_csv(csv_filename)


	def Conn_analysis(self, hazard, path_to_guids, retrofit_key, n_samples):

		for fast in range(2):
			if fast == 0:
				fast_mult = 1
			elif fast == 1: 
				fast_mult = 0.5
			
			# repair time parameters for roads
			if hazard == 'eq':
				roadway_rep_time_mu = np.array([0.9, 2.2, 21])
				roadway_rep_time_std = np.array([0.05, 1.8, 16])

				bridge_rep_time_mu = np.array([0.6, 2.5, 75, 230])*fast_mult
				bridge_rep_time_std = np.array([0.6, 2.7, 42, 110])*fast_mult

			elif hazard == 'tsu':
				roadway_rep_time_mu = np.array([1, 3, 20, 30])
				roadway_rep_time_std = roadway_rep_time_mu*0.5
		
				bridge_rep_time_mu = np.array([1, 4, 30, 120])*fast_mult
				bridge_rep_time_std = bridge_rep_time_mu*0.5*fast_mult

			elif hazard == 'cumulative':
				""" assuming that the repair time parameters for cumulative 
					damage are the max of eq and tsu. """
				roadway_rep_time_mu = np.array([1, 3, 20, 30])
				roadway_rep_time_std = roadway_rep_time_mu*0.5

				bridge_rep_time_mu = np.array([1, 4, 75, 230])*fast_mult
				bridge_rep_time_std = np.array([0.6, 2.7, 42, 110])


			# COV of repair time
			roadway_rep_time_cov = roadway_rep_time_std/roadway_rep_time_mu  
			# lognormal parameters for repair time model
			roadway_rep_time_log_med = np.log(roadway_rep_time_mu/
												np.sqrt(roadway_rep_time_cov**2+1)) 
			roadway_rep_time_beta = np.sqrt(np.log(roadway_rep_time_cov**2+1))
			roadway_rep_time_covm = roadway_rep_time_beta[:, None]*\
									roadway_rep_time_beta
			
			# repair time parameters for bridges
			bridge_rep_time_cov = bridge_rep_time_std/bridge_rep_time_mu
			bridge_rep_time_log_med = np.log(bridge_rep_time_mu/np.sqrt(bridge_rep_time_cov**2+1))
			bridge_rep_time_beta = np.sqrt(np.log(bridge_rep_time_cov**2+1))
			bridge_rep_time_covm = bridge_rep_time_beta[:,None]*bridge_rep_time_beta

			rts = [100, 250, 500, 1000, 2500, 5000, 10000]
			column_keys = ['iter_{}' .format(i) for i in range(n_samples)]
			guids = os.listdir(path_to_guids)

			bldg_dataset_id = "5df40388b9219c06cf8b0c80"    # building dataset
			road_dataset_id = "5d25118eb9219c0692cd7527"    # road network
			bridge_dset_id = "5d6ede5db9219c34b56fc20b"     # bridges
			bldg_to_network_id = "5d260a6eb9219c0692db4888" # links buildings to road edges

			conn = TransConnectivity(self.client)

			conn.load_remote_input_dataset("buildings", bldg_dataset_id)
			conn.load_remote_input_dataset("road_dataset", road_dataset_id)
			conn.load_remote_input_dataset("bridge_dataset", bridge_dset_id)
			conn.load_remote_input_dataset("building_to_network", bldg_to_network_id)

			conn.set_parameter("remove_road_DS", 3)
			conn.set_parameter("remove_bridge_DS", 2)

			conn.set_parameter('roadway_reptime_log_med', roadway_rep_time_log_med)
			conn.set_parameter('roadway_reptime_covm', roadway_rep_time_covm)
			conn.set_parameter('bridge_reptime_log_med', bridge_rep_time_log_med)
			conn.set_parameter('bridge_reptime_covm', bridge_rep_time_covm)

			# --- performing connectivity analysis
			func = {}
			rep = {}
			for rt_i, rt in enumerate(rts):
				print_msg = '\tconn_analysis: {}, rt_{}, {}, fast_{}: ' \
								.format(hazard, rt, retrofit_key, fast)
				conn.set_parameter('prnt_msg', print_msg)
				road_dmg_file = 'road_DS_{}_{}yr.csv' .format(hazard, rt)
				road_dmg_file = os.path.join(self.mc_path, road_dmg_file)
				bridge_dmg_file = 'bridge_DS_{}_{}yr_{}.csv' .format(hazard, rt, retrofit_key)
				bridge_dmg_file = os.path.join(self.mc_path, bridge_dmg_file)

				# ---
				road_dmg_dset = Dataset.from_file(road_dmg_file,
					"ergo:LinkDamageInventory")
				conn.set_input_dataset("road_dmg", road_dmg_dset)

				bridge_damage_dataset = Dataset.from_file(bridge_dmg_file, 
					"ergo:NearEdgeDamageInventory")
				conn.set_input_dataset("bridge_dmg", bridge_damage_dataset)
				func[rt], rep[rt] = conn.run()
			
			# --- writing results for each guid
			for guid_i, guid in enumerate(guids):
				prnt_msg = 'writing {} guids' .format(len(guids))
				self.print_percent_complete(prnt_msg, guid_i, len(guids))

				o_path = os.path.join(path_to_guids, 
									  guid, 
									  'mc_results', 
									  'transportation',
									  )
				if not os.path.exists(o_path):
					os.makedirs(o_path)

				o_file_func = os.path.join(o_path, 
									 'func_{}_trans_{}_fast{}.gz' 
									 .format(hazard, retrofit_key, fast))
				
				o_file_rep = os.path.join(o_path, 
									 'reptime_{}_trans_{}_fast{}.gz' 
									 .format(hazard, retrofit_key, fast))

				temp_data_func = np.zeros((len(rts), n_samples))
				temp_data_rep = np.zeros((len(rts), n_samples))
				for rt_i, rt in enumerate(rts):
					temp_data_func[rt_i] = func[rt].loc[guid]
					temp_data_rep[rt_i] = rep[rt].loc[guid]

				o_df_func = pd.DataFrame(temp_data_func, index=rts, columns=column_keys)
				o_df_func.to_csv(o_file_func, compression='gzip')

				o_df_rep = pd.DataFrame(temp_data_rep, index=rts, columns=column_keys)
				o_df_rep.to_csv(o_file_rep, compression='gzip')


	def print_percent_complete(self, msg, i, n_i):
		i, n_i = int(i)+1, int(n_i)
		sys.stdout.write('\r')
		sys.stdout.write("{} ({:.1f}%)" .format(msg, (100/(n_i)*i)))
		sys.stdout.flush()
		if i==n_i:
			print()

if __name__ == "__main__":

	n_samples = 1000
	for retrofit_key_val in [0]:
		np.random.seed(1337)
		print('retrofit_key_val: {}' .format(retrofit_key_val))
		damage_output_folder = 'damage_output_r{}' .format(retrofit_key_val)
		retrofit_key = 'retrofit{}' .format(retrofit_key_val)

		output_path = os.path.join(os.getcwd(), 
								   '..', 
								   'data',
								   'pyincore_damage', 
								   'transportation'
								   )
		tax_lot_path = os.path.join(os.getcwd(),
									'..',
									'data', 
									'parcels')
		mc_path = os.path.join(output_path, 'mc_results')

		trns_dmg = transportation_damage(output_path, retrofit_key_val)


		# --- performing damage anlaysis ---
		# trns_dmg.run_road_damage(haz_type='eq')
		# trns_dmg.run_road_damage(haz_type='tsu')
		# trns_dmg.run_cumulative_road_damage()
		# trns_dmg.run_bridge_damage(haz_type='eq', retrofit_key=retrofit_key)
		# trns_dmg.run_bridge_damage(haz_type='tsu', retrofit_key=retrofit_key)
		# trns_dmg.run_cumulative_bridge_damage(retrofit_key=retrofit_key)

		# trns_dmg.DS_mc_sample(hazard = 'eq', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)
		# trns_dmg.DS_mc_sample(hazard = 'tsu', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)	
		# trns_dmg.DS_mc_sample(hazard = 'cumulative', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)


		# --- performing connectivity anlaysis ---
		# trns_dmg.Conn_analysis(hazard='eq',
		# 					   path_to_guids = tax_lot_path,
		# 					   retrofit_key = retrofit_key,
		# 					   n_samples = n_samples)
		# trns_dmg.Conn_analysis(hazard='tsu',
		# 					   path_to_guids = tax_lot_path,
		# 					   retrofit_key = retrofit_key,
		# 					   n_samples = n_samples)
		trns_dmg.Conn_analysis(hazard='cumulative',
							   path_to_guids = tax_lot_path,
							   retrofit_key = retrofit_key,
							   n_samples = n_samples)















