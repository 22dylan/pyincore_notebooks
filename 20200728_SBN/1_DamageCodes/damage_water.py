from pyincore import IncoreClient, Dataset
from pyincore.analyses.pipelinedamage import PipelineDamage
from pyincore.analyses.pipelinedamagerepairrate import PipelineDamageRepairRate

from pyincore.analyses.waterfacilitydamage import WaterFacilityDamage
from connectivityOSU import WterConnectivity, Wter2ElecConnectivity

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

class water_damage():
	def __init__(self, output_path, retrofit_key_val):
		self.client = IncoreClient()

		self.output_path = os.path.join(output_path, 
							'retrofit{}' .format(retrofit_key_val))
		
		self.mc_path = os.path.join(self.output_path, 'mc_results')
		self.wterfclty_output_path = os.path.join(self.output_path,
						'wterfclty_damage_output')

		self.pipe_output_path = os.path.join(self.output_path, 
						'pipe_damage_output')

		self.makedir(self.output_path)
		self.makedir(self.mc_path)
		self.makedir(self.wterfclty_output_path)
		self.makedir(self.pipe_output_path)

	def makedir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def run_pipeline_damage(self, haz_type):
		""" pipeline damage using pyincore"""
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]

		# Seaside pipes
		pipe_dataset_id = "5d2666b5b9219c3c5595ee65"

		if haz_type == 'eq':
			hazard_type = "earthquake"
			rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
							  250: "5dfa41aab9219c934b64d4b2",
							  500: "5dfa4300b9219c934b64d4d0",
							  1000: "5dfa3e36b9219c934b64c231",
							  2500: "5dfa4417b9219c934b64d4d3", 
							  5000: "5dfbca0cb9219c101fd8a58d",
							 10000: "5dfa51bfb9219c934b68e6c2"}

			fragility_key = "pgv"

			# seaside pipe fragility mappng for EQ
			mapping_id = "5b47c227337d4a38464efea8"
			pipeline_dmg = PipelineDamageRepairRate(self.client)

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

			# seaside pipe fragility mappng for tsunami
			mapping_id = "5d320a87b9219c6d66398b45"
			pipeline_dmg = PipelineDamage(self.client)


		# test tsunami pipeline
		pipeline_dmg.load_remote_input_dataset("pipeline", pipe_dataset_id)
		pipeline_dmg.set_parameter("mapping_id", mapping_id)
		pipeline_dmg.set_parameter("hazard_type", hazard_type)
		pipeline_dmg.set_parameter("fragility_key",fragility_key)
		pipeline_dmg.set_parameter("num_cpu", 1)

		for rt_val in rt:
			print('\tpipe_dmg: {} rt_{}' .format(haz_type, rt_val))
			result_name = os.path.join(self.pipe_output_path, 
									   'pipe_{}_{}yr_dmg' 
									   .format(haz_type, rt_val))
			hazard_id = rt_hazard_dict[rt_val]

			pipeline_dmg.set_parameter("hazard_id", hazard_id)
			pipeline_dmg.set_parameter("result_name",result_name)

			# Run pipeline damage analysis
			result = pipeline_dmg.run_analysis()


	def DS_mc_sample_pipes(self, hazard, retrofit_key, n_samples=100):
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
		pipe_files = ['pipe_{}_{}yr_dmg.csv' 
						.format(hazard, i) for i in rts]
		
		pipe_data = {}
		for i, _ in enumerate(pipe_files):
			pipe_data[rts[i]] = pd.read_csv(os.path.join(self.pipe_output_path, 
														 pipe_files[i]))            


		pipe_guids = list(pipe_data[rts[0]]['guid'])

		column_keys = ['iter_{}' .format(i) for i in range(n_samples)]

		# --- mc for pipes 
		for rt_i, rt in enumerate(rts):
			prnt_msg = '\tmc_pipe_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)

			ds_results = np.zeros((len(pipe_guids), n_samples))
			temp_data = pipe_data[rt]

			if hazard == 'eq':
				""" eq analysis passes back T/F for failure/no-failure """
				rv = np.random.uniform(low=0., high=1., size=(len(temp_data),n_samples))
				for i in range(n_samples):
					self.print_percent_complete(prnt_msg, i, n_samples)
					ds_results[:,i] = rv[:,i] <= temp_data['failprob']

			if hazard == 'tsu':
				""" tsu analysis passes back T/F for failure/no-failure """
				for guid_i, guid in enumerate(pipe_guids):
					self.print_percent_complete(prnt_msg, guid_i, len(pipe_guids))
					row = pipe_data[rt].loc[pipe_data[rt]['guid']==guid].T.squeeze()
					bins = np.array([row['ls-slight'], 
									 row['ls-moderat'], 
									 row['ls-extensi'], 
									 row['ls-complet'],
									 0])
					bins[0] += 1e-9
					rv = np.random.uniform(low=0., high=1., size=(n_samples))
					temp = np.digitize(rv, bins, right=True)
					ds_results[guid_i] = temp>1	# assuming failed when DS is mod, ext, or comp.
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = pipe_guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_path,
										'pipe_DS_{}_{}yr_{}.csv'
										.format(hazard, rt, retrofit_key))

			d_out.to_csv(csv_filename)


	def run_cumulative_pipeline_damage(self):
		""" 
		multi-hazard pipe damage according to hazus calcuations
		"""
		""" PWP1 = brittle
			PWP2 = ductile """

		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		# rt = [100]

		for rt_val in rt:
			print('\tmc_pipe_dmg: cumulative rt_{}' .format(rt_val))
			# --- reading in damage results from above analysis
			eq_damage_results_csv = os.path.join(self.mc_path, 
												 'pipe_DS_eq_{}yr_{}.csv' 
												 .format(rt_val, retrofit_key))
			tsu_damage_results_csv = os.path.join(self.mc_path, 
												  'pipe_DS_tsu_{}yr_{}.csv'
												  .format(rt_val, retrofit_key))
			eq_df = pd.read_csv(eq_damage_results_csv)
			tsu_df = pd.read_csv(tsu_damage_results_csv)

			eq_df.set_index('guid', inplace=True)
			tsu_df.set_index('guid', inplace=True)

			column_keys = list(eq_df.columns)

			cum_df = np.logical_or(eq_df.values, tsu_df.values).astype(int)
			cum_df = pd.DataFrame(cum_df, index=eq_df.index, columns=column_keys)
			

			result_name = os.path.join(self.mc_path, 
									   'pipe_DS_cumulative_{}yr_{}.csv' 
										.format(rt_val, retrofit_key))

			cum_df.to_csv(result_name, index=True)



	def run_wtrfclty_damage(self, haz_type, retrofit_key=None):
		""" water facility damage using pyincore"""

		rt = [100, 250, 500, 1000, 2500, 5000, 10000]

		fclty_dataset_id = "5d266507b9219c3c5595270c"	# water facilities
		
		if haz_type == 'eq':
			hazard_type = "earthquake"
			rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
							  250: "5dfa41aab9219c934b64d4b2",
							  500: "5dfa4300b9219c934b64d4d0",
							  1000: "5dfa3e36b9219c934b64c231",
							  2500: "5dfa4417b9219c934b64d4d3", 
							  5000: "5dfbca0cb9219c101fd8a58d",
							 10000: "5dfa51bfb9219c934b68e6c2"}

			fragility_key = "pga"       # EQ

			# wterfclty Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d39e010b9219cc18bd0b0b6"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5edfe6f04aedc80001f87dfd'	# eq retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = '5d39e010b9219cc18bd0b0b6'	# not retrofitted
			elif retrofit_key == 'retrofit3':
				mapping_id = '5edfe6f04aedc80001f87dfd'	# eq retrofitted


		elif haz_type == 'tsu':

			hazard_type = "tsunami"
			rt_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
							  250: "5df910abb9219cd00cf5f0a5",
							  500: "5df90e07b9219cd00ce971e7",
							  1000: "5df90137b9219cd00cb774ec",
							  2500: "5df90761b9219cd00ccff258",
							  5000: "5df90871b9219cd00ccff273",
							  10000: "5d27b986b9219c3c55ad37d0"}

			fragility_key = "Non-Retrofit inundationDepth Fragility ID Code"  # seaside bridges - TSU

			#  wterfclty Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d31f737b9219c6d66398521"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5d31f737b9219c6d66398521'	# not retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = '5d31f737b9219c6d66398521'	# tsu retrofitted
				raise ValueError('Ensure that waterfacilitydamage.py hazard_val is modified for raising wps')
			elif retrofit_key == 'retrofit3':
				mapping_id = '5d31f737b9219c6d66398521'	# tsu retrofitted
				raise ValueError('Ensure that waterfacilitydamage.py hazard_val is modified for raising wps')

		wterfclty_dmg = WaterFacilityDamage(self.client)

		# test tsunami pipeline
		wterfclty_dmg.load_remote_input_dataset("water_facilities", fclty_dataset_id)
		wterfclty_dmg.set_parameter("mapping_id", mapping_id)
		wterfclty_dmg.set_parameter("hazard_type", hazard_type)
		wterfclty_dmg.set_parameter("fragility_key",fragility_key)
		wterfclty_dmg.set_parameter("num_cpu", 1)

		for rt_val in rt:
			print('\twterfclty_dmg: {} rt_{}' .format(haz_type, rt_val))
			result_name = os.path.join(self.wterfclty_output_path, 
									   'wterfclty_{}_{}yr_dmg' 
									   .format(haz_type, rt_val))
			hazard_id = rt_hazard_dict[rt_val]

			wterfclty_dmg.set_parameter("hazard_id", hazard_id)
			wterfclty_dmg.set_parameter("result_name",result_name)

			# Run pipeline damage analysis
			result = wterfclty_dmg.run_analysis()



	def run_cumulative_wtrfclty_damage(self):
		""" 
		multi-hazard pipe damage according to hazus calcuations
		"""

		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		
		for rt_val in rt:
			print('\twterfclty_dmg: cumulative rt_{}' .format(rt_val))
			# --- reading in damage results from above analysis
			eq_damage_results_csv = os.path.join(self.wterfclty_output_path, 
												 'wterfclty_eq_{}yr_dmg.csv' 
												 .format(rt_val))
			tsu_damage_results_csv = os.path.join(self.wterfclty_output_path, 
												  'wterfclty_tsu_{}yr_dmg.csv' 
												  .format(rt_val))
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

			result_name = os.path.join(self.wterfclty_output_path, 
									   'wterfclty_cumulative_{}yr_dmg.csv' 
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
			



	def DS_mc_sample_wterfclty(self, hazard, retrofit_key, n_samples=100):
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
		wterfclty_files = ['wterfclty_{}_{}yr_dmg.csv' 
						.format(hazard, i) for i in rts]
		
		wterfclty_data = {}
		for i, _ in enumerate(wterfclty_files):
			wterfclty_data[rts[i]] = pd.read_csv(os.path.join(self.wterfclty_output_path, 
														 wterfclty_files[i]))            
		
		wterfclty_guids = list(wterfclty_data[rts[0]]['guid'])
	
		column_keys = ['iter_{}' .format(i) for i in range(n_samples)]

		# --- mc for bridges 
		for rt_i, rt in enumerate(rts):
			prnt_msg = '\tmc_wterfclty_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)

			ds_results = np.zeros((len(wterfclty_guids), n_samples))
			temp_data = wterfclty_data[rt]

			for guid_i, guid in enumerate(wterfclty_guids):
				self.print_percent_complete(prnt_msg, guid_i, len(wterfclty_guids))
				row = wterfclty_data[rt].loc[wterfclty_data[rt]['guid']==guid].T.squeeze()
				bins = np.array([row['ls-slight'], 
								 row['ls-moderat'], 
								 row['ls-extensi'], 
								 row['ls-complet'],
								 0])
				bins[0] += 1e-9
				rv = np.random.uniform(low=0., high=1., size=(n_samples))
				ds_results[guid_i] = np.digitize(rv, bins, right=True)
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = wterfclty_guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_path,
										'wterfclty_DS_{}_{}yr_{}.csv'
										.format(hazard, rt, retrofit_key))

			d_out.to_csv(csv_filename)


	def epn_wter_conn(self, hazard, retrofit_epn_key_val):
		elec_mc_path = os.path.join(self.output_path,
						'..',
						'..',
						'electric',
						'retrofit{}' .format(retrofit_epn_key_val),
						'mc_results'
						)
		
		for efast in range(2):
			if efast == 0:
				fast_mult = 1
			elif efast == 1: 
				fast_mult = 0.5
			
			# repair time parameters for poles
			if hazard == 'eq':
				pole_rep_time_mu = np.array([0.3, 1.0, 3.0, 7.0])*fast_mult 
				pole_rep_time_std = np.array([0.2, 0.5, 1.5, 3.0])*fast_mult        

			elif hazard == 'tsu':
				pole_rep_time_mu = np.array([1, 5, 20, 90])*fast_mult
				pole_rep_time_std = np.array([1, 5, 20, 90])*0.5*fast_mult

			elif hazard == 'cumulative':
				""" assuming that the repair time parameters for cumulative 
					damage are the max of eq and tsu. """
				pole_rep_time_mu = np.array([1, 5, 20, 90])*fast_mult
				pole_rep_time_std = np.array([1, 5, 20, 90])*0.5*fast_mult

			# COV of repair time  
			pole_rep_time_cov = pole_rep_time_std/pole_rep_time_mu  
			# lognormal parameters for repair time model
			pole_rep_time_log_med = np.log(pole_rep_time_mu/
												np.sqrt(pole_rep_time_cov**2+1)) 
			pole_rep_time_beta = np.sqrt(np.log(pole_rep_time_cov**2+1))
			pole_rep_time_covm = pole_rep_time_beta[:, None]*pole_rep_time_beta
			
			rts = [100, 250, 500, 1000, 2500, 5000, 10000]
			column_keys = ['iter_{}' .format(i) for i in range(n_samples)]

			line_dset_id = "5d263df6b9219cf93c056c37"	# line data
			pole_dset_id = "5d263f08b9219cf93c056c68"    # pole data


			""" the way critical nodes is setup is best given through an example:
				with the setup below, the connectivity analysis
				determines whether each tax-lot is connected to:
					- (node 229 OR node 230) AND (node 300)
				
				so the nodes in each inner lists undergo a logical_or 
				statement, whereas these results undergo a logical_and.

			"""

			from_nodes = {66: '402388b9-f14c-4402-9ad9-8b0965de0937', 	# wtp
						  1: '584f8368-c4ac-42bd-a39a-230517a5e13e', 	# wps1
						  318:'9b38bb0c-a2e9-4068-90e8-4bf0479b3a9e', 	# wps2
						  244: '0d268d61-1733-48ee-bab9-8126b3609522'	# wps3
						  }
						  # wtp, wps1, wps2, wps3

			critical_nodes = [[211]]        # substation node

			conn = Wter2ElecConnectivity(self.client)

			conn.load_remote_input_dataset("line_dataset", line_dset_id)
			conn.load_remote_input_dataset("pole_dataset", pole_dset_id)

			conn.set_parameter('from_nodes', from_nodes)
			conn.set_parameter("critical_nodes", critical_nodes)
			conn.set_parameter("remove_pole_DS", 3)

			conn.set_parameter('pole_reptime_log_med', pole_rep_time_log_med)
			conn.set_parameter('pole_reptime_covm', pole_rep_time_covm)
			
			for rt_i, rt in enumerate(rts):
				print_msg = '\tepn_wter_conn: {}, rt_{}, elec_retrofit{}, elec_fast{}: ' \
								.format(hazard, rt, retrofit_epn_key_val, efast)
				conn.set_parameter('prnt_msg', print_msg)
				pole_dmg_file = 'electric_DS_{}_{}yr_retrofit{}.csv' .format(hazard, rt, retrofit_epn_key_val)
				pole_dmg_file = os.path.join(elec_mc_path, pole_dmg_file)

				# ---
				pole_dmg_dset = Dataset.from_file(pole_dmg_file,"ergo:NodeDamageInventory")
				conn.set_input_dataset("pole_dmg", pole_dmg_dset)

				func, rep = conn.run()

				o_path = os.path.join(self.output_path,
									'..',
									'wter2elec')
				if not os.path.exists(o_path):
					os.makedirs(o_path)

				o_file_func = os.path.join(o_path,
								'func_{}_{}yr_wter2elec_eretrofit{}_efast{}.csv'
								.format(hazard, rt, retrofit_epn_key_val, efast))
				o_file_rep= os.path.join(o_path,
								'reptime_{}_{}yr_wter2elec_eretrofit{}_efast{}.csv'
								.format(hazard, rt, retrofit_epn_key_val, efast))
				func.to_csv(o_file_func)
				rep.to_csv(o_file_rep)
				


	def Conn_analysis(self, hazard, path_to_guids, retrofit_key, eretrofit, n_samples):
		"""
		performing connectivity analysis bw each parcel and their respective 
			pumping station (1, 2, 3).
		There are two prior dependencies:
			each pumping station is connected to the water treatment plant
			each pumping station and treatment plant is connected to electricity
		"""
		for fast in range(2):
			if fast == 0:
				n_workers = 32
				fast_mult = 1.
			elif fast == 1: 
				n_workers = 64
				fast_mult = 0.5
			""" using the probability of failure, rather than leak/break.
				assuming that the repair rate is the average of the leak/break
				repair rates from hazus.
						break 	leak	avg.
				> 20" - 0.33 	0.66	0.5
				< 20"	0.5		1.0 	0.75
			"""
			pipe_reprate = [0.5, 0.75]	# Fixed pipes per Day per Worker (>20", <20" diameter)

			# repair time parameters for roads
			if hazard == 'eq':
				wtp_rep_time_mu = np.array([0.9, 1.9, 32, 95])*fast_mult # mean repair time for water treatement plants for DS2-DS5
				wtp_rep_time_std = np.array([0.3, 1.2, 31, 65])*fast_mult # std for repair time		

				wps_rep_time_mu = np.array([0.9, 3.1, 13.5, 35])*fast_mult # mean repair time for water treatement plants for DS2-DS5
				wps_rep_time_std = np.array([0.3, 2.7, 10, 18])*fast_mult # std for repair time		

			elif hazard == 'tsu':
				wtp_rep_time_mu = np.array([1, 6, 20, 90])*fast_mult # mean repair time for water treatement plants for DS2-DS5
				wtp_rep_time_std = np.array([1, 6, 20, 90])*fast_mult # std for repair time		

				wps_rep_time_mu = np.array([1, 6, 20, 240])*fast_mult # mean repair time for water treatement plants for DS2-DS5
				wps_rep_time_std = np.array([1, 6, 20, 120])*fast_mult # std for repair time		

			elif hazard == 'cumulative':
				""" assuming that the repair time parameters for cumulative 
					damage are the max of eq and tsu. """
				wtp_rep_time_mu = np.array([1, 6, 32, 95])*fast_mult
				wtp_rep_time_std = np.array([1, 6, 31, 65])*fast_mult

				wps_rep_time_mu = np.array([1, 6, 20, 240])*fast_mult
				wps_rep_time_std = np.array([1, 6, 20, 120])*fast_mult

			wtp_rep_time_cov = wtp_rep_time_std/wtp_rep_time_mu # COV of repiar time
			wtp_rep_time_log_med = np.log(wtp_rep_time_mu/np.sqrt(wtp_rep_time_cov**2+1)) # lognormal parameters for repair time model
			wtp_rep_time_beta = np.sqrt(np.log(wtp_rep_time_cov**2+1))
			wtp_rep_time_covm = wtp_rep_time_beta[:,None]*wtp_rep_time_beta

			wps_rep_time_cov = wps_rep_time_std/wps_rep_time_mu # COV of repiar time
			wps_rep_time_log_med = np.log(wps_rep_time_mu/np.sqrt(wps_rep_time_cov**2+1)) # lognormal parameters for repair time model
			wps_rep_time_beta = np.sqrt(np.log(wps_rep_time_cov**2+1))
			wps_rep_time_covm = wps_rep_time_beta[:,None]*wps_rep_time_beta

			rts = [100, 250, 500, 1000, 2500, 5000, 10000]
			column_keys = ['iter_{}' .format(i) for i in range(n_samples)]
			guids = os.listdir(path_to_guids)

			bldg_dataset_id = "5df40388b9219c06cf8b0c80"    # building dataset
			pipe_dataset_id = "5d2666b5b9219c3c5595ee65"    # water pipes
			wterfclty_dataset_id = "5d266507b9219c3c5595270c"
			bldg_to_network_id = "5d260b12b9219c0692dca091" # links buildings to road edges
			
			""" the way critical nodes is setup is best given through an example:
				with the setup below, the connectivity analysis
				determines whether each tax-lot is connected to:
					- (node 229 OR node 230) AND (node 300)
				
				so the nodes in each inner lists undergo a logical_or 
				statement, whereas these results undergo a logical_and.

			"""

			conn = WterConnectivity(self.client)

			conn.load_remote_input_dataset("buildings", bldg_dataset_id)
			conn.load_remote_input_dataset("pipe_dataset", pipe_dataset_id)
			conn.load_remote_input_dataset("wterfclty_dataset", wterfclty_dataset_id)
			conn.load_remote_input_dataset("building_to_network", bldg_to_network_id)

			conn.set_parameter('n_workers', n_workers)
			conn.set_parameter('pipe_reprate', pipe_reprate)
			conn.set_parameter('wtp_rep_time_log_med', wtp_rep_time_log_med)
			conn.set_parameter('wtp_rep_time_covm', wtp_rep_time_covm)
			conn.set_parameter('wps_rep_time_log_med', wps_rep_time_log_med)
			conn.set_parameter('wps_rep_time_covm', wps_rep_time_covm)

			for efast in range(2):
				# --- performing connectivity analysis
				func = {}
				rep = {}
				for rt_i, rt in enumerate(rts):
					print_msg = '\tconn_analysis: {}, rt_{}, {}, fast{}, eretrofit{}, efast{}:' \
									.format(hazard, rt, retrofit_key, fast, eretrofit, efast)
					
					conn.set_parameter('prnt_msg', print_msg)

					wter2elec_func = 'func_cumulative_{}yr_wter2elec_eretrofit{}_efast{}.csv' \
									.format(rt, eretrofit, efast)
					wter2elec_func = os.path.join(self.output_path,'..','wter2elec',wter2elec_func)

					wter2elec_rept = 'reptime_cumulative_{}yr_wter2elec_eretrofit{}_efast{}.csv' \
									.format(rt, eretrofit, efast)
					wter2elec_rept = os.path.join(self.output_path,'..','wter2elec',wter2elec_rept)

					pipe_dmg_file = 'pipe_DS_{}_{}yr_{}.csv' .format(hazard, rt, retrofit_key)
					pipe_dmg_file = os.path.join(self.mc_path, pipe_dmg_file)
					
					wterfclty_dmg_file = 'wterfclty_DS_{}_{}yr_{}.csv' .format(hazard, rt, retrofit_key)
					wterfclty_dmg_file = os.path.join(self.mc_path, wterfclty_dmg_file)

					# ---
					wter2elec_func_dset = Dataset.from_file(wter2elec_func, "ergo:DamageInventory")
					conn.set_input_dataset("wter2elec_func", wter2elec_func_dset)

					wter2elec_rept_dset = Dataset.from_file(wter2elec_rept, "ergo:DamageInventory")
					conn.set_input_dataset("wter2elec_rep", wter2elec_rept_dset)
					
					pipe_dmg_dset = Dataset.from_file(pipe_dmg_file, "ergo:DamageInventory")
					conn.set_input_dataset("pipe_dmg", pipe_dmg_dset)

					wterfclty_damage_dataset = Dataset.from_file(wterfclty_dmg_file, "ergo:DamageInventory")
					conn.set_input_dataset("wterfclty_dmg", wterfclty_damage_dataset)
					
					func[rt], rep[rt] = conn.WterConn_run()
					
					# temp_func = func[rt].head(5)
					# temp_rep = rep[rt].head(5)
					# print(temp_func.mean(axis=1))
					# print(temp_rep.mean(axis=1))

				# --- writing results for each guid
				for guid_i, guid in enumerate(guids):
					prnt_msg = 'writing {} guids' .format(len(guids))
					self.print_percent_complete(prnt_msg, guid_i, len(guids))

					o_path = os.path.join(path_to_guids, 
										  guid, 
										  'mc_results', 
										  'water',
										  )
					if not os.path.exists(o_path):
						os.makedirs(o_path)

					o_file_func = os.path.join(o_path, 
										 'func_{}_wter_{}_fast{}_eretrofit{}_efast{}.gz' 
										 .format(hazard, retrofit_key, fast, eretrofit, efast))
					
					o_file_rep = os.path.join(o_path, 
										 'reptime_{}_wter_{}_fast{}_eretrofit{}_efast{}.gz' 
										 .format(hazard, retrofit_key, fast, eretrofit, efast))

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
								   'water'
								   )
		tax_lot_path = os.path.join(os.getcwd(),
									'..',
									'data', 
									'parcels')
		mc_path = os.path.join(output_path, 'mc_results')

		wter_dmg = water_damage(output_path, retrofit_key_val)

		""" Pipe damage is different than other damage analysis. 
			Process as follows:
				1) earthquake pipe damage.
				2) tsunami pipe damage.
				3) eq MC damage (results in fail T/F)
				4) tsu MC damage (results in fail T/F)
				5) cumulative MC damage (results in fail T/F)

		"""
		# wter_dmg.run_pipeline_damage(haz_type='eq')
		# wter_dmg.run_pipeline_damage(haz_type='tsu')

		# wter_dmg.DS_mc_sample_pipes(hazard = 'eq', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)

		# wter_dmg.DS_mc_sample_pipes(hazard = 'tsu', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)

		# wter_dmg.run_cumulative_pipeline_damage()
		

		# --- performing water facility damage anlaysis ---
		# wter_dmg.run_wtrfclty_damage(haz_type='eq', retrofit_key=retrofit_key)
		# wter_dmg.run_wtrfclty_damage(haz_type='tsu', retrofit_key=retrofit_key)
		# wter_dmg.run_cumulative_wtrfclty_damage()
		

		# wter_dmg.DS_mc_sample_wterfclty(hazard = 'eq', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)
		# wter_dmg.DS_mc_sample_wterfclty(hazard = 'tsu', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)	
		# wter_dmg.DS_mc_sample_wterfclty(hazard = 'cumulative', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)



		for epn_retrofit in [0, 1, 2, 3]:
			if retrofit_key_val == 0:
				# wter_dmg.epn_wter_conn(hazard='eq', retrofit_epn_key_val=epn_retrofit)
				# wter_dmg.epn_wter_conn(hazard='tsu', retrofit_epn_key_val=epn_retrofit)
				# wter_dmg.epn_wter_conn(hazard='cumulative', retrofit_epn_key_val=epn_retrofit)
				pass

			# wter_dmg.Conn_analysis(hazard='eq',
			# 						path_to_guids = tax_lot_path,
			# 						retrofit_key = retrofit_key,
			# 						eretrofit = epn_retrofit,
			# 						n_samples = n_samples)
			# wter_dmg.Conn_analysis(hazard='tsu',
			# 						path_to_guids = tax_lot_path,
			# 						retrofit_key = retrofit_key,
			# 						eretrofit = epn_retrofit,
			# 						n_samples = n_samples)
			wter_dmg.Conn_analysis(hazard='cumulative',
									path_to_guids = tax_lot_path,
									retrofit_key = retrofit_key,
									eretrofit = epn_retrofit,
									n_samples = n_samples)














