
from pyincore import IncoreClient, Dataset
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.cumulativebuildingdamage import CumulativeBuildingDamage

import os
import sys
import numpy as np
import pandas as pd

"""
this code uses pyincore to perform a building damage anlaysis for Seaside, OR.
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

class building_damage():
	def __init__(self, output_path, guid_path):
		self.client = IncoreClient()

		# checking if output path exists, and creating if it doesn't
		self.output_path = output_path
		self.guid_path = guid_path
		self.bldg_dmg_path = os.path.join(self.output_path, 'building_damage_output')
		self.mc_dmg_path = os.path.join(self.output_path, 'mc_results')

		self.makedir(self.output_path)
		self.makedir(self.guid_path)
		self.makedir(self.bldg_dmg_path)
		self.makedir(self.mc_dmg_path)

	def makedir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def run_eq_damage(self, retrofit_val):
		# --- earthquake damage using pyincore
		hazard_type = "earthquake"
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
						  250: "5dfa41aab9219c934b64d4b2",
						  500: "5dfa4300b9219c934b64d4d0",
						  1000: "5dfa3e36b9219c934b64c231",
						  2500: "5dfa4417b9219c934b64d4d3", 
						  5000: "5dfbca0cb9219c101fd8a58d",
						 10000: "5dfa51bfb9219c934b68e6c2"}

		bldg_dmg = BuildingDamage(self.client)   # initializing pyincore
		# defining building dataset (GIS point layer)
		bldg_dataset_id = "5df40388b9219c06cf8b0c80"
		bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)

		# specifiying mapping id from fragilites to building types
		if retrofit_val == 'retrofit0':
			mapping_id = "5d2789dbb9219c3c553c7977"        # retrofit_0 
		elif retrofit_val == 'retrofit1':
			mapping_id = "5e99cd959c68d00001cacf4a"        # retrofit_1
		elif retrofit_val == 'retrofit2':
			mapping_id = "5e99d0bff2935b0001190085"        # retrofit_2
		elif retrofit_val == 'retrofit3':
			mapping_id = "5e99d145f2935b00011900a4"        # retrofit_3

		bldg_dmg.set_parameter("hazard_type", hazard_type)
		bldg_dmg.set_parameter("mapping_id", mapping_id)
		bldg_dmg.set_parameter("num_cpu", 4)

		for rt_val in rt:
			print('\tearthquake: rt_{}' .format(rt_val))
			result_name = os.path.join(self.bldg_dmg_path,
									   'buildings_eq_{}yr_dmg_{}' 
									   .format(rt_val, retrofit_val))
			hazard_id = rt_hazard_dict[rt_val]
			bldg_dmg.set_parameter("hazard_id", hazard_id)
			bldg_dmg.set_parameter("result_name", result_name)

			bldg_dmg.run_analysis()


	def run_tsu_damage(self, retrofit_val):
		# --- tsunami damage using pyincore
		hazard_type = "tsunami"
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		rt_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
						  250: "5df910abb9219cd00cf5f0a5",
						  500: "5df90e07b9219cd00ce971e7",
						  1000: "5df90137b9219cd00cb774ec",
						  2500: "5df90761b9219cd00ccff258",
						  5000: "5df90871b9219cd00ccff273",
						  10000: "5d27b986b9219c3c55ad37d0"}

		bldg_dmg = BuildingDamage(self.client)
		bldg_dataset_id = "5df40388b9219c06cf8b0c80"
		bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)
		
		# specifiying mapping id from fragilites to building types
		if retrofit_val == 'retrofit0':
			mapping_id = "5d279bb9b9219c3c553c7fba"        # retrofit_0 
		elif retrofit_val == 'retrofit1':
			mapping_id = "5e99d887217aaf0001e55562"        # retrofit_1
		elif retrofit_val == 'retrofit2':
			mapping_id = "5e99d923f2935b0001190db1"        # retrofit_2
		elif retrofit_val == 'retrofit3':
			mapping_id = "5e99d99c9c68d00001cafaba"        # retrofit_3

		bldg_dmg.set_parameter("hazard_type", hazard_type)
		bldg_dmg.set_parameter("mapping_id", mapping_id)
		bldg_dmg.set_parameter("num_cpu", 4)

		for rt_val in rt:
			print('\ttsunami: rt_{}' .format(rt_val))
			result_name = os.path.join(self.bldg_dmg_path, 
									   'buildings_tsu_{}yr_dmg_{}'
									   .format(rt_val, retrofit_val))
			hazard_id = rt_hazard_dict[rt_val]
			bldg_dmg.set_parameter("hazard_id", hazard_id)
			bldg_dmg.set_parameter("result_name", result_name)

			bldg_dmg.run_analysis()

	def run_cumulative_damage(self, retrofit_val):
		# --- multi-hazard building damage using pyincore
		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		cumulative_bldg_dmg = CumulativeBuildingDamage(self.client)
		# setting number of cpus for parallel processing
		cumulative_bldg_dmg.set_parameter("num_cpu", 4)

		for rt_val in rt:
			print('\tcumulative: rt_{}' .format(rt_val))
			# reading in damage results from above analysis
			eq_damage_results_csv = os.path.join(self.bldg_dmg_path,
												 'buildings_eq_{}yr_dmg_{}.csv' 
												 .format(rt_val, retrofit_val))
			tsu_damage_results_csv = os.path.join(self.bldg_dmg_path,
												  'buildings_tsu_{}yr_dmg_{}.csv' 
												  .format(rt_val, retrofit_val))
			
			# loading datasets from CSV files into pyincore
			eq_damage_dataset = Dataset.from_file(eq_damage_results_csv, 
													"ergo:buildingDamageVer4")
			tsu_damage_dataset = Dataset.from_file(tsu_damage_results_csv, 
													"ergo:buildingDamageVer4")
			
			cumulative_bldg_dmg.set_input_dataset("eq_bldg_dmg", 
												  eq_damage_dataset)
			cumulative_bldg_dmg.set_input_dataset("tsunami_bldg_dmg", 
												  tsu_damage_dataset)
			
			# defining path to output 
			result_name = os.path.join(self.bldg_dmg_path,
									   'buildings_cumulative_{}yr_dmg_{}' 
									   .format(rt_val, retrofit_val))
			cumulative_bldg_dmg.set_parameter("result_name", result_name)

			# running analysis
			cumulative_bldg_dmg.run_analysis()

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
		print(hazard)
		rts = [100, 250, 500, 1000, 2500, 5000, 10000]
		# --- new code
		column_keys = ['iter_{}' .format(i) for i in range(n_samples)]

		for rt_i, rt in enumerate(rts):
			prnt_msg = '\tmc_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)
			file = 'buildings_{}_{}yr_dmg_{}.csv' .format(hazard, rt, retrofit_key)
			pyincore_data = os.path.join(self.bldg_dmg_path, file)
			pyincore_data = pd.read_csv(pyincore_data)


			ds_results = np.zeros((len(pyincore_data), n_samples))
			guids = pyincore_data['guid']
			for guid_i, guid in enumerate(guids):
				self.print_percent_complete(prnt_msg, guid_i, len(guids))
				row = pyincore_data.loc[pyincore_data['guid']==guid].T.squeeze()
				bins = np.array([ 
								row['complete'] + row['heavy'] + row['moderate'],
								row['complete'] + row['heavy'],
								row['complete'],
								0
								])
				bins[0] += 1e-9
				rv = np.random.uniform(low=0., high=1., size=(n_samples))
				ds_results[guid_i] = np.digitize(rv, bins, right=True)
			# print()
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_dmg_path,
										'building_DS_{}_{}yr_{}.csv'
										.format(hazard, rt, retrofit_key))

			d_out.to_csv(csv_filename)


	def func_reptime_calcs(self, hazard, retrofit_key):
		""" determining the functionality and repair time of each building. 
			functinoality defined as DS <= 1 (insignificant) 

			inputs:
				- hazard: 'eq', 'tsu', 'cumulative'
				- path_out: path to where MC results should be written. 
				- retrofit_key: 0, 1, 2, 3 corresponding to minimum retrofit
				  levels as status quo, pre-, moderate-, or high-code.
		"""
		print(hazard)
		rts = [100, 250, 500, 1000, 2500, 5000, 10000]
		fast_vals = [1, 0.75, 0.5, 0.25]

		# --- new code ---
		func = {}
		rep = {}
		np.random.seed(1338)
		for rt_i, rt in enumerate(rts):

			bldg_dmg_file = 'building_DS_{}_{}yr_{}.csv' .format(hazard, rt, retrofit_key)
			bldg_dmg_file = os.path.join(self.mc_dmg_path, bldg_dmg_file)
			bldg_dmg = pd.read_csv(bldg_dmg_file)
			bldg_dmg.set_index('guid', inplace=True)

			# functionality analysis
			func[rt] = bldg_dmg

			# repair time analysis
			rep[rt] = {}
			n_tax_lots, n_iter = np.shape(bldg_dmg)

			prnt_msg = '\tfunc/rep: {} rt_{}, {}' .format(hazard, rt, retrofit_key)
			count = 0
			n_count = len(fast_vals)*n_iter
			for fast_i, fast_mult in enumerate(fast_vals):
				# restoration times; HAZUS-MH table 15.10 and 15.11
				rep_time_mu = np.array([0.5, 60, 360, 720])*fast_mult
				# std for repair time
				rep_time_std = np.array([0.5, 0.5, 0.5, 0.5])*rep_time_mu
				# COV of repair time
				rep_time_cov = rep_time_std/rep_time_mu 
				# lognormal parameters for repair time model
				rep_time_log_med = np.log(rep_time_mu/np.sqrt(rep_time_cov**2+1))
				rep_time_beta = np.sqrt(np.log(rep_time_cov**2+1))
				rep_time_covm = rep_time_beta[:, None]*rep_time_beta

				rep_time_all = np.exp(np.random.multivariate_normal(
														rep_time_log_med, 
														rep_time_covm, 
														(n_tax_lots,n_iter)))
				
				reptime_save = np.zeros(np.shape(bldg_dmg))
				for key_i, key in enumerate(bldg_dmg.keys()):
					self.print_percent_complete(prnt_msg, count, n_count)
					temp = rep_time_all[:,key_i, :]
					# getting repair times for specific damage states (0,1,2,or 3)
					rep_time = [temp[int(ii), int(obj)] for ii, obj in enumerate(bldg_dmg[key])]
					reptime_save[:, key_i] = rep_time
					count += 1
				rep[rt][fast_i] = pd.DataFrame(reptime_save, index=bldg_dmg.index)
		
		guids = bldg_dmg.index
		column_keys = list(bldg_dmg.columns)  # getting iteration names
		n_samples = len(column_keys)

		prnt_msg = 'writing {} guids' .format(len(guids))
		for guid_i, guid in enumerate(guids):
			self.print_percent_complete(prnt_msg, guid_i, len(guids))
			o_path = os.path.join(self.guid_path, 
								  guid, 
								  'mc_results', 
								  'building',
								  )
			if not os.path.exists(o_path):
				os.makedirs(o_path)

			for fast_i, fast in enumerate(fast_vals):
				o_file_func = os.path.join(o_path, 
									 'func_{}_bldg_{}_fast{}.gz' 
									 .format(hazard, retrofit_key, fast_i))
				
				o_file_rep = os.path.join(o_path, 
									 'reptime_{}_bldg_{}_fast{}.gz' 
									 .format(hazard, retrofit_key, fast_i))

				temp_data_func = np.zeros((len(rts), n_samples))
				temp_data_rep = np.zeros((len(rts), n_samples))
				for rt_i, rt in enumerate(rts):
					temp_data_func[rt_i,:] = func[rt].loc[guid]
					temp_data_rep[rt_i,:] = rep[rt][fast_i].loc[guid]
				
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
	for retrofit_key_val in [0, 1, 2, 3]:
		np.random.seed(1337)
		print('retrofit_key_val: {}' .format(retrofit_key_val))
		damage_output_folder = 'retrofit{}' .format(retrofit_key_val)
		retrofit_key = 'retrofit{}' .format(retrofit_key_val)

		output_path = os.path.join(os.getcwd(), 
								   '..', 
								   'data',
								   'pyincore_damage', 
								   'buildings',
								   damage_output_folder)

		guid_path = os.path.join(os.getcwd(),
									'..',
									'data', 
									'parcels')

		bldg_dmg = building_damage(output_path, guid_path)

		# --- performing building damage anlaysis ---
		# print('\n--- Running Pyincore ---')
		# bldg_dmg.run_eq_damage(retrofit_key)
		# bldg_dmg.run_tsu_damage(retrofit_key)
		# bldg_dmg.run_cumulative_damage(retrofit_key)


		# print('\n--- MC DS sampling ---')
		# bldg_dmg.DS_mc_sample(hazard = 'eq', 
		#                       retrofit_key = retrofit_key, 
		#                       n_samples = n_samples)
		# bldg_dmg.DS_mc_sample(hazard = 'tsu', 
		#                       retrofit_key = retrofit_key, 
		#                       n_samples = n_samples)
		# bldg_dmg.DS_mc_sample(hazard = 'cumulative', 
		#                       retrofit_key = retrofit_key, 
		#                       n_samples = n_samples)

		""" from debugging, the seed below needs to be different from the one
		    specified above; unsure of reason why. presuming it has to do with
		    np.random.uniform() called in both """
		print('\n--- MC Performance Sampling ---')
		# bldg_dmg.func_reptime_calcs(hazard = 'eq', 
		#                                retrofit_key = retrofit_key)
		# bldg_dmg.func_reptime_calcs(hazard = 'tsu', 
		#                                retrofit_key = retrofit_key)
		bldg_dmg.func_reptime_calcs(hazard = 'cumulative', 
		                               retrofit_key = retrofit_key)
		













