from pyincore import IncoreClient, Dataset
from pyincore.analyses.epfdamage import EpfDamage
from backend.connectivityOSU import EPNConnectivity

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

class electric_damage():
	def __init__(self, output_path, retrofit_key_val):
		self.client = IncoreClient()

		self.output_path = os.path.join(output_path, 
							'retrofit{}' .format(retrofit_key_val))
		
		self.electric_output_path = os.path.join(self.output_path, 
						'electric_damage_output')
		self.mc_path = os.path.join(self.output_path, 'mc_results')

		self.makedir(self.output_path)
		self.makedir(self.electric_output_path)
		self.makedir(self.mc_path)

	def makedir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)


	def run_electric_damage(self, haz_type):
		rts = [100, 250, 500, 1000, 2500, 5000, 10000]
		poles_ss_id = "5d263f08b9219cf93c056c68"     # elelctric power poles and substation
			
		if haz_type == 'eq':
			hazard_type = 'earthquake'
			rt_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
							  250: "5dfa41aab9219c934b64d4b2",
							  500: "5dfa4300b9219c934b64d4d0",
							  1000: "5dfa3e36b9219c934b64c231",
							  2500: "5dfa4417b9219c934b64d4d3", 
							  5000: "5dfbca0cb9219c101fd8a58d",
							 10000: "5dfa51bfb9219c934b68e6c2"}

			fragility_key = "pga"       # EQ

			# Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d489aa1b9219c0689f1988e"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5ec6b65fa6fcb40001d00d49'	# eq retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = "5d489aa1b9219c0689f1988e"	# not retrofitted
			elif retrofit_key == 'retrofit3':
				mapping_id = '5ec6b65fa6fcb40001d00d49'	# eq retrofitted

		elif haz_type == 'tsu':
			hazard_type = "tsunami"
			rt_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
							  250: "5df910abb9219cd00cf5f0a5",
							  500: "5df90e07b9219cd00ce971e7",
							  1000: "5df90137b9219cd00cb774ec",
							  2500: "5df90761b9219cd00ccff258",
							  5000: "5df90871b9219cd00ccff273",
							  10000: "5d27b986b9219c3c55ad37d0"}
			fragility_key = "Non-Retrofit inundationDepth Fragility ID Code"  # TSU

			# Fragility Mapping on incore-service
			if retrofit_key == 'retrofit0':
				mapping_id = "5d31eb7fb9219c6d66398445"	# not retrofitted
			elif retrofit_key == 'retrofit1':
				mapping_id = '5d31eb7fb9219c6d66398445'	# not retrofitted
			elif retrofit_key == 'retrofit2':
				mapping_id = '5ec6ba10a6dffd00017ec46b'	# tsu retrofitted
			elif retrofit_key == 'retrofit3':
				mapping_id = '5ec6ba10a6dffd00017ec46b'	# tsu retrofitted


		# Run epf damage
		epf_dmg = EpfDamage(self.client)

		epf_dmg.load_remote_input_dataset("epfs", poles_ss_id)
		epf_dmg.set_parameter("mapping_id", mapping_id)
		epf_dmg.set_parameter("hazard_type", hazard_type)
		epf_dmg.set_parameter("num_cpu", 1)
		epf_dmg.set_parameter('fragility_key', fragility_key)

		# in loop
		for rt_val in rts:
			print('\telectric_dmg: {} rt_{}' .format(haz_type, rt_val))
			result_name = os.path.join(self.electric_output_path, 
										'electric_{}_{}yr_dmg' 
										.format(haz_type, rt_val))
			hazard_id = rt_hazard_dict[rt_val]

			epf_dmg.set_parameter("hazard_id", hazard_id)
			epf_dmg.set_parameter("result_name", result_name)

			# Run Analysis
			epf_dmg.run_analysis()

			""" some of the epf probabilites were negative (issue in hazus 
				fragilities). here I'm updating those probabilties such that
				they are not negative """
			result_name = result_name + '.csv'
			output = pd.read_csv(result_name)
			cruc_cols = ['ds-none', 'ds-slight', 'ds-moderat', 'ds-extensi', 'ds-complet']
			if any(output[cruc_cols]<0):
				for col_i, col in enumerate(cruc_cols):
					temp_data = output[col].loc[output[col]<0]

					output[col].loc[output[col]<0] = 0
					output[cruc_cols[col_i-1]].loc[temp_data.index] = \
						output[cruc_cols[col_i-1]].loc[temp_data.index]+temp_data

				output['ls-slight'] = output['ds-slight'] + output['ds-moderat'] + \
										output['ds-extensi'] + output['ds-complet']
				output['ls-moderat'] = output['ds-moderat']	+ output['ds-extensi'] + \
										output['ds-complet']
				output['ls-extensi'] = output['ds-extensi'] + output['ds-complet']
				output['ls-complet'] = output['ls-complet']
				output.to_csv(result_name, index=False)



	def run_cumulative_electric_damage(self):
		""" 
		multi-hazard road damage according to hazus calcuations
		"""

		rt = [100, 250, 500, 1000, 2500, 5000, 10000]
		
		for rt_val in rt:
			print('\telectric_dmg: cumulative rt_{}' .format(rt_val))
			# --- reading in damage results from above analysis
			eq_damage_results_csv = os.path.join(self.electric_output_path, 
												 'electric_eq_{}yr_dmg.csv' 
												 .format(rt_val))
			tsu_damage_results_csv = os.path.join(self.electric_output_path, 
												  'electric_tsu_{}yr_dmg.csv' 
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

			result_name = os.path.join(self.electric_output_path, 
									   'electric_cumulative_{}yr_dmg.csv' 
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
		column_keys = ['iter_{}' .format(i) for i in range(n_samples)]

		rts = [100, 250, 500, 1000, 2500, 5000, 10000]
		for rt_i, rt in enumerate(rts):
			file = 'electric_{}_{}yr_dmg.csv' .format(hazard, rt)
			data = pd.read_csv(os.path.join(self.electric_output_path, file))
			guids = data['guid']
		

			prnt_msg = '\tmc_electric_dmg: {} rt_{}, {}' .format(hazard, rt, retrofit_key)

			ds_results = np.zeros((len(guids), n_samples))
			
			for guid_i, guid in enumerate(guids):
				self.print_percent_complete(prnt_msg, guid_i, len(guids))
				row = data.loc[data['guid']==guid].T.squeeze()
				bins = np.array([row['ls-slight'], 
								 row['ls-moderat'], 
								 row['ls-extensi'], 
								 row['ls-complet'],
								 0])
				bins[0] += 1e-9
				rv = np.random.uniform(low=0., high=1., size=(n_samples))
				ds_results[guid_i] = np.digitize(rv, bins, right=True)
			d_out = pd.DataFrame(ds_results, columns=column_keys)
			d_out.index = guids
			d_out.index.name = 'guid'
			csv_filename = os.path.join(self.mc_path,
										'electric_DS_{}_{}yr_{}.csv'
										.format(hazard, rt, retrofit_key))

			d_out.to_csv(csv_filename)


	def Conn_analysis(self, hazard, path_to_guids, retrofit_key, n_samples):

		for fast in range(2):
			if fast == 0:
				fast_mult = 1
			elif fast == 1: 
				fast_mult = 0.5
			
			# repair time parameters for roads
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
			guids = os.listdir(path_to_guids)

			bldg_dset_id = "5df40388b9219c06cf8b0c80"    # building dataset
			line_dset_id = "5d263df6b9219cf93c056c37"	# line data
			pole_dset_id = "5d263f08b9219cf93c056c68"    # pole data
			bldg_to_network_id = "5d26050fb9219c0692d6d936" # links buildings to pole info


			""" the way critical nodes is setup is best given through an example:
				with the setup below, the connectivity analysis
				determines whether each tax-lot is connected to:
					- (node 229 OR node 230) AND (node 300)
				
				so the nodes in each inner lists undergo a logical_or 
				statement, whereas these results undergo a logical_and.

			"""
			
			""" node that node 211 is the nearest to the substation, not the 
				actual substation node """
			critical_nodes = [[211]]        # substation node 

			conn = EPNConnectivity(self.client)

			conn.load_remote_input_dataset("buildings", bldg_dset_id)
			conn.load_remote_input_dataset("line_dataset", line_dset_id)
			conn.load_remote_input_dataset("pole_dataset", pole_dset_id)
			conn.load_remote_input_dataset("building_to_network", bldg_to_network_id)

			conn.set_parameter("critical_nodes", critical_nodes)
			conn.set_parameter("remove_pole_DS", 3)

			conn.set_parameter('pole_reptime_log_med', pole_rep_time_log_med)
			conn.set_parameter('pole_reptime_covm', pole_rep_time_covm)
			
			func = {}
			rep = {}
			for rt_i, rt in enumerate(rts):
				print_msg = '\tconn_analysis: {}, rt_{}, {}, fast_{}: ' \
								.format(hazard, rt, retrofit_key, fast)
				conn.set_parameter('prnt_msg', print_msg)
				pole_dmg_file = 'electric_DS_{}_{}yr_{}.csv' .format(hazard, rt, retrofit_key)
				pole_dmg_file = os.path.join(self.mc_path, pole_dmg_file)

				# ---
				pole_dmg_dset = Dataset.from_file(pole_dmg_file,"ergo:NodeDamageInventory")
				conn.set_input_dataset("pole_dmg", pole_dmg_dset)

				func[rt], rep[rt] = conn.run()

			# guids = guids[0:1]    # note: temporary
			for guid_i, guid in enumerate(guids):

				prnt_msg = 'writing {} guids' .format(len(guids))
				self.print_percent_complete(prnt_msg, guid_i, len(guids))
				o_path = os.path.join(path_to_guids, 
									  guid, 
									  'mc_results', 
									  'electric',
									  )
				if not os.path.exists(o_path):
					os.makedirs(o_path)

				o_file_func = os.path.join(o_path, 
									 'func_{}_elec_{}_fast{}.gz' 
									 .format(hazard, retrofit_key, fast))
				
				o_file_rep = os.path.join(o_path, 
									 'reptime_{}_elec_{}_fast{}.gz' 
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
								   'electric'
								   )
		tax_lot_path = os.path.join(os.getcwd(),
									'..',
									'data', 
									'parcels')
		mc_path = os.path.join(output_path, 'mc_results')

		elec_dmg = electric_damage(output_path, retrofit_key_val)

		# # --- performing damage anlaysis ---
		# elec_dmg.run_electric_damage(haz_type='eq')
		# elec_dmg.run_electric_damage(haz_type='tsu')
		# elec_dmg.run_cumulative_electric_damage()

		# elec_dmg.DS_mc_sample(hazard = 'eq', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)
		# elec_dmg.DS_mc_sample(hazard = 'tsu', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)	
		# elec_dmg.DS_mc_sample(hazard = 'cumulative', 
		# 					  retrofit_key = retrofit_key, 
		# 					  n_samples = n_samples)


		# --- performing connectivity anlaysis ---
		# elec_dmg.Conn_analysis(hazard='eq',
		# 					   path_to_guids = tax_lot_path,
		# 					   retrofit_key = retrofit_key,
		# 					   n_samples = n_samples)
		# elec_dmg.Conn_analysis(hazard='tsu',
		# 					   path_to_guids = tax_lot_path,
		# 					   retrofit_key = retrofit_key,
		# 					   n_samples = n_samples)
		elec_dmg.Conn_analysis(hazard='cumulative',
							   path_to_guids = tax_lot_path,
							   retrofit_key = retrofit_key,
							   n_samples = n_samples)















