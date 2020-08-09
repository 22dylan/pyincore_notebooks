import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as st
import json
import itertools
import h5py
import ast


"""
TODO:
    - comment code

"""

class build_CPTs():
    def __init__(self, dmg_path, tax_lot_path, n_samples, h5path,
                hzrd_key='cumulative'):
        self.tax_lot_path = tax_lot_path
        self.hzrd_key = hzrd_key
        self.n_samples = n_samples

        # reading data into dataframe
        # self.guids = os.listdir(tax_lot_path)
        # self.guids = self.guids[0:9] # note: remove this later


        self.define_global_vars()
        self.create_h5(h5path)

        self.n_guids = len(self.guids)
        print('Building CPTs for {} guids' .format(self.n_guids))

    def create_h5(self, h5path):

        if not os.path.exists(h5path):
            print('\th5 directory not found. creating')
            os.makedirs(h5path)

        h5file = os.path.join(h5path, 'CPTs.h5')
        if not os.path.isfile(h5file):
             print('\th5 file not found. creating')
             f = h5py.File(h5file, "w")
             f.close()

        self.h5file = h5py.File(h5file, 'r+')
        self.guids = list(self.h5file.keys())   # note: temporary

        with h5py.File(h5file, 'r+') as f:
            h5_guids = set(list(f.keys()))
            guids = set(self.guids)
            missing_guids = guids-h5_guids

            prnt_msg = '\tmissing {} guids in h5. creating' .format(len(missing_guids))
            if len(missing_guids) > 0:
                for guid_i, guid in enumerate(missing_guids):
                    self.print_percent_complete(prnt_msg, guid_i, len(missing_guids))
                    f.create_group(guid)



    def event_cpt(self):
        prnt_msg = '\tevent cpts'
        event_dist = {i:1/len(self.RT_keys) for i in self.RT_keys}

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(event_dist, 'DSCRT_event_dist', guid)

    def goal_cpt(self):
        prnt_msg = '\tgoal cpts'
        goal_dist = {i:1/len(self.Goal_keys) for i in self.Goal_keys}

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(goal_dist, 'DSCRT_goal_dist', guid)

    def ex_ante_building_cpt(self):
        prnt_msg = '\tex-ante building cpts'
        ex_ante_dist = {i:1/len(self.Retrofit_building_keys) for i 
                        in self.Retrofit_building_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_ante_dist, 'DSCRT_ex_ante_building_dist', guid)

    def ex_ante_electric_cpt(self):
        prnt_msg = '\tex-ante electric cpts'
        ex_ante_dist = {i:1/len(self.Retrofit_electric_keys) for i 
                        in self.Retrofit_electric_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_ante_dist, 'DSCRT_ex_ante_electric_dist', guid)
    
    def ex_ante_transportation_cpt(self):
        prnt_msg = '\tex-ante transportation cpts'
        ex_ante_dist = {i:1/len(self.Retrofit_transportation_keys) for i 
                        in self.Retrofit_transportation_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_ante_dist, 'DSCRT_ex_ante_transportation_dist', guid)
    
    def ex_ante_water_cpt(self):
        prnt_msg = '\tex-ante water cpts'
        ex_ante_dist = {i:1/len(self.Retrofit_water_keys) for i 
                        in self.Retrofit_water_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_ante_dist, 'DSCRT_ex_ante_water_dist', guid)

    def ex_post_building_cpt(self):
        prnt_msg = '\tex-post building cpts'
        ex_post_dist = {i:1/len(self.Fast_building_keys) for i 
                        in self.Fast_building_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_post_dist, 'DSCRT_ex_post_building_dist', guid)

    def ex_post_electric_cpt(self):
        prnt_msg = '\tex-post electric cpts'
        ex_post_dist = {i:1/len(self.Fast_electric_keys) for i 
                        in self.Fast_electric_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_post_dist, 'DSCRT_ex_post_electric_dist', guid)

    def ex_post_transportation_cpt(self):
        prnt_msg = '\tex-post transportation cpts'
        ex_post_dist = {i:1/len(self.Fast_transportation_keys) for i 
                        in self.Fast_transportation_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_post_dist, 'DSCRT_ex_post_transportation_dist', guid)

    def ex_post_water_cpt(self):
        prnt_msg = '\tex-post water cpts'
        ex_post_dist = {i:1/len(self.Fast_water_keys) for i 
                        in self.Fast_water_keys}
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            self.write_to_h5(ex_post_dist, 'DSCRT_ex_post_water_dist', guid)

    def functionality_building_cpt(self):
        bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])

        prnt_msg = '\tfunctionality (bldg) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)

            CPT_func = np.zeros((len(self.RT_keys),
                                len(self.Retrofit_building_keys),
                                 len(self.functionality_bldg_keys)))

            for retrofit in range(len(self.Retrofit_building_keys)):
                func_file = 'func_{}_bldg_retrofit{}_fast0.gz' .format(self.hzrd_key, retrofit)
                data = self.read_func('building',func_file, guid, ret_array=True)
                for rt in range(len(self.RT_keys)):
                    temp = data[rt,:]
                    ds0 = np.sum(temp==0)/len(temp)
                    ds1 = np.sum(temp==1)/len(temp)
                    ds2 = np.sum(temp==2)/len(temp)
                    ds3 = np.sum(temp==3)/len(temp)
                    CPT_func[rt, retrofit,:] = np.array([ds0, ds1, ds2, ds3])
            
            CPT_func = self.adjust_cpt(CPT_func)
            CPT_func = self.format_cpt(CPT_func, 
                                       self.RT_keys,
                                       self.Retrofit_building_keys,
                                       self.functionality_bldg_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_bldg', guid)

    def functionality_electric_cpt(self):
        # bins = np.array([-np.inf, 0.1, 0.2, 0.3, 0.4, 
        #                 0.5, 0.6, 0.7, 0.8, 0.9, np.inf])
        prnt_msg = '\tfunctionality (elec) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)

            CPT_func = np.zeros((len(self.RT_keys),
                                len(self.Retrofit_electric_keys),
                                 len(self.functionality_elec_keys)))

            for retrofit in range(len(self.Retrofit_electric_keys)):
                func_file = 'func_{}_elec_retrofit{}_fast0.gz' .format(self.hzrd_key, retrofit)
                data = self.read_func('electric',func_file, guid, ret_array=True)
                func_yes = np.average(data, axis=1)
                func_no = 1 - func_yes

                CPT_func[:,retrofit] = np.column_stack([func_yes, func_no])
            
            CPT_func = self.adjust_cpt(CPT_func)
            CPT_func = self.format_cpt(CPT_func, 
                                       self.RT_keys,
                                       self.Retrofit_electric_keys,
                                       self.functionality_elec_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_elec', guid)

    def functionality_transportation_cpt(self):
        bins = np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf])
        prnt_msg = '\tfunctionality (trns) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)

            CPT_func = np.zeros((len(self.RT_keys),
                                len(self.Retrofit_transportation_keys),
                                 len(self.functionality_trns_keys)))

            for retrofit_i in range(len(self.Retrofit_transportation_keys)):
                func_file = 'func_{}_trans_retrofit{}_fast0.gz' .format(self.hzrd_key, retrofit_i)
                data = self.read_func('transportation',func_file, guid, ret_array=True)

                for rt_i in range(len(self.RT_keys)):
                    temp = data[rt_i,:]
                    temp += np.random.uniform(0,1e-8, len(temp))
                    CPT_func[rt_i, retrofit_i] = self.calc_CPT(temp, bins)

            CPT_func = self.adjust_cpt(CPT_func)
            CPT_func = self.format_cpt(CPT_func, 
                                       self.RT_keys,
                                       self.Retrofit_transportation_keys,
                                       self.functionality_trns_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_trns', guid)
   
    def functionality_water_cpt(self):
        prnt_msg = '\tfunctionality (wter) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)

            CPT_func = np.zeros((len(self.RT_keys),
                                len(self.Retrofit_water_keys),
                                len(self.Retrofit_electric_keys),
                                 len(self.functionality_wter_keys)))

            for retrofit in range(len(self.Retrofit_water_keys)):
                for eretrofit in range(len(self.Retrofit_electric_keys)):

                    func_file = 'func_{}_wter_retrofit{}_fast0_eretrofit{}_efast0.gz' .format(self.hzrd_key, retrofit, eretrofit)
                    data = self.read_func('water',func_file, guid, ret_array=True)
                    func_yes = np.average(data, axis=1)
                    func_no = 1 - func_yes

                    CPT_func[:,retrofit, eretrofit] = np.column_stack([func_yes, func_no])
            
            CPT_func = self.adjust_cpt(CPT_func)
            CPT_func = self.format_cpt(CPT_func, 
                                       self.RT_keys,
                                       self.Retrofit_water_keys,
                                       self.Retrofit_electric_keys,
                                       self.functionality_wter_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_wter', guid)
      
    def reptime_building_cpt(self):
        prnt_msg = '\tbuilding reptime cpts'
        R_threshold = np.array([-np.inf,3,7,15,30,90,
                                180,360,2*360,3*360,np.inf])
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            rep_data = np.zeros((4,4,7,self.n_samples))   # [retroft, fast, rt, i]

            for retrofit_i, retrofit in enumerate(self.Retrofit_building_keys):
                for fast_i, fast in enumerate(self.Fast_building_keys):
                    filename = 'reptime_{}_bldg_retrofit{}_fast{}.gz' \
                                .format(self.hzrd_key, retrofit_i, fast_i)
                    rep_data[retrofit_i, fast_i] = self.read_rep('building', 
                                                filename, guid, ret_array=True)

            CPT_rep_time = np.zeros((len(self.RT_keys),
                                     len(self.Retrofit_building_keys),
                                     len(self.Fast_building_keys),
                                     len(self.reptime_bldg_keys)))

            for rt_i in range(len(self.RT_keys)):
                for retrofit_i in range(len(self.Retrofit_building_keys)):
                    for fast_i in range(len(self.Fast_building_keys)):
                        temp = rep_data[retrofit_i,fast_i,rt_i,:]
                        temp += np.random.uniform(0,1e-8, len(temp))
                        CPT_rep_time[rt_i,retrofit_i,fast_i] = self.calc_CPT(temp, R_threshold)

            CPT_rep_time = self.adjust_cpt(CPT_rep_time)
            CPT_rep_time = self.format_cpt(CPT_rep_time, 
                                       self.RT_keys,
                                       self.Retrofit_building_keys,
                                       self.Fast_building_keys,
                                       self.reptime_bldg_keys)
            self.write_to_h5(CPT_rep_time, 'CPT_reptime_building', guid)

    def reptime_electric_cpt(self):
        prnt_msg = '\telectric reptime cpts'
        R_threshold = np.array([-np.inf,3,7,15,30,90,
                                180,360,2*360,3*360,np.inf])
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            rep_data = np.zeros((4,2,7,self.n_samples))   # [retroft, fast, rt, i]

            for retrofit_i, retrofit in enumerate(self.Retrofit_electric_keys):
                for fast_i, fast in enumerate(self.Fast_electric_keys):
                    filename = 'reptime_{}_elec_retrofit{}_fast{}.gz' \
                                .format(self.hzrd_key, retrofit_i, fast_i)
                    rep_data[retrofit_i, fast_i] = self.read_rep('electric', 
                                                filename, guid, ret_array=True)

            CPT_rep_time = np.zeros((len(self.RT_keys),
                                     len(self.Retrofit_electric_keys),
                                     len(self.Fast_electric_keys),
                                     len(self.reptime_elec_keys)))

            for rt_i in range(len(self.RT_keys)):
                for retrofit_i in range(len(self.Retrofit_electric_keys)):
                    for fast_i in range(len(self.Fast_electric_keys)):
                        temp = rep_data[retrofit_i,fast_i,rt_i,:]
                        temp += np.random.uniform(0,1e-8, len(temp))
                        CPT_rep_time[rt_i,retrofit_i,fast_i] = self.calc_CPT(temp, R_threshold)

            CPT_rep_time = self.adjust_cpt(CPT_rep_time)
            CPT_rep_time = self.format_cpt(CPT_rep_time, 
                                       self.RT_keys,
                                       self.Retrofit_electric_keys,
                                       self.Fast_electric_keys,
                                       self.reptime_elec_keys)
            self.write_to_h5(CPT_rep_time, 'CPT_reptime_electric', guid)

    def reptime_transportation_cpt(self):
        prnt_msg = '\ttransportation reptime cpts'
        R_threshold = np.array([-np.inf,3,7,15,30,90,
                                180,360,2*360,3*360,np.inf])
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            rep_data = np.zeros((4,2,7,self.n_samples))   # [retroft, fast, rt, i]

            for retrofit_i, retrofit in enumerate(self.Retrofit_transportation_keys):
                for fast_i, fast in enumerate(self.Fast_transportation_keys):
                    filename = 'reptime_{}_trans_retrofit{}_fast{}.gz' \
                                .format(self.hzrd_key, retrofit_i, fast_i)
                    rep_data[retrofit_i, fast_i] = self.read_rep('transportation', 
                                                filename, guid, ret_array=True)

            CPT_rep_time = np.zeros((len(self.RT_keys),
                                     len(self.Retrofit_transportation_keys),
                                     len(self.Fast_transportation_keys),
                                     len(self.reptime_trns_keys)))

            for rt_i in range(len(self.RT_keys)):
                for retrofit_i in range(len(self.Retrofit_transportation_keys)):
                    for fast_i in range(len(self.Fast_transportation_keys)):
                        temp = rep_data[retrofit_i,fast_i,rt_i,:]
                        temp += np.random.uniform(0,1e-8, len(temp))
                        CPT_rep_time[rt_i,retrofit_i,fast_i] = self.calc_CPT(temp, R_threshold)

            CPT_rep_time = self.adjust_cpt(CPT_rep_time)
            CPT_rep_time = self.format_cpt(CPT_rep_time, 
                                       self.RT_keys,
                                       self.Retrofit_transportation_keys,
                                       self.Fast_transportation_keys,
                                       self.reptime_trns_keys)
            self.write_to_h5(CPT_rep_time, 'CPT_reptime_transportation', guid)
    
    def reptime_water_cpt(self):
        prnt_msg = '\twter reptime cpts'
        R_threshold = np.array([-np.inf,3,7,15,30,90,
                                180,360,2*360,3*360,np.inf])
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            rep_data = np.zeros((4,4,2,2,7,self.n_samples))   # [retroft, eretrofit, fast, efast, rt, i]

            for retrofit_i, retrofit in enumerate(self.Retrofit_water_keys):
                for eretrofit_i, eretrofit in enumerate(self.Retrofit_electric_keys):
                    for fast_i, fast in enumerate(self.Fast_water_keys):
                        for efast_i, efast in enumerate(self.Fast_electric_keys):
                            filename = 'reptime_{}_wter_retrofit{}_fast{}_eretrofit{}_efast{}.gz' \
                                        .format(self.hzrd_key, retrofit_i, fast_i, eretrofit_i, efast_i)

                            rep_data[retrofit_i, eretrofit_i, fast_i, efast_i] = \
                                self.read_rep('water', filename, guid, ret_array=True)

            CPT_rep_time = np.zeros((len(self.RT_keys),
                                     len(self.Retrofit_water_keys),
                                     len(self.Retrofit_electric_keys),
                                     len(self.Fast_water_keys),
                                     len(self.Fast_electric_keys),
                                     len(self.reptime_wter_keys)))

            for rt_i in range(len(self.RT_keys)):
                for retrofit_i in range(len(self.Retrofit_water_keys)):
                    for eretrofit_i in range(len(self.Retrofit_electric_keys)):
                        for fast_i in range(len(self.Fast_water_keys)):
                            for efast_i in range(len(self.Fast_electric_keys)):
                                temp = rep_data[retrofit_i,eretrofit_i, fast_i,efast_i,rt_i,:]
                                temp += np.random.uniform(0,1e-8, len(temp))
                                CPT_rep_time[rt_i,retrofit_i,eretrofit_i, fast_i, efast_i] = self.calc_CPT(temp, R_threshold)

            CPT_rep_time = self.adjust_cpt(CPT_rep_time)
            CPT_rep_time = self.format_cpt(CPT_rep_time, 
                                       self.RT_keys,
                                       self.Retrofit_water_keys,
                                       self.Retrofit_electric_keys,
                                       self.Fast_water_keys,
                                       self.Fast_electric_keys,
                                       self.reptime_wter_keys)
            self.write_to_h5(CPT_rep_time, 'CPT_reptime_water', guid)


    def functionality_target_bldg_cpt(self):
        prnt_msg = '\tfunctionality target (bldg) cpts'
        functionality_targets = np.array([[2, 2, 2, 2, 3, 3, 3],
                                          [1, 1, 1, 2, 2, 2, 2],
                                          [0, 0, 1, 1, 1, 2, 2]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            functionality_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.functionality_target_bldg_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_per = int(functionality_targets[goal, rt])
                    functionality_target[goal, rt, idx_per] = 1.

            CPT_func = self.format_cpt(functionality_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.functionality_target_bldg_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_target_bldg', guid)

    def functionality_target_elec_cpt(self):
        prnt_msg = '\tfunctionality target (elec) cpts'
        """ - the below is all ones to correspond to Func_target_elec_no 
            - expecting electricity to be nonfunctional and accept it as is.
            - electric resilience depends on rapidity only
        """ 
        functionality_targets = np.array([[1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            functionality_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.functionality_target_elec_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_per = int(functionality_targets[goal, rt])
                    functionality_target[goal, rt, idx_per] = 1.

            CPT_func = self.format_cpt(functionality_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.functionality_target_elec_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_target_elec', guid)

    def functionality_target_trns_cpt(self):
        prnt_msg = '\tfunctionality target (trns) cpts'
        functionality_targets = np.array([[2, 2, 1, 0, 0, 0, 0],
                                          [3, 3, 2, 1, 1, 1, 1],
                                          [4, 4, 3, 2, 2, 2, 2]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            functionality_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.functionality_target_trns_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_per = int(functionality_targets[goal, rt])
                    functionality_target[goal, rt, idx_per] = 1.

            CPT_func = self.format_cpt(functionality_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.functionality_target_trns_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_target_trns', guid)

    def functionality_target_wter_cpt(self):
        prnt_msg = '\tfunctionality target (wter) cpts'
        """ the below is all ones to correspond to Func_target_wter_no """ 
        functionality_targets = np.array([[1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            functionality_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.functionality_target_wter_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_per = int(functionality_targets[goal, rt])
                    functionality_target[goal, rt, idx_per] = 1.

            CPT_func = self.format_cpt(functionality_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.functionality_target_wter_keys)
            self.write_to_h5(CPT_func, 'CPT_functionality_target_wter', guid)

    def reptime_target_bldg_cpt(self):
        prnt_msg = '\treptime target (bldg) cpts'
        reptime_targets = np.array([[6, 7, 7, 8, 8, 8, 8],
                                    [5, 6, 7, 8, 8, 8, 8],
                                    [4, 5, 6, 7, 7, 8, 8]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            reptime_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.reptime_target_bldg_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_rep = int(reptime_targets[goal, rt])

                    reptime_target[goal, rt, idx_rep] = 1.

            CPT_reptime_target = self.format_cpt(reptime_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.reptime_target_bldg_keys)
            self.write_to_h5(CPT_reptime_target, 'CPT_reptime_target_bldg', guid)

    def reptime_target_elec_cpt(self):
        prnt_msg = '\treptime target (elec) cpts'
        reptime_targets = np.array([[3, 4, 4, 4, 5, 6, 6],
                                    [2, 3, 4, 4, 4, 5, 5],
                                    [1, 2, 3, 3, 3, 4, 4]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            reptime_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.reptime_target_elec_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_rep = int(reptime_targets[goal, rt])

                    reptime_target[goal, rt, idx_rep] = 1.

            CPT_reptime_target = self.format_cpt(reptime_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.reptime_target_elec_keys)
            self.write_to_h5(CPT_reptime_target, 'CPT_reptime_target_elec', guid)

    def reptime_target_trns_cpt(self):
        prnt_msg = '\treptime target (trns) cpts'
        reptime_targets = np.array([[2, 3, 4, 4, 5, 5, 5],
                                    [1, 2, 3, 3, 4, 4, 4],
                                    [0, 1, 2, 2, 3, 3, 3]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            reptime_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.reptime_target_trns_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_rep = int(reptime_targets[goal, rt])

                    reptime_target[goal, rt, idx_rep] = 1.

            CPT_reptime_target = self.format_cpt(reptime_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.reptime_target_trns_keys)
            self.write_to_h5(CPT_reptime_target, 'CPT_reptime_target_trns', guid)

    def reptime_target_wter_cpt(self):
        prnt_msg = '\treptime target (wter) cpts'
        reptime_targets = np.array([[4, 5, 5, 5, 6, 6, 6],
                                    [3, 4, 4, 4, 5, 5, 5],
                                    [2, 3, 3, 3, 4, 4, 4]])

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            reptime_target = np.zeros((len(self.Goal_keys), 
                                            len(self.RT_keys), 
                                            len(self.reptime_target_wter_keys)))
            for goal in range(len(self.Goal_keys)):
                for rt in range(len(self.RT_keys)):
                    idx_rep = int(reptime_targets[goal, rt])

                    reptime_target[goal, rt, idx_rep] = 1.

            CPT_reptime_target = self.format_cpt(reptime_target,
                                       self.Goal_keys, 
                                       self.RT_keys,
                                       self.reptime_target_wter_keys)
            self.write_to_h5(CPT_reptime_target, 'CPT_reptime_target_wter', guid)


    def robustness_building_cpt(self):
        prnt_msg = '\trobustness (bldg) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_robust = np.zeros((len(self.functionality_bldg_keys), 
                                   len(self.functionality_target_bldg_keys),
                                   len(self.robust_bldg_keys)))

            for p in range(len(self.functionality_bldg_keys)):
                p_val = int(self.functionality_bldg_keys[p].split('_')[-1])

                for pt in range(len(self.functionality_target_bldg_keys)):
                    pt_val = int(self.functionality_target_bldg_keys[pt].split('_')[-1])
                    if p_val <= pt_val:
                        cpt_val = np.array([1.,0.])   # robustness met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])   # robustness met [yes=0, no=1]
                    CPT_robust[p, pt] = cpt_val

            CPT_robust = self.format_cpt(CPT_robust,
                                       self.functionality_bldg_keys,
                                       self.functionality_target_bldg_keys,
                                       self.robust_bldg_keys)
            self.write_to_h5(CPT_robust, 'CPT_robust_bldg', guid)

    def robustness_electric_cpt(self):
        prnt_msg = '\trobustness (elec) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_robust = np.zeros((len(self.functionality_elec_keys), 
                                   len(self.functionality_target_elec_keys),
                                   len(self.robust_elec_keys)))

            for p in range(len(self.functionality_elec_keys)):
                p_val = self.functionality_elec_keys[p].split('_')[-1]
                for pt in range(len(self.functionality_target_elec_keys)):
                    pt_val = self.functionality_target_elec_keys[pt].split('_')[-1]

                    if (p_val == 'yes') and (pt_val == 'yes'):
                        cpt_val = np.array([1.,0.])

                    if (p_val == 'yes') and (pt_val == 'no'):
                        cpt_val = np.array([1.,0.])

                    if (p_val == 'no') and (pt_val == 'yes'):
                        cpt_val = np.array([0.,1.])

                    if (p_val == 'no') and (pt_val == 'no'):
                        cpt_val = np.array([1.,0.])

                    CPT_robust[p, pt] = cpt_val

            CPT_robust = self.format_cpt(CPT_robust,
                                       self.functionality_elec_keys,
                                       self.functionality_target_elec_keys,
                                       self.robust_elec_keys)
            self.write_to_h5(CPT_robust, 'CPT_robust_elec', guid)

    def robustness_transportation_cpt(self):
        prnt_msg = '\trobustness (trns) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_robust = np.zeros((len(self.functionality_trns_keys), 
                                   len(self.functionality_target_trns_keys),
                                   len(self.robust_trns_keys)))

            for p in range(len(self.functionality_trns_keys)):
                p_val = int(self.functionality_trns_keys[p].split('_')[-1])

                for pt in range(len(self.functionality_target_trns_keys)):
                    pt_val = int(self.functionality_target_trns_keys[pt].split('_')[-1])
                    if p_val >= pt_val:
                        cpt_val = np.array([1.,0.])   # rapidity met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])   # rapidity met [yes=0, no=1]
                    CPT_robust[p, pt] = cpt_val

            CPT_robust = self.format_cpt(CPT_robust,
                                       self.functionality_trns_keys,
                                       self.functionality_target_trns_keys,
                                       self.robust_trns_keys)
            self.write_to_h5(CPT_robust, 'CPT_robust_trns', guid)

    def robustness_water_cpt(self):
        prnt_msg = '\trobustness (wter) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_robust = np.zeros((len(self.functionality_wter_keys), 
                                   len(self.functionality_target_wter_keys),
                                   len(self.robust_wter_keys)))

            for p in range(len(self.functionality_wter_keys)):
                p_val = self.functionality_wter_keys[p].split('_')[-1]
                for pt in range(len(self.functionality_target_wter_keys)):
                    pt_val = self.functionality_target_wter_keys[pt].split('_')[-1]

                    if (p_val == 'yes') and (pt_val == 'yes'):
                        cpt_val = np.array([1.,0.])

                    if (p_val == 'yes') and (pt_val == 'no'):
                        cpt_val = np.array([1.,0.])

                    if (p_val == 'no') and (pt_val == 'yes'):
                        cpt_val = np.array([0.,1.])

                    if (p_val == 'no') and (pt_val == 'no'):
                        cpt_val = np.array([1.,0.])

                    CPT_robust[p, pt] = cpt_val

            CPT_robust = self.format_cpt(CPT_robust,
                                       self.functionality_wter_keys,
                                       self.functionality_target_wter_keys,
                                       self.robust_wter_keys)
            self.write_to_h5(CPT_robust, 'CPT_robust_wter', guid)

    def rapidity_building_cpt(self):
        prnt_msg = '\trapidity (bldg) cpts'

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_rapid = np.zeros((len(self.reptime_bldg_keys), 
                                   len(self.reptime_target_bldg_keys),
                                   len(self.rapid_bldg_keys)))

            for r in range(len(self.reptime_bldg_keys)):
                r_val = int(self.reptime_bldg_keys[r].split('_')[-1])
                for rt in range(len(self.reptime_target_bldg_keys)):
                    rt_val = int(self.reptime_target_bldg_keys[rt].split('_')[-1])
                    if r_val <= rt_val:
                        cpt_val = np.array([1.,0.])       # rapidity met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])       # rapidity met [yes=0, no=1]
                    CPT_rapid[r, rt] = cpt_val

            CPT_rapid = self.format_cpt(CPT_rapid,
                                       self.reptime_bldg_keys,
                                       self.reptime_target_bldg_keys,
                                       self.rapid_bldg_keys)
            self.write_to_h5(CPT_rapid, 'CPT_rapid_bldg', guid)

    def rapidity_electric_cpt(self):
        prnt_msg = '\trapidity (elec) cpts'

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_rapid = np.zeros((len(self.reptime_elec_keys), 
                                   len(self.reptime_target_elec_keys),
                                   len(self.rapid_elec_keys)))

            for r in range(len(self.reptime_elec_keys)):
                r_val = int(self.reptime_elec_keys[r].split('_')[-1])
                for rt in range(len(self.reptime_target_elec_keys)):
                    rt_val = int(self.reptime_target_elec_keys[rt].split('_')[-1])
                    if r_val <= rt_val:
                        cpt_val = np.array([1.,0.])       # rapidity met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])       # rapidity met [yes=0, no=1]
                    CPT_rapid[r, rt] = cpt_val

            CPT_rapid = self.format_cpt(CPT_rapid,
                                       self.reptime_elec_keys,
                                       self.reptime_target_elec_keys,
                                       self.rapid_elec_keys)
            self.write_to_h5(CPT_rapid, 'CPT_rapid_elec', guid)

    def rapidity_transportation_cpt(self):
        prnt_msg = '\trapidity (trns) cpts'

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_rapid = np.zeros((len(self.reptime_trns_keys), 
                                   len(self.reptime_target_trns_keys),
                                   len(self.rapid_trns_keys)))

            for r in range(len(self.reptime_trns_keys)):
                r_val = int(self.reptime_trns_keys[r].split('_')[-1])
                for rt in range(len(self.reptime_target_trns_keys)):
                    rt_val = int(self.reptime_target_trns_keys[rt].split('_')[-1])
                    if r_val <= rt_val:
                        cpt_val = np.array([1.,0.])       # rapidity met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])       # rapidity met [yes=0, no=1]
                    CPT_rapid[r, rt] = cpt_val

            CPT_rapid = self.format_cpt(CPT_rapid,
                                       self.reptime_trns_keys,
                                       self.reptime_target_trns_keys,
                                       self.rapid_trns_keys)
            self.write_to_h5(CPT_rapid, 'CPT_rapid_trns', guid)

    def rapidity_water_cpt(self):
        prnt_msg = '\trapidity (wter) cpts'

        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            CPT_rapid = np.zeros((len(self.reptime_wter_keys), 
                                   len(self.reptime_target_wter_keys),
                                   len(self.rapid_wter_keys)))

            for r in range(len(self.reptime_wter_keys)):
                r_val = int(self.reptime_wter_keys[r].split('_')[-1])
                for rt in range(len(self.reptime_target_wter_keys)):
                    rt_val = int(self.reptime_target_wter_keys[rt].split('_')[-1])
                    if r_val <= rt_val:
                        cpt_val = np.array([1.,0.])       # rapidity met [yes=1, no=0]
                    else:
                        cpt_val = np.array([0.,1.])       # rapidity met [yes=0, no=1]
                    CPT_rapid[r, rt] = cpt_val

            CPT_rapid = self.format_cpt(CPT_rapid,
                                       self.reptime_wter_keys,
                                       self.reptime_target_wter_keys,
                                       self.rapid_wter_keys)
            self.write_to_h5(CPT_rapid, 'CPT_rapid_wter', guid)


    def resilience_bldg_cpt(self):
        prnt_msg = '\tresilience (bldg) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_resilient = np.zeros((len(self.robust_bldg_keys), 
                                      len(self.rapid_bldg_keys),
                                      len(self.resilient_bldg_keys)))
            for ro in range(len(self.robust_bldg_keys)):
                ro_val = self.robust_bldg_keys[ro].split('_')[-1]
                for ra in range(len(self.rapid_bldg_keys)):
                    ra_val = self.robust_bldg_keys[ra].split('_')[-1]
                    if ro_val == 'yes' and ra_val=='yes':
                        cpt_val = np.array([1., 0.])
                    else:
                        cpt_val = np.array([0., 1.])

                    CPT_resilient[ro, ra] = cpt_val

            CPT_resilient = self.format_cpt(CPT_resilient,
                                       self.robust_bldg_keys,
                                       self.rapid_bldg_keys,
                                       self.resilient_bldg_keys)
            self.write_to_h5(CPT_resilient, 'CPT_resilient_bldg', guid)

    def resilience_elec_cpt(self):
        prnt_msg = '\tresilience (elec) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_resilient = np.zeros((len(self.robust_elec_keys), 
                                      len(self.rapid_elec_keys),
                                      len(self.resilient_elec_keys)))
            for ro in range(len(self.robust_elec_keys)):
                ro_val = self.robust_elec_keys[ro].split('_')[-1]
                for ra in range(len(self.rapid_elec_keys)):
                    ra_val = self.robust_elec_keys[ra].split('_')[-1]
                    if ro_val == 'yes' and ra_val=='yes':
                        cpt_val = np.array([1., 0.])
                    else:
                        cpt_val = np.array([0., 1.])

                    CPT_resilient[ro, ra] = cpt_val

            CPT_resilient = self.format_cpt(CPT_resilient,
                                       self.robust_elec_keys,
                                       self.rapid_elec_keys,
                                       self.resilient_elec_keys)
            self.write_to_h5(CPT_resilient, 'CPT_resilient_elec', guid)

    def resilience_trns_cpt(self):
        prnt_msg = '\tresilience (trns) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_resilient = np.zeros((len(self.robust_trns_keys), 
                                      len(self.rapid_trns_keys),
                                      len(self.resilient_trns_keys)))
            for ro in range(len(self.robust_trns_keys)):
                ro_val = self.robust_trns_keys[ro].split('_')[-1]
                for ra in range(len(self.rapid_trns_keys)):
                    ra_val = self.robust_trns_keys[ra].split('_')[-1]
                    if ro_val == 'yes' and ra_val=='yes':
                        cpt_val = np.array([1., 0.])
                    else:
                        cpt_val = np.array([0., 1.])

                    CPT_resilient[ro, ra] = cpt_val

            CPT_resilient = self.format_cpt(CPT_resilient,
                                       self.robust_trns_keys,
                                       self.rapid_trns_keys,
                                       self.resilient_trns_keys)
            self.write_to_h5(CPT_resilient, 'CPT_resilient_trns', guid)

    def resilience_wter_cpt(self):
        prnt_msg = '\tresilience (wter) cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_resilient = np.zeros((len(self.robust_wter_keys), 
                                      len(self.rapid_wter_keys),
                                      len(self.resilient_wter_keys)))
            for ro in range(len(self.robust_wter_keys)):
                ro_val = self.robust_wter_keys[ro].split('_')[-1]
                for ra in range(len(self.rapid_wter_keys)):
                    ra_val = self.robust_wter_keys[ra].split('_')[-1]
                    if ro_val == 'yes' and ra_val=='yes':
                        cpt_val = np.array([1., 0.])
                    elif ro_val == 'yes' and ra_val == 'no':
                        cpt_val = np.array([0., 1.])
                    elif ro_val == 'no' and ra_val == 'yes':
                        cpt_val = np.array([0., 1.])
                    elif ro_val == 'no' and ra_val == 'no':
                        cpt_val = np.array([0., 1.])

                    CPT_resilient[ro, ra] = cpt_val

            CPT_resilient = self.format_cpt(CPT_resilient,
                                       self.robust_wter_keys,
                                       self.rapid_wter_keys,
                                       self.resilient_wter_keys)
            self.write_to_h5(CPT_resilient, 'CPT_resilient_wter', guid)

    def robust_cpt(self):
        prnt_msg = 'robustness cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_robust = np.zeros((len(self.robust_bldg_keys), 
                                      len(self.robust_elec_keys),
                                      len(self.robust_trns_keys),
                                      len(self.robust_wter_keys),
                                      len(self.robustness_keys)))
            for rb in range(len(self.robust_bldg_keys)):
                rb_val = self.robust_bldg_keys[rb].split('_')[-1]
                for re in range(len(self.robust_elec_keys)):
                    re_val = self.robust_elec_keys[re].split('_')[-1]
                    for rt in range(len(self.robust_trns_keys)):
                        rt_val = self.robust_trns_keys[rt].split('_')[-1]
                        for rw in range(len(self.robust_wter_keys)):
                            rw_val = self.robust_wter_keys[rw].split('_')[-1]

                            if (rb_val == 'yes') and (re_val == 'yes') and (rt_val == 'yes') and (rw_val == 'yes'):
                                cpt_val = np.array([1., 0.])
                            else:
                                cpt_val = np.array([0., 1.])

                            CPT_robust[rb, re, rt, rw] = cpt_val

            CPT_robust = self.format_cpt(CPT_robust,
                                       self.robust_bldg_keys,
                                       self.robust_elec_keys,
                                       self.robust_trns_keys,
                                       self.robust_wter_keys,
                                       self.robustness_keys)
            self.write_to_h5(CPT_robust, 'CPT_robust', guid)

    def rapid_cpt(self):
        prnt_msg = 'rapid cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_rapid = np.zeros((len(self.rapid_bldg_keys), 
                                      len(self.rapid_elec_keys),
                                      len(self.rapid_trns_keys),
                                      len(self.rapid_wter_keys),
                                      len(self.rapidity_keys)))
            for rb in range(len(self.rapid_bldg_keys)):
                rb_val = self.rapid_bldg_keys[rb].split('_')[-1]
                for re in range(len(self.rapid_elec_keys)):
                    re_val = self.rapid_elec_keys[re].split('_')[-1]
                    for rt in range(len(self.rapid_trns_keys)):
                        rt_val = self.rapid_trns_keys[rt].split('_')[-1]
                        for rw in range(len(self.rapid_wter_keys)):
                            rw_val = self.rapid_wter_keys[rw].split('_')[-1]


                            if (rb_val == 'yes') and (re_val == 'yes') and (rt_val == 'yes') and (rw_val == 'yes'):
                                cpt_val = np.array([1., 0.])
                            else:
                                cpt_val = np.array([0., 1.])

                            CPT_rapid[rb, re, rt, rw] = cpt_val

            CPT_rapid = self.format_cpt(CPT_rapid,
                                       self.rapid_bldg_keys,
                                       self.rapid_elec_keys,
                                       self.rapid_trns_keys,
                                       self.rapid_wter_keys,
                                       self.rapidity_keys)
            self.write_to_h5(CPT_rapid, 'CPT_rapid', guid)

    def resilience_cpt(self):
        prnt_msg = 'resilience cpts'
        for guid_i, guid in enumerate(self.guids):
            self.print_percent_complete(prnt_msg, guid_i, self.n_guids)
            # --- resilience cpt
            CPT_resilient = np.zeros((len(self.resilient_bldg_keys), 
                                      len(self.resilient_elec_keys),
                                      len(self.resilient_trns_keys),
                                      len(self.resilient_wter_keys),
                                      len(self.resilient_keys)))
            for rb in range(len(self.resilient_bldg_keys)):
                rb_val = self.resilient_bldg_keys[rb].split('_')[-1]
                for re in range(len(self.resilient_elec_keys)):
                    re_val = self.resilient_elec_keys[re].split('_')[-1]
                    for rt in range(len(self.resilient_trns_keys)):
                        rt_val = self.resilient_trns_keys[rt].split('_')[-1]
                        for rw in range(len(self.resilient_wter_keys)):
                            rw_val = self.resilient_wter_keys[rw].split('_')[-1]
                            if (rb_val == 'yes') and (re_val == 'yes') and (rt_val == 'yes') and (rw_val == 'yes'):
                                cpt_val = np.array([1., 0.])
                            else:
                                cpt_val = np.array([0., 1.])

                            CPT_resilient[rb, re, rt, rw] = cpt_val

            CPT_resilient = self.format_cpt(CPT_resilient,
                                       self.resilient_bldg_keys,
                                       self.resilient_elec_keys,
                                       self.resilient_trns_keys,
                                       self.resilient_wter_keys,
                                       self.resilient_keys)
            self.write_to_h5(CPT_resilient, 'CPT_resilient', guid)

    def format_cpt(self, data, *args):
        n_rows = data.size
        perm = [s for s in itertools.product(*args)]
        data = data.flatten()
        new_cpt = []
        for row in range(n_rows):
            temp = list(perm[row])
            temp.append(data[row])
            new_cpt.append(temp)
        return new_cpt

    def write_to_h5(self, data, var_str, guid):
        s = np.array(data).astype('S')
        if var_str in list(self.h5file[guid].keys()):
            del self.h5file[guid][var_str]
        self.h5file[guid].create_dataset(var_str, data=s)


    # def write_json(self, var, var_str, guid):
    #     s = np.array(var).astype('S')
    #     if var_str in list(self.h5file[guid].keys()):
    #         del self.h5file[guid][var_str]
    #     self.h5file[guid].create_dataset(var_str, data=s)
    #     temp = np.array(self.h5file[guid][var_str]).astype(str).tolist() 
    #     fds

    #     # s = json.dumps(var, separators=(',',': '))
    #     # if var_str in list(self.h5file[guid].keys()):
    #     #     del self.h5file[guid][var_str]
    #     # self.h5file[guid].create_dataset(var_str, data=np.array(s, dtype='S'))


    def define_global_vars(self):
        # decision  nodes (10)
        self.RT_keys = ['RT_100', 'RT_250', 'RT_500', 
                           'RT_1000', 'RT_2500', 'RT_5000', 'RT_10000']
        self.Goal_keys = ['Easy', 'Moderate', 'Difficult']
        
        self.Retrofit_building_keys = ['building_retrofit_{}' .format(i) for i in range(4)]
        self.Retrofit_electric_keys = ['electric_retrofit_{}' .format(i) for i in range(4)]
        self.Retrofit_transportation_keys = ['transportation_retrofit_{}' .format(i) for i in range(4)]
        self.Retrofit_water_keys = ['water_retrofit_{}' .format(i) for i in range(4)]

        self.Fast_building_keys = ['building_fast_{}' .format(i) for i in range(4)]
        self.Fast_electric_keys = ['electric_fast_{}' .format(i) for i in range(2)]
        self.Fast_transportation_keys = ['transportation_fast_{}' .format(i) for i in range(2)]
        self.Fast_water_keys = ['water_fast_{}' .format(i) for i in range(2)]

        # for cpts (29)
        self.functionality_bldg_keys = ['Func_bldg_{}' .format(i) for i in range(4)]
        self.functionality_elec_keys = ['Func_elec_yes', 'Func_elec_no']
        self.functionality_trns_keys = ['Func_trns_{}' .format(i) for i in range(5)]
        self.functionality_wter_keys = ['Func_wter_yes', 'Func_wter_no']

        self.reptime_bldg_keys = ['Reptime_bldg_{}' .format(i) for i in range(10)]
        self.reptime_elec_keys = ['Reptime_elec_{}' .format(i) for i in range(10)]
        self.reptime_trns_keys = ['Reptime_trns_{}' .format(i) for i in range(10)]
        self.reptime_wter_keys = ['Reptime_wter_{}' .format(i) for i in range(10)]

        self.functionality_target_bldg_keys = ['Func_target_bldg_{}' .format(i) for i in range(4)]
        self.functionality_target_elec_keys = ['Func_target_elec_yes', 'Func_target_elec_no']
        self.functionality_target_trns_keys = ['Func_target_trns_{}' .format(i) for i in range(5)]
        self.functionality_target_wter_keys = ['Func_target_wter_yes', 'Func_target_wter_no']

        self.reptime_target_bldg_keys = ['Reptime_target_bldg_{}' .format(i) for i in range(9)]
        self.reptime_target_elec_keys = ['Reptime_target_elec_{}' .format(i) for i in range(9)]
        self.reptime_target_trns_keys = ['Reptime_target_trns_{}' .format(i) for i in range(9)]
        self.reptime_target_wter_keys = ['Reptime_target_wter_{}' .format(i) for i in range(9)]
        
        self.robust_bldg_keys = ['Robust_bldg_yes', 'Robust_bldg_no']
        self.robust_elec_keys = ['Robust_elec_yes', 'Robust_elec_no']
        self.robust_trns_keys = ['Robust_trns_yes', 'Robust_trns_no']
        self.robust_wter_keys = ['Robust_wter_yes', 'Robust_wter_no']
        
        self.rapid_bldg_keys = ['Rapid_bldg_yes', 'Rapid_bldg_no']
        self.rapid_elec_keys = ['Rapid_elec_yes', 'Rapid_elec_no']
        self.rapid_trns_keys = ['Rapid_trns_yes', 'Rapid_trns_no']
        self.rapid_wter_keys = ['Rapid_wter_yes', 'Rapid_wter_no']
        
        self.resilient_bldg_keys = ['Resilient_bldg_yes', 'Resilient_bldg_no']
        self.resilient_elec_keys = ['Resilient_elec_yes', 'Resilient_elec_no']
        self.resilient_trns_keys = ['Resilient_trns_yes', 'Resilient_trns_no']
        self.resilient_wter_keys = ['Resilient_wter_yes', 'Resilient_wter_no']

        self.robustness_keys = ['Robust_yes', 'Robust_no']
        self.rapidity_keys = ['Rapid_yes', 'Rapid_no']
        self.resilient_keys = ['Resilient_yes', 'Resilient_no']

    def adjust_cpt(self, CPT, threshold=1e-4):
        """
        CPT - the input CPT table
            all CPT entries < threshold are set equal to threshold
            correspondingly, the other entries of the CPT are modified
        """
        temp = CPT[:]
        org_shp = np.shape(temp)
        new_shp = (np.prod(org_shp[:-1]), org_shp[-1])
        
        temp = np.reshape(CPT, new_shp)
        for i in range(len(temp)):
            row = temp[i,:]
            ind1 = row<=threshold
            ind2 = row>threshold

            add_pf = np.sum(ind1)*threshold
            minus_pf = add_pf/np.sum(ind2)
            
            row[ind1] = threshold
            row[ind2] = row[ind2] - minus_pf
            temp[i,:] = row
        CPT_adjust = np.reshape(temp, org_shp)
    
        return CPT_adjust

    def calc_CPT(self, data, thrshld):
        CPT = []
        for i in range(len(thrshld)-1):
            if len(data) > 0:
                CPT.append(st.gaussian_kde(data).integrate_box_1d(thrshld[i], 
                                thrshld[i+1])) 
            else:
                CPT.append(0)
        return np.array(CPT)

    def read_func(self, inf_key, filename, guid, ret_array=False):
        func_df = os.path.join(self.tax_lot_path, 
                              guid, 
                              'mc_results', 
                              inf_key,
                              filename)

        func_df = pd.read_csv(func_df, index_col=0, compression='gzip')
        
        if ret_array == True:
            func_df = func_df.values

        return func_df
        
    def read_rep(self, inf_key, filename, guid, ret_array=False):
        rep_df = os.path.join(self.tax_lot_path, 
                      guid, 
                      'mc_results', 
                      inf_key,
                      filename)
        rep_df = pd.read_csv(rep_df, index_col=0, compression='gzip')
        if ret_array == True:
            rep_df = rep_df.values

        return rep_df

    def print_percent_complete(self, msg, i, n_i):
        i, n_i = int(i)+1, int(n_i)
        sys.stdout.write('\r')
        sys.stdout.write("{} {}/{} ({:.1f}%)" .format(msg, i, n_i, (100/(n_i)*i)))
        sys.stdout.flush()
        if i==n_i:
            print()

if __name__ == "__main__":
    dmg_path = os.path.join(os.getcwd(), 
                                   '..', 
                                   'data',
                                   'pyincore_damage')
    tax_lot_path = os.path.join(os.getcwd(), 
                                   '..', 
                                   'data',
                                   'parcels')
    h5path = os.path.join(os.getcwd(),
                                    '..',
                                    'data',
                                    'hdf5_files')
    bld_cpt = build_CPTs(dmg_path, 
                         tax_lot_path, 
                         n_samples=1000,
                         h5path=h5path,
                         hzrd_key='cumulative')


    """ 41 nodes to each BN """ 
    
    # --- decision nodes (10)
    # bld_cpt.event_cpt()
    # bld_cpt.goal_cpt()
    
    # bld_cpt.ex_ante_building_cpt()
    # bld_cpt.ex_ante_electric_cpt()
    # bld_cpt.ex_ante_transportation_cpt()
    # bld_cpt.ex_ante_water_cpt()

    # bld_cpt.ex_post_building_cpt()
    # bld_cpt.ex_post_electric_cpt()
    # bld_cpt.ex_post_transportation_cpt()
    # bld_cpt.ex_post_water_cpt()

    # --- CPTs (31)
    # bld_cpt.functionality_building_cpt()
    # bld_cpt.functionality_electric_cpt()
    # bld_cpt.functionality_transportation_cpt()
    # bld_cpt.functionality_water_cpt()

    # bld_cpt.reptime_building_cpt()
    # bld_cpt.reptime_electric_cpt()
    # bld_cpt.reptime_transportation_cpt()
    # bld_cpt.reptime_water_cpt()
    
    # bld_cpt.robustness_building_cpt()
    # bld_cpt.robustness_electric_cpt()
    # bld_cpt.robustness_transportation_cpt()
    # bld_cpt.robustness_water_cpt()
    
    # bld_cpt.rapidity_building_cpt()
    # bld_cpt.rapidity_electric_cpt()
    # bld_cpt.rapidity_transportation_cpt()
    # bld_cpt.rapidity_water_cpt()
    
    # bld_cpt.functionality_target_bldg_cpt()
    # bld_cpt.functionality_target_elec_cpt()
    # bld_cpt.functionality_target_trns_cpt()
    # bld_cpt.functionality_target_wter_cpt()
    
    # bld_cpt.reptime_target_bldg_cpt()
    # bld_cpt.reptime_target_elec_cpt()
    # bld_cpt.reptime_target_trns_cpt()
    # bld_cpt.reptime_target_wter_cpt()
    
    # bld_cpt.resilience_bldg_cpt()
    # bld_cpt.resilience_elec_cpt()
    # bld_cpt.resilience_trns_cpt()
    # bld_cpt.resilience_wter_cpt()

    # bld_cpt.robust_cpt()
    # bld_cpt.rapid_cpt()
    # bld_cpt.resilience_cpt()



