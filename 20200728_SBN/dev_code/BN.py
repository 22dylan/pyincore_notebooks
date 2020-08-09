import os
import sys
import time
import pandas as pd
import pomegranate as pmg
import numpy as np
import h5py
import ast
import multiprocessing as mp


"""
Spatial Bayesian network. 
this code written by Dylan Sanderson (March, 2020)
"""

"""
TO-DO:
    - comment code
"""



class Building_BN():
    def __init__(self, cpt_file, guid):
        self.CPT = h5py.File(cpt_file, 'r')
        self.build_network(guid)
 
    def read_h5(self, guid, key, ret_dict=False):
        temp = np.array(self.CPT[guid][key]).astype(str).tolist()
        if ret_dict == True:
            temp = ast.literal_eval(temp)
        else:
            new = []
            for sblst in temp:
                sblst[-1] = float(sblst[-1])
                new.append(sblst)
            temp = new[:]
        return temp

    def build_network(self, guid):
        """ --- reading cpt from h5 file """
        # -- decision nodes (10)

        event_dist = self.read_h5(guid, 'DSCRT_event_dist', ret_dict=True)
        goal_dist = self.read_h5(guid, 'DSCRT_goal_dist', ret_dict=True)

        ex_ante_bldg_dist = self.read_h5(guid, 'DSCRT_ex_ante_building_dist', ret_dict=True)
        ex_ante_elec_dist = self.read_h5(guid, 'DSCRT_ex_ante_electric_dist', ret_dict=True)
        ex_ante_trns_dist = self.read_h5(guid, 'DSCRT_ex_ante_transportation_dist', ret_dict=True)
        ex_ante_wter_dist = self.read_h5(guid, 'DSCRT_ex_ante_water_dist', ret_dict=True)
        
        ex_post_bldg_dist = self.read_h5(guid, 'DSCRT_ex_post_building_dist', ret_dict=True)
        ex_post_elec_dist = self.read_h5(guid, 'DSCRT_ex_post_electric_dist', ret_dict=True)
        ex_post_trns_dist = self.read_h5(guid, 'DSCRT_ex_post_transportation_dist', ret_dict=True)
        ex_post_wter_dist = self.read_h5(guid, 'DSCRT_ex_post_water_dist', ret_dict=True)

        # -- CPTs (29)
        functionality_bldg_cpt = self.read_h5(guid, 'CPT_functionality_bldg')
        functionality_elec_cpt = self.read_h5(guid, 'CPT_functionality_elec')
        functionality_trns_cpt = self.read_h5(guid, 'CPT_functionality_trns')
        functionality_wter_cpt = self.read_h5(guid, 'CPT_functionality_wter')
        
        reptime_bldg_cpt = self.read_h5(guid, 'CPT_reptime_building')
        reptime_elec_cpt = self.read_h5(guid, 'CPT_reptime_electric')
        reptime_trns_cpt = self.read_h5(guid, 'CPT_reptime_transportation')
        reptime_wter_cpt = self.read_h5(guid, 'CPT_reptime_water')
        
        robustness_bldg_cpt = self.read_h5(guid, 'CPT_robust_bldg')
        robustness_elec_cpt = self.read_h5(guid, 'CPT_robust_elec')
        robustness_trns_cpt = self.read_h5(guid, 'CPT_robust_trns')
        robustness_wter_cpt = self.read_h5(guid, 'CPT_robust_wter')
        
        rapidity_bldg_cpt = self.read_h5(guid, 'CPT_rapid_bldg')
        rapidity_elec_cpt = self.read_h5(guid, 'CPT_rapid_elec')
        rapidity_trns_cpt = self.read_h5(guid, 'CPT_rapid_trns')
        rapidity_wter_cpt = self.read_h5(guid, 'CPT_rapid_wter')
        
        functionality_target_bldg_cpt = self.read_h5(guid, 'CPT_functionality_target_bldg')
        functionality_target_elec_cpt = self.read_h5(guid, 'CPT_functionality_target_elec')
        functionality_target_trns_cpt = self.read_h5(guid, 'CPT_functionality_target_trns')
        functionality_target_wter_cpt = self.read_h5(guid, 'CPT_functionality_target_wter')

        reptime_target_bldg_cpt = self.read_h5(guid, 'CPT_reptime_target_bldg')
        reptime_target_elec_cpt = self.read_h5(guid, 'CPT_reptime_target_elec')
        reptime_target_trns_cpt = self.read_h5(guid, 'CPT_reptime_target_trns')
        reptime_target_wter_cpt = self.read_h5(guid, 'CPT_reptime_target_wter')

        resilience_bldg_cpt = self.read_h5(guid, 'CPT_resilient_bldg')
        resilience_elec_cpt = self.read_h5(guid, 'CPT_resilient_elec')
        resilience_trns_cpt = self.read_h5(guid, 'CPT_resilient_trns')
        resilience_wter_cpt = self.read_h5(guid, 'CPT_resilient_wter')

        robustness_cpt = self.read_h5(guid, 'CPT_robust')
        rapid_cpt = self.read_h5(guid, 'CPT_rapid')
        resilient_cpt = self.read_h5(guid, 'CPT_resilient')


        """ --- setting up pomegranate distributions """
        # decision nodes (10)
        event = pmg.DiscreteDistribution(event_dist)
        goal = pmg.DiscreteDistribution(goal_dist)

        ex_ante_bldg = pmg.DiscreteDistribution(ex_ante_bldg_dist)
        ex_ante_elec = pmg.DiscreteDistribution(ex_ante_elec_dist)
        ex_ante_trns = pmg.DiscreteDistribution(ex_ante_trns_dist)
        ex_ante_wter = pmg.DiscreteDistribution(ex_ante_wter_dist)

        ex_post_bldg = pmg.DiscreteDistribution(ex_post_bldg_dist)
        ex_post_elec = pmg.DiscreteDistribution(ex_post_elec_dist)
        ex_post_trns = pmg.DiscreteDistribution(ex_post_trns_dist)
        ex_post_wter = pmg.DiscreteDistribution(ex_post_wter_dist)
        
        # -- CPTs (31)
        functionality_bldg = pmg.ConditionalProbabilityTable(
                                    functionality_bldg_cpt, 
                                    [event,
                                     ex_ante_bldg,
                                     ]
                                    )
        functionality_elec = pmg.ConditionalProbabilityTable(
                                    functionality_elec_cpt, 
                                    [event,
                                     ex_ante_elec,
                                     ]
                                    )
        functionality_trns = pmg.ConditionalProbabilityTable(
                                    functionality_trns_cpt, 
                                    [event,
                                     ex_ante_trns,
                                     ]
                                    )
        functionality_wter = pmg.ConditionalProbabilityTable(
                                    functionality_wter_cpt, 
                                    [event,
                                     ex_ante_wter,
                                     ex_ante_elec
                                     ]
                                    )

        reptime_bldg = pmg.ConditionalProbabilityTable(
                                    reptime_bldg_cpt,
                                    [event,
                                     ex_ante_bldg,
                                     ex_post_bldg
                                    ]
                                    )
        reptime_elec = pmg.ConditionalProbabilityTable(
                                    reptime_elec_cpt,
                                    [event,
                                     ex_ante_elec,
                                     ex_post_elec
                                    ]
                                    )
        reptime_trns = pmg.ConditionalProbabilityTable(
                                    reptime_trns_cpt,
                                    [event,
                                     ex_ante_trns,
                                     ex_post_trns
                                    ]
                                    )
        reptime_wter = pmg.ConditionalProbabilityTable(
                                    reptime_wter_cpt,
                                    [event,
                                     ex_ante_wter,
                                     ex_ante_elec,
                                     ex_post_wter,
                                     ex_post_elec
                                    ]
                                    )



        functionality_target_bldg = pmg.ConditionalProbabilityTable(
                                    functionality_target_bldg_cpt, 
                                    [goal,
                                     event]
                                    )
        functionality_target_elec = pmg.ConditionalProbabilityTable(
                                    functionality_target_elec_cpt, 
                                    [goal,
                                     event]
                                    )
        functionality_target_trns = pmg.ConditionalProbabilityTable(
                                    functionality_target_trns_cpt, 
                                    [goal,
                                     event]
                                    )
        functionality_target_wter = pmg.ConditionalProbabilityTable(
                                    functionality_target_wter_cpt, 
                                    [goal,
                                     event]
                                    )
        
        reptime_target_bldg = pmg.ConditionalProbabilityTable(
                                    reptime_target_bldg_cpt, 
                                    [goal,
                                     event]
                                    )
        reptime_target_elec = pmg.ConditionalProbabilityTable(
                                    reptime_target_elec_cpt, 
                                    [goal,
                                     event]
                                    )
        reptime_target_trns = pmg.ConditionalProbabilityTable(
                                    reptime_target_trns_cpt, 
                                    [goal,
                                     event]
                                    )
        reptime_target_wter = pmg.ConditionalProbabilityTable(
                                    reptime_target_wter_cpt, 
                                    [goal,
                                     event]
                                    )



        Robustness_bldg = pmg.ConditionalProbabilityTable(
                                    robustness_bldg_cpt, 
                                    [functionality_bldg, 
                                     functionality_target_bldg]
                                    )
        Robustness_elec = pmg.ConditionalProbabilityTable(
                                    robustness_elec_cpt, 
                                    [functionality_elec, 
                                     functionality_target_elec]
                                    )
        Robustness_trns = pmg.ConditionalProbabilityTable(
                                    robustness_trns_cpt, 
                                    [functionality_trns, 
                                     functionality_target_trns]
                                    )
        Robustness_wter = pmg.ConditionalProbabilityTable(
                                    robustness_wter_cpt, 
                                    [functionality_wter, 
                                     functionality_target_wter]
                                    )

        Rapidity_bldg = pmg.ConditionalProbabilityTable(
                                    rapidity_bldg_cpt, 
                                    [reptime_bldg, 
                                     reptime_target_bldg]
                                    )
        Rapidity_elec = pmg.ConditionalProbabilityTable(
                                    rapidity_elec_cpt, 
                                    [reptime_elec, 
                                     reptime_target_elec]
                                    )
        Rapidity_trns = pmg.ConditionalProbabilityTable(
                                    rapidity_trns_cpt, 
                                    [reptime_trns, 
                                     reptime_target_trns]
                                    )
        Rapidity_wter = pmg.ConditionalProbabilityTable(
                                    rapidity_wter_cpt, 
                                    [reptime_wter, 
                                     reptime_target_wter]
                                    )


        Resilience_bldg = pmg.ConditionalProbabilityTable(
                                    resilience_bldg_cpt,
                                    [Robustness_bldg,
                                    Rapidity_bldg]
                                    )
        Resilience_elec = pmg.ConditionalProbabilityTable(
                                    resilience_elec_cpt,
                                    [Robustness_elec,
                                    Rapidity_elec]
                                    )
        Resilience_trns = pmg.ConditionalProbabilityTable(
                                    resilience_trns_cpt,
                                    [Robustness_trns,
                                    Rapidity_trns]
                                    )
        Resilience_wter = pmg.ConditionalProbabilityTable(
                                    resilience_wter_cpt,
                                    [Robustness_wter,
                                    Rapidity_wter]
                                    )



        # --- robustness node
        Robustness = pmg.ConditionalProbabilityTable(
                                    robustness_cpt,
                                    [Robustness_bldg,
                                    Robustness_elec,
                                    Robustness_trns,
                                    Robustness_wter])


        # --- rapidity node

        Rapidity = pmg.ConditionalProbabilityTable(
                                    rapid_cpt,
                                    [Rapidity_bldg,
                                    Rapidity_elec,
                                    Rapidity_trns,
                                    Rapidity_wter])


        # --- Resilience node
        Resilience = pmg.ConditionalProbabilityTable(
                                    resilient_cpt,
                                    [Resilience_bldg,
                                    Resilience_elec,
                                    Resilience_trns,
                                    Resilience_wter])


        """ building network in pomegranate """
        # decision nodes (10)
        event_node = pmg.Node(event, name='event')
        goal_node = pmg.Node(goal, name='goal')
        
        ex_ante_bldg_node = pmg.Node(ex_ante_bldg, name='ex_ante_bldg')
        ex_ante_elec_node = pmg.Node(ex_ante_elec, name='ex_ante_elec')
        ex_ante_trns_node = pmg.Node(ex_ante_trns, name='ex_ante_trns')
        ex_ante_wter_node = pmg.Node(ex_ante_wter, name='ex_ante_wter')

        ex_post_bldg_node = pmg.Node(ex_post_bldg, name='ex_post_bldg')
        ex_post_elec_node = pmg.Node(ex_post_elec, name='ex_post_elec')
        ex_post_trns_node = pmg.Node(ex_post_trns, name='ex_post_trns')
        ex_post_wter_node = pmg.Node(ex_post_wter, name='ex_post_wter')

        # -- CPTs (29)
        functionality_bldg_node = pmg.Node(functionality_bldg, name='functionality_bldg')
        functionality_elec_node = pmg.Node(functionality_elec, name='functionality_elec')
        functionality_trns_node = pmg.Node(functionality_trns, name='functionality_trns')
        functionality_wter_node = pmg.Node(functionality_wter, name='functionality_wter')
        
        reptime_bldg_node = pmg.Node(reptime_bldg, name='reptime_bldg')
        reptime_elec_node = pmg.Node(reptime_elec, name='reptime_elec')
        reptime_trns_node = pmg.Node(reptime_trns, name='reptime_trns')
        reptime_wter_node = pmg.Node(reptime_wter, name='reptime_wter')

        functionality_target_bldg_node = pmg.Node(functionality_target_bldg, name='functionality_target_bldg')
        functionality_target_elec_node = pmg.Node(functionality_target_elec, name='functionality_target_elec')
        functionality_target_trns_node = pmg.Node(functionality_target_trns, name='functionality_target_trns')
        functionality_target_wter_node = pmg.Node(functionality_target_wter, name='functionality_target_wter')
        
        reptime_target_bldg_node = pmg.Node(reptime_target_bldg, name='reptime_target_bldg')
        reptime_target_elec_node = pmg.Node(reptime_target_elec, name='reptime_target_elec')
        reptime_target_trns_node = pmg.Node(reptime_target_trns, name='reptime_target_trns')
        reptime_target_wter_node = pmg.Node(reptime_target_wter, name='reptime_target_wter')
        
        Robustness_bldg_node = pmg.Node(Robustness_bldg, name='Robustness_bldg')
        Robustness_elec_node = pmg.Node(Robustness_elec, name='Robustness_elec')
        Robustness_trns_node = pmg.Node(Robustness_trns, name='Robustness_trns')
        Robustness_wter_node = pmg.Node(Robustness_wter, name='Robustness_wter')
        
        Rapidity_bldg_node = pmg.Node(Rapidity_bldg, name='Rapidity_bldg')
        Rapidity_elec_node = pmg.Node(Rapidity_elec, name='Rapidity_elec')
        Rapidity_trns_node = pmg.Node(Rapidity_trns, name='Rapidity_trns')
        Rapidity_wter_node = pmg.Node(Rapidity_wter, name='Rapidity_wter')

        Resilience_bldg_node = pmg.Node(Resilience_bldg, name='Resilience_bldg')
        Resilience_elec_node = pmg.Node(Resilience_elec, name='Resilience_elec')
        Resilience_trns_node = pmg.Node(Resilience_trns, name='Resilience_trns')
        Resilience_wter_node = pmg.Node(Resilience_wter, name='Resilience_wter')

        Robustness_node = pmg.Node(Robustness, name='Robustness')
        Rapidity_node = pmg.Node(Rapidity, name='Rapidity')
        Resilience_node = pmg.Node(Resilience, name='Resilience')

        self.network = pmg.BayesianNetwork(guid)
        self.network.add_states(event_node, 
                                goal_node,
 
                                ex_ante_bldg_node,
                                ex_ante_elec_node,
                                ex_ante_trns_node,
                                ex_ante_wter_node,

                                ex_post_bldg_node,
                                ex_post_elec_node,
                                ex_post_trns_node,
                                ex_post_wter_node,

                                functionality_bldg_node,
                                functionality_elec_node,
                                functionality_trns_node,
                                functionality_wter_node, 
                                
                                reptime_bldg_node,
                                reptime_elec_node,
                                reptime_trns_node,
                                reptime_wter_node,

                                functionality_target_bldg_node,
                                functionality_target_elec_node,
                                functionality_target_trns_node,
                                functionality_target_wter_node,
                                
                                reptime_target_bldg_node,
                                reptime_target_elec_node,
                                reptime_target_trns_node,
                                reptime_target_wter_node,
                                
                                Robustness_bldg_node,
                                Robustness_elec_node,
                                Robustness_trns_node,
                                Robustness_wter_node,
                                
                                Rapidity_bldg_node,
                                Rapidity_elec_node,
                                Rapidity_trns_node,
                                Rapidity_wter_node,
                                
                                Resilience_bldg_node,
                                Resilience_elec_node,
                                Resilience_trns_node,
                                Resilience_wter_node,

                                Robustness_node,
                                Rapidity_node,
                                Resilience_node
                                )

        # --- functionality nodes
        self.network.add_transition(event_node, functionality_bldg_node)
        self.network.add_transition(ex_ante_bldg_node, functionality_bldg_node)

        self.network.add_transition(event_node, functionality_elec_node)
        self.network.add_transition(ex_ante_elec_node, functionality_elec_node)

        self.network.add_transition(event_node, functionality_trns_node)
        self.network.add_transition(ex_ante_trns_node, functionality_trns_node)

        self.network.add_transition(event_node, functionality_wter_node)
        self.network.add_transition(ex_ante_wter_node, functionality_wter_node)
        self.network.add_transition(ex_ante_elec_node, functionality_wter_node)


        # --- reptime nodes
        self.network.add_transition(event_node, reptime_bldg_node)
        self.network.add_transition(ex_ante_bldg_node, reptime_bldg_node)
        self.network.add_transition(ex_post_bldg_node, reptime_bldg_node)

        self.network.add_transition(event_node, reptime_elec_node)
        self.network.add_transition(ex_ante_elec_node, reptime_elec_node)
        self.network.add_transition(ex_post_elec_node, reptime_elec_node)

        self.network.add_transition(event_node, reptime_trns_node)
        self.network.add_transition(ex_ante_trns_node, reptime_trns_node)
        self.network.add_transition(ex_post_trns_node, reptime_trns_node)

        self.network.add_transition(event_node, reptime_wter_node)
        self.network.add_transition(ex_ante_wter_node, reptime_wter_node)
        self.network.add_transition(ex_post_wter_node, reptime_wter_node)
        self.network.add_transition(ex_ante_elec_node, reptime_wter_node)
        self.network.add_transition(ex_post_elec_node, reptime_wter_node)

        # --- func target nodes
        self.network.add_transition(goal_node, functionality_target_bldg_node)
        self.network.add_transition(event_node, functionality_target_bldg_node)
        
        self.network.add_transition(goal_node, functionality_target_elec_node)
        self.network.add_transition(event_node, functionality_target_elec_node)
        
        self.network.add_transition(goal_node, functionality_target_trns_node)
        self.network.add_transition(event_node, functionality_target_trns_node)
        
        self.network.add_transition(goal_node, functionality_target_wter_node)
        self.network.add_transition(event_node, functionality_target_wter_node)


        # --- reptime target node
        self.network.add_transition(goal_node, reptime_target_bldg_node)
        self.network.add_transition(event_node, reptime_target_bldg_node)

        self.network.add_transition(goal_node, reptime_target_elec_node)
        self.network.add_transition(event_node, reptime_target_elec_node)
        
        self.network.add_transition(goal_node, reptime_target_trns_node)
        self.network.add_transition(event_node, reptime_target_trns_node)
        
        self.network.add_transition(goal_node, reptime_target_wter_node)
        self.network.add_transition(event_node, reptime_target_wter_node)


        # --- robustness nodes
        self.network.add_transition(functionality_bldg_node, Robustness_bldg_node)
        self.network.add_transition(functionality_target_bldg_node, Robustness_bldg_node)

        self.network.add_transition(functionality_elec_node, Robustness_elec_node)
        self.network.add_transition(functionality_target_elec_node, Robustness_elec_node)

        self.network.add_transition(functionality_trns_node, Robustness_trns_node)
        self.network.add_transition(functionality_target_trns_node, Robustness_trns_node)

        self.network.add_transition(functionality_wter_node, Robustness_wter_node)
        self.network.add_transition(functionality_target_wter_node, Robustness_wter_node)

        # --- rapidity nodes
        self.network.add_transition(reptime_bldg_node, Rapidity_bldg_node)
        self.network.add_transition(reptime_target_bldg_node, Rapidity_bldg_node)

        self.network.add_transition(reptime_elec_node, Rapidity_elec_node)
        self.network.add_transition(reptime_target_elec_node, Rapidity_elec_node)

        self.network.add_transition(reptime_trns_node, Rapidity_trns_node)
        self.network.add_transition(reptime_target_trns_node, Rapidity_trns_node)
        
        self.network.add_transition(reptime_wter_node, Rapidity_wter_node)
        self.network.add_transition(reptime_target_wter_node, Rapidity_wter_node)

        # --- resilient nodes
        self.network.add_transition(Robustness_bldg_node, Resilience_bldg_node)
        self.network.add_transition(Rapidity_bldg_node, Resilience_bldg_node)
        
        self.network.add_transition(Robustness_elec_node, Resilience_elec_node)
        self.network.add_transition(Rapidity_elec_node, Resilience_elec_node)

        self.network.add_transition(Robustness_trns_node, Resilience_trns_node)
        self.network.add_transition(Rapidity_trns_node, Resilience_trns_node)

        self.network.add_transition(Robustness_wter_node, Resilience_wter_node)
        self.network.add_transition(Rapidity_wter_node, Resilience_wter_node)


        # --- Robustness node
        self.network.add_transition(Robustness_bldg_node, Robustness_node)
        self.network.add_transition(Robustness_elec_node, Robustness_node)
        self.network.add_transition(Robustness_trns_node, Robustness_node)
        self.network.add_transition(Robustness_wter_node, Robustness_node)
        

        # --- Rapidity node
        self.network.add_transition(Rapidity_bldg_node, Rapidity_node)
        self.network.add_transition(Rapidity_elec_node, Rapidity_node)
        self.network.add_transition(Rapidity_trns_node, Rapidity_node)
        self.network.add_transition(Rapidity_wter_node, Rapidity_node)


        # --- resilient node
        self.network.add_transition(Resilience_bldg_node, Resilience_node)
        self.network.add_transition(Resilience_elec_node, Resilience_node)
        self.network.add_transition(Resilience_trns_node, Resilience_node)
        self.network.add_transition(Resilience_wter_node, Resilience_node)

        self.network.bake()

    def eval_network(self, observations):
        names = [i.name for i in self.network.states]
        del observations['guid']
        probs = self.network.predict_proba(observations)
        prob_dict = {}
        for i, name in enumerate(names):
            if type(probs[i]) == str:
                prob_dict[name] = probs[i]
            else:
                keys = probs[i].keys()
                values = probs[i].values()
                prob_dict[name] = {keys[i]:values[i] for i in range(len(keys))}
            
        return prob_dict

    def save_network(self, guid,  path_out):
        BN_name = 'BN_{}.json' .format(guid)
        BN_out = os.path.join(path_out, guid, BN_name)
        network_json = self.network.to_json(separators=(',',': '), indent=4)
        with open(BN_out, 'w') as fout:
            fout.write(network_json)
        
    def load_network(self, path_in):
        with open(path_in) as f:
            network_json = f.read()
        self.network = pmg.from_json(network_json)



class Seaside_buildings_BN():
    def __init__(self, cpt_path):
        self.cpt_file = cpt_path

    def evaluate_BNs(self, input_data_bldg, input_data_cmty, outfilename, n_process,
                    write_out=True):
        n_guids = len(input_data_bldg)
        t = time.time()

        observations = []
        i = 0
        print('\nEvaluating: {}' .format(outfilename))
        print('\tnumber of parcles: {}' .format(n_guids))
        print('\tevent: {}' .format(input_data_cmty.loc['event', 'Value']))
        print('\tex_ante_elec: {}' .format(input_data_cmty.loc['ex_ante_elec', 'Value']))
        print('\tex_ante_trns: {}' .format(input_data_cmty.loc['ex_ante_trns', 'Value']))
        print('\tex_ante_wter: {}' .format(input_data_cmty.loc['ex_ante_wter', 'Value']))
        print('\tex_post_elec: {}' .format(input_data_cmty.loc['ex_post_elec', 'Value']))
        print('\tex_post_trns: {}' .format(input_data_cmty.loc['ex_post_trns', 'Value']))
        print('\tex_post_wter: {}' .format(input_data_cmty.loc['ex_post_wter', 'Value']))
        
        for row_i, row in input_data_bldg.iterrows():
            guid = row['guid']
            observations.append({'guid': guid,
                                'goal': row['goal'],
                                'event': input_data_cmty.loc['event', 'Value'],
                                'ex_ante_bldg': row['ex_ante'],
                                'ex_post_bldg': row['ex_post'],
                                'ex_ante_elec': input_data_cmty.loc['ex_ante_elec', 'Value'],
                                'ex_post_elec': input_data_cmty.loc['ex_post_elec', 'Value'],
                                'ex_ante_trns': input_data_cmty.loc['ex_ante_trns', 'Value'],
                                'ex_post_trns': input_data_cmty.loc['ex_post_trns', 'Value'],
                                'ex_ante_wter': input_data_cmty.loc['ex_ante_wter', 'Value'],
                                'ex_post_wter': input_data_cmty.loc['ex_post_wter', 'Value'],
                                })
            i+=1

        with mp.Pool(n_process) as p:
            results = p.map(self.wrapper, observations)
        
        out_df = pd.DataFrame(results)

        if write_out == True:
            out_df = out_df[['guid', 

                            'robustness_bldg',
                            'robustness_elec',
                            'robustness_trns',
                            'robustness_wter',
                            'robustness',

                            'rapidity_bldg',
                            'rapidity_elec',
                            'rapidity_trns',
                            'rapidity_wter',
                            'rapidity',

                            'resilience_bldg',
                            'resilience_elec',
                            'resilience_trns',
                            'resilience_wter',
                            'resilience'
                            ]]
            
            out_df.set_index('guid', inplace=True)
            out_df.to_csv(outfilename, index=True)
        
        elapsed_eval = time.time()-t
        
        print('\n\nEval complete!')

        print('\trun time:    {:>11.4f}s' .format(elapsed_eval))
        print('\tnumber of BNs: {0:>12}\n' .format(n_guids))        
        print('\tavg resilience_bldg: {:>11.4f}' .format(out_df['resilience_bldg'].mean()))
        print('\tavg resilience_elec: {:>11.4f}' .format(out_df['resilience_elec'].mean()))
        print('\tavg resilience_trns: {:>11.4f}' .format(out_df['resilience_trns'].mean()))
        print('\tavg resilience_wter: {:>11.4f}' .format(out_df['resilience_wter'].mean()))
        print('\n\tavg resilience: {:>11.4f}' .format(out_df['resilience'].mean()))
        
        print('\n-----')
        return out_df

    def wrapper(self, observation):
        guid = observation['guid']
        
        BN = Building_BN(self.cpt_file, guid)
        results = BN.eval_network(observation)

        out_dict = {}
        out_dict['guid'] = guid
        out_dict['resilience_bldg'] = results['Resilience_bldg']['Resilient_bldg_yes']
        out_dict['resilience_elec'] = results['Resilience_elec']['Resilient_elec_yes']
        out_dict['resilience_trns'] = results['Resilience_trns']['Resilient_trns_yes']
        out_dict['resilience_wter'] = results['Resilience_wter']['Resilient_wter_yes']
        
        out_dict['robustness_bldg'] = results['Robustness_bldg']['Robust_bldg_yes']
        out_dict['robustness_elec'] = results['Robustness_elec']['Robust_elec_yes']
        out_dict['robustness_trns'] = results['Robustness_trns']['Robust_trns_yes']
        out_dict['robustness_wter'] = results['Robustness_wter']['Robust_wter_yes']
        
        out_dict['rapidity_bldg'] = results['Rapidity_bldg']['Rapid_bldg_yes']
        out_dict['rapidity_elec'] = results['Rapidity_elec']['Rapid_elec_yes']
        out_dict['rapidity_trns'] = results['Rapidity_trns']['Rapid_trns_yes']
        out_dict['rapidity_wter'] = results['Rapidity_wter']['Rapid_wter_yes']
        
        rapidity_temp = results['Rapidity']['Rapid_yes']
        robustness_temp = results['Robustness']['Robust_yes']
        resilience_temp = results['Resilience']['Resilient_yes']

        out_dict['robustness'] = robustness_temp
        out_dict['rapidity'] = rapidity_temp
        out_dict['resilience'] = resilience_temp
        return out_dict

    def print_percent_complete(self, msg, i, n_i):
        i, n_i = int(i)+1, int(n_i)
        sys.stdout.write('\r')
        sys.stdout.write("{} {}/{} ({:.1f}%)" .format(msg, i, n_i, (100/(n_i)*i)))
        sys.stdout.flush()
        if i/n_i == 1:
            print()


if __name__ == "__main__":
    cpt_path = os.path.join(os.getcwd(), '..', 'data', 'hdf5_files', 'CPTs.h5')
    files = [
                # ['building_BN_input0_EsyTarget.csv', 'community_BN_input0.csv', 'SBNo_EsyTarget.csv'],
                # ['building_BN_input1_ModTarget.csv', 'community_BN_input0.csv', 'SBNo_ModTarget.csv'],
                # ['building_BN_input2_DifTarget.csv', 'community_BN_input0.csv', 'SBNo_DifTarget.csv'],
                
                # fig 9, 10, & 11a
                # ['fig9_bldg_input.csv', 'fig9_cmty_input.csv', 'Fig9-10-11a_SNBo.csv'],
              
                # fig 12b
                ['fig12b_bldg_input.csv', 'fig12b_cmty_input.csv', 'Fig12b_SBNo_TEST.csv'],

                # # fig 13
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt100.csv', 'Fig13_SBNo_rt100.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt250.csv', 'Fig13_SBNo_rt250.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt500.csv', 'Fig13_SBNo_rt500.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt1000.csv', 'Fig13_SBNo_rt1000.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt2500.csv', 'Fig13_SBNo_rt2500.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt5000.csv', 'Fig13_SBNo_rt5000.csv'],
                # ['fig13_bldg_input.csv', 'fig13_cmty_rt10000.csv', 'Fig13_SBNo_rt10000.csv'],
                
            ]

    path_to_input = os.path.join(os.getcwd(), 'BN_input')
    path_to_output = os.path.join(os.getcwd(), 'BN_output')

    guids = list(h5py.File(cpt_path, 'r').keys())
    # guids = guids[0:100] # note: remove this

    SBBN = Seaside_buildings_BN(cpt_path)

    # --- building BNs
    for file in files:
        input_data_bldg = os.path.join(path_to_input, file[0])
        input_data_bldg = pd.read_csv(input_data_bldg)
        input_data_bldg = input_data_bldg[input_data_bldg['guid'].isin(guids)]

        input_data_cmty = os.path.join(path_to_input, file[1])
        input_data_cmty = pd.read_csv(input_data_cmty, index_col=0)

        outfilename = file[2]
        # --- evaluating BNs
        SBBN.evaluate_BNs(input_data_bldg = input_data_bldg, 
                        input_data_cmty = input_data_cmty,
                        outfilename = outfilename, 
                        write_out=True
                        )










