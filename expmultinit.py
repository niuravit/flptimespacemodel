import networkx as nx
import sys
import importlib
import os

import matplotlib.pyplot as plt
import pickle as pk

from datetime import datetime
from datetime import timedelta
from datetime import time
import pandas as pd
import numpy as np
import random
from copy import deepcopy
import time

import heapq
import json

import timespacenetwork_modules as tsmd
import experiment_on_tsn_modules as exp

class experiment_configuration:
    def __init__(self, arg):
        '''configurations
            inst_size: <int> specifying the size(#node) of instance
            time_lim: <int> specifying time limit of each heuristic in seconds
            instance_list: list<int> specifying instance id
            initialization_list: list<strings> from mgcp, sp specifying the initilization method
            imp_heuristics_list: list<strings> from vol-mgcp, org_vol-mgcp, ssp, vol-mgcp_w_grasp,org_vol-mgcp_w_grasp, adssp specifying the heuristics method ,
            demand_scaler: <int> constant positive factor to scale demand
            name_suff: <string> any name tag for the experiment
            e.g., >>> python3 expmultinit.py 12 300 all mgcp,sp mgcp,ssp testrun
            this command will run instance of size 12 nodes, each 300 s, for all 10 instances, 
            comparing init using mgcp and sp construction, comparing improvement heuristics using mgcp and ssp,
            with additional label on the result out file testrun
        '''
        self.inst_size = None
        self.time_lim = None
        self.instance_list = None
        self.initialization_list = None
        self.imp_heuristics_list = None
        self.demand_scaler = None
        self.name_suff = ""
        self.static_config = None
        
        self.cmd_arg = arg
        self.read_command_line_config(len(arg), arg)
        self.read_json_config()
        
    def read_command_line_config(self,l, argv):
        print("reading configuration from command line...")
        if l == 7:
            self.inst_size = int(argv[1])
            self.time_lim = int(argv[2])
            self.instance_list = self.__parse_instance_list(argv[3])
            self.initialization_list = argv[4].split(',')
            self.imp_heuristics_list = argv[5].split(',')
            self.demand_scaler = float(argv[6])
        elif l == 8:
            self.inst_size = int(argv[1])
            self.time_lim = int(argv[2])
            self.instance_list = self.__parse_instance_list(argv[3])
            self.initialization_list = argv[4].split(',')
            self.imp_heuristics_list = argv[5].split(',')
            self.demand_scaler = float(argv[6])
            self.name_suff = argv[7]
        else:
            raise Exception('invalid command, re-check the definition!')
            
        print(f"\t instance size:{self.inst_size}")
        print(f"\t time limit:{self.time_lim}")
        print(f"\t instance list:{self.instance_list}")
        print(f"\t initialization procedures:{self.initialization_list}")
        print(f"\t improving heuristics:{self.imp_heuristics_list}")
        print(f"\t demand_scaler:{self.demand_scaler}")
        print(f"\t additional name suff:{self.name_suff}")
        
    def __parse_instance_list(self, inst_list_arg):
        if (inst_list_arg == 'all'):
            return [i for i in range(10)]
        else:
            return [int(i) for i in inst_list_arg.split(',')]
        
    def read_json_config(self,):
        # Read the JSON configuration file
        with open('expconfig.json', 'r') as file:
            json_text = file.read()
        self.static_config  = json.loads(json_text)
        # store this as a list
        self.static_config['model']['service_types'] = self.static_config['model']['service_types'].split(",")
    
    def export_json_configs(self, f_name):
        out_dict = self.static_config
        out_dict['cmd_config'] = {
            "inst_size":self.inst_size,
            "time_lim":self.time_lim,
            "instance_list":self.instance_list,
            "initialization_list":self.initialization_list,
            "imp_heuristics_list":self.imp_heuristics_list,
            "demand_scaler":self.demand_scaler,
            "name_suff":self.name_suff,
        }
        
        with open(f'{f_name}.json', 'w') as file:
            json.dump(out_dict, file, indent=4)
    
    def get_rearrange_col_log(self):
        rearrange_col = []
        instance_spec_col = ['nodes_no', 'arcs_no', 'total_dem', 'trail_cap', 'min_dem', 'max_dem']
        rearrange_col += instance_spec_col

        imp_obj_set = ['imp%','t_imp%','s_imp%','obj','tcost','scost']    
        util_set = ['ud_avg']
        time_set = ['iter','rtime']

        # obj set
        for init_idx in range(1,len(self.initialization_list)+1):
            init_proc = self.initialization_list[init_idx-1]
            init_re_col = [f'init-{init_proc}-{init_idx}_{val}' for val in ['obj','tcost','scost']]
            rearrange_col += init_re_col
            
            for imp_idx in range(1,len(self.imp_heuristics_list)+1):
                imp_proc = self.imp_heuristics_list[imp_idx-1]
                
                imp_re_col = [f'init-{init_proc}-{init_idx}-imp-{imp_proc}-{imp_idx}_{val}' for val in imp_obj_set]
                rearrange_col += imp_re_col

        # utilization set
        for init_idx in range(1,len(self.initialization_list)+1):
            init_proc = self.initialization_list[init_idx-1]
            init_re_col = [f'init-{init_proc}-{init_idx}_ud_avg']
            rearrange_col += init_re_col
            
            for imp_idx in range(1,len(self.imp_heuristics_list)+1):
                imp_proc = self.imp_heuristics_list[imp_idx-1]
                imp_re_col = [f'init-{init_proc}-{init_idx}-imp-{imp_proc}-{imp_idx}_{val}' for val in util_set]
                rearrange_col += imp_re_col

        # rtime itertime set
        for init_idx in range(1,len(self.initialization_list)+1):
            init_proc = self.initialization_list[init_idx-1] 
            # no init col
            for imp_idx in range(1,len(self.imp_heuristics_list)+1):
                imp_proc = self.imp_heuristics_list[imp_idx-1]
                imp_re_col = [f'init-{init_proc}-{init_idx}-imp-{imp_proc}-{imp_idx}_{val}' for val in time_set]
                rearrange_col += imp_re_col
        return rearrange_col

def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        # If it doesn't exist, create the folder
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")
    else:
        print(f"Folder '{folder}' already exists.")

# read configurations
exp_config = experiment_configuration(sys.argv)
# configuration from json config
constant_dict = exp_config.static_config['model']
instance_path = exp_config.static_config['data']['instance_path']
module_path = exp_config.static_config['data']['module_path']
sys.path.insert(0, module_path)


inst_f_list = os.listdir(instance_path)

# configuration from command line args
inst_size = exp_config.inst_size
time_lim = exp_config.time_lim
instance_id_list = exp_config.instance_list
initialization_list = exp_config.initialization_list
imp_heuristics_list = exp_config.imp_heuristics_list
demand_scaler = exp_config.demand_scaler
name_suff = exp_config.name_suff

# option = 1

inst_names = [n for n in inst_f_list if f"inst_{inst_size}n" in n]
inst_names = sorted(inst_names)
inst_list = [exp.read_instance(instance_path+name) for name in inst_names]

time_stamp = datetime.now().strftime("%H%M-%b%d%y")
result_file_name = f"multicom-tsnetwork-{inst_size}n-{time_lim}tl-{time_stamp}{name_suff}"
json_output_name = f"{result_file_name}-jsonconfig"

print(result_file_name)
log_collection = dict()
result_folder = "playgrounddump/"
plot_folder = "plots/"

# create result and plot folders
create_folder_if_not_exist(result_folder)
create_folder_if_not_exist(plot_folder)

plt.figure(2)
for i in instance_id_list:
    print(f'Staring fixInstanceExperiment... instance {inst_size}n id {i}')
    i_log = exp.fixInstanceExperiment(inst_list[i:i+1], i, constant_dict, initialization_list, imp_heuristics_list, 
                                      time_limit=time_lim, demand_scaling_factor = demand_scaler)
    
    log_collection[i+1] = i_log[1]
    result_tab = pd.DataFrame(log_collection).T
    # arrange col before save as .csv
    rearrange_col = exp_config.get_rearrange_col_log()
    result_tab[rearrange_col].to_csv(result_folder+result_file_name+".csv")
    plt.savefig(f'{plot_folder}scobjplots_inst{i}-{inst_size}n-{time_lim}tl-{time_stamp}{name_suff}.png',bbox_inches='tight')
plt.close()
# recording the json-config 
exp_config.export_json_configs(result_folder+json_output_name)
