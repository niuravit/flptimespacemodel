# import networkx as nx
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
            obj_func: <string> from step or fix or modified
            initialization_list: list<strings> from mgcp, sp_earliest or sp_exact specifying the initilization method
                                * load from presave init: load-mgcp or load-sp_earliest or load-sp_exactdate
            imp_heuristics_list: list<strings> from vol-mgcp, org_vol-mgcp, multiphase-ssp,singlephase-ssp, vol-mgcp_w_grasp,org_vol-mgcp_w_grasp, adssp specifying the heuristics method ,
                                * special mode: "save_init_sol" no imp heuristics is running, but save out init solution as pickle
            demand_scaler: list<int> constant positive factor to scale demand
            sort_scaler: list<int> constant positive factor to scale sort cost
            name_suff: <string> any name tag for the experiment
            e.g., >>> python3 expmultinit.py 12 300 all mgcp,sp mgcp,ssp testrun
            this command will run instance of size 12 nodes, each 300 s, for all 10 instances, 
            comparing init using mgcp and sp construction, comparing improvement heuristics using mgcp and ssp,
            with additional label on the result out file testrun
        '''
        self.inst_size = None
        self.time_lim = None
        self.instance_list = None
        self.obj_func = None
        self.initialization_list = None
        self.imp_heuristics_list = None
        self.demand_scaler = None
        self.sort_scaler = None
        self.name_suff = ""
        self.static_config = None
        
        self.cmd_arg = arg
        self.read_command_line_config(len(arg), arg)
        self.read_json_config()
        
    def read_command_line_config(self,l, argv):
        print("reading configuration from command line...")
        if l == 10:
            self.inst_size = argv[1]
            self.time_lim = int(argv[2])
            self.instance_list = self.__parse_instance_list(argv[3])
            self.obj_func = argv[4]
            self.initialization_list = argv[5].split(',')
            self.imp_heuristics_list = argv[6].split(',')
            self.demand_scaler = [float(d) for d in argv[7].split(',')]
            self.sort_scaler = [float(s) for s in argv[8].split(',')]
            self.name_suff = argv[9]
        else:
            raise Exception('invalid command, re-check the definition!')
            
        print(f"\t instance size:{self.inst_size}")
        print(f"\t time limit:{self.time_lim}")
        print(f"\t instance list:{self.instance_list}")
        print(f"\t obj_func:{self.obj_func}")
        print(f"\t initialization procedures:{self.initialization_list}")
        print(f"\t improving heuristics:{self.imp_heuristics_list}")
        print(f"\t demand_scaler:{self.demand_scaler}")
        print(f"\t sort_scaler:{self.sort_scaler}")
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
            "obj_func":self.obj_func,
            "initialization_list":self.initialization_list,
            "imp_heuristics_list":self.imp_heuristics_list,
            "demand_scaler":self.demand_scaler,
            "sort_scaler":self.sort_scaler,
            "name_suff":self.name_suff,
        }
        
        with open(f'{f_name}.json', 'w') as file:
            json.dump(out_dict, file, indent=4)
    
    def get_rearrange_col_log(self, gen_init_sol_flg = False):
        rearrange_col = []
        instance_spec_col = ['inst_id','dem_sc','s_sc','nodes_no', 'arcs_no', 'total_dem', 'trail_cap', 'min_dem', 'max_dem',]
        rearrange_col += instance_spec_col
        if (gen_init_sol_flg):
            for init_idx in range(1,len(self.initialization_list)+1):
                init_proc = self.initialization_list[init_idx-1]
                init_re_col = [f'init-{init_proc}-{init_idx}_{val}' for val in ['obj','tcost','scost','ud_avg','path_len']]
                rearrange_col += init_re_col
        else:
            imp_obj_set = ['imp%','t_imp%','s_imp%','obj','tcost','scost']    
            util_set = ['ud_avg','path_len']
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
                init_re_col = [f'init-{init_proc}-{init_idx}_{val}' for val in ['ud_avg','path_len']]
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
    if not(os.path.exists(folder)):
        # If it doesn't exist, create the folder
        print(f"Folder '{folder}' not exists, create a new one!")
        os.makedirs(folder,exist_ok=True)
        
    else:
        print(f"Folder '{folder}' already exists.")

# read configurations
exp_config = experiment_configuration(sys.argv)
# configuration from json config
constant_dict = exp_config.static_config['model']
instance_path = exp_config.static_config['data']['instance_path']
init_sol_path = exp_config.static_config['data']['init_sol_path']
module_path = exp_config.static_config['data']['module_path']
sys.path.insert(0, module_path)

inst_f_list = os.listdir(instance_path)

# configuration from command line args
inst_size = exp_config.inst_size
time_lim = exp_config.time_lim
instance_id_list = exp_config.instance_list
obj_func = exp_config.obj_func
initialization_list = exp_config.initialization_list
imp_heuristics_list = exp_config.imp_heuristics_list
demand_scaler_list = exp_config.demand_scaler
sort_scaler_list = exp_config.sort_scaler
name_suff = exp_config.name_suff


inst_names = [n for n in inst_f_list if f"inst_{inst_size}n" in n]
inst_names = sorted(inst_names)
inst_list = [exp.read_instance(instance_path+name) for name in inst_names]

time_stamp = datetime.now().strftime("%H%M-%b%d%y")
batch_name = f"multicom-tsnetwork-{inst_size}n-{time_lim}tl-{time_stamp}{name_suff}"
json_output_name = f"{batch_name}-jsonconfig"

print(batch_name)
log_collection = dict()
result_folder = "playgrounddump/"
result_subfolder = f"{result_folder}{batch_name}/"
plot_folder = "plots/"
plot_subfolder = f"{plot_folder}{batch_name}/"


# create result and plot folders
create_folder_if_not_exist(result_folder)
create_folder_if_not_exist(result_subfolder)
create_folder_if_not_exist(plot_folder)
create_folder_if_not_exist(plot_subfolder)

exp_uid = 0
for demand_scaler in demand_scaler_list:
    for sort_scaler in sort_scaler_list:
        for i in instance_id_list:
            result_file_name = f"{time_stamp}{name_suff}_log"
            print(f'Staring fixInstanceExperiment... instance {inst_size}n, id {i}, d-sc{demand_scaler}, s-sc{sort_scaler}')
            i_log = exp.fixInstanceExperiment(inst_list[i],inst_names[i], i, constant_dict, initialization_list, imp_heuristics_list, 
                                            time_limit=time_lim, demand_scaling_factor = demand_scaler, sortc_scaling_factor = sort_scaler, obj_mode = obj_func,
                                            plot_folder=plot_subfolder, save_instance_path = init_sol_path)
            log_collection[exp_uid] = i_log[1]
            result_tab = pd.DataFrame(log_collection).T
            if (imp_heuristics_list[0]=="save_init_sol"):
                # no improvement is running
                rearrange_col = exp_config.get_rearrange_col_log(gen_init_sol_flg = True)
                # continue
            else:
                # arrange col before save as .csv
                rearrange_col = exp_config.get_rearrange_col_log(gen_init_sol_flg = False)
            result_tab[rearrange_col].to_csv(result_subfolder+result_file_name+".csv")
            # update exp uid
            exp_uid+=1
    
# recording the json-config 
exp_config.export_json_configs(result_subfolder+json_output_name)
