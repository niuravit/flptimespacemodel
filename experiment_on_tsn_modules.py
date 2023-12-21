import sys
import importlib
from datetime import datetime
from datetime import timedelta
from datetime import time

import numpy as np
import os

from itertools import combinations,permutations 
import random 
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pk
import nltk
import copy
from copy import deepcopy
import time

from math import atan
from collections import namedtuple
import concurrent.futures
import timespacenetwork_modules as tsmd
import heapq

def read_instance(f_path):
    with open(f_path,'rb') as handle:
        inst_obj = pk.load(handle)
    return inst_obj

def solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit, lane_selection_mode = "vol",
                                     reflow_mode = "volume_based", init_proc_text="", obj_mode = 'step'):
    # mgcp improving heuristics
    flow_arc_mgcp_imp = deepcopy(init_flow_arc)
    flow_comarc_mgcp_imp = deepcopy(init_flowcom_arc)
    mgcpsolver = tsmd.MarginalCostPathSolver(network, flow_arc_mgcp_imp, flow_comarc_mgcp_imp, init_alpha, obj_mode)
    # for plot labeling
    mgcpsolver.init_proc_text = init_proc_text
    mgcp_iter, mgcp_runtime = mgcpsolver.mgcp_single_lane_improvement_with_time_limit(time_limit ,
                                                                                      lane_selection_mode = lane_selection_mode,
                                                                                      reflow_mode = reflow_mode)
    it_cost,is_cost = tsmd.get_obj(init_flow_arc, mgcpsolver.distance_matrix, mgcpsolver.trailer_cap, mgcpsolver.handling_cost, mgcpsolver.obj_mode)
    init_obj = it_cost+is_cost
    mgcpt_cost,mcgps_cost = tsmd.get_obj(mgcpsolver.flow_arc, mgcpsolver.distance_matrix, mgcpsolver.trailer_cap, mgcpsolver.handling_cost, mgcpsolver.obj_mode)
    mgcp_obj = mgcpt_cost+mcgps_cost
    print(f'{reflow_mode} mcgp improving heuristic:{mgcp_obj}  (t{mgcpt_cost},s{mcgps_cost}), imp\%:{round(100*(mgcp_obj-init_obj)/(init_obj),2)}\%'+
          f' (t{round((mgcpt_cost-it_cost)*100/it_cost,2)}% , s{round((mcgps_cost-is_cost)*100/is_cost,2)}%)')
    # validate solution
    # mgcpsolver.validate_demand_delivered(flow_arc_mgcp_imp,flow_comarc_mgcp_imp)
    mgcpsolver.validate_demand_delivered(mgcpsolver.flow_arc,mgcpsolver.flowcom_arc)
    # retrieve path sol
    mgcp_path_sol = mgcpsolver.convert_arc_sol_to_path_sol(mgcpsolver.flowcom_arc)
    # output solution is stored in self.flow_arc and self.flowcom_arc
    mgcp_output = (mgcp_obj, mgcpsolver.flow_arc, mgcpsolver.flowcom_arc, mgcp_path_sol, mgcp_iter, mgcp_runtime, mgcpsolver.plot_objs, mgcpsolver.logobjval) 
    return mgcp_output

def solve_adaptive_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, 
                                                      init_proc_text="", obj_mode = 'step'):
    # slope scaling improving heuristics
    flow_arc_ass_imp = deepcopy(init_flow_arc)
    flow_comarc_ass_imp = deepcopy(init_flowcom_arc)
    sssolver = tsmd.SlopeScalingSolver(network, flow_arc_ass_imp, flow_comarc_ass_imp, obj_mode); 
    sssolver.init_proc_text = init_proc_text;
    min_sol, iter_log = sssolver.adaptive_slope_scalling_with_time_limit(time_limit=time_limit, plot_slope=True)
    flow_arc_ass_imp = min_sol['flow_arc']
    flow_comarc_ass_imp = min_sol['flowcom_arc']
    path_sol_ass_imp = min_sol['path_sol']
    it_cost,is_cost = tsmd.get_obj(init_flow_arc,sssolver.distance_matrix, sssolver.trailer_cap, sssolver.handling_cost, sssolver.obj_mode);
    init_obj = it_cost+is_cost
    ss_obj = min_sol['obj'];
    ss_tcost = min_sol['tcost']; ss_scost = min_sol['scost']
    print('adaptive (a)ss improving heuristic:{}, imp\%:{}'.format(ss_obj, round(100*(ss_obj-init_obj)/(init_obj),2)) +
          f' (t{round((ss_tcost-it_cost)*100/it_cost,2)}% , s{round((ss_scost-is_cost)*100/is_cost,2)}%)')
    sssolver.validate_demand_duedate_satisfaction(min_sol['path_sol'])
    ss_output = (ss_obj,flow_arc_ass_imp,flow_comarc_ass_imp,path_sol_ass_imp,len(iter_log), iter_log[len(iter_log)-1]['timestamp'], sssolver.plot_objs,iter_log)             
    return ss_output

def solve_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, 
                                             init_proc_text="",obj_mode = 'step', phases = ['P0']):
    # slope scaling improving heuristics
    flow_arc_ss_imp = deepcopy(init_flow_arc)
    flow_comarc_ss_imp = deepcopy(init_flowcom_arc)
    sssolver = tsmd.SlopeScalingSolver(network, flow_arc_ss_imp, flow_comarc_ss_imp, obj_mode); 
    sssolver.init_proc_text = init_proc_text;
    # min_sol, iter_log = sssolver.concurrent_slope_scalling_with_time_limit(time_limit=time_limit, plot_slope=True)
    min_sol, iter_log = sssolver.concurrent_slope_scalling_multiphase_with_time_limit(time_limit=time_limit,
                                                                                       plot_slope=True,
                                                                                       phases = phases)
    flow_arc_ss_imp = min_sol['flow_arc']
    flow_comarc_ss_imp = min_sol['flowcom_arc']
    path_sol_ss_imp = min_sol['path_sol']
    it_cost,is_cost = tsmd.get_obj(init_flow_arc, sssolver.distance_matrix, sssolver.trailer_cap, sssolver.handling_cost, sssolver.obj_mode);
    init_obj = it_cost+is_cost
    ss_obj = min_sol['obj'];
    ss_tcost = min_sol['tcost']; ss_scost = min_sol['scost']
    print('ss improving heuristic:{}, imp\%:{}'.format(ss_obj, round(100*(ss_obj-init_obj)/(init_obj),2)) +
          f' (t{round((ss_tcost-it_cost)*100/it_cost,2)}% , s{round((ss_scost-is_cost)*100/is_cost,2)}%)')
    sssolver.validate_demand_duedate_satisfaction(min_sol['path_sol'])
    ss_output = (ss_obj,flow_arc_ss_imp,flow_comarc_ss_imp,path_sol_ss_imp,len(iter_log),iter_log[len(iter_log)-1]['timestamp'], sssolver.plot_objs,iter_log)             
    return ss_output

def get_util_dist_average(flow_arc,distance_matrix,trailer_cap):
    total_dist_pos_arc = sum([distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
    total_util_dist = sum([(flow_arc[a]/(np.ceil(flow_arc[a]/trailer_cap)*trailer_cap))*distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
    return total_util_dist/total_dist_pos_arc

def generate_initial_solution(network, init_mode, obj_mode):
    '''return: flow_arc, flowcom_arc, alpha'''
    if (init_mode == "mgcp"):
        mgcpsolver = tsmd.MarginalCostPathSolver(network,{}, {}, {},obj_mode)
        flow_arc_mgcpinit1,flowcom_arc_mgcpinit1 = mgcpsolver.mgcp_construction(plot_network = False, save_to_img = False)
        it_cost,is_cost = tsmd.get_obj(flow_arc_mgcpinit1, mgcpsolver.distance_matrix, mgcpsolver.trailer_cap, mgcpsolver.handling_cost, mgcpsolver.obj_mode)
        # init_obj = it_cost+is_cost
        path_sol_mgcpinit1 = mgcpsolver.convert_arc_sol_to_path_sol(flowcom_arc_mgcpinit1)
        return (it_cost,is_cost), flow_arc_mgcpinit1, flowcom_arc_mgcpinit1,path_sol_mgcpinit1, mgcpsolver.alpha
    elif ("sp" in init_mode):
        init_proc, init_arr_forcing = init_mode.split("_")
        sssolver = tsmd.SlopeScalingSolver(network, {}, {},obj_mode)
        spp_fa_init, spp_fca_init, spp_path_sol_init = sssolver.get_initial_shortest_path_sol(fixing_intree = True, 
                                                                                              arrival_forcing = init_arr_forcing)
        it_cost,is_cost = tsmd.get_obj(spp_fa_init, sssolver.distance_matrix, sssolver.trailer_cap, sssolver.handling_cost, sssolver.obj_mode)
        # init_obj = it_cost+is_cost
        return (it_cost,is_cost), spp_fa_init, spp_fca_init, spp_path_sol_init, sssolver.alpha
    else:
        raise Exception(f"Invalid init mode {init_mode}")
    
def get_initial_solution(init_proc, init_sol_path, demand_scaling_factor, inst_name, network, obj_mode):
    if "load" in init_proc: # either load_sp or load_mgcp
        ip = init_proc.split('-')[1]
        fname = f"init_sol_{ip}_demscale{demand_scaling_factor}_{inst_name}" 
        instance = read_initial_sol(f"{init_sol_path}{fname}")
        flow_arc = __get_fca_from_fa(instance['flowcom_arc'])
        (it_cost, is_cost), init_fa, init_fca, init_path_sol, alpha = ((instance['t_cost'],instance['s_cost']),flow_arc, instance['flowcom_arc'], instance['path_sol'], instance['alpha'] )
        return (it_cost, is_cost), init_fa, init_fca, init_path_sol, alpha, instance['network']
    else:
        (it_cost, is_cost), init_fa, init_fca, init_path_sol, alpha = generate_initial_solution(network, init_proc, obj_mode)
        return (it_cost, is_cost), init_fa, init_fca, init_path_sol, alpha, network

def __get_fca_from_fa(flowcom_arc):
    flow_arc = dict()
    for dc in flowcom_arc:
        for arc in flowcom_arc[dc]:
            if arc in flow_arc.keys():
                flow_arc[arc] += flowcom_arc[dc][arc]
            else:
                flow_arc[arc] = flowcom_arc[dc][arc]
    return flow_arc

def runimprovement(network, init_fa, init_fca, alpha, imp_proc, time_limit, init_proc_text = "", obj_mode = "step" ):
    # create the replications of the init solution 
    init_flow_arc = deepcopy(init_fa)
    init_flowcom_arc = deepcopy(init_fca)
    init_alpha = deepcopy(alpha)
    # mode split by -, for mgcp, we can specify the lane selection mode and reflow mode
    _temp = imp_proc.split("-")
    if (len(_temp)==1): 
        imp_proc = _temp[0]
        lane_sel_mode = 'vol'
    elif (len(_temp)==2): 
        (mode,imp_proc) = _temp
        if (imp_proc == 'mgcp'):
            lane_sel_mode = mode
        elif (imp_proc == 'ssp'):
            phase_mode = mode
            if (phase_mode == "singlephase"):
                phases = ["P0"]
            elif (phase_mode == "multiphase"):
                phases = ["P1","P2","P3","P4"]
    else:
        raise Exception(f"Invalid imp proc mode {imp_proc}")
    
    if (imp_proc == "mgcp"):
        # mgcp improving heuristics
        mgcp_output = solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit, 
                                                      lane_selection_mode = lane_sel_mode ,reflow_mode = "volume_based", 
                                                      obj_mode = obj_mode,
                                                      init_proc_text = init_proc_text)
        (mgcp_obj,flow_arc_mgcp_imp,flow_comarc_mgcp_imp,path_sol_mgcp_imp, mgcp_iter, mgcp_runtime, plot_objs,iter_log) = mgcp_output
        return (flow_arc_mgcp_imp,flow_comarc_mgcp_imp,path_sol_mgcp_imp,mgcp_iter,mgcp_runtime,plot_objs,iter_log)
    elif (imp_proc == "mgcp_w_grasp"):
        # grasp mgcp improving heuristics 
        mgcp_output = solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit,
                                                      lane_selection_mode = lane_sel_mode, reflow_mode = "grasp",
                                                      obj_mode = obj_mode,
                                                      init_proc_text = init_proc_text)
        (mgcp_obj,flow_arc_mgcp_imp,flow_comarc_mgcp_imp, path_sol_mgcp_imp, mgcp_iter, mgcp_runtime,plot_objs,iter_log) = mgcp_output
        return (flow_arc_mgcp_imp,flow_comarc_mgcp_imp,path_sol_mgcp_imp,mgcp_iter,mgcp_runtime,plot_objs,iter_log)
    elif (imp_proc == "ssp"):
        # slope scaling improving heuristics: sc don't need alpha
        ss_output = solve_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, 
                                                             init_proc_text = init_proc_text,
                                                             obj_mode = obj_mode, phases = phases )
        (ss_obj,flow_arc_ss_imp,flow_comarc_ss_imp,path_sol_ss_imp, iter_ss_imp, rtime_ss_imp,plot_objs,iter_log) = ss_output           
        return (flow_arc_ss_imp,flow_comarc_ss_imp,path_sol_ss_imp,iter_ss_imp,rtime_ss_imp,plot_objs,iter_log)
    elif (imp_proc == "assp"):
        # slope scaling improving heuristics: sc don't need alpha
        ass_output = solve_adaptive_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit,
                                                                       init_proc_text = init_proc_text,
                                                                       obj_mode = obj_mode,)
        (ss_obj,flow_arc_ass_imp,flow_comarc_ass_imp,path_sol_ss_imp,iter_ass_imp, rtime_ass_imp,plot_objs,iter_log) = ass_output           
        return (flow_arc_ass_imp,flow_comarc_ass_imp,path_sol_ss_imp,iter_ass_imp,rtime_ass_imp,plot_objs,iter_log)
    else:
        raise Exception(f"invalid heuristic algo: {imp_proc}")

def _demand_scaler(network, scaler):
    x = 0;
    for fc in network.demand_by_fc:
        for a in network.demand_by_fc[fc]:
            x = network.demand_by_fc[fc][a]*scaler
            network.demand_by_fc[fc][a] = x
    print(f'Done scaling all demand by a factor of {scaler}');
    


def fixInstanceExperiment(inst_list,inst_name_list, inst_id, constant_dict, initialization_list, imp_heuristics_list,
                          iter_log = None, 
                          time_limit = 120, 
                          demand_scaling_factor = 1,
                          sortc_scaling_factor = 1,
                          obj_mode = "step",
                          plot_folder = "/plots",
                          save_instance_path = None): # seconds):
    HANDLING_COST = constant_dict["handling_cost"]*sortc_scaling_factor
    TRAILER_CAP = constant_dict["trailer_cap"]
    num_days = constant_dict['num_days']
    periods_per_day = constant_dict['periods_per_day']
    hub_capacity = None
    simplified_sort_label = constant_dict['simplified_sort_label']
    velocity = constant_dict['velocity']

    plot_collections = []

    if (iter_log is None):
        iter_log = dict(); itx_ct = 0;
    else:
        itx_ct = len(iter_log)
        
    for i in range(len(inst_list)):
        inst = inst_list[i]
        itx_ct+=1  
        travel_hr_matrix = dict([(e,inst['distance_matrix'][e]/velocity) for e in inst['distance_matrix']])
        # generate time-expanded network from instance
        network = tsmd.TimeExpandedNetwork(inst, num_days, periods_per_day, hub_capacity, travel_hr_matrix, 
                                   simplified_sort_label,TRAILER_CAP, HANDLING_COST)
        network.remove_infeasible_demand()
        _demand_scaler(network, demand_scaling_factor);
        flatten_dem = [network.demand_by_fc[dc][a] for dc in network.demand_by_fc for a in network.demand_by_fc[dc]]
        # create new log dict for each instance
        log = {}; _logger = logger(log, obj_mode)
        # log statistic of instance
        log['inst_id'] = i; log['dem_sc'] = demand_scaling_factor; log['s_sc'] = sortc_scaling_factor;
        log['nodes_no'] = len(network.nodes); log['arcs_no'] = len(network.edges);
        log['total_dem'] = round(sum(flatten_dem),2); log['trail_cap'] = TRAILER_CAP
        log['min_dem'] = round(min(flatten_dem),2); log['max_dem'] = round(max(flatten_dem),2); 
        
        # initialize the feasible solution
        init_ct = 0
        for init_proc in initialization_list:
            init_ct+=1
            # get initial solution: either newly generated or presaved init solution
            (it_cost, is_cost), init_fa, init_fca, init_path_sol, alpha, network = get_initial_solution(init_proc, save_instance_path, demand_scaling_factor, inst_name_list[i], network, obj_mode)
            # add-hoc sort-cost scaler adjustment
            print(f"Adjusting sorting cost from {constant_dict['handling_cost']} to {HANDLING_COST}")
            network.handling_cost = HANDLING_COST
            network.update_phases_for_multi_phases()
            # (it_cost, is_cost), init_fa, init_fca, alpha = generate_initial_solution(network, init_proc, obj_mode)
            _logger.log_initial_solution(network, init_fa, init_fca, init_path_sol, init_proc, init_ct)

            imp_ct = 0
            # run improving heuristics
            for imp_proc in imp_heuristics_list:
                imp_ct+=1
                # special mode add for saving init_sol of big instance: 50n
                if (imp_proc == "save_init_sol"):
                     __pack_instance_obj_and_save(_logger, log, obj_mode, 
                                                  it_cost, is_cost, init_fca, init_path_sol,
                                                  alpha, network, 
                                                  demand_scaling_factor, init_proc, 
                                                  save_instance_path, inst_name_list[i])
                    
                else:
                    print(f"==={itx_ct}: init-proc {init_proc}-{init_ct}, imp-proc {imp_proc}-{imp_ct} ===");print(" ")
                    (imp_fa, imp_fca, imp_path_sol, imp_iter, imp_rtime, imp_plot_objs, iter_log) = runimprovement(network, init_fa, init_fca, alpha, imp_proc, time_limit,
                                                                                        obj_mode = obj_mode, 
                                                                                        init_proc_text = f"inst{inst_id}-init{init_proc}{init_ct}-imp{imp_proc}{imp_ct}" )
                    _logger.init_proc_text = f"inst{inst_id}-init{init_proc}{init_ct}-imp{imp_proc}{imp_ct}"
                    # imp_fc, imp_fca = ({},{})
                    # wait for updating
                    # imp_path_sol = {}
                    _logger.log_improved_solution(network, imp_fa, imp_fca, imp_path_sol, imp_iter, imp_rtime, (it_cost, is_cost), f"init-{init_proc}-{init_ct}", f"imp-{imp_proc}-{imp_ct}")
                    # save the plot
                    _logger.save_plots(imp_plot_objs,f'{plot_folder}{time_limit}tl')
                    # save imp log
                    _logger.save_imp_log(iter_log,f'{plot_folder}{time_limit}tl')

        iter_log[itx_ct] = log
    return iter_log

def __pack_instance_obj_and_save(logger, log, obj_mode, it_cost, is_cost, init_fca, init_path_sol, alpha, network, demand_scaling_factor, init_proc, save_instance_path, inst_name):
    inst_init_sol = dict()
    inst_init_sol['nodes_no'] = log['nodes_no']
    inst_init_sol['arcs_no'] = log['arcs_no']
    inst_init_sol['obj_mode'] = obj_mode
    inst_init_sol['t_cost'] = it_cost
    inst_init_sol['s_cost'] = is_cost
    inst_init_sol['flowcom_arc'] = init_fca
    inst_init_sol['path_sol'] = init_path_sol
    inst_init_sol['alpha'] = alpha
    inst_init_sol['network'] = network
    inst_init_sol['demand_scaling_factor'] = demand_scaling_factor
    fname = save_instance_path + f"init_sol_{init_proc}_demscale{demand_scaling_factor}_{inst_name}"
    # save as pickle, the initial solution
    logger.save_instance(inst_init_sol, fname)



class logger:
    def __init__(self, init_log, obj_mode):
        self.log = init_log
        self.obj_mode = obj_mode
    
    def log_initial_solution(self, network, init_fa, init_fca, init_path_sol, init_proc, init_id):
        # solver for cal obj
        __solver = tsmd.MarginalCostPathSolver(network, init_fa, init_fca,{}, self.obj_mode)
        it_cost,is_cost = tsmd.get_obj(init_fa, __solver.distance_matrix, __solver.trailer_cap, __solver.handling_cost, __solver.obj_mode)
         # we just load the initial solution. the objective cost calculation maybe differ from when it's generated, so recaluculate again.
        print(f'init-sol recal cost: handling cost used: {network.handling_cost}, tcost{it_cost}, scost{is_cost}')

        init_obj = it_cost+is_cost
        # trailer util
        init_util_dist_avg = self.get_util_dist_average(init_fa, network.distance_matrix, network.trailer_cap)
        init_avg_path_length = self.get_path_length_average(init_path_sol)
        
        self.log[f"init-{init_proc}-{init_id}_obj"] = round(init_obj,2)
        self.log[f"init-{init_proc}-{init_id}_tcost"] = round(it_cost,2)
        self.log[f"init-{init_proc}-{init_id}_scost"] = round(is_cost,2)
        self.log[f"init-{init_proc}-{init_id}_ud_avg"] = round(init_util_dist_avg,5); 
        self.log[f"init-{init_proc}-{init_id}_path_len"] = round(init_avg_path_length,5); 
    
    def log_improved_solution(self, network, imp_fa, imp_fca, imp_path_sol, imp_iter, imp_rtime, init_obj, init_proc, imp_proc):
        it_cost, is_cost = init_obj
        # solver for cal obj
        __solver = tsmd.MarginalCostPathSolver(network, imp_fa, imp_fca,{}, self.obj_mode)
        imp_tcost,imp_scost = tsmd.get_obj(imp_fa, __solver.distance_matrix, __solver.trailer_cap, __solver.handling_cost, __solver.obj_mode)
       
        imp_obj = imp_tcost+imp_scost
        # trailer util
        imp_util_dist_avg = self.get_util_dist_average(imp_fa, network.distance_matrix, network.trailer_cap)
        imp_avg_path_length = self.get_path_length_average(imp_path_sol)
        
        self.log[f"{init_proc}-{imp_proc}_obj"] = round(imp_obj,2)
        self.log[f"{init_proc}-{imp_proc}_tcost"] = round(imp_tcost,2)
        self.log[f"{init_proc}-{imp_proc}_scost"] = round(imp_scost,2)

        self.log[f"{init_proc}-{imp_proc}_iter"] = round(imp_iter)
        self.log[f"{init_proc}-{imp_proc}_rtime"] = round(imp_rtime,2)
        self.log[f"{init_proc}-{imp_proc}_imp%"] = round((imp_obj-sum(init_obj))/sum(init_obj),5); 
        self.log[f"{init_proc}-{imp_proc}_t_imp%"] = round((imp_tcost-it_cost)/it_cost,5); 
        self.log[f"{init_proc}-{imp_proc}_s_imp%"] = round((imp_scost-is_cost)/is_cost,5); 

        self.log[f"{init_proc}-{imp_proc}_ud_avg"] = round(imp_util_dist_avg,5); 
        self.log[f"{init_proc}-{imp_proc}_path_len"] = round(imp_avg_path_length,5); 
        
    def get_path_length_average(self, path_sol):
        acc_length = 0
        for p_key in path_sol:
            acc_length += (len(path_sol[p_key][1])-1)
        if (len(path_sol) == 0): return 0
        else: return acc_length/len(path_sol)

    def get_util_dist_average(self, flow_arc, distance_matrix, trailer_cap):
        total_dist_pos_arc = sum([distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
        if total_dist_pos_arc < 1e-5 :
            return 0
        total_util_dist = sum([(flow_arc[a]/(np.ceil(flow_arc[a]/trailer_cap)*trailer_cap))*distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
        return total_util_dist/total_dist_pos_arc
    
    def save_plots(self, plot_objs, plot_folder):
        for (name, (fig,ax)) in plot_objs.items():
            plt.savefig(f'{plot_folder}{name}.png', bbox_inches='tight')
            plt.clf()
    
    def save_imp_log(self, iter_log, save_folder):
        file_path = f'{save_folder}{self.init_proc_text}.txt'
        with open(file_path, 'w') as file:
            objs = [iter_log[i]['obj'] for i in range(0,len(iter_log)-1)]
            tcosts = [iter_log[i]['tcost'] for i in range(0,len(iter_log)-1)]
            scosts = [iter_log[i]['scost'] for i in range(0,len(iter_log)-1)]
            appcosts = [iter_log[i]['appcost'] for i in range(0,len(iter_log)-1)]
            # Convert each inner list to a string and write it to the file
            file.write("objs," +','.join(map(str, objs)) + '\n')
            file.write("tcosts," +','.join(map(str, tcosts)) + '\n')
            file.write("scosts," +','.join(map(str, scosts)) + '\n')
            file.write("appcosts," +','.join(map(str, appcosts)) + '\n')

    
    def save_instance(self, instance, file_path):
        with open(file_path, 'wb') as file:
            pk.dump(instance, file)

        print(f"Instace saved as pickle file: {file_path}")
    
def read_initial_sol(file_path):
    with open(file_path, 'rb') as file:
        loaded_object = pk.load(file)
    print(f"Loaded object from path {file_path}")
    return loaded_object

