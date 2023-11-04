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

def fixInstanceSCImprovingSPPInitExperiment(inst_list, constant_dict,
                                                        iter_log = None, time_limit = 120): # seconds):
    HANDLING_COST = constant_dict["HANDLING_COST"]
    TRAILER_CAP = constant_dict["TRAILER_CAP"]
    SERVICE_TYPES = constant_dict["SERVICE_TYPES"]
    num_days = constant_dict['num_days']
    periods_per_day = constant_dict['periods_per_day']
    hub_capacity = constant_dict['hub_capacity']
    simplified_sort_label = constant_dict['simplified_sort_label']
    velocity = constant_dict['velocity']
    
    if (iter_log is None):
        iter_log = dict(); itx_ct = 0;
    else:
        itx_ct = len(iter_log)
        
    for inst in inst_list:
        itx_ct+=1 
        print(f'=== Iteration {itx_ct} ===')
        travel_hr_matrix = dict([(e,inst['distance_matrix'][e]/velocity) for e in inst['distance_matrix']])
        # generate time-expanded network from instance
        network = tsmd.TimeExpandedNetwork(inst, num_days, periods_per_day, hub_capacity, travel_hr_matrix, 
                                   simplified_sort_label,TRAILER_CAP, HANDLING_COST)
        network.remove_infeasible_demand()
        flatten_dem = [network.demand_by_fc[dc][a] for dc in network.demand_by_fc for a in network.demand_by_fc[dc]]
        # create log dict
        log = {lk:None for lk in constant_dict['log_keys']}
        
        # log statistic of instance
        log['nodes_no'] = len(network.nodes); log['arcs_no'] = len(network.edges);
        log['total_dem'] = sum(flatten_dem); log['trail_cap'] = TRAILER_CAP
        log['min_dem'] = min(flatten_dem); log['max_dem'] = max(flatten_dem); 
        
        # arc-based model...
        abm_obj = None    
        sample_no = 2
        for i in range(sample_no):
            # init: random sequence of MGCP 
            sppinit_id = i + 1
            print(f"\n === Instance {itx_ct}, SPP init-soln {sppinit_id} ===");print(" ")
            sssolver = tsmd.SlopeScalingSolver(network, {},{})
            flow_arc_sppinit, flowcom_arc_sppinit, spp_paths = sssolver.get_initial_shortest_path_sol()

            # slope scaling improving heuristics
            ss_output = solve_slope_scaling_improvement_from_sol(network, flow_arc_sppinit, flowcom_arc_sppinit, time_limit)

            # add to solution log
            name_id = sppinit_id
            (init_fa,init_fca) = (flow_arc_sppinit,flowcom_arc_sppinit); 
            (ss_obj,ss_fa,ss_fca,ss_iter,ss_rtime) = ss_output;
            
            # calculating obj
            it_cost, is_cost = sssolver.get_obj(init_fa)
            init_obj = it_cost + is_cost
            sst_cost, sss_cost = sssolver.get_obj(ss_fa)
            ss_obj = sst_cost + sss_cost
            
            # trailer util
            init_util_dist_avg = get_util_dist_average(init_fa,network.distance_matrix,network.trailer_cap)
            ss_util_dist_avg = get_util_dist_average(ss_fa,network.distance_matrix,network.trailer_cap)
            
            log["Init{}_obj".format(name_id)] = init_obj
            log["Init{}_truck".format(name_id)] = it_cost
            log["Init{}_sort".format(name_id)] = is_cost
            log["ss{}_obj".format(name_id)] = ss_obj
            log["ss{}_truck".format(name_id)] = sst_cost
            log["ss{}_sort".format(name_id)] = sss_cost

            log["ss{}_iter".format(name_id)] = ss_iter
            log["ss{}_rtime".format(name_id)] = ss_rtime
            
            log["Init{}_gap".format(name_id)] = None #round((init_obj-abm_obj)/abm_obj,5); 
            log["imp\%_ss{}".format(name_id)] = round((ss_obj-init_obj)/init_obj,5);
            log["imp\%_ss_truck{}".format(name_id)] = round((sst_cost-it_cost)/it_cost,5);
            log["imp\%_ss_sort{}".format(name_id)] = round((sss_cost-is_cost)/is_cost,5);
            
            log["Init{}_ud_avg".format(name_id)] = init_util_dist_avg; 
            log["ss{}_ud_avg".format(name_id)] = ss_util_dist_avg

        iter_log[itx_ct] = log
    return iter_log

def fixInstanceSCImprovingMGCPInitExperiment(inst_list, constant_dict,
                                                        iter_log = None, time_limit = 120): # seconds):
    HANDLING_COST = constant_dict["HANDLING_COST"]
    TRAILER_CAP = constant_dict["TRAILER_CAP"]
    SERVICE_TYPES = constant_dict["SERVICE_TYPES"]
    num_days = constant_dict['num_days']
    periods_per_day = constant_dict['periods_per_day']
    hub_capacity = constant_dict['hub_capacity']
    simplified_sort_label = constant_dict['simplified_sort_label']
    velocity = constant_dict['velocity']
    
    if (iter_log is None):
        iter_log = dict(); itx_ct = 0;
    else:
        itx_ct = len(iter_log)
        
    for inst in inst_list:
        itx_ct+=1 
        print(f'=== Iteration {itx_ct} ===')
        travel_hr_matrix = dict([(e,inst['distance_matrix'][e]/velocity) for e in inst['distance_matrix']])
        # generate time-expanded network from instance
        network = tsmd.TimeExpandedNetwork(inst, num_days, periods_per_day, hub_capacity, travel_hr_matrix, 
                                   simplified_sort_label,TRAILER_CAP, HANDLING_COST)
        network.remove_infeasible_demand()
        flatten_dem = [network.demand_by_fc[dc][a] for dc in network.demand_by_fc for a in network.demand_by_fc[dc]]
        # create log dict
        log = {lk:None for lk in constant_dict['log_keys']}
        
        # log statistic of instance
        log['nodes_no'] = len(network.nodes); log['arcs_no'] = len(network.edges);
        log['total_dem'] = sum(flatten_dem); log['trail_cap'] = TRAILER_CAP
        log['min_dem'] = min(flatten_dem); log['max_dem'] = max(flatten_dem); 
        
        # arc-based model...
        abm_obj = None
        
        sample_no = 2
        for i in range(sample_no):
            # init: random sequence of MGCP 
            mgcpinit_id = i + 1
            print(f"\n === Instance {itx_ct}, MGCP init-soln {mgcpinit_id} ===");print(" ")
            mgcpsolver = tsmd.MarginalCostPathSolver(network,{}, {})
            flow_arc_mgcpinit1,flowcom_arc_mgcpinit1 = mgcpsolver.mgcp_construction(plot_network = False, save_to_img = False)
            
            # slope scaling improving heuristics
            ss_output = solve_slope_scaling_improvement_from_sol(network, flow_arc_mgcpinit1, flowcom_arc_mgcpinit1, time_limit)

            # add to solution log
            name_id = mgcpinit_id
            (init_fa,init_fca) = (flow_arc_mgcpinit1,flowcom_arc_mgcpinit1); 
            (ss_obj,ss_fa,ss_fca,ss_iter,ss_rtime) = ss_output;
            
            # calculating obj            
            it_cost, is_cost = mgcpsolver.get_obj(init_fa)
            init_obj = it_cost + is_cost
            sst_cost, sss_cost = mgcpsolver.get_obj(ss_fa)
            ss_obj = sst_cost + sss_cost
            
            # trailer util
            init_util_dist_avg = get_util_dist_average(init_fa,network.distance_matrix,network.trailer_cap)
            ss_util_dist_avg = get_util_dist_average(ss_fa,network.distance_matrix,network.trailer_cap)
            
            log["Init{}_obj".format(name_id)] = init_obj
            log["Init{}_truck".format(name_id)] = it_cost
            log["Init{}_sort".format(name_id)] = is_cost
            log["ss{}_obj".format(name_id)] = ss_obj
            log["ss{}_truck".format(name_id)] = sst_cost
            log["ss{}_sort".format(name_id)] = sss_cost

            log["ss{}_iter".format(name_id)] = ss_iter
            log["ss{}_rtime".format(name_id)] = ss_rtime
            
            log["Init{}_gap".format(name_id)] = None #round((init_obj-abm_obj)/abm_obj,5); 
            log["imp\%_ss{}".format(name_id)] = round((ss_obj-init_obj)/init_obj,5);
            log["imp\%_ss_truck{}".format(name_id)] = round((sst_cost-it_cost)/it_cost,5);
            log["imp\%_ss_sort{}".format(name_id)] = round((sss_cost-is_cost)/is_cost,5);
            
            log["Init{}_ud_avg".format(name_id)] = init_util_dist_avg; 
            log["ss{}_ud_avg".format(name_id)] = ss_util_dist_avg

        iter_log[itx_ct] = log
    return iter_log



def fixInstanceImprovingOnMultiInitializationExperiment(inst_list, constant_dict,
                                                        iter_log = None, time_limit = 120): # seconds):
    HANDLING_COST = constant_dict["HANDLING_COST"]
    TRAILER_CAP = constant_dict["TRAILER_CAP"]
    SERVICE_TYPES = constant_dict["SERVICE_TYPES"]
    num_days = constant_dict['num_days']
    periods_per_day = constant_dict['periods_per_day']
    hub_capacity = constant_dict['hub_capacity']
    simplified_sort_label = constant_dict['simplified_sort_label']
    velocity = constant_dict['velocity']
    
    if (iter_log is None):
        iter_log = dict(); itx_ct = 0;
    else:
        itx_ct = len(iter_log)
        
    for inst in inst_list:
        itx_ct+=1 
        print(f'=== Iteration {itx_ct} ===')
        travel_hr_matrix = dict([(e,inst['distance_matrix'][e]/velocity) for e in inst['distance_matrix']])
        # generate time-expanded network from instance
        network = tsmd.TimeExpandedNetwork(inst, num_days, periods_per_day, hub_capacity, travel_hr_matrix, 
                                   simplified_sort_label,TRAILER_CAP, HANDLING_COST)
        network.remove_infeasible_demand()
        flatten_dem = [network.demand_by_fc[dc][a] for dc in network.demand_by_fc for a in network.demand_by_fc[dc]]
        # create log dict
        log = {lk:None for lk in constant_dict['log_keys']}
        
        # log statistic of instance
        log['nodes_no'] = len(network.nodes); log['arcs_no'] = len(network.edges);
        log['total_dem'] = sum(flatten_dem); log['trail_cap'] = TRAILER_CAP
        log['min_dem'] = min(flatten_dem); log['max_dem'] = max(flatten_dem); 
        
        # arc-based model...
        abm_obj = None
        
        sample_no = 2
        ct_id = 0
        for i in range(sample_no):
            # init: random sequence of MGCP 
            mgcpinit_id = ct_id
            print(f"\n === Instance {itx_ct}, MGCP init-soln {mgcpinit_id} ===");print(" ")
            mgcpsolver = tsmd.MarginalCostPathSolver(network,{}, {}, {})
            flow_arc_mgcpinit1,flowcom_arc_mgcpinit1 = mgcpsolver.mgcp_construction(plot_network = False, save_to_img = False)
            (mgcp1_sol,ss1_sol) = solveMultiHeuristicsFromSol(network, flow_arc_mgcpinit1,flowcom_arc_mgcpinit1,
                                                              mgcpsolver.alpha, time_limit)
            # add to solution log
            add_solution_stats_for_output(log,abm_obj,(flow_arc_mgcpinit1,flowcom_arc_mgcpinit1),mgcp1_sol,ss1_sol,network,str(mgcpinit_id))
            ct_id+=1

            # NEED ADJUSTMENT: NOT WORKING BECAUSE MGCP NEEDS IN_TREE CONSTRAINT
            # # init: first iter sc
            # sc1init_id = ct_id
            # print(f"\n === Instance {itx_ct}, SC init-sol {sc1init_id} ===");print(" ")
            # scsolver = tsmd.SlopeScalingSolver(network, {}, {})
            # min_sol, sc_log = scsolver.concurrent_slope_scalling_with_time_limit(time_limit = time_limit,iteration_limit=1, plot_slope=False)
            # flow_arc_scinit1,flowcom_arc_scinit1 = (min_sol['flow_arc'],min_sol['flowcom_arc'])
            # (mgcp1_sol,sc1_sol) = solveMultiHeuristicsFromSol(network, flow_arc_scinit1,flowcom_arc_scinit1, time_limit)
            # # add to solution log
            # add_solution_stats_for_output(log,abm_obj,(flow_arc_scinit1,flowcom_arc_scinit1),mgcp1_sol,sc1_sol,network,str(sc1init_id))
            # ct_id+=1

        # init: SPP iter
        sppinit_id = ct_id
        print(f"\n === Instance {itx_ct}, SPP init-sol {sppinit_id} ===");print(" ")
        sssolver = tsmd.SlopeScalingSolver(network, {}, {})
        spp_fa_init, spp_fca_init, _ = sssolver.get_initial_shortest_path_sol()
        (mgcp1_sol,ss1_sol) = solveMultiHeuristicsFromSol(network, spp_fa_init, spp_fca_init, 
                                                            sssolver.alpha, time_limit)
        # add to solution log
        add_solution_stats_for_output(log,abm_obj,(spp_fa_init,spp_fca_init),mgcp1_sol,ss1_sol,network,str(sppinit_id))

        iter_log[itx_ct] = log
    return iter_log

def solveMultiHeuristicsFromSol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit ):
    # mgcp improving heuristics
    mgcp_output = solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit)
    
    # slope scaling improving heuristics
    ss_output = solve_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit)
    return mgcp_output, ss_output

def solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit, lane_selection_mode = "vol",
                                     reflow_mode = "default", init_proc_text=""):
    # mgcp improving heuristics
    flow_arc_mgcp_imp = deepcopy(init_flow_arc)
    flow_comarc_mgcp_imp = deepcopy(init_flowcom_arc)
    mgcpsolver = tsmd.MarginalCostPathSolver(network, flow_arc_mgcp_imp, flow_comarc_mgcp_imp, init_alpha)
    # for plot labeling
    mgcpsolver.init_proc_text = init_proc_text
    mgcp_iter, mgcp_runtime = mgcpsolver.mgcp_single_lane_improvement_with_time_limit(time_limit ,
                                                                                      lane_selection_mode = lane_selection_mode,
                                                                                      reflow_mode = reflow_mode)
    it_cost,is_cost = mgcpsolver.get_obj(init_flow_arc)
    init_obj = it_cost+is_cost
    mgcpt_cost,mcgps_cost = mgcpsolver.get_obj(mgcpsolver.flow_arc)
    mgcp_obj = mgcpt_cost+mcgps_cost
    print(f'{reflow_mode} mcgp improving heuristic:{mgcp_obj}  (t{mgcpt_cost},s{mcgps_cost}), imp\%:{round(100*(mgcp_obj-init_obj)/(init_obj),2)}\%'+
          f' (t{round((mgcpt_cost-it_cost)*100/it_cost,2)}% , s{round((mcgps_cost-is_cost)*100/is_cost,2)}%)')
    # validate solution
    # mgcpsolver.validate_demand_delivered(flow_arc_mgcp_imp,flow_comarc_mgcp_imp)
    mgcpsolver.validate_demand_delivered(mgcpsolver.flow_arc,mgcpsolver.flowcom_arc)
    mgcp_output = (mgcp_obj,flow_arc_mgcp_imp,flow_comarc_mgcp_imp, mgcp_iter, mgcp_runtime) 
    return mgcp_output

def solve_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, init_proc_text=""):
    # slope scaling improving heuristics
    flow_arc_ass_imp = deepcopy(init_flow_arc)
    flow_comarc_ass_imp = deepcopy(init_flowcom_arc)
    sssolver = tsmd.SlopeScalingSolver(network, flow_arc_ass_imp, flow_comarc_ass_imp); 
    sssolver.init_proc_text = init_proc_text;
    min_sol, iter_log = sssolver.adaptive_slope_scalling_with_time_limit(time_limit=time_limit, plot_slope=True)
    flow_arc_ass_imp = min_sol['flow_arc']
    flow_comarc_ass_imp = min_sol['flowcom_arc']
    it_cost,is_cost = sssolver.get_obj(init_flow_arc);
    init_obj = it_cost+is_cost
    ss_obj = min_sol['obj'];
    ss_tcost = min_sol['tcost']; ss_scost = min_sol['scost']
    print('adaptive (a)ss improving heuristic:{}, imp\%:{}'.format(ss_obj, round(100*(ss_obj-init_obj)/(init_obj),2)) +
          f' (t{round((ss_tcost-it_cost)*100/it_cost,2)}% , s{round((ss_scost-is_cost)*100/is_cost,2)}%)')
    sssolver.validate_demand_duedate_satisfaction(min_sol['path_sol'])
    ss_output = (ss_obj,flow_arc_ass_imp,flow_comarc_ass_imp, len(iter_log), iter_log[len(iter_log)-2]['timestamp'])             
    return ss_output

def solve_adaptive_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, init_proc_text=""):
    # slope scaling improving heuristics
    flow_arc_ss_imp = deepcopy(init_flow_arc)
    flow_comarc_ss_imp = deepcopy(init_flowcom_arc)
    sssolver = tsmd.SlopeScalingSolver(network, flow_arc_ss_imp, flow_comarc_ss_imp); 
    sssolver.init_proc_text = init_proc_text;
    min_sol, iter_log = sssolver.concurrent_slope_scalling_with_time_limit(time_limit=time_limit, plot_slope=True)
    flow_arc_ss_imp = min_sol['flow_arc']
    flow_comarc_ss_imp = min_sol['flowcom_arc']
    it_cost,is_cost = sssolver.get_obj(init_flow_arc);
    init_obj = it_cost+is_cost
    ss_obj = min_sol['obj'];
    ss_tcost = min_sol['tcost']; ss_scost = min_sol['scost']
    print('ss improving heuristic:{}, imp\%:{}'.format(ss_obj, round(100*(ss_obj-init_obj)/(init_obj),2)) +
          f' (t{round((ss_tcost-it_cost)*100/it_cost,2)}% , s{round((ss_scost-is_cost)*100/is_cost,2)}%)')
    sssolver.validate_demand_duedate_satisfaction(min_sol['path_sol'])
    ss_output = (ss_obj,flow_arc_ss_imp,flow_comarc_ss_imp, len(iter_log), iter_log[len(iter_log)-2]['timestamp'])             
    return ss_output



def add_solution_stats_for_output(output,abm_obj,init_sol,mgcp_sol,ss_sol,network,name_id=""):
    (init_fa,init_fca) = init_sol; 
    (mgcp_obj,mgcp_fa,mgcp_fca,mgcp_iter,mgcp_rtime) = mgcp_sol;
    (ss_obj,ss_fa,ss_fca,ss_iter,ss_rtime) = ss_sol;
    
    # solver for cal obj
    mgcpsolver = tsmd.MarginalCostPathSolver(network, init_sol[0], init_sol[1],{})
    
    init_obj = mgcpsolver.get_obj(init_fa)
    mgcp_obj = mgcpsolver.get_obj(mgcp_fa)
    ss_obj = mgcpsolver.get_obj(ss_fa)
    
    # trailer util
    init_util_dist_avg = get_util_dist_average(init_fa,network.distance_matrix,network.trailer_cap)
    mgcp_util_dist_avg = get_util_dist_average(mgcp_fa,network.distance_matrix,network.trailer_cap)
    ss_util_dist_avg = get_util_dist_average(ss_fa,network.distance_matrix,network.trailer_cap)
    
    output["Init{}_obj".format(name_id)] = init_obj
    output["mgcp{}_obj".format(name_id)] = mgcp_obj
    output["ss{}_obj".format(name_id)] = ss_obj
    
    output["mgcp{}_iter".format(name_id)] = mgcp_iter
    output["ss{}_iter".format(name_id)] = ss_iter
    output["mgcp{}_rtime".format(name_id)] = mgcp_rtime
    output["ss{}_rtime".format(name_id)] = ss_rtime
    
    output["Init{}_gap".format(name_id)] = None#round((init_obj-abm_obj)/abm_obj,5); 
    output["imp\%_mgcp{}".format(name_id)] = round((mgcp_obj-init_obj)/init_obj,5); 
    output["imp\%_ss{}".format(name_id)] = round((ss_obj-init_obj)/init_obj,5);
    
    output["Init{}_ud_avg".format(name_id)] = init_util_dist_avg; 
    output["mgcp{}_ud_avg".format(name_id)] = mgcp_util_dist_avg; 
    output["ss{}_ud_avg".format(name_id)] = ss_util_dist_avg
def get_util_dist_average(flow_arc,distance_matrix,trailer_cap):
    total_dist_pos_arc = sum([distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
    total_util_dist = sum([(flow_arc[a]/(np.ceil(flow_arc[a]/trailer_cap)*trailer_cap))*distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
    return total_util_dist/total_dist_pos_arc

def generate_initial_solution(network, mode):
    '''return: flow_arc, flowcom_arc, alpha'''
    if (mode == "mgcp"):
        mgcpsolver = tsmd.MarginalCostPathSolver(network,{}, {}, {})
        flow_arc_mgcpinit1,flowcom_arc_mgcpinit1 = mgcpsolver.mgcp_construction(plot_network = False, save_to_img = False)
        it_cost,is_cost = mgcpsolver.get_obj(flow_arc_mgcpinit1)
        # init_obj = it_cost+is_cost
        return (it_cost,is_cost), flow_arc_mgcpinit1, flowcom_arc_mgcpinit1, mgcpsolver.alpha
    elif (mode == "sp"):
        sssolver = tsmd.SlopeScalingSolver(network, {}, {})
        spp_fa_init, spp_fca_init, _ = sssolver.get_initial_shortest_path_sol()
        it_cost,is_cost = sssolver.get_obj(spp_fa_init)
        # init_obj = it_cost+is_cost
        return (it_cost,is_cost), spp_fa_init, spp_fca_init, sssolver.alpha

def runimprovement(network, init_fa, init_fca, alpha, imp_proc, time_limit, init_proc_text = "" ):
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
        (lane_sel_mode,imp_proc) = _temp
    else:
        raise Exception(f"Invalid imp proc mode {imp_proc}")
    
    if (imp_proc == "mgcp"):
        # mgcp improving heuristics
        mgcp_output = solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit, 
                                                      lane_selection_mode = lane_sel_mode ,reflow_mode = "default", 
                                                      init_proc_text = init_proc_text)
        (mgcp_obj,flow_arc_mgcp_imp,flow_comarc_mgcp_imp, mgcp_iter, mgcp_runtime) = mgcp_output
        return (flow_arc_mgcp_imp,flow_comarc_mgcp_imp,mgcp_iter,mgcp_runtime)
    elif (imp_proc == "mgcp_w_grasp"):
        # grasp mgcp improving heuristics 
        mgcp_output = solve_mgcp_single_lane_from_sol(network, init_flow_arc, init_flowcom_arc, init_alpha, time_limit * 4,
                                                      lane_selection_mode = lane_sel_mode, reflow_mode = "grasp",
                                                      init_proc_text = init_proc_text)
        (mgcp_obj,flow_arc_mgcp_imp,flow_comarc_mgcp_imp, mgcp_iter, mgcp_runtime) = mgcp_output
        return (flow_arc_mgcp_imp,flow_comarc_mgcp_imp,mgcp_iter,mgcp_runtime)
    elif (imp_proc == "ssp"):
        # slope scaling improving heuristics: sc don't need alpha
        ss_output = solve_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, init_proc_text)
        (ss_obj,flow_arc_ss_imp,flow_comarc_ss_imp, iter_ss_imp, rtime_ss_imp) = ss_output           
        return (flow_arc_ss_imp,flow_comarc_ss_imp,iter_ss_imp,rtime_ss_imp)
    elif (imp_proc == "assp"):
        # slope scaling improving heuristics: sc don't need alpha
        ass_output = solve_adaptive_slope_scaling_improvement_from_sol(network, init_flow_arc, init_flowcom_arc, time_limit, init_proc_text)
        (ss_obj,flow_arc_ass_imp,flow_comarc_ass_imp, iter_ass_imp, rtime_ass_imp) = ass_output           
        return (flow_arc_ass_imp,flow_comarc_ass_imp,iter_ass_imp,rtime_ass_imp)
    else:
        raise Exception(f"invalid heuristic algo: {imp_proc}")

def _demand_scaler(network, scaler):
    x = 0;
    for fc in network.demand_by_fc:
        for a in network.demand_by_fc[fc]:
            x = network.demand_by_fc[fc][a]*scaler
            # x = 0.8*2000
            # x = 2000
            network.demand_by_fc[fc][a] = x
    print(f'Done fixing all demand to {x}');
    # print(f'Done scaling all demand by a factor of {scaler}');
    


def fixInstanceExperiment(inst_list, inst_id, constant_dict, initialization_list, imp_heuristics_list,
                          iter_log = None, 
                          time_limit = 120, 
                          demand_scaling_factor = 1): # seconds):
    HANDLING_COST = constant_dict["handling_cost"]
    TRAILER_CAP = constant_dict["trailer_cap"]
    num_days = constant_dict['num_days']
    periods_per_day = constant_dict['periods_per_day']
    hub_capacity = None
    simplified_sort_label = constant_dict['simplified_sort_label']
    velocity = constant_dict['velocity']

    if (iter_log is None):
        iter_log = dict(); itx_ct = 0;
    else:
        itx_ct = len(iter_log)
        
    for inst in inst_list:
        itx_ct+=1  
        travel_hr_matrix = dict([(e,inst['distance_matrix'][e]/velocity) for e in inst['distance_matrix']])
        # generate time-expanded network from instance
        network = tsmd.TimeExpandedNetwork(inst, num_days, periods_per_day, hub_capacity, travel_hr_matrix, 
                                   simplified_sort_label,TRAILER_CAP, HANDLING_COST)
        network.remove_infeasible_demand()
        _demand_scaler(network, demand_scaling_factor);
        flatten_dem = [network.demand_by_fc[dc][a] for dc in network.demand_by_fc for a in network.demand_by_fc[dc]]
        # create new log dict for each instance
        log = {}; _logger = logger(log)
        # log statistic of instance
        log['nodes_no'] = len(network.nodes); log['arcs_no'] = len(network.edges);
        log['total_dem'] = sum(flatten_dem); log['trail_cap'] = TRAILER_CAP
        log['min_dem'] = min(flatten_dem); log['max_dem'] = max(flatten_dem); 
        
        # initialize the feasible solution
        init_ct = 0
        for init_proc in initialization_list:
            init_ct+=1
            (it_cost, is_cost), init_fa, init_fca, alpha = generate_initial_solution(network, init_proc)
            # init_fa, init_fca, alpha = ({},{},{})
            _logger.log_initial_solution(network, init_fa, init_fca, init_proc, init_ct)

            imp_ct = 0
            # run improving heuristics
            for imp_proc in imp_heuristics_list:
                imp_ct+=1
                print(f"==={itx_ct}: init-proc {init_proc}-{init_ct}, imp-proc {imp_proc}-{imp_ct} ===");print(" ")
                (imp_fa, imp_fca, imp_iter, imp_rtime) = runimprovement(network, init_fa, init_fca, alpha, imp_proc, time_limit, init_proc_text = f"inst{inst_id}-init{init_proc}{init_ct}-imp{imp_proc}{imp_ct}" )
                # imp_fc, imp_fca = ({},{})
                _logger.log_improved_solution(network, imp_fa, imp_fca, imp_iter, imp_rtime, (it_cost, is_cost), f"init-{init_proc}-{init_ct}", f"imp-{imp_proc}-{imp_ct}")
        
        iter_log[itx_ct] = log
    return iter_log

class logger:
    def __init__(self, init_log):
        self.log = init_log
    
    def log_initial_solution(self, network, init_fa, init_fca, init_proc, init_id):
        # solver for cal obj
        __solver = tsmd.MarginalCostPathSolver(network, init_fa, init_fca,{})
        it_cost,is_cost = __solver.get_obj(init_fa)
        init_obj = it_cost+is_cost
        # trailer util
        init_util_dist_avg = self.get_util_dist_average(init_fa, network.distance_matrix, network.trailer_cap)
        
        self.log[f"init-{init_proc}-{init_id}_obj"] = init_obj
        self.log[f"init-{init_proc}-{init_id}_tcost"] = it_cost
        self.log[f"init-{init_proc}-{init_id}_scost"] = is_cost
        self.log[f"init-{init_proc}-{init_id}_ud_avg"] = init_util_dist_avg; 
    
    def log_improved_solution(self, network, imp_fa, imp_fca, imp_iter, imp_rtime, init_obj, init_proc, imp_proc):
        it_cost, is_cost = init_obj
        # solver for cal obj
        __solver = tsmd.MarginalCostPathSolver(network, imp_fa, imp_fca,{})
        imp_tcost,imp_scost = __solver.get_obj(imp_fa)
        imp_obj = imp_tcost+imp_scost
        # trailer util
        imp_util_dist_avg = self.get_util_dist_average(imp_fa, network.distance_matrix, network.trailer_cap)
        
        self.log[f"{init_proc}-{imp_proc}_obj"] = imp_obj
        self.log[f"{init_proc}-{imp_proc}_tcost"] = imp_tcost
        self.log[f"{init_proc}-{imp_proc}_scost"] = imp_scost

        self.log[f"{init_proc}-{imp_proc}_iter"] = imp_iter
        self.log[f"{init_proc}-{imp_proc}_rtime"] = imp_rtime
        self.log[f"{init_proc}-{imp_proc}_imp%"] = round((imp_obj-sum(init_obj))/sum(init_obj),5); 
        self.log[f"{init_proc}-{imp_proc}_t_imp%"] = round((imp_tcost-it_cost)/it_cost,5); 
        self.log[f"{init_proc}-{imp_proc}_s_imp%"] = round((imp_scost-is_cost)/is_cost,5); 
        
        self.log[f"{init_proc}-{imp_proc}_ud_avg"] = imp_util_dist_avg; 
        
    def get_util_dist_average(self, flow_arc, distance_matrix, trailer_cap):
        total_dist_pos_arc = sum([distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
        if total_dist_pos_arc < 1e-5 :
            return 0
        total_util_dist = sum([(flow_arc[a]/(np.ceil(flow_arc[a]/trailer_cap)*trailer_cap))*distance_matrix[(a[0][0],a[1][0])] for a in flow_arc if flow_arc[a]>1e-5])
        return total_util_dist/total_dist_pos_arc
    
