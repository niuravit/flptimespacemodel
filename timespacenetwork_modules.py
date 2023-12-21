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

import heapq

import folium
import geopandas as gpd
import geopy

from collections import defaultdict


COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

class TimeExpandedNetwork:
    def __init__(self, instance_obj,num_days,periods_per_day,hub_capacity,travel_time_matrix, 
                 simplified_sort_label,trailer_cap,handling_cost
                   ):
        self.buildings = instance_obj['nodes']
        self.num_days = num_days
        self.periods_per_day = periods_per_day
        self.hub_capacity = hub_capacity
        self.travel_time_matrix = travel_time_matrix
        self.simplified_sort_label = simplified_sort_label
        self.distance_matrix = instance_obj['distance_matrix']
        self.trailer_cap = trailer_cap
        self.handling_cost = handling_cost
        self.demand_by_fc = deepcopy(instance_obj['demand_by_fc'])
        self.desagg_commodities =  deepcopy(instance_obj['desagg_commodities'])
        # define sunrise as a preload with id 1
        self.des_sort = 1

        # Create nodes for each building, day, and period as tuples
        self.nodes = {(building, day, period): {'capacity': hub_capacity}
                      for building in self.buildings
                      for day in range(1, num_days + 1)
                      for period in range(1, periods_per_day + 1)}

        # Create arcs based on valid arrival times
        self.edges = {}
        self.outbound_edges = {}
        self.inbound_edges = {}
        for building in self.buildings:
            for day in range(1, num_days + 1):
                for period in range(1, periods_per_day+1):
                    from_node = (building, day, period)
                    self.outbound_edges[from_node] = set();
                    for next_building in self.buildings:
                        if building != next_building:
                            travel_time = int(self.travel_time_matrix[(building,next_building)])
                            to_node = self.get_next_node(building, day, period, next_building )
                            if (to_node[1]>num_days): continue
                            # use distance as a weight here
                            self.edges[(from_node, to_node)] = {'weight': self.distance_matrix[(building,next_building)]}
                            self.outbound_edges[from_node] = self.outbound_edges[from_node].union(set([(from_node, to_node)]))
                            # add subsequent sorts
                            for i in range(1,periods_per_day):
                                next_day = to_node[1] + np.floor((to_node[2]+i-1)/periods_per_day)
                                next_period = (to_node[2]+i)%periods_per_day
                                if next_period==0:next_period=4
                                if (next_day>num_days): continue
                                sub_to_node = (next_building,int(next_day),int(next_period))
                                # use distance as a weight here
                                self.edges[(from_node, sub_to_node)] = {'weight': self.distance_matrix[(building,next_building)]}
                                self.outbound_edges[from_node] = self.outbound_edges[from_node].union(set([(from_node, sub_to_node)]))
        
        self.nodes_df =  pd.DataFrame(self.nodes.keys())
        self.nodes_df['nodes'] =  self.nodes.keys()
        self.edges_df = pd.DataFrame(self.edges.keys())
        self.edges_df['edges'] = list(self.edges.keys())
        self.edges_df['uday'] = self.edges_df[0].apply(lambda x: x[1])
        self.edges_df['vday'] = self.edges_df[1].apply(lambda x: x[1])

        for n in self.nodes:
            inbound_arc = self.edges_df[self.edges_df[1]==n]['edges'].values
            if len(inbound_arc)>0 :
                self.inbound_edges[n] = inbound_arc
        
        # sets of edges for multi-phase SS
        self.update_phases_for_multi_phases()
        # self.EOLtoBBs = self.getEOLtoBBs()
        # self.BBtoEOLs = self.getBBtoEOLs()
        # self.BBtoBBs = self.getBBtoBBs()
        # self.EOLtoEOLs = self.getEOLtoEOLs()
    
    def update_phases_for_multi_phases(self):
        self.EOLtoBBs = self.getEOLtoBBs()
        self.BBtoEOLs = self.getBBtoEOLs()
        self.BBtoBBs = self.getBBtoBBs()
        self.EOLtoEOLs = self.getEOLtoEOLs()


    def get_nodes_edges_between_days(self, start_day, end_day):
        out_nodes = self.nodes_df.loc[(self.nodes_df[1] >= start_day) 
                                      & (self.nodes_df[1] <= end_day)]['nodes'].values
        out_edges = self.edges_df.loc[(self.edges_df['uday'] >= start_day) 
                                      & (self.edges_df['vday'] <= end_day)]['edges'].values
        return out_nodes, out_edges
    
    def get_next_node(self, f_building, f_day, f_period, t_building, ):
        travel_time = np.ceil(self.travel_time_matrix[(f_building,t_building)])
        # find the earliest arrival time 
        t_day,t_period = self.get_next_day_period(f_day,f_period,travel_time)
        to_node = (t_building, t_day, t_period)
        return to_node
        
    def get_next_day_period(self,f_day,f_period,travel_time):
        dep_time = self.simplified_sort_label[str(f_period)]['eoh']
        raw_arr_time = dep_time + travel_time # shift to 24 hr a day
        res_day = np.floor(raw_arr_time/24)
        res_hr = raw_arr_time%24 # shift back to 3-27 hr
        diff_from_sort_eoh = res_hr%24 - np.array([3, 11, 17, 23])
        min_diff = max(diff_from_sort_eoh[diff_from_sort_eoh<=0])
        res_period = np.where(diff_from_sort_eoh==min_diff)[0][0] + 1
        
        t_day = f_day + res_day
        t_period = res_period
        return int(t_day),int(t_period)
    
    def remove_infeasible_demand(self):
        _del_list = []
        for dc in self.demand_by_fc:
            for org_node in self.demand_by_fc[dc]:
                if (org_node[1] + get_day_from_dc(dc) > self.num_days):
                    _del_list.append((dc,org_node))
        del_val = 0
        for del_obj in _del_list:
            del_val+= self.demand_by_fc[del_obj[0]][del_obj[1]]
            del self.demand_by_fc[del_obj[0]][del_obj[1]]
        rem_dem = sum([self.demand_by_fc[dc][a] for dc in self.demand_by_fc for a in self.demand_by_fc[dc]])
        print(f"removed infeasible demands of value {del_val}, remaining demand {rem_dem}")
    
    # multi-phases SS
    def getEOLtoBBs(self):
        ts_edges = pd.DataFrame(self.edges.keys())
        ts_edges['org_sort'] = ts_edges[0].apply(lambda x: x[2])
        return ts_edges.loc[ts_edges['org_sort']==1][[0,1]].values

    def getBBtoEOLs(self):
        ts_edges = pd.DataFrame(self.edges.keys())
        ts_edges['des_sort'] = ts_edges[1].apply(lambda x: x[2])
        return ts_edges.loc[ts_edges['des_sort']==2][[0,1]].values

    def getBBtoBBs(self):
        ts_edges = pd.DataFrame(self.edges.keys())
        ts_edges['org_sort'] = ts_edges[0].apply(lambda x: x[2])
        ts_edges['des_sort'] = ts_edges[1].apply(lambda x: x[2])
        return ts_edges.loc[(ts_edges['des_sort']!=2) & (ts_edges['org_sort']!=1)][[0,1]].values

    def getEOLtoEOLs(self):
        ts_edges = pd.DataFrame(self.edges.keys())
        ts_edges['org_sort'] = ts_edges[0].apply(lambda x: x[2])
        ts_edges['des_sort'] = ts_edges[1].apply(lambda x: x[2])
        return ts_edges.loc[(ts_edges['des_sort']==2) & (ts_edges['org_sort']==1)][[0,1]].values


    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges
    
def get_obj(flow_arc, distance_matrix, trailer_cap, handling_cost, obj_mode, trailer_coeff = 1):
    ''' Global function for calculate the objective cost
    accept two modes: step or fix
    '''
    if (obj_mode == "step"):
        t_cost = 0
        s_cost = 0
        for a in flow_arc:
            if (flow_arc[a] > 1e-5):
                # positive flow, charges stepping cost by number of trailers
                t_cost += distance_matrix[_to_physical_arc(a)]*np.ceil(flow_arc[a]/trailer_cap)*trailer_coeff
                s_cost += handling_cost*flow_arc[a]
        return t_cost,s_cost
    elif (obj_mode == "fix"):
        t_cost = 0
        s_cost = 0
        for a in flow_arc:
            if (flow_arc[a] > 1e-5):
                # positive flow, charges single fix charge
                t_cost += distance_matrix[_to_physical_arc(a)]*trailer_coeff
                s_cost += handling_cost*flow_arc[a]
        return t_cost,s_cost
    else:
        raise Exception(f"Invalid obj mode {obj_mode}");
    
def _to_physical_arc(tsarc):
        return (tsarc[0][0],tsarc[1][0])

def get_day_from_dc(dc):
    dstr = dc[1]
    if (dstr == '1D'): return 1
    elif (dstr == '2D'): return 2
    elif (dstr == '3D'): return 3
    elif (dstr == 'GND'): return 7

def init_alpha_key(n, dc, alpha, init_val):
    '''init val is the latest due date (end of horizon)'''
    if (n not in alpha.keys()):
        alpha[n] = dict()
    if (dc not in alpha[n].keys()):
        alpha[n][dc] = init_val

def get_inout_positive_edges(inorout_edges_dict, n, dc, flowcom_arc):
    inorout_pos_arcs = []
    for a in inorout_edges_dict[n]:
        if (a in flowcom_arc[dc]):
            if (flowcom_arc[dc][a]>1e-5):
                inorout_pos_arcs.append(a)
    return inorout_pos_arcs

def update_alpha(n, dc, alpha, network, flowcom_arc):
    '''update alpha for flow adding'''
    latest_due = min(n[1] + get_day_from_dc(dc), network.num_days)
    inbound_pos_arcs = [a for a in flowcom_arc[dc] if (a[1]==n)&(flowcom_arc[dc][a]>1e-5)] 
    a_inbound_pos_arcs = []
    for a in inbound_pos_arcs:
        if (a[0] in alpha.keys()):
            if (dc in alpha[a[0]].keys()):
                a_inbound_pos_arcs.append(alpha[a[0]][dc])
    a_inbound_pos_arcs.append(latest_due)
    nalpha = min(a_inbound_pos_arcs)
    init_alpha_key(n, dc, alpha, np.inf)
    if (nalpha >= n[1]):
        # let alpha propagate, there is the case where change occurs in way down down downstream 
        alpha[n][dc] = nalpha
        outbound_pos_arcs = [a for a in flowcom_arc[dc] if (a[0]==n) & (flowcom_arc[dc][a]>1e-5)] # only positive flow arc
        for (_,nextn) in outbound_pos_arcs:
            update_alpha(nextn, dc, alpha, network, flowcom_arc)
    else:
        print(f'dc {dc}, inbound_pos_arcs {[alpha[a[0]][dc] for a in inbound_pos_arcs]}, nalpha {nalpha}, latest_due {latest_due}, n {n}')
        raise Exception("Some demand will be late")
        
def update_alpha_removal(n, dc, alpha, network, flowcom_arc):
    '''update alpha for flow removing'''
    inbound_pos_arcs = [a for a in flowcom_arc[dc] if (a[1]==n) & (flowcom_arc[dc][a]>1e-5)] 
    # reset alpha if no inbound flows of the same dc
    if len(inbound_pos_arcs) == 0 :
        if (n in network.demand_by_fc[dc].keys()):
            # has originating demand, reset of due of this demand
            alpha[n][dc] = n[1] + get_day_from_dc(dc)
        else: 
            if (n in  alpha.keys()):
                # no dc passing this n, del alpha
                if (dc in alpha[n].keys()):
                    del alpha[n][dc]
                # dc never ever passed this n, stop recurse
                else: return None
            else: return None
        outbound_pos_arcs = [a for a in flowcom_arc[dc] if (a[0]==n)]
        for (_,nextn) in outbound_pos_arcs:
            update_alpha_removal(nextn, dc, alpha, network, flowcom_arc)

def update_alpha_of_path(dc, path, alpha):
    '''update alpha for new added path'''
    first_node = path[0]
    due_date = first_node[1] + get_day_from_dc(dc)
    for nn in path:
        if (due_date >= nn[1]):
            init_alpha_key(nn,dc,alpha,due_date)
            if (alpha[nn][dc] > due_date):
                alpha[nn][dc] = due_date
        else:
            raise Exception(f"path {path} is not time feasible for {dc} with due-date {due_date}, curn {nn}")

def update_alpha_path_based_solution(path_sol, alpha):
    '''update alpha for path-based solution'''
    print('Updating alpha label for path-based solution...')
    for ((dc,o),v)in path_sol.items():
        # print(f'updating alpha of path {dc, v[1], alpha} ')
        update_alpha_of_path(dc, v[1], alpha)
    print('done.')
    return alpha

def convert_node_path_to_arc_path(npath):
    return [(npath[i],npath[i+1]) for i in range(len(npath)-1)]

class MultiHeuristicSolver:
    def __init__(self, network,flow_arc,flowcom_arc, alpha, obj_mode):
        self.network = network
        self.flow_arc = flow_arc
        self.flowcom_arc = flowcom_arc
        self.distance_matrix = network.distance_matrix
        self.trailer_cap = network.trailer_cap
        self.handling_cost = network.handling_cost
        self.deleted_demand = []
        self.alpha = alpha
        self.obj_mode = obj_mode
        self.logobjval = {}

        # some text for labeling the graph
        self.init_proc_text = ""

        # plot objs
        self.plot_objs = {}


class MarginalCostPathSolver:
    def __init__(self, network,flow_arc,flowcom_arc, alpha, obj_mode):
        self.network = network
        self.flow_arc = flow_arc
        self.flowcom_arc = flowcom_arc
        self.distance_matrix = network.distance_matrix
        self.trailer_cap = network.trailer_cap
        self.handling_cost = network.handling_cost
        self.deleted_demand = []
        self.alpha = alpha
        self.obj_mode = obj_mode
        self.logobjval = {}

        # some text for labeling the graph
        self.init_proc_text = ""

        # plot objs
        self.plot_objs = {}
    
    def add_flow_to_path(self, f, dc, path, flow_arc, flowcom_arc):
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            arc = (u,v)
            if (arc in flow_arc.keys()):
                flow_arc[arc] += f
            else:
                flow_arc[arc] = f
            if (dc not in flowcom_arc.keys()):
                flowcom_arc[dc] = dict()
                
            if (arc in flowcom_arc[dc].keys()):
                flowcom_arc[dc][arc] += f
            else:
                flowcom_arc[dc][arc] = f

    def initialize_alpha_from_demand(self):
        for dc in self.network.demand_by_fc:
            for org in self.network.demand_by_fc[dc]:
                due_date = org[1] + get_day_from_dc(dc)
                if (org not in self.alpha.keys()):
                    self.alpha[org] = dict()
                if (dc not in self.alpha[org].keys()):
                    self.alpha[org][dc] = due_date

    def mgcp_construction(self, flow_arc = None, flowcom_arc = None, plot_network = False, save_to_img = False):
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc

        s_time = time.time()
        cost = 0
        demand_by_fc = self.network.demand_by_fc
        # sorted by ascending order of total volume from all orgs
        sorted_dict = dict(sorted(demand_by_fc.items(), key=lambda item: sum(item[1].values())))
        dc_list = list(sorted_dict.keys())
        # random.shuffle(dc_list)
        
        i = 0 
        if save_to_img: 
            time_stp = datetime.now().strftime("%H%M-%b%d%y")
            folder_name = f"mgcp-contruction_{time_stp}"
            if not(os.path.exists(f"plots/{folder_name}")):
                os.mkdir(f"plots/{folder_name}")
        for fc in dc_list:
            d = fc[0]
            # sorted by ascending order of volume by org
            org_dict = dict(sorted(demand_by_fc[fc].items(), key=lambda item: item[1]))
            os_list = list(org_dict.keys())
            # random.shuffle(os_list)
            for o in os_list:
                i+=1
                f = org_dict[o]
                feasible_choices = self.marginal_cost_path(f,fc,o,(None,None,None),self.flow_arc,self.flowcom_arc,intree = True,mode = "C")
                if (len(feasible_choices) == 0): # no feasible path, delete this demand from origin
                    self.delete_fc(fc,o)
                    continue # skip to next
                # get cheapest choice
                sorted_feasible_choices = sorted(feasible_choices.items(), key=lambda x:x[1][1])
                (_, (path,label)) = sorted_feasible_choices[0]
                cost+=label
                self.add_flow_to_path(f,fc,path,flow_arc,flowcom_arc)
                # update alpha dc, n
                update_alpha(o, fc, self.alpha, self.network, flowcom_arc)
                
                if plot_network:
                    if save_to_img: 
                        fname = f"plots/{folder_name}/{i}-{fc}-{round(f)}-{o}"
                    else: fname = None
                    # plot newly added path
                    plot_base_network_with_paths(self.network,flow_arc,[(fc,tuple(path))], save_name = fname)
            
        print(f"Total acc cost label: {cost}")
        print(f"Obj-mode-{self.obj_mode} t_cost, s_cost: {get_obj(flow_arc, self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)}")
        print(f"Construction time: {time.time() - s_time}")
        return flow_arc,flowcom_arc
    
    def delete_fc(self, fc, org_node):
        print(f'del {fc} {org_node}')
        self.deleted_demand.append(((fc,org_node),self.network.demand_by_fc[fc][org_node]))
        del self.network.demand_by_fc[fc][org_node];
    
    def marginal_cost_path(self, flow_vol, dc, org_node, des_node, flow_arc = None, flowcom_arc = None, intree = False, mode = "C"): 
        '''mode: C - construction, starting from empty flow, I - improvement, starting from feasible solution'''
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc


        labels = {};
        labels[org_node] = 0
        predecessors = {}
        priority_queue = [(0, org_node)]
        feasible_choices = {}
        
        # only building is specified, need to check the due date of the service class
        if des_node == (None,None,None):
            fix_des_mode = False
            if (mode == "C"):
                # use only the originating demand to determine the due date
                due_day = org_node[1] + self._get_day_from_dc(dc)
            elif (mode == "I"):
                # use alpha at the tail of the first arc (origin) to get the earliest due date
                due_day = self.alpha[org_node][dc]
            due_period = 1 # destionation sort is always sunrise
            des_node = (dc[0],due_day,due_period)
        else: # no duedate is needed, destination ts-node is specified
            fix_des_mode = True

        # forcing intree
        if intree: 
            dcs_out_from_node = self._get_dcs_out_by_tsnode(flowcom_arc)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if (fix_des_mode):
                if current_node == des_node:
                    feasible_choices[(current_node)] = (self._reconstruct_path(predecessors, current_node), labels[current_node])
            else:
                # allow arriving early
                if (current_node[0] == des_node[0]) and (current_node[1] <= due_day) and (current_node[2] == 1):# sunrise sort
                    feasible_choices[(current_node)] = (self._reconstruct_path(predecessors, current_node), labels[current_node])
                
            for ub_edge in self.network.outbound_edges[current_node]:
                (u,v) = ub_edge

                # forcing intree
                if (intree): 
                    if not(self._check_intree_outflow(dc, ub_edge, dcs_out_from_node)): 
                        continue
                        
                # check if the arrival time satisfies the constraints
                if self._check_arrival_time(v,des_node):
                    if (ub_edge in flow_arc.keys()):
                        if (flow_arc[ub_edge]<1e-5):
                            oflow = 0
                            ocost = 0
                        else:
                            oflow = flow_arc[ub_edge]
                            ocost = self.__get_marginal_cost_on_edge(ub_edge, oflow)
                            # ocost = np.ceil(oflow/self.trailer_cap)*self.distance_matrix[(u[0],v[0])]+(self.handling_cost*oflow)
                    else:
                        oflow = 0
                        ocost = 0
                    new_flow = oflow + flow_vol
                    cost_af = self.__get_marginal_cost_on_edge(ub_edge, new_flow)
                    # cost_af = (np.ceil((new_flow)/self.trailer_cap)*self.distance_matrix[(u[0],v[0])])+(self.handling_cost*new_flow)
                    marginal_cost = cost_af - ocost
                    # if (marginal_cost<0):
                    #     print(f'edge {ub_edge}, oflow {oflow}, ocost {ocost} ')
                    #     print(f' flow added {flow_vol}, new_flow {new_flow}, new cost {cost_af}')
                    #     print(f'ocost: {ocost}, newcost {cost_af}, mgncost {marginal_cost}') 
                    
                    if (v in labels.keys()):
                        if (current_distance + marginal_cost < labels[v]): 
                            # update new minimum marginal cost
                            labels[v] = current_distance + marginal_cost
                            predecessors[v] = u
                            heapq.heappush(priority_queue, (labels[v], v))
                        else:
                            pass
                    else: # never reach to this before, so initialize the new label
                        labels[v] =  marginal_cost + current_distance 
                        predecessors[v] = u
                        heapq.heappush(priority_queue, (labels[v], v))

        return feasible_choices
    
    def __get_marginal_cost_on_edge(self, ts_edge, flow, trailer_coeff = 1 ):
        (u,v) = ts_edge
        flow = round(flow, 6)
        if (self.obj_mode == 'step'):
            return ((np.ceil((flow)/self.trailer_cap)*self.distance_matrix[(u[0],v[0])])*trailer_coeff) + (self.handling_cost*flow)
        elif (self.obj_mode == 'fix'):
            if (flow > 1e-6): return (self.distance_matrix[(u[0],v[0])]*trailer_coeff) + (self.handling_cost*flow)
            else: return 0
        else:
            raise Exception(f"Invalid obj mode {self.obj_mode}")
    
    def _check_arrival_time(self,next_node, destination_node):
        '''given next time space node and final time space destination node'''
        if (next_node[1] < destination_node[1]): return True #before the day
        elif (next_node[1] == destination_node[1]):
            if (next_node[2] <= destination_node[2]): return True
            else: return False
        else: return False
    
    def _check_arrive_ontime(self, ts_ord, des_b, dc, n_node):
        '''given next time space node, origin time space node, current destination and flow class'''
        trav_days = self._get_day_from_dc(dc)
        return (n_node[0]==des_b and n_node[1]<=(ts_ord[1]+trav_days) and n_node[2]==1)
        
    def _reconstruct_path(self, predecessors, destination):
        path = [destination]
        while destination in predecessors:
            destination = predecessors[destination]
            path.append(destination)
        return list(reversed(path))
    
    def _get_day_from_dc(self,dc):
        dstr = dc[1]
        if (dstr == '1D'): return 1
        elif (dstr == '2D'): return 2
        elif (dstr == '3D'): return 3
        elif (dstr == 'GND'): return 7
        
    def _get_dcs_out_by_tsnode(self, flowcom_arc):
        # get dc outflow from hub
        dc_out_from_tsnode = dict()
        for dc in flowcom_arc:
            for arc in flowcom_arc[dc]:
                if arc[0] not in dc_out_from_tsnode.keys():
                    dc_out_from_tsnode[arc[0]] = dict();
                if arc not in dc_out_from_tsnode[arc[0]].keys():
                    dc_out_from_tsnode[arc[0]][arc] = set()
                if(flowcom_arc[dc][arc]>1e-5):
                    dc_out_from_tsnode[arc[0]][arc].add(dc)
        return dc_out_from_tsnode
    
    def _check_intree_outflow(self, dc, narc, dc_out_from_tsnode):
        (u,v) = narc
        ult_dest = dc[0]
        arc_obey_intree = False

        dcs_in_ob = list()
        dcs_in_narc = list()

        if u not in dc_out_from_tsnode.keys():
            return True

        for arc in dc_out_from_tsnode[u]:
            for ob_dc in dc_out_from_tsnode[u][arc]:
                dcs_in_ob.append(ob_dc)
                if (arc==narc):dcs_in_narc.append(ob_dc)

        if (dc not in dcs_in_ob):
            # none dc appears in the outbound -> true
            arc_obey_intree = True
        else:
            # dc appears this _narc -> true
            arc_obey_intree = (dc in dcs_in_narc)
        return arc_obey_intree
    
    
    def mgcp_single_lane_improvement_with_time_limit(self, time_limit = 120, lane_selection_mode = 'vol', reflow_mode = "volume_based",
                                                     plot_obj = True):
        if lane_selection_mode == 'vol':
            sorted_target_arcs = sorted(self.flow_arc.items(),  key=lambda x:x[1], reverse=True)
        elif lane_selection_mode == 'org_vol':
            # sorted_target_arcs = sorted(self.flow_arc.items(),   key=lambda x:(x[0][0],x[1]), reverse=True)
            fa_df = pd.DataFrame(self.flow_arc, index = ['flow']).T
            fa_df.index.names = ['u','v']
            org_by_vol = fa_df.groupby('u').sum('flow').sort_values(by = 'flow', ascending = False)
            fa_df['sumorgflow'] = fa_df.apply(lambda x: org_by_vol.loc[[x.name[0]]]['flow'][0], axis = 1)
            sorted_target_arcs = fa_df.sort_values(by = ['sumorgflow','u', 'flow'], ascending = False)['flow'].to_dict().items()
        else:
            raise Exception(f'{lane_selection_mode} is invalid lane selection mode')

        start_timer = time.time()
        lane_num = len(sorted_target_arcs)
        print(f'MGCP single lane w timelimit {time_limit}, mode {lane_selection_mode}-{reflow_mode}, total #lanes: {lane_num}')
        hit_limit_in_for = False
        iteration = 0
        old_cost = 0                                                
        while ((time.time()-start_timer)< time_limit):
            for (a,_) in sorted_target_arcs:
                if ((time.time()-start_timer)> time_limit): 
                    hit_limit_in_for = True; break;
                if (self.flow_arc[a] < 1e-5):
                    print("Zero flow, skip!!")
                    continue
                iteration+=1
                # different modes  
                cost = self.mgcp_single_lane_improvement(a, reflow_mode)
                # if (reflow_mode == "default"):
                #     cost = self.mgcp_single_lane_improvement(a,)
                # elif (reflow_mode == "grasp"):
                #     cost = self.mgcp_single_lane_improvement_with_grasp(a,)
                # else:
                #     raise Exception(f"{reflow_mode} is invalid mode for mgcp single lane improvement")
                self.logobjval[iteration] = cost

            if (hit_limit_in_for): break;
            if (abs(old_cost-cost)<1e-5): print('Obj not change!'); break
            old_cost = cost
        runtime = (time.time()-start_timer)
        #### Plot stats
        if (plot_obj):
            self.plot_stat()
        return (iteration, runtime)   
    
    
    def plot_stat(self, ):
        iter_log = self.logobjval
        fig, ax = plt.subplots()
        random.seed(len(iter_log))
        line_color = random.choice(COLOR_PALETTE)  # Random hexadecimal color
        if "grasp" in self.init_proc_text:
            line_stl = "-"
            # m = 'x'
        else:
            line_stl = "--"
            # m = 'o'

        se_afmgcp_cost = pd.Series(iter_log.values())
        ax.plot(se_afmgcp_cost, color=line_color, linestyle = line_stl,label=f"{self.init_proc_text}-obj")
        ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel("iterations",  size = 20)
        ax.set_ylabel("obj", size = 20)     

        self.plot_objs[self.init_proc_text] = (fig, ax)

    
    def _arrange_dc_for_reflow(self, dc_flow_removed, mode="volume_based", num_replication = 4, shuffling_ele = 3):
        if (mode=="volume_based"):
            sorted_volume_based_dc = sorted(dc_flow_removed.items(), key=lambda x:x[1], reverse=True)
            return sorted_volume_based_dc
        elif (mode=="grasp"):
            # greedy randomized adaptive search procedure
            sorted_volume_based_dc = sorted(dc_flow_removed.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_volume_based_dc)==1:
                return sorted_volume_based_dc
            else:
                grasp_volume_based_dc = []
                if (len(sorted_volume_based_dc) < 10):
                    basket_size = min(3,len(sorted_volume_based_dc))
                else:
                    basket_size = int(0.3*len(sorted_volume_based_dc))
                basket = sorted_volume_based_dc[:basket_size]

                for i in range(len(sorted_volume_based_dc)):
                    sel_item_i = random.randint(0,len(basket)-1)
                    sel_item = basket.pop(sel_item_i)
                    grasp_volume_based_dc.append(sel_item)

                    if (i+basket_size < len(sorted_volume_based_dc)):
                        basket.append(sorted_volume_based_dc[i+basket_size]);
                # print(f'sorted_vol_dc: {sorted_volume_based_dc}')
                # print(f'grasp_vol_dc: {grasp_volume_based_dc}')
                return grasp_volume_based_dc

            '''
            old replication implementation
            # prevent exception when size of dc is small
            # t = min(shuffling_ele, len(sorted_volume_based_dc))
            # replications = []
            # for i in range(num_replication):
            #     replication = sorted_volume_based_dc.copy()
            #     # set different seed to get different result
            #     random.seed(i)
            #     # retrive and suffle the first t elements
            #     shuffled_part = replication[:t]
            #     random.shuffle(shuffled_part)
            #     # replace the first t elements with shuffled one
            #     replication[:t] = shuffled_part
            #     # append to the storage
            #     replications.append(replication)
            # return replications
            '''
        else:
            raise Exception("Invalid mode")

    def sequencial_dc_sequence_mgcp_submission(self, dc_sequence_collections, ots_node):
        repath_results = []
        for dc_seq in dc_sequence_collections:
            fa = deepcopy(self.flow_arc)
            fca = deepcopy(self.flowcom_arc)
            (cost_added, dc_mgcp_repath, fa, fca, dc_seq) = self.repath_dc_sequence_with_mgcp(dc_seq, ots_node,fa,fca)
            repath_results.append((cost_added, dc_mgcp_repath, fa, fca, dc_seq))
        return repath_results        

    def concurrent_dc_sequence_mgcp_submission(self, dc_sequence_collections, ots_node):
        # create input collection
        con_input_sets = [(dc_seq, ots_node, deepcopy(self.flow_arc), deepcopy(self.flowcom_arc)) for dc_seq in dc_sequence_collections] 

        # Create a concurrent executor
        with concurrent.futures.ThreadPoolExecutor(max_workers = 12) as executor:
            # Submit mgcp computations concurrently.
            futures = {executor.submit(self.repath_dc_sequence_with_mgcp, \
                                       inp_set[0],inp_set[1],inp_set[2],inp_set[3]): inp_set for inp_set in con_input_sets}
            # Collect the results as they become available.
            repath_results = []
            for future in concurrent.futures.as_completed(futures):
                cost_added, dc_mgcp_repath, fa, fca, dc_seq = future.result()
                repath_results += [(cost_added, dc_mgcp_repath, fa, fca, dc_seq)]
        return repath_results

    def repath_dc_sequence_with_mgcp(self, dc_sequence, ots_node, flow_arc, flowcom_arc):
        cost_added = 0
        # mgcp reflow dc back
        dc_mgcp_repath = dict()
        for (dc,f) in dc_sequence:    
            ult_d = dc[0]
            feasible_choices = self.marginal_cost_path(f,dc,ots_node,(None,None,None),flow_arc,flowcom_arc,intree = True,mode = "I")
            if (len(feasible_choices) == 0): # no feasible path, delete this demand from origin
                 raise Exception("Cannot put flow back, something wrong!")
            # get cheapest choice of all feasible paths
            (_, (path,label)) = sorted(feasible_choices.items(), key=lambda x:x[1][1])[0]
            cost_added+=label
            self.add_flow_to_path(f,dc,path,flow_arc,flowcom_arc)
            dc_mgcp_repath[dc] = path

        # print(f'repath dc seq: {[k[0] for k in dc_sequence]}, cost added: {cost_added} ')
        return (cost_added, dc_mgcp_repath, flow_arc, flowcom_arc, dc_sequence)

    def mgcp_single_lane_improvement(self, lane_arc, reflow_mode):
        (ots_node, dts_node) = lane_arc
        (ot_cost,os_cost) = get_obj(self.flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        o_cost = ot_cost + os_cost
        # return dict each dc and volume that needs to be reflowed
        dc_flow_removed, cost_removed, dc_removed_path = self.remove_flowlane_from_downstream(lane_arc)
        dc_sequence = self._arrange_dc_for_reflow(dc_flow_removed,mode=reflow_mode)
        # repath dc sequence with mgcp: we can use self.flow_arc and self.flowcom_arc directly
        (cost_added, dc_mgcp_repath, _t1, _t2, _t3) = self.repath_dc_sequence_with_mgcp(dc_sequence, ots_node, self.flow_arc, self.flowcom_arc)

        aft_cost,afs_cost = get_obj(self.flow_arc, self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        cost_af_repath = aft_cost + afs_cost
        print(f'Lane {lane_arc}, objmode {self.obj_mode}, cost-change: {round(cost_added+cost_removed,3)} (recheck cost-change: {round(cost_af_repath-o_cost,3)}) ({round((cost_added+cost_removed)*100/(o_cost),2)} %)')

        if (cost_added+cost_removed<=-1e-5):
            # update alpha
            for (dc,f) in dc_sequence: 
                # updating for old path (removed)
                # print(f'updating alpha for removed path... from {lane_arc[1]}') 
                update_alpha_removal(lane_arc[1], dc, self.alpha, self.network, self.flowcom_arc)
                # updating for new path
                path = dc_mgcp_repath[dc]
                first_org = path[1]
                # print(f'updating alpha for new path... from {first_org}')
                update_alpha(first_org, dc, self.alpha, self.network, self.flowcom_arc)
        else:
            print("Cost increased!, undo to original plan")
            for (dc,f) in dc_sequence: 
                remove_path = dc_mgcp_repath[dc]
                self.add_flow_to_path(-f,dc,remove_path,self.flow_arc,self.flowcom_arc)
                readd_path = dc_removed_path[dc]
                self.add_flow_to_path(f,dc,readd_path,self.flow_arc,self.flowcom_arc)
            aft_cost,afs_cost = get_obj(self.flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
            cost_af_repath = aft_cost + afs_cost

        return cost_af_repath
    
    def get_path_solution_by_od_commodity(self, flow_arc=None, flowcom_arc=None):
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        dummy_flowcom_arc = deepcopy(flowcom_arc)
        dummy_flow_arc = deepcopy(flow_arc)
        paths = dict()
        print("Constructing path-based solution...")
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                path = self.find_downstream_path(org_h,dc[0],dc,dummy_flowcom_arc)
                key = (dc, org_h, path[-1], flow)
                paths[key] = path
        return paths
    
    def remove_flowlane_from_downstream(self, lane_arc):
        (o_arc,d_arc) = lane_arc
        
        dcs_on_lane = dict()
        dcs_removed_path = dict()
        for dc in self.flowcom_arc:
            if (lane_arc in self.flowcom_arc[dc].keys()):
                if( self.flowcom_arc[dc][lane_arc]>1e-5 ):
                    dcs_on_lane[dc] =  self.flowcom_arc[dc][lane_arc]

        # trace path first and remove later
        bft_cost, bfs_cost = get_obj(self.flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        cost_before_removed = bft_cost + bfs_cost
        dcs = list(dcs_on_lane.keys())
        for dc in dcs:
            remove_f = dcs_on_lane[dc]
            path = self.trace_downstream_for_removing(lane_arc[0],lane_arc[1],dc)
            if path is None:
                # in tree violation, don't consider this dc
                del dcs_on_lane[dc]
                print(f'in tree violdation skip {dc}')
                continue
            else:
                # remove here
                self.add_flow_to_path(-remove_f,dc,path,self.flow_arc,self.flowcom_arc)
                dcs_removed_path[dc] = path
        # calculate cost after removing flow
        aft_cost, afs_cost = get_obj(self.flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        cost_af_removed = aft_cost+afs_cost
        cost_removed = cost_af_removed - cost_before_removed
        dc_flow_removed = dcs_on_lane
        return dc_flow_removed, cost_removed, dcs_removed_path
    
    def trace_downstream_for_removing(self, org, des, dc, flowcom_arc = None):
        '''only support intree solution'''
        if (flowcom_arc == None):
            flowcom_arc = self.flowcom_arc

        trav_days = self._get_day_from_dc(dc)
        cnode = org
        path = [cnode]
        # print('o-d:',org,des,'dc:',dc)
        dc_tree = flowcom_arc[dc]
        notarrived = True
        while (notarrived):
            ob_arcs = [a for a in dc_tree if (dc_tree[a]>1e-5)&(a[0]==cnode)]
            if (len(ob_arcs)>1):
                print(f'more than one outgoing arc available {dc} {ob_arcs}')
                path = None
                break
            (_,nloc) = ob_arcs[0]
            path = path + [nloc]
            cnode = nloc
            if self._check_arrive_ontime(org,dc[0],dc,nloc):
                notarrived = False
        return path

    def change_dclane_from_downstream(self, dc, f, arc, acc_cost = 0, path = [], 
                                      flow_arc=None, flowcom_arc=None):
        '''This module is for removing flow only, no validity guarantee in adding flow usage'''
        # if None, change will be made to the solver flow_arc and flowcom_arc directly
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
            
        (o_arc,d_arc) = arc
        # initialize path
        if (len(path)==0): path = [o_arc]
        
        # calculate cost before making change
        trailer_bf = np.ceil(flow_arc[arc]/self.trailer_cap)
        volume_bf = flow_arc[arc]
        phys_arc = _to_physical_arc(arc)
        cost_bf = self.distance_matrix[phys_arc]*trailer_bf +  self.handling_cost*volume_bf
        
        # add/remove flow: recommended to use for deleting only, adding behavior is not guarantee
        # when adding, it will trace the flow of same dc
        if (arc not in flowcom_arc[dc]): flowcom_arc[dc][arc] = 0;
        flowcom_arc[dc][arc] += f

        if (arc not in flow_arc): flow_arc[arc] = 0;
        flow_arc[arc] += f

        # calculate the cost after making change
        trailer_af = np.ceil(flow_arc[arc]/self.trailer_cap)
        volume_af = flow_arc[arc]
        cost_af = self.distance_matrix[phys_arc]*trailer_af + self.handling_cost*volume_af
        
        # accumulate the cost and path
        acc_cost += (cost_af-cost_bf)
        path += [d_arc]
        # check if flow arrive at it destination, arriving early than the deadline is allowed
        if not(self._check_arrive_ontime(o_arc,dc[0],dc,d_arc)):
            dc_arcs = pd.Series(flowcom_arc[dc])
            dc_arcs.index.names = ['u','v']
            # find next arc
            next_arcs = dc_arcs[(dc_arcs.index.get_level_values('u')==d_arc)
                               & (dc_arcs>1e-5)
                               & (abs(dc_arcs+f)>-1e-5)]
            next_arc = next_arcs.first_valid_index()
            if (next_arc is None):
                raise Exception("Cannot find next arc")
            # recursively remove flow from next arc (downstream) until reaching the final destination
            acc_cost, path = self.change_dclane_from_downstream(dc, f, next_arc, acc_cost, path, flow_arc, flowcom_arc)

        # Validate flow value
        self.validate_flow(flow_arc, flowcom_arc)   
        return acc_cost, path
    
    def validate_flow(self, flow_arc, flowcom_arc):
        ''' validate consistency between flow_arc and flowcom_arc, and verify if there exist negative flow arc'''
        flow_arc_conservation = [1e-5>abs(flow_arc[a]-sum([flowcom_arc[dc][a] for dc in flowcom_arc if (a in flowcom_arc[dc].keys())])) for a in flow_arc.keys()]
        if not all(flow_arc_conservation):
            raise Exception("Flow-conservation violated on arc")
        nonnegative_flow = [(flow_arc[a]>=-1e-5) for a in flow_arc]
        if not all(nonnegative_flow):
            print([(a,flow_arc[a]) for a in flow_arc if (flow_arc[a]<-1e-5)])
            raise Exception("Negative flow arc")

        nonnegative_flowcom = [(flowcom_arc[dc][a]>=-1e-5) for dc in flowcom_arc for a in flowcom_arc[dc]]
        if not all(nonnegative_flowcom):
            print([(dc,a,flowcom_arc[dc][a]) for dc in flowcom_arc for a in flowcom_arc[dc] if (flowcom_arc[dc][a]<1e-5)])
            raise Exception("Negative flowcom arc")
    
    def validate_demand_satisfaction(self, flow_arc=None, flowcom_arc=None):
        ''' (old version) validate total demand satisfaction (volume satisfaction only), 
            only support intree solution,
            find demand paths, delete them from their origins and check that the remaining flow must be zero
        '''
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        dummy_flowcom_arc = deepcopy(flowcom_arc)
        dummy_flow_arc = deepcopy(flow_arc)
        print("Validating demand satisfaction...")
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                print(f'finding ds path from {org_h},{dc}')
                path = self.find_downstream_path(org_h,dc[0],dc,dummy_flowcom_arc)
                print(f'removing dc{dc}, o{org_h}, flow:{flow} from path:{path}')
                _ = self.change_dclane_from_downstream(dc, -flow, (path[0],path[1]), 0, [],
                                                       dummy_flow_arc, dummy_flowcom_arc)
        if (sum(dummy_flow_arc.values())>1e-5):
            print(dummy_flow_arc.values())
            raise Exception("Violate demand satisfaction")
        
    def convert_arc_sol_to_path_sol(self, flowcom_arc=None):
        ''' 
            only support intree solution, if not, path sol cannot be reconstructed
        '''
        path_sol = {}
        print("Reconstructing path solution...")
        no_dc = len(self.network.demand_by_fc)
        dc_cnt = 0
        for dc in self.network.demand_by_fc:
            if (dc_cnt % int(0.1*no_dc) == 0):
                print(f'reconstructed paths done for {dc_cnt}/{no_dc} dcs')

            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                dco_key = (dc, org_h)
                # reconstruct path
                path = self.find_downstream_path(org_h,dc[0],dc,flowcom_arc)
                path_sol[dco_key] = (flow, path)
            dc_cnt+=1

        return path_sol
            
    def validate_intree_violation(self, flow_arc=None, flowcom_arc=None):
        ''' validate intree path for each flow class'''
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        for dc in flowcom_arc:
            for arc in flowcom_arc[dc]:
                se = pd.DataFrame(flowcom_arc[dc].keys(),index =range(len(flowcom_arc[dc])))
                se['flow'] = flowcom_arc[dc].values()
                outbound_pos_arc = se.loc[(se['flow']>1e-5) & (se[0]==arc[1])]
                count_outbound_pos_arc = outbound_pos_arc.shape[0]
                if count_outbound_pos_arc>1:
                    print(dc , outbound_pos_arc)
                    raise Exception("In-tree violation!")
                
    def validate_demand_delivered(self, flow_arc=None, flowcom_arc=None):
        ''' validate total demand satisfaction (volume satisfaction only)'''
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        total_demand = sum([self.network.demand_by_fc[dc][a] for dc in self.network.demand_by_fc for a in self.network.demand_by_fc[dc]])
        dem_delivered = 0
        nodes_df = self.network.nodes_df; edges_df = self.network.edges_df
        for des_sort in nodes_df.loc[nodes_df[2]==1]['nodes'].values:
            inbound_edges = edges_df.loc[edges_df[1]==des_sort]['edges']
            inbound_flow = sum([flow_arc[x] for x in inbound_edges if x in flow_arc])
            outbound_edges = edges_df.loc[edges_df[0]==des_sort]['edges']
            outbound_flow = sum([flow_arc[x] for x in outbound_edges if x in flow_arc])
            dem_delivered += (inbound_flow-outbound_flow)
        if abs(total_demand-dem_delivered)>1e-5:
            print(f'total demand: {total_demand}, dem_delivered: {dem_delivered}')
            raise Exception("Demand not satisfied!")
        else:
            print('All demands are delivered!')
            print(f'total demand: {total_demand}, dem_delivered: {dem_delivered}')

    def find_downstream_path(self, org, des, dc, flowcom_arc):
        trav_days = self._get_day_from_dc(dc)
        cnode = org
        path = [cnode]
        dc_tree = flowcom_arc[dc]
        notarrived = True
        while (notarrived):
            # assume in-tree holds and give first outbount arc
            ob_arcs = [a for a in dc_tree if (dc_tree[a]>1e-5)&(a[0]==cnode)]
            (_,nloc) = ob_arcs[0]
            path = path + [nloc]
            cnode = nloc
            if self._check_arrive_ontime(org,des,dc,nloc):
                notarrived = False
        return path

# Slope Scaling for time-expanded network
class SlopeScalingSolver:
    def __init__(self, network, flow_arc, flowcom_arc, obj_mode):
        self.network = network
        self.flow_arc = flow_arc
        self.flowcom_arc = flowcom_arc
        self.distance_matrix = network.distance_matrix
        self.trailer_cap = network.trailer_cap
        self.handling_cost = network.handling_cost
        self.deleted_demand = []
        self.rho = dict()
        self.timespacearcs = network.get_edges().keys()
        self.obj_mode = obj_mode
        
        # initialize rho with the base slope
        self.initialize_rho()
        self.tcost, self.scost = get_obj(flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        self.init_obj = self.tcost + self.scost
        self.alpha = dict()

        # some text for labeling the graph
        self.init_proc_text = ""

        # dict for plot object
        self.plot_objs = {}

        # dict store size of each edge group
        self.edge_set = {
            "P0": self.timespacearcs,
            "P1": self.network.EOLtoBBs,
            "P2": self.network.BBtoEOLs,
            "P3": self.network.BBtoBBs,
            "P4": self.network.EOLtoEOLs,
        } 
        
        
    def initialize_rho(self, init_rho = None):
        print('Initialize rho...')
        if (init_rho is None):
            if (self.obj_mode == 'step'):
                self.rho = dict([(a, (self.distance_matrix[_to_physical_arc(a)]/self.trailer_cap) + self.handling_cost) for a in self.network.get_edges().keys()])
            elif (self.obj_mode == 'fix'):
                self.rho = dict([(a, (self.distance_matrix[_to_physical_arc(a)]) + self.handling_cost) for a in self.network.get_edges().keys()])
            # self.rho = dict([(a, (self.distance_matrix[self._to_physical_arc(a)]/self.trailer_cap)) for a in self.network.get_edges().keys()])
        else:
            self.rho = deepcopy(init_rho)
        
    # def update_rho(self, flow_arc = None, flowcom_arc = None ):
    #     self.rho = self.get_linearized_slope(self.rho, flow_arc, flowcom_arc)
    
    def get_linearized_slope(self, prev_rho, flow_arc = None, flowcom_arc = None ):
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        rho = dict()
        
        for a in self.timespacearcs:
            if (a in flow_arc.keys() and flow_arc[a]>1e-5):
                phys_a = _to_physical_arc(a)
                if (self.obj_mode == 'step'):
                    # linearized step charge 
                    rho[a] = ((self.distance_matrix[phys_a]*np.ceil(flow_arc[a]/self.trailer_cap))/flow_arc[a]) + self.handling_cost
                elif (self.obj_mode == 'fix'):
                    # linearized fix charge
                    rho[a] = (self.distance_matrix[phys_a]/flow_arc[a]) + self.handling_cost

            else:
                rho[a] = prev_rho[a]
            # if (abs(rho[a] - prev_rho[a]) > 1e-5):
            #     print(a,rho[a], prev_rho[a],(self.distance_matrix[phys_a]/self.trailer_cap) + self.handling_cost , flow_arc[a])
        return rho
        
    def get_time_space_des_agg_feasible_orgs(self,):
        '''grouping all possible origin demands that can arrive to each des, only need demand data don't need current flow'''
        des_agg_orgs = dict()
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                for feasible_arr_day in range(org_h[1]+1, org_h[1] + self._get_day_from_dc(dc)+1):
                    if (feasible_arr_day <= self.network.num_days):
                        ts_des = (dc[0],feasible_arr_day,1) 
                        if (ts_des in des_agg_orgs):
                            des_agg_orgs[ts_des].append((dc, org_h, flow))
                        else:
                            des_agg_orgs[ts_des] = [(dc, org_h, flow)]
        return des_agg_orgs
    
    def get_time_space_des_exact_day_agg_feasible_orgs(self,):
        '''grouping all possible origin demands that can arrive to each des EXACT day'''
        des_agg_orgs = dict()
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                feasible_arr_day = org_h[1] + self._get_day_from_dc(dc)
                if (feasible_arr_day <= self.network.num_days):
                    ts_des = (dc[0],feasible_arr_day,1) 
                    if (ts_des in des_agg_orgs):
                        des_agg_orgs[ts_des].append((dc, org_h, flow))
                    else:
                        des_agg_orgs[ts_des] = [(dc, org_h, flow)]                    
        return des_agg_orgs
        
    def get_sol_diff(self, flowcom_arc_new, flowcom_arc_prev):
        '''computing fro-norm between two solution'''
        sol_diff = 0 
        for fc in self.network.demand_by_fc:
            # some demand might be deleted bc of infeasibility
            if len(self.network.demand_by_fc[fc])>0:
                for a in self.timespacearcs:
                    if (a not in flowcom_arc_new[fc]): f_fc_new = 0;
                    else: f_fc_new = flowcom_arc_new[fc][a]
                    if (a not in flowcom_arc_prev[fc]): f_fc_old = 0;
                    else: f_fc_old = flowcom_arc_prev[fc][a]
                    sol_diff+=abs(f_fc_old-f_fc_new)
        return sol_diff
    
    def get_initial_shortest_path_sol(self, fixing_intree = True, arrival_forcing = "earliest"):
        '''heuristic construction using shortest path separated by flow class'''
        # des_org_com = self.get_time_space_des_exact_day_agg_feasible_orgs()
        if (arrival_forcing == "earliest"):
            des_agg_orgs = self.get_time_space_des_agg_feasible_orgs()
        elif (arrival_forcing == "exactdate"):
            des_agg_orgs = self.get_time_space_des_exact_day_agg_feasible_orgs()
        else:
            raise Exception(f"Invalid arrival forcing argument {arrival_forcing}")
        spp_input_list = []
        # construct input set for the spp solver: multiple ts-orgs -> single ts-des
        for (des,orgs) in des_agg_orgs.items():
            sday = min([o[1] for d,o,f in orgs])
            eday = des[1]
            sub_nodes,sub_edges = self.network.get_nodes_edges_between_days(sday,eday)
            input_set = (sub_nodes,sub_edges,des,orgs)
            spp_input_list.append(input_set)

        # solving shortest path for each commodity
        flow_arc_new = dict()
        flowcom_arc_new = dict()
        path_solution = dict()
        feasible_path = dict()

        # make rho equal the distance matrix     
        self.initialize_rho()
        start_t = time.time()

        # solving sc concurrently
        shortest_path_trees = self.concurrent_shortest_path_submission(spp_input_list)

        # Sequentially update the network instance based on the computed shortest paths
        # construct the dict that store only the best path 
        feasible_path = dict()
        for result_item in shortest_path_trees:
            (paths,label) = result_item
            for (dcof,path) in paths:
                if (path is not None):
                    if dcof in feasible_path.keys():
                        feasible_path[dcof].append((path,label[dcof[1]]))
                    else:
                        feasible_path[dcof] = [(path,label[dcof[1]])]
                # else: 
                #     # does not exist path that can deliver on time, del demand
                #     self.delete_fc(dcof[0],dcof[1])
        # check now and delete fc that has no feasible path
        for result_item in shortest_path_trees:
            (paths,label) = result_item
            for (dcof,path) in paths:
                if not(dcof in feasible_path):
                    self.delete_fc(dcof[0],dcof[1])
                    
        # update the flow
        for dcof in feasible_path:
            sorted_list = sorted(feasible_path[dcof], key=lambda x:(x[1],x[0][-1][1]))
            path,l = sorted_list[0]
            # shortcut if it pass through (arrive!) at des, sort-1 early
            path = self.__shortcutting_path(dcof[0],path)
            self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc_new, flowcom_arc_new)
            path_solution[(dcof[0],dcof[1])] = (dcof[2],path)

        print(f"finished generated initial spp solution: {time.time()-start_t},  pathed {len(spp_input_list)} o-d commodities")
        self.validate_demand_delivered(flow_arc_new,flowcom_arc_new)
        self.validate_demand_duedate_satisfaction(path_solution)

        if (fixing_intree):
            print(f'start intree-fixing procedure...')
            self.alpha = self.intree_fixing_path_based_solution(path_solution,flow_arc_new,flowcom_arc_new)
    
        return flow_arc_new, flowcom_arc_new, path_solution
    
    def __shortcutting_path(self,dc,path):
        ''' trim the remaining visits after it reaches destination at des-sort (1-sunrise)'''
        new_path = []
        for n in path:
            if (n[0] != dc[0]) or (n[2] != self.network.des_sort):
                new_path.append(n)
            else:
                new_path.append(n)
                break
        return new_path
        
    def slope_scaling_with_time_limit(self, time_limit = 120, plot_slope = False):
        ''' run sequencial slope scaling with time limit (one dc at a time)'''
        TIMELIMIT_MET = False; TOLERANCE_MET = False
        print(f'Slope scaling w timelimit {time_limit}')
        iter_ct = 1
        iteration_log = dict()
        iteration_log[0] = {
                        'flow_arc': dict(),
                        'flowcom_arc': dict(),
                        'rho_arc':self.rho,
                        'obj': self.init_obj,
                        'tcost': self.tcost,
                        'scost': self.scost
                       }
        # init the slope (cost) on each edge
        self.update_rho()
        
        # storage of the best solution
        min_sol_dict = {'obj': np.inf}
        
        # get o-d pair agg by des from current solution
        t_s = time.time()
        spp_input_list = []
        des_agg_orgs = self.get_time_space_des_agg_feasible_orgs()
        for (des,orgs) in des_agg_orgs.items():
            sday = min([o[1] for d,o,f in orgs])
            eday = des[1]
            sub_nodes,sub_edges = self.network.get_nodes_edges_between_days(sday,eday)
            input_set = (sub_nodes,sub_edges,des,orgs)
            spp_input_list.append(input_set)
        # print(f'constructing batch inputs:{time.time()-t_s}')
        
        start_timer = time.time()
        while not (TIMELIMIT_MET or TOLERANCE_MET):
            flow_arc_new = dict()
            flowcom_arc_new = dict()
            path_solution = dict()
            
            t_s = time.time()
            feasible_path = dict()
            for spp in spp_input_list:
                shortest_path_tree, s_labels = dijkstra_shortest_path_tree(spp[0],spp[1],self.rho,spp[2],spp[3])
                # construct the dict that store all feasible paths
                for dcof,path in shortest_path_tree.items():
                    if path is not None:
                        if dcof in feasible_path.keys():
                            feasible_path[dcof].append((path,s_labels[dcof[1]]))
                        else:
                            feasible_path[dcof] = [(path,s_labels[dcof[1]])]

            # update the flow with the best minimum path
            t_s = time.time()         
            for dcof in feasible_path:
                sorted_list = sorted(feasible_path[dcof], key=lambda x:(x[1],-x[0][-1][1]))
                path,l = sorted_list[0]
                self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc_new, flowcom_arc_new)
                path_solution[(dcof[0],dcof[1])] = (dcof[2],path)
            # print(f'add results to paths:{time.time()-t_s}')

            tcost,scost = get_obj(flow_arc_new,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode); 
            cost_new = tcost+scost
            cost_old = iteration_log[iter_ct-1]['obj']
            approx_cost = self.get_approx_obj(flow_arc_new,self.rho)
            
            iteration_log[iter_ct] = dict()
            iteration_log[iter_ct]['flow_arc'] = flow_arc_new.copy()
            iteration_log[iter_ct]['flowcom_arc'] = flowcom_arc_new.copy()
            iteration_log[iter_ct]['rho_arc'] = self.rho.copy()
            iteration_log[iter_ct]['path_sol'] = path_solution.copy()
            iteration_log[iter_ct]['obj'] = cost_new
            iteration_log[iter_ct]['tcost'] = tcost
            iteration_log[iter_ct]['scost'] = scost
            iteration_log[iter_ct]['appcost'] = approx_cost
            iteration_log[iter_ct]['timestamp'] = time.time() - start_timer
            
            # update the slop for next iteration
            self.update_rho(flow_arc_new, flowcom_arc_new)

            if (min_sol_dict['obj']>cost_new):
                print('Improved solution found, saving...')
                min_sol_dict = iteration_log[iter_ct].copy()
                
            # update terminating condition
            sol_diff = np.inf
            if (iter_ct-1>0):
                sol_diff = self.get_sol_diff(flowcom_arc_new,iteration_log[iter_ct-1]['flowcom_arc'])

            iter_ct+=1

            if (sol_diff <= 1e-5): TOLERANCE_MET = True
            if ((time.time()-start_timer)> time_limit): TIMELIMIT_MET = True

            print("SC-Iteration {}: {} {}, change {}".format(iter_ct,TOLERANCE_MET,TIMELIMIT_MET,cost_new-cost_old))
            

        #### Plot stats
        if (plot_slope):
            self.plot_stat(iteration_log)
        return min_sol_dict, iteration_log
    
    def plot_stat(self, iter_log):
        # for bigger size problem, rho plot doesn't tell anything....
        # plt.figure(1)
        # for i in range(0,len(iter_log)-1):
        #     ser1 = pd.Series(iter_log[i]['rho_arc'])
        #     ser1.plot(x="arcs", y="rho")
        #     plt.xlabel("arcs",  size = 20)
        #     plt.ylabel("rho", size = 20)
        # plt.legend(['Iter{}'.format(i) for i in range(0,len(iter_log)-1)])

        fig, ax = plt.subplots()
        line_color1 = random.choice(COLOR_PALETTE)  # Random hexadecimal color
        line_color2 = random.choice(COLOR_PALETTE)  # Random hexadecimal color

        # separate data by phase
        roundup_cost_series = []; roundup_cost_basket = []
        approx_cost_series = []; approx_cost_basket = []
        phases_series = ['P0']
        curr_phase = ''
        prev_phase = iter_log[0]['phase']
        for i in range(0,len(iter_log)):
            curr_phase = iter_log[i]['phase']
            roundupcost = iter_log[i]['obj']
            approxcost = iter_log[i]['appcost']
            if (curr_phase!=prev_phase):
                phases_series.append(curr_phase)
                roundup_cost_series.append(roundup_cost_basket)
                approx_cost_series.append(approx_cost_basket)
                # reset the basket
                roundup_cost_basket = [roundupcost]
                approx_cost_basket = [approxcost]
            else:
                roundup_cost_basket.append(roundupcost)
                approx_cost_basket.append(approxcost)
            prev_phase = curr_phase
            # print(curr_phase,phases_series)
            # print(roundup_cost_basket,roundup_cost_series)
        roundup_cost_series.append(roundup_cost_basket)
        approx_cost_series.append(approx_cost_basket)

        begin_x = 0
        end_x = 0
        for i in range(len(phases_series)):
            end_x += len(roundup_cost_series[i])
            if i % 2 == 0:
                linesty = '-'
            else:
                linesty = '--'
            # print(phases_series[i],range(begin_x,end_x),roundup_cost_series[i],linesty)
            if (i == len(phases_series)-1):
                x_range = [j for j in range(begin_x,end_x)]
                y_roundup_range = roundup_cost_series[i]
                y_app_range = approx_cost_series[i]
            else:
                x_range = [j for j in range(begin_x,end_x+1)]
                y_roundup_range = roundup_cost_series[i]+[roundup_cost_series[i+1][0]]
                y_app_range = approx_cost_series[i]+[approx_cost_series[i+1][0]]
            
            ax.plot(x_range,y_roundup_range, color=line_color1,linestyle=linesty, label=f"{self.init_proc_text}-roundup-obj-{phases_series[i]}")
            ax.plot(x_range,y_app_range, color=line_color2,linestyle=linesty, label=f"{self.init_proc_text}-approx-obj-{phases_series[i]}")
            begin_x = end_x

        ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel("iterations",  size = 20)
        ax.set_ylabel("obj", size = 20)

        self.plot_objs[self.init_proc_text] = (fig, ax)

    def update_rho(self, updating_edges, flow_arc = None, flowcom_arc = None, phase = "P0",release_all_edges_flag=False):
        start_time = time.time()
        # calculate all rhos
        new_rho = self.get_linearized_slope(self.rho, flow_arc, flowcom_arc)
        size_of_new_arcs = 0
        # specify edges
        if (phase == "P0"):
            # update rho for all edges
            updating_edges = self.timespacearcs
            self.rho = new_rho
        elif (phase != "P4"):
            if phase == "P1":
                # get EOL -> BBs edges
                new_edges = pd.DataFrame(self.network.EOLtoBBs)
                size_of_new_arcs = len(new_edges)
            elif phase == "P2":
                # get BB -> EOLs edges
                new_edges = pd.DataFrame(self.network.BBtoEOLs)
                size_of_new_arcs = len(new_edges)
            elif phase == "P3":
                # get BB -> BB edges
                new_edges = pd.DataFrame(self.network.BBtoBBs)
                size_of_new_arcs = len(new_edges)
            else:
                raise Exception(f"Invalid phase {phase}")
            # update edge and rho
            updating_edges_set = set(updating_edges)
            # Identify rows to keep using vectorized operations
            mask = new_edges.set_index([0, 1]).index.isin(updating_edges_set)
            # Filter the DataFrame based on the mask: focus on not previously in the group
            new_edges = new_edges[~mask]
            # Create a dictionary to store the maximum values for each group
            max_values = {}
            if not(release_all_edges_flag):
                # Iterate over groups and find the maximum link: groupby 0(org)
                for k1, group_df in new_edges.groupby(0):
                    # distance as a key
                    max_key = max(group_df[1], key=lambda k: self.distance_matrix[_to_physical_arc((k1, k))])
                    # new rho as a key
                    # max_key = max(group_df[1], key=lambda k: new_rho[(k1, k)])
                    max_values[k1] = (k1, max_key)
                # Update the updating_edges list with maximum links
                updating_edges.extend(max_values.values())
            else:
                # add all remaining edges
                updating_edges = updating_edges + [tuple(lst) for lst in new_edges.values]
                release_all_edges_flag = False
            
            # Bulk update approach
            updating_dict = dict.fromkeys(updating_edges)
            for key in updating_dict:
                self.rho[key] = new_rho[key]

        elif (phase == "P4"):
            # just check if eol-eol edge ever added to the updating set
            new_edges_array = self.network.EOLtoEOLs
            size_of_new_arcs = len(new_edges_array)
            if (tuple(new_edges_array[0]) not in updating_edges):
                # add all edges simulteneously
                updating_edges = updating_edges + [tuple(lst) for lst in new_edges_array]
            # Bulk update approach
            updating_dict = dict.fromkeys(updating_edges)
            for key in updating_dict:
                self.rho[key] = new_rho[key]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time {phase}: {elapsed_time} seconds, size of edges {size_of_new_arcs}")
        return updating_edges,release_all_edges_flag

    def concurrent_slope_scalling_multiphase_with_time_limit(self, time_limit = 120, iteration_limit = np.inf, phases = ['P0'] , plot_slope = False):
        ''' run concurrent slope scaling with time limit (one dc at a time)'''
        TIMELIMIT_MET = False; TOLERANCE_MET = False; ITERATION_MET = False;
        print(f'(Con)Slope scaling w timelimit {time_limit},  iterlimit {iteration_limit}, phases {phases}')
        
        # init the slope (cost) on each edge
        self.update_rho([])

        iter_ct = 1
        iteration_log = dict()
        iteration_log[0] = {
                        'flow_arc': self.flow_arc,
                        'flowcom_arc': self.flowcom_arc,
                        'rho_arc':self.rho,
                        'obj': self.init_obj,
                        'tcost': self.tcost,
                        'scost': self.scost,
                        'appcost': self.get_approx_obj(self.flow_arc,self.rho),
                        'phase':"P0"
                    }

        # storage of the best solution
        min_sol_dict = {'obj': np.inf}

        # get o-d pair agg by des from current solution
        t_s = time.time()
        spp_input_list = []
        des_agg_orgs = self.get_time_space_des_agg_feasible_orgs()
        for (des,orgs) in des_agg_orgs.items():
            sday = min([o[1] for d,o,f in orgs])
            eday = des[1]
            sub_nodes,sub_edges = self.network.get_nodes_edges_between_days(sday,eday)
            input_set = (sub_nodes,sub_edges,des,orgs)
            spp_input_list.append(input_set)
        # print(f'constructing batch inputs:{time.time()-t_s}')
        
        # multi-phase configuration
        phase_tol_flg = {p:False for p in phases}
        phase_tol_icnt = {p:0 for p in phases}
        processed_phases = []
        # get initial phase to work on
        curr_phase = phases.pop(0)
        phase_cnt = 1
        processed_phases.append(curr_phase)
        updating_edges = []
        cur_total_edge_size = len(self.edge_set[curr_phase])
        release_all_edges_flag = False
        
        start_timer = time.time()
        while not (TIMELIMIT_MET or TOLERANCE_MET or ITERATION_MET):
            flow_arc_new = dict()
            flowcom_arc_new = dict()
            path_solution = dict()
            # solving sc concurrently
            shortest_path_trees = self.concurrent_shortest_path_submission(spp_input_list)
            # print(f'solving shortest paths concurrently:{time.time()-t_s}')
            t_s = time.time()
            # Sequentially update the network instance based on the computed shortest paths.
            # construct the dict that store only the best path 
            feasible_path = defaultdict(list)
            for paths, label in shortest_path_trees:
                for dcof, path in paths:
                    if path is not None:
                        feasible_path[dcof].append((path, label[dcof[1]]))
            # for result_item in shortest_path_trees:
            #     (paths,label) = result_item
            #     for (dcof,path) in paths:
            #         if path is not None:
            #             if dcof in feasible_path.keys():
            #                 feasible_path[dcof].append((path,label[dcof[1]]))
            #             else:
            #                 feasible_path[dcof] = [(path,label[dcof[1]])]

            # update the flow
            for dcof, path_list in feasible_path.items():
                # Sort the path_list based on the specified key
                sorted_list = sorted(path_list, key=lambda x: (x[1], x[0][-1][1]))
                # print(sorted_list)
                path,l = sorted_list[0]
                self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc_new, flowcom_arc_new)
                path_solution[(dcof[0],dcof[1])] = (dcof[2],path)

            tcost,scost = get_obj(flow_arc_new,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode); 
            cost_new = tcost+scost 
            cost_old = iteration_log[iter_ct-1]['obj']
            approx_cost = self.get_approx_obj(flow_arc_new,self.rho)

            iteration_log[iter_ct] = dict()
            iteration_log[iter_ct]['flow_arc'] = flow_arc_new.copy()
            iteration_log[iter_ct]['flowcom_arc'] = flowcom_arc_new.copy()
            iteration_log[iter_ct]['rho_arc'] = self.rho.copy()
            iteration_log[iter_ct]['path_sol'] = path_solution.copy()
            iteration_log[iter_ct]['obj'] = cost_new
            iteration_log[iter_ct]['tcost'] = tcost
            iteration_log[iter_ct]['scost'] = scost
            iteration_log[iter_ct]['appcost'] = approx_cost
            iteration_log[iter_ct]['timestamp'] = time.time() - start_timer
            iteration_log[iter_ct]['phase'] = curr_phase

            # update the slop for next iteration
            updating_edges,release_all_edges_flag = self.update_rho(updating_edges,flow_arc_new, flowcom_arc_new,curr_phase,release_all_edges_flag)

            if (min_sol_dict['obj']>cost_new):
                print('Improved solution found, saving...')
                min_sol_dict = iteration_log[iter_ct].copy()


            # update terminating condition
            sol_diff = np.inf
            if (iter_ct-1>0):
                sol_diff = self.get_sol_diff(flowcom_arc_new,iteration_log[iter_ct-1]['flowcom_arc'])

            # after saving, clear large variable to save memory: flow_arc, flowcom_arc, path_sol of previous iteration
            if (iter_ct > 2): 
                del iteration_log[iter_ct-2]['flow_arc']
                del iteration_log[iter_ct-2]['flowcom_arc']
                del iteration_log[iter_ct-2]['rho_arc']
                del iteration_log[iter_ct-2]['path_sol']

            iter_ct+=1

            if (sol_diff <= 1e-5): TOLERANCE_MET = (True & (len(phases)==0))
            if ((time.time()-start_timer)> time_limit): TIMELIMIT_MET = True
            if (iteration_limit <= iter_ct): ITERATION_MET = True

            print(f"SC-Iteration {iter_ct}: TOL-{TOLERANCE_MET} TIME-{TIMELIMIT_MET}, PHASE-{curr_phase}, change {cost_new-cost_old}, LINK LEN-{len(updating_edges)}")
            # print(f"Prev solution SPP cost: {approx_cost}")
            print(f"New cost: {cost_new}, new approx cost: {self.get_approx_obj(flow_arc_new,self.rho)}")
            
            # phase tol checking
            cost_diff = cost_new-cost_old
            if (abs(cost_diff/cost_old) < 0.003): 
                phase_tol_icnt[curr_phase] += 1
            else:
                phase_tol_icnt[curr_phase] = 0 
            
            # two either or conditions: tol met of time limit met (give timelimit evenly to each phase)
            phase_time_lim = ((time.time()-start_timer) > (time_limit*phase_cnt/4))
            phase_tol_limt = (phase_tol_icnt[curr_phase] >= 10) 
            if ((phase_tol_limt or phase_time_lim) and (len(phases)>0)):
                if (len(updating_edges) < cur_total_edge_size):
                    print(f"PhaseTol {phase_tol_limt} - PhaseTimeLim {phase_time_lim} met, add all edges of curr phases: {len(updating_edges)}/{cur_total_edge_size}")
                    # reset the flag and tol cnt
                    release_all_edges_flag = True
                    phase_tol_icnt[curr_phase] = 0 
                else:
                    print(f"Tolerance met, and all edges included. Phase changes: Changing phase from {curr_phase} to {phases[0]}")
                    curr_phase = phases.pop(0)
                    phase_cnt += 1
                    processed_phases.append(curr_phase)
                    cur_total_edge_size = len(set([tuple(e) for p in processed_phases for e in self.edge_set[p]]))
            
        #### Plot stats
        if (plot_slope):
            self.plot_stat(iteration_log)
        return min_sol_dict, iteration_log
        




    # def concurrent_slope_scalling_with_time_limit(self, time_limit = 120, iteration_limit = np.inf, plot_slope = False):
    #     ''' run concurrent slope scaling with time limit (one dc at a time)'''
    #     TIMELIMIT_MET = False; TOLERANCE_MET = False; ITERATION_MET = False;
    #     print(f'(Con)Slope scaling w timelimit {time_limit},  iterlimit {iteration_limit}')

    #     # init the slope (cost) on each edge
    #     self.update_rho()

    #     iter_ct = 1
    #     iteration_log = dict()
    #     iteration_log[0] = {
    #                     'flow_arc': self.flow_arc,
    #                     'flowcom_arc': self.flowcom_arc,
    #                     'rho_arc':self.rho,
    #                     'obj': self.init_obj,
    #                     'appcost': self.get_approx_obj(self.flow_arc,self.rho)
    #                    }
        
        
    #     # storage of the best solution
    #     min_sol_dict = {'obj': np.inf}
        
    #     # get o-d pair agg by des from current solution
    #     t_s = time.time()
    #     spp_input_list = []
    #     des_agg_orgs = self.get_time_space_des_agg_feasible_orgs()
    #     for (des,orgs) in des_agg_orgs.items():
    #         sday = min([o[1] for d,o,f in orgs])
    #         eday = des[1]
    #         sub_nodes,sub_edges = self.network.get_nodes_edges_between_days(sday,eday)
    #         input_set = (sub_nodes,sub_edges,des,orgs)
    #         spp_input_list.append(input_set)
    #     # print(f'constructing batch inputs:{time.time()-t_s}')
        
    #     start_timer = time.time()
    #     while not (TIMELIMIT_MET or TOLERANCE_MET or ITERATION_MET):
    #         flow_arc_new = dict()
    #         flowcom_arc_new = dict()
    #         path_solution = dict()
    #         # solving sc concurrently
    #         shortest_path_trees = self.concurrent_shortest_path_submission(spp_input_list)
    #         # print(f'solving shortest paths concurrently:{time.time()-t_s}')
    #         t_s = time.time()
    #         # Sequentially update the network instance based on the computed shortest paths.
    #         # construct the dict that store only the best path 
    #         feasible_path = dict()
    #         for result_item in shortest_path_trees:
    #             (paths,label) = result_item
    #             for (dcof,path) in paths:
    #                 if path is not None:
    #                     if dcof in feasible_path.keys():
    #                         feasible_path[dcof].append((path,label[dcof[1]]))
    #                     else:
    #                         feasible_path[dcof] = [(path,label[dcof[1]])]

    #         # update the flow
    #         for dcof in feasible_path:
    #             sorted_list = sorted(feasible_path[dcof], key=lambda x:(x[1],x[0][-1][1]))
    #             # print(sorted_list)
    #             path,l = sorted_list[0]
    #             self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc_new, flowcom_arc_new)
    #             path_solution[(dcof[0],dcof[1])] = (dcof[2],path)

    #         tcost,scost = get_obj(flow_arc_new,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode); 
    #         cost_new = tcost+scost 
    #         cost_old = iteration_log[iter_ct-1]['obj']
    #         approx_cost = self.get_approx_obj(flow_arc_new,self.rho)

    #         iteration_log[iter_ct] = dict()
    #         iteration_log[iter_ct]['flow_arc'] = flow_arc_new.copy()
    #         iteration_log[iter_ct]['flowcom_arc'] = flowcom_arc_new.copy()
    #         iteration_log[iter_ct]['rho_arc'] = self.rho.copy()
    #         iteration_log[iter_ct]['path_sol'] = path_solution.copy()
    #         iteration_log[iter_ct]['obj'] = cost_new
    #         iteration_log[iter_ct]['tcost'] = tcost
    #         iteration_log[iter_ct]['scost'] = scost
    #         iteration_log[iter_ct]['appcost'] = approx_cost
    #         iteration_log[iter_ct]['timestamp'] = time.time() - start_timer

    #         # update the slop for next iteration
    #         self.update_rho(flow_arc_new, flowcom_arc_new)

    #         if (min_sol_dict['obj']>cost_new):
    #             print('Improved solution found, saving...')
    #             min_sol_dict = iteration_log[iter_ct].copy()
                       
                
    #         # update terminating condition
    #         sol_diff = np.inf
    #         if (iter_ct-1>0):
    #             sol_diff = self.get_sol_diff(flowcom_arc_new,iteration_log[iter_ct-1]['flowcom_arc'])

    #         # after saving, clear large variable to save memory: flow_arc, flowcom_arc, path_sol of previous iteration
    #         if (iter_ct > 2): 
    #             del iteration_log[iter_ct-2]['flow_arc']
    #             del iteration_log[iter_ct-2]['flowcom_arc']
    #             del iteration_log[iter_ct-2]['rho_arc']
    #             del iteration_log[iter_ct-2]['path_sol']

            
    #         iter_ct+=1

    #         if (sol_diff <= 1e-5): TOLERANCE_MET = True
    #         if ((time.time()-start_timer)> time_limit): TIMELIMIT_MET = True
    #         if (iteration_limit <= iter_ct): ITERATION_MET = True
            
    #         print("SC-Iteration {}: {} {}, change {}".format(iter_ct,TOLERANCE_MET,TIMELIMIT_MET,cost_new-cost_old))
    #         print(f"Prev solution SPP cost: {approx_cost}")
    #         print(f"New cost: {cost_new}, new approx cost: {self.get_approx_obj(flow_arc_new,self.rho)}")
            
    #     #### Plot stats
    #     if (plot_slope):
    #         self.plot_stat(iteration_log)
    #     return min_sol_dict, iteration_log

    def concurrent_shortest_path_submission(self, spp_input_list):
        # Create a concurrent executor (you can choose between ThreadPoolExecutor or ProcessPoolExecutor).
        with concurrent.futures.ThreadPoolExecutor(max_workers = 12) as executor:
            # Submit shortest path computations concurrently.
            futures = {executor.submit(dijkstra_shortest_path_tree,spp[0],spp[1],self.rho,spp[2],spp[3]): spp for spp in spp_input_list}
    
            # Collect the results as they become available.
            shortest_paths = []
            for future in concurrent.futures.as_completed(futures):
                shortest_path_tree, s_labels = future.result()
                shortest_paths += [(shortest_path_tree.items(),s_labels)]
        return shortest_paths

    # def get_feasible_path_from_spp_tree(self,shortest_path_trees):
    #     feasible_path = dict()
    #     for result_item in shortest_path_trees:
    #         (paths,label) = result_item
    #         for (dcof,path) in paths:
    #             if path is not None:
    #                 if dcof in feasible_path.keys():
    #                     feasible_path[dcof].append((path,label[dcof[1]]))
    #                 else:
    #                     feasible_path[dcof] = [(path,label[dcof[1]])]
    #     return feasible_path

    # def update_flow_with_feasible_path(self,feasible_path,flow_arc, flowcom_arc, path_sol):
    #     # update the flow
    #     for dcof in feasible_path:
    #         sorted_list = sorted(feasible_path[dcof], key=lambda x:(x[1],x[0][-1][1]))
    #         # print(sorted_list)
    #         path,l = sorted_list[0]
    #         self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc, flowcom_arc)
    #         path_sol[(dcof[0],dcof[1])] = (dcof[2],path)
        


    def adaptive_slope_scalling_with_time_limit(self, time_limit = 120, iteration_limit = np.inf, plot_slope = False):
        ''' run adaptive slope scaling with time limit (one dc at a time)'''
        TIMELIMIT_MET = False; TOLERANCE_MET = False; ITERATION_MET = False;
        print(f'(Con)Slope scaling w timelimit {time_limit},  iterlimit {iteration_limit}')

        # init the slope (cost) on each edge
        self.update_rho()

        iter_ct = 1
        iteration_log = dict()
        iteration_log[0] = {
                        'flow_arc': self.flow_arc,
                        'flowcom_arc': self.flowcom_arc,
                        'rho_arc':self.rho,
                        'obj': self.init_obj,
                        'appcost': self.get_approx_obj(self.flow_arc,self.rho)
                       }
        
        # storage of the best solution
        min_sol_dict = {'obj': np.inf}
        
        # get o-d pair agg by des from current solution
        t_s = time.time()
        # dict key by des (physical) node: {des_b: [ spp_input_list<(sub_nodes,sub_edges,des,orgs)> ]}
        spp_input_list_by_des = dict()

        des_agg_orgs = self.get_time_space_des_agg_feasible_orgs()
        # sort by des
        des_agg_orgs = dict(sorted(des_agg_orgs.items(), key=lambda item: item[0][0])) # by des physical node

        for (des,orgs) in des_agg_orgs.items():
            sday = min([o[1] for d,o,f in orgs])
            eday = des[1]
            sub_nodes,sub_edges = self.network.get_nodes_edges_between_days(sday,eday)
            input_set = (sub_nodes,sub_edges,des,orgs)
            if (des[0] in spp_input_list_by_des):
                spp_input_list_by_des[des[0]].append(input_set)
            else:
                spp_input_list_by_des[des[0]] = [input_set]
            # spp_input_list.append(input_set)

        # print(f'constructing batch inputs:{time.time()-t_s}')
        # this define the proceeding sequence of des physical node
        des_proceeding_seq = list(spp_input_list_by_des.keys())
        n_des = len(des_proceeding_seq)
        
        start_timer = time.time()
        while not (TIMELIMIT_MET or TOLERANCE_MET or ITERATION_MET):
            flow_arc_new = dict()
            flowcom_arc_new = dict()
            path_solution = dict()

            # shuffle list of processing des every iteration
            des_proceeding_seq_idx = random.sample(range(n_des), n_des)

            # index proceeding commodity as k_cnt
            for k_cnt in des_proceeding_seq_idx:
                proc_des = des_proceeding_seq[k_cnt]
                # list of des-agg commodity (batch of single des but multiple time nodes)
                spp_input_list = spp_input_list_by_des[proc_des]
                # solving sc concurrently only for one physical des node
                shortest_path_trees = self.concurrent_shortest_path_submission(spp_input_list)

                # Sequentially update the network instance based on the computed shortest paths.
                # construct the dict that store only the best path 
                feasible_path = dict()
                for result_item in shortest_path_trees:
                    (paths,label) = result_item
                    for (dcof,path) in paths:
                        if path is not None:
                            if dcof in feasible_path.keys():
                                feasible_path[dcof].append((path,label[dcof[1]]))
                            else:
                                feasible_path[dcof] = [(path,label[dcof[1]])]

                # update the flow
                for dcof in feasible_path:
                    sorted_list = sorted(feasible_path[dcof], key=lambda x:(x[1],x[0][-1][1]))
                    # print(sorted_list)
                    path,l = sorted_list[0]
                    self.add_flow_to_path(dcof[2], dcof[0], path, flow_arc_new, flowcom_arc_new)
                    path_solution[(dcof[0],dcof[1])] = (dcof[2],path)

                # est approx_cost, if after update rho, we will have the same cost
                approx_cost = self.get_approx_obj(flow_arc_new,self.rho)
                # update the slope for the next des-agg input list
                self.update_rho(flow_arc_new, flowcom_arc_new)

            tcost,scost = get_obj(flow_arc_new,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode); 
            cost_new = tcost+scost 
            cost_old = iteration_log[iter_ct-1]['obj']
            # approx_cost = self.get_approx_obj(flow_arc_new,self.rho)

            iteration_log[iter_ct] = dict()
            iteration_log[iter_ct]['flow_arc'] = flow_arc_new.copy()
            iteration_log[iter_ct]['flowcom_arc'] = flowcom_arc_new.copy()
            iteration_log[iter_ct]['rho_arc'] = self.rho.copy()
            iteration_log[iter_ct]['path_sol'] = path_solution.copy()
            iteration_log[iter_ct]['obj'] = cost_new
            iteration_log[iter_ct]['tcost'] = tcost
            iteration_log[iter_ct]['scost'] = scost
            iteration_log[iter_ct]['appcost'] = approx_cost
            iteration_log[iter_ct]['timestamp'] = time.time() - start_timer

            # update the slop for next iteration
            self.update_rho(flow_arc_new, flowcom_arc_new)

            if (min_sol_dict['obj']>cost_new):
                print('Improved solution found, saving...')
                min_sol_dict = iteration_log[iter_ct].copy()
                
            # update terminating condition
            sol_diff = np.inf
            if (iter_ct-1>0):
                sol_diff = self.get_sol_diff(flowcom_arc_new,iteration_log[iter_ct-1]['flowcom_arc'])
            
            # after saving, clear large variable to save memory: flow_arc, flowcom_arc, path_sol of previous iteration
            if (iter_ct > 2):
                del iteration_log[iter_ct-2]['flow_arc']
                del iteration_log[iter_ct-2]['flowcom_arc']
                del iteration_log[iter_ct-2]['rho_arc']
                del iteration_log[iter_ct-2]['path_sol']

            iter_ct+=1

            if (sol_diff <= 1e-5): TOLERANCE_MET = True
            if ((time.time()-start_timer)> time_limit): TIMELIMIT_MET = True
            if (iteration_limit <= iter_ct): ITERATION_MET = True
            
            print("SC-Iteration {}: {} {}, change {}".format(iter_ct,TOLERANCE_MET,TIMELIMIT_MET,cost_new-cost_old))
            print(f"Prev solution SPP cost: {approx_cost}")
            print(f"New cost {self.obj_mode}-mode: {cost_new}, new approx cost: {self.get_approx_obj(flow_arc_new,self.rho)}")
            
        #### Plot stats
        if (plot_slope):
            self.plot_stat(iteration_log)
        return min_sol_dict, iteration_log

    
    def get_approx_obj(self, flow_arc, rho):
        approx_cost = 0
        for a in flow_arc:
            if (flow_arc[a] > 1e-5):
                approx_cost += flow_arc[a]*rho[a]
        return approx_cost

    def get_lp_frac_obj(self,flow_arc):
        # need to add handling cost 
        cost = sum([(((flow_arc[a]/self.network.trailer_cap)*self.network.distance_matrix[(a[0][0],a[1][0])]) + (flow_arc[a]*self.handling_cost) ) for a in flow_arc])
        return cost
    
    def get_new_up_obj(self, flow_arc):
        ''' THIS IS NOT VALID'''
        # need to add handling cost 
        cost = sum([((flow_arc[a]/self.network.trailer_cap)*self.network.distance_matrix[(a[0][0],a[1][0])]) + (flow_arc[a]*self.handling_cost) + (0.5*self.network.distance_matrix[(a[0][0],a[1][0])]) for a in flow_arc])
        return cost
    
    def _get_day_from_dc(self,dc):
        dstr = dc[1]
        if (dstr == '1D'): return 1
        elif (dstr == '2D'): return 2
        elif (dstr == '3D'): return 3
        elif (dstr == 'GND'): return 7
    
    def _check_arrive_ontime(self, ts_ord, des_b, dc, n_node):
        '''given next time space node, origin time space node, current destination and flow class'''
        trav_days = self._get_day_from_dc(dc)
        return (n_node[0]==des_b and n_node[1]<=(ts_ord[1]+trav_days) and n_node[2]==1)
    
    def delete_fc(self, fc, org_node):
        print(f'del {fc} {org_node}')
        self.deleted_demand.append(((fc,org_node),self.network.demand_by_fc[fc][org_node]))
        del self.network.demand_by_fc[fc][org_node];

    def find_downstream_path(self, org, des, dc, flowcom_arc):
        trav_days = self._get_day_from_dc(dc)
        cnode = org
        path = [cnode]
        dc_tree = flowcom_arc[dc]
        notarrived = True
        while (notarrived):
            # assume in-tree holds and give first outbount arc
            ob_arcs = [a for a in dc_tree if (dc_tree[a]>1e-5)&(a[0]==cnode)]
            (_,nloc) = ob_arcs[0]
            path = path + [nloc]
            cnode = nloc
            if self._check_arrive_ontime(org,des,dc,nloc):
                notarrived = False
        return path
    
    def get_path_solution_by_od_commodity(self, flow_arc=None, flowcom_arc=None):
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        dummy_flowcom_arc = deepcopy(flowcom_arc)
        dummy_flow_arc = deepcopy(flow_arc)
        paths = dict()
        print("Constructing path-based solution...")
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                flow = self.network.demand_by_fc[dc][org_h]
                path = self.find_downstream_path(org_h,dc[0],dc,dummy_flowcom_arc)
                key = (dc, org_h, path[-1], flow)
                paths[key] = path
        return paths
    
    def add_flow_to_path(self, f, dc, path, flow_arc = None, flowcom_arc = None):
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc

        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            arc = (u,v)
            if (arc in flow_arc.keys()):
                flow_arc[arc] += f
            else:
                flow_arc[arc] = f
            if (dc not in flowcom_arc.keys()):
                flowcom_arc[dc] = dict()
                
            if (arc in flowcom_arc[dc].keys()):
                flowcom_arc[dc][arc] += f
            else:
                flowcom_arc[dc][arc] = f


    def intree_fixing_path_based_solution(self, path_sol, flow_arc, flowcom_arc):
        ''' given a path-solution, heuristically fixes all intree breaking by 
        merging them to one the single earliest-arrival path'''
        ot_cost,os_cost = get_obj(flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        old_obj = ot_cost+os_cost 
        for dc in flowcom_arc:
            itbn = self.get_intree_breaking_nodes(dc, path_sol, flowcom_arc[dc])
            for bn in itbn:
                print(f'fixing intree-breaking at {bn} {itbn[bn].keys()} of {dc}')
                self.merge_to_earliest_branch(dc, itbn[bn],flow_arc, flowcom_arc, path_sol)

        # update alpha of the new solution
        alpha = {}
        update_alpha_path_based_solution(path_sol,alpha)
        nt_cost,ns_cost =  get_obj(flow_arc,self.distance_matrix, self.trailer_cap, self.handling_cost, self.obj_mode)
        new_obj = nt_cost+ns_cost
        # validating approve solution
        self.validate_demand_delivered(flow_arc,flowcom_arc)
        self.validate_demand_duedate_satisfaction(path_sol)

        print(f'obj before fixing:{old_obj} (t{ot_cost},s{os_cost}), obj after fixing: {new_obj} (t{nt_cost},s{ns_cost}), ' + \
              f'change: {round((new_obj-old_obj)*100/old_obj,2)} % (t{round((nt_cost-ot_cost)*100/ot_cost,2)}% , s{round((ns_cost-os_cost)*100/os_cost,2)}%)')
        return alpha

    def merge_to_earliest_branch(self, dc, itb_paths, flow_arc, flowcom_arc, path_sol):
        ''' given the intree breaking point, collasp all of them to a single outbound arc(path)'''
        all_paths = [(barc,p) for (barc,val)in itb_paths.items() for p in val]
        # sort them by the arrival date
        all_paths = sorted(all_paths, key=lambda x: x[1][1][-1][1]) 
        # fix the earliest arrival one
        (ear_barc, (ear_flow,ear_path)) = all_paths[0]
        # move all other paths to the fixed one
        for (barc,(f,p)) in all_paths[1:]:
            if ( flowcom_arc[dc][barc]-f >= -1e-5 ):
                # remove from old path
                self.add_flow_to_path(-f, dc, p, flow_arc, flowcom_arc)
                del path_sol[(dc, p[0])]

                # constructing the new path that merge to the earliest arrival path
                # use the original path before the breaking point
                first_section = p[:p.index(barc[0]) + 1]
                # use the selected path before the remaing part after the breaking point
                second_section = ear_path[ear_path.index(barc[0]) + 1:]
                
                # add flow back to the earliest arrival path
                new_path = first_section + second_section
                self.add_flow_to_path(f, dc, new_path, flow_arc, flowcom_arc)
                path_sol[(dc, new_path[0])] = (f,new_path)

    def get_intree_breaking_nodes(self, dc, path_sol, flowcom_arc_dc):
        nthru = set()
        outgoing_arc_by_org = {} # storing outbound positive arcs
        for a in flowcom_arc_dc:
            if (flowcom_arc_dc[a] > 1e-5):
                nthru.add(a[0])
                if (a[0] not in outgoing_arc_by_org.keys()):
                    outgoing_arc_by_org[a[0]] = [a]
                else:
                    outgoing_arc_by_org[a[0]].append(a)

        itbn = {} # intree-breaking nodes
        for n in nthru:
            ogarc = outgoing_arc_by_org[n]
            if len(ogarc)>=2:
                for a in ogarc:
                    # iterate over path of dc at any origin
                    for ((k,o),v) in path_sol.items():
                        apath = convert_node_path_to_arc_path(v[1])
                        if (dc == k) and (a in apath):
                            if (n not in itbn):
                                itbn[n] = {}
                            if (a not in itbn[n]):
                                itbn[n][a] = []
                            itbn[n][a].append(v) # (flow vol, path)
        
        return itbn

    def validate_demand_delivered(self, flow_arc=None, flowcom_arc=None):
        ''' validate total demand satisfaction (volume satisfaction only)'''
        if (flow_arc == None) and (flowcom_arc == None):
            flow_arc = self.flow_arc
            flowcom_arc = self.flowcom_arc
        total_demand = sum([self.network.demand_by_fc[dc][a] for dc in self.network.demand_by_fc for a in self.network.demand_by_fc[dc]])
        dem_delivered = 0
        nodes_df = self.network.nodes_df; edges_df = self.network.edges_df
        for des_sort in nodes_df.loc[nodes_df[2]==1]['nodes'].values:
            inbound_edges = edges_df.loc[edges_df[1]==des_sort]['edges']
            inbound_flow = sum([flow_arc[x] for x in inbound_edges if x in flow_arc])
            outbound_edges = edges_df.loc[edges_df[0]==des_sort]['edges']
            outbound_flow = sum([flow_arc[x] for x in outbound_edges if x in flow_arc])
            # print(f'out: {outbound_flow},in: {inbound_flow}')
            dem_delivered += (inbound_flow-outbound_flow)
        if abs(total_demand-dem_delivered)>1e-5:
            print(f'total demand: {total_demand},dem_delivered: {dem_delivered}')
            raise Exception("Demand not satisfied!")
        else:
            print('All demands are delivered!')
            print(f'total demand: {total_demand},dem_delivered: {dem_delivered}')

    def validate_demand_duedate_satisfaction(self, path_sol):
        ''' validate demand path with its deadline started from the origin'''
        dummy_path_sol = deepcopy(path_sol)
        print("Validating demand satisfaction...")
        for dc in self.network.demand_by_fc:
            for org_h in self.network.demand_by_fc[dc]:
                dco = (dc, org_h)
                dem = self.network.demand_by_fc[dc][org_h]
                flow,path = dummy_path_sol.pop(dco)
                # print(f'sending dem/flow {dem}/{flow} for {dc} from {org_h}: path {path}')
                des = path[-1]
                if (abs(dem-flow)>1e-5): raise Exception("Invalid demand flow!")
                if (org_h[1]+self._get_day_from_dc(dc) < des[1]): 
                    raise Exception("Late!")

        if (len(dummy_path_sol)>0):
            print(dummy_path_sol)
            raise Exception("Exceed demand send!")

def dijkstra_shortest_path_tree(nodes, edges, dist_mat, sink, dcofs):
    # Initialize distances to all nodes as infinity except the source node.
    node_backward_neighbors = {node: [edge for edge in edges if edge[1]==node] for node in nodes }
    distances = dict(zip(nodes,[1e7] * len(nodes)))
    distances[sink] = 0

    # Initialize the shortest path tree as an empty dictionary.
    shortest_path_tree = {}
    
    # Initialize the predecessor dictionary.
    predecessors = {}
    
    # Priority queue to keep track of the nodes to visit.
    priority_queue = [(0, sink)]
    
    while priority_queue:
        # Pop the node with the smallest distance.
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If the current node is already in the shortest path tree, skip it.
        if current_node in shortest_path_tree:
            continue
        
        # Add the current node to the shortest path tree.
        shortest_path_tree[current_node] = current_distance
        
        # Update distances to neighbors.
        for neighbor in node_backward_neighbors[current_node]:
            weight = dist_mat[neighbor]
            distance = current_distance + weight
            
            # If the new distance is shorter than the recorded distance, update it.
            if distance < distances[neighbor[0]]:
                distances[neighbor[0]] = distance
                predecessors[neighbor[0]] = current_node
                heapq.heappush(priority_queue, (distance, neighbor[0]))
    
    # Extract shortest paths from sources to the sink.
    shortest_paths = {}
    for dcof in dcofs:
        source = dcof[1]
        if source == sink:
            continue  # Skip the sink itself
        path = []
        current_node = source
        while current_node != sink:
            path.insert(0, current_node)
            if (current_node in predecessors.keys()):
                current_node = predecessors[current_node]
            else:
                # print(f'No pred found for {current_node}, cur path {path}, return none')
                path = None
                break
        if path is None:
            shortest_paths[dcof] = None
        else:
            path.insert(0, sink)
            path.reverse()
            shortest_paths[dcof] = path
        # {dcof: path}, label {tsorg: cost}
    return shortest_paths,shortest_path_tree

# MIP model



# FLAT MODEL
class timespace_LTL_model():
    def __init__(self, _nodes, _arcs, _dist_mat, _desagg_commodities, _demand_by_fc, _constant_dict,_model_name = "flat_net"):
        self.nodes = _nodes
        self.arcs = _arcs
        self.distance_matrix = _dist_mat
        self.desagg_commodities = _desagg_commodities
        self.demand_by_fc = _demand_by_fc
        self.flat_net_model = Model(_model_name)
        
        self.B_i_d = dict(zip(self.nodes,[[]]*len(self.nodes)))
        for h in self.nodes:
            for d in self.desagg_commodities:
                if (h==d[0]):
                    if (len(self.B_i_d[h])==0):self.B_i_d[h] = dict(zip([d],[-sum([orgh_dem for orgh_dem in self.demand_by_fc[d].values()])]))
                    else: self.B_i_d[h][d] = -sum([orgh_dem for orgh_dem in self.demand_by_fc[d].values()])
                else:
                    if (len(self.B_i_d[h])==0): self.B_i_d[h] = dict(zip([d],[self.demand_by_fc[d][h]]))
                    else: self.B_i_d[h][d] = self.demand_by_fc[d][h]

        
        self.constant_dict = _constant_dict
        self.HANDLING_COST = _constant_dict['HANDLING_COST']
        self.TRAILER_CAP = _constant_dict['TRAILER_CAP']
 
    def generateVariables(self):
        print("generating variables...")
        # init vars
        # x_ij
        self.x_ij = self.flat_net_model.addVars(self.arcs, lb=0,vtype=GRB.CONTINUOUS, name='freightflow')

        # tau_ij
        self.tau_ij = self.flat_net_model.addVars(self.arcs, lb=0,vtype=GRB.INTEGER, name='trailer')

        #x_d_ij
        self.x_d_ij = dict()
        for d_com in self.desagg_commodities:
            self.x_d_ij[d_com] = self.flat_net_model.addVars(self.arcs, lb=0,vtype=GRB.CONTINUOUS, name='freightflow_{}'.format(d_com))

        #y_d_ij
        self.y_d_ij = dict()
        for d_com in self.desagg_commodities:
            self.y_d_ij[d_com] = self.flat_net_model.addVars(self.arcs, lb=0,vtype=GRB.BINARY, name='indicator_{}'.format(d_com))

    def generateObjective(self):
        # obj
        print("generating objective...")
        self.flat_net_model.setObjective( quicksum(self.distance_matrix[a]*self.tau_ij[a]+self.HANDLING_COST*self.x_ij[a] for a in self.arcs) ,
                                        sense=GRB.MINIMIZE)
    
    def generateConstaints(self):
        print("generating constraints...")
        # constraints
        # flow balance
        self.flowbalance = (
                quicksum(self.x_d_ij[d][a] for a in self.arcs if a[0]==i) - 
                quicksum(self.x_d_ij[d][a] for a in self.arcs if a[1]==i) == self.B_i_d[i][d] \
                                     for i in self.nodes for d in self.desagg_commodities)

        self.flat_net_model.addConstrs(self.flowbalance,name='flowbalance')

        # in-tree constrain
        self.intree = (quicksum(self.y_d_ij[d][a] for a in self.arcs if a[0]==i) <= 1 \
                                     for i in self.nodes for d in self.desagg_commodities)

        self.flat_net_model.addConstrs(self.intree,name='intree')

        # upper bound of flowcom
        self.upcomflow = (self.x_d_ij[d][a] <= quicksum(self.demand_by_fc[d].values())*self.y_d_ij[d][a] \
                                     for d in self.desagg_commodities for a in self.arcs)

        self.flat_net_model.addConstrs(self.upcomflow,name='upcomflow')

        # upper bound of flow
        self.upflow = (self.x_ij[a] <= self.TRAILER_CAP*self.tau_ij[a] \
                                   for a in self.arcs)
        self.flat_net_model.addConstrs(self.upflow,name='upflow')

        # sum of flowcom
        self.flowcomdef = (self.x_ij[a] == quicksum(self.x_d_ij[d][a] for d in self.desagg_commodities) \
                                   for a in self.arcs)
        self.flat_net_model.addConstrs(self.flowcomdef,name='flowcomdef')
    
    def solveModel(self):
        self.flat_net_model.update()
        self.flat_net_model.optimize()
    
    def getArcUtil(self):
        self.arc_util = pd.DataFrame([[self.tau_ij[a].X,round(self.x_ij[a].X,3)] for a in self.arcs],columns=['trailers','flows'],index=self.arcs)
        pd.set_option('display.chop_threshold', 0.00001)
        # arc_util.loc[arc_util['flows']<0.0001].loc[:,'flows'] = 0
        # arc_util.loc[arc_util['trailers']<0.0001].loc[:,'trailers'] = int(0)

        self.arc_util.loc[:,'avgutil'] = self.arc_util['flows']/(self.arc_util['trailers']*self.TRAILER_CAP)
        self.arc_util = self.arc_util.fillna(0)
        return self.arc_util
     
    def getFlowTreeByCom(self, _color_list, _arc_config_dict):
        flow_trees_by_com = dict()
        for idx in range(len(self.desagg_commodities)):
            d = self.desagg_commodities[idx]
            color_lab = _color_list[idx%len(_color_list)]
            print("Solution Plot: {}".format(d))
            flow_tree_of_com = []
            for arc in self.x_d_ij[d]:
                flow_com = self.x_d_ij[d][arc].X
                trailer_ij = round(self.tau_ij[arc].X)
                if (flow_com>0.00001):
                    arc_config = dict(zip(['arcs_list','config'],[[arc],_arc_config_dict.copy()]))
                    arc_config['config']['name'] = "f{}, t{}".format(round(flow_com,2), trailer_ij)
                    arc_config['config']['line_color'] = color_lab
                    arc_config['config']['line_width'] = trailer_ij*4
                    flow_tree_of_com.append(arc_config)
            flow_trees_by_com[d] = flow_tree_of_com
        return flow_trees_by_com






# Function to plot the time-expanded network
def plot_time_expanded_network(network):
    G = nx.DiGraph()

    # Add nodes to the graph
    for node, data in network.get_nodes().items():
        G.add_node(node, size = 20)

    # Add edges to the graph
    for edge, data in network.get_edges().items():
        G.add_edge(edge[0], edge[1], weight=data['weight'])

    # Define node positions for better visualization
    node_dict = G.nodes()
    node_keys = list(node_dict.keys())
    node_by_buildings = dict()
    for n in node_keys:
        b = n[0]
        if (b in node_by_buildings):
            node_by_buildings[b].append(n)
        else:
            node_by_buildings[b] = [n]

    building_keys = list(node_by_buildings.keys())
    pos = {n:(n[1]+(n[2]/4), -int(b_i))for b_i in range(len(building_keys)) for n in node_by_buildings[building_keys[b_i]]}

    # Create labels for nodes (optional)
    node_labels = {node: f"{node[0]}\n{node[1]},{node[2]}" for node, data in G.nodes(data=True)}

    
    # Create labels for edges (optional)
#     edge_labels = {(u, v): f"Tt: {data['weight']}" for u,v, data in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#     print(edge_labels)
    # Plot the network
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("Time-Expanded Network Representation")
    plt.axis('on')
    plt.show()
    
# Function to plot the time-expanded network
def plot_flowarcs_on_time_expanded_network(network, flowarcs):
    G = nx.DiGraph()

    # Add nodes to the graph
    for node, data in network.get_nodes().items():
        G.add_node(node, size = 20)
    # Define node positions for better visualization
    pos = {node: ((node[1]+(node[2]/4)), -int(node[0].split("_")[-1])) for node in G.nodes()}

    # Create labels for nodes (optional)
    node_labels = {node: f"{node[0]}\n{node[1]},{node[2]}" for node, data in G.nodes(data=True)}
    
    # Add edges to the graph
    for arc in flowarcs:
        G.add_edge(arc[0], arc[1], weight= round(flowarcs[arc],2))
    
    # Create labels for edges (optional)
#     edge_labels = {(u, v): f"Tt: {data['weight']}" for u,v, data in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#     print(edge_labels)
    # Plot the network
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("Time-Expanded Network Representation")
    plt.axis('on')
    plt.show()



from matplotlib.patches import Patch
def plot_paths_on_network(base_network, paths, save_name = None, figsize = (16, 8)):
    # Create a copy of the base network for visualization
    G = nx.DiGraph()
    # Add nodes from the graph
    G.add_nodes_from(base_network.get_nodes())
    

    # Generate random colors for paths
    path_colors = {}
    for dc,path in paths:
        color = random.choice(COLOR_PALETTE)  # Random hexadecimal color
        path_colors[path] = color

    # Create labels for paths (optional)
    path_labels = {path: f"Path {dc} o{path[0]}" for i, path in enumerate(paths)}
    
    # Plot the base network
    # Create labels for nodes (optional)
    node_labels = {node: f"{node[0]}\n{node[1]},{node[2]}" for node, data in G.nodes(data=True)}
    pos = {node: ((node[1]+(node[2]/4)), -int(node[0].split("_")[-1])) for node in G.nodes()}

    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    

    
    # Create labels for edges (optional)
    edge_labels = {(u, v): f"{data['weight']}%u" for u,v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Plot the paths on top of the base network
    legend_elements = []
    for i, (dc,path) in enumerate(paths):
        path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
        # Add edges to the graph
        for arc in path_edges:
            G.add_edge(arc[0], arc[1])
        path_color = path_colors[path]
        nx.draw(G.subgraph(path), pos, with_labels=False, node_size=1000, node_color=path_color, edge_color=path_color, width=2, arrows=True, label=path_labels.get(path, f"Path {dc} o{path[0]}"))
        legend_elements.append(Patch(facecolor=path_color, edgecolor=path_color, 
                                     label=path_labels.get(path, f"Path {dc} o{path[0]}")))

    plt.legend(handles=legend_elements, loc='upper right')

    # Add path labels to the legend
#     legend_colors = [path_colors.get(path) for path in paths]
#     legend_labels = [path_labels.get(path, f"Path {i + 1}") for i, path in enumerate(paths)]

    plt.title("Base Network with Paths")
    plt.axis('off')
    
    if save_name is not None:
        # Save the plot as an image
        plt.savefig(fname=f"{save_name}.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_base_network_with_paths(base_network, flowarcs, paths, save_name = None, figsize = (16, 8)):
    # Create a copy of the base network for visualization
    G = nx.DiGraph()
    # Add nodes from the graph
    G.add_nodes_from(base_network.get_nodes())
    # Add edges to the graph
    for arc in flowarcs:
        G.add_edge(arc[0], arc[1], weight= round((flowarcs[arc]%base_network.trailer_cap)*100/base_network.trailer_cap,0))

    # Generate random colors for paths
    path_colors = {}
    for dc,path in paths:
        color = random.choice(COLOR_PALETTE)  # Random hexadecimal color
        path_colors[path] = color

    # Create labels for paths (optional)
    path_labels = {path: f"Path {dc} o{path[0]}" for i, path in enumerate(paths)}
    
    # Plot the base network
    # Create labels for nodes (optional)
    node_labels = {node: f"{node[0]}\n{node[1]},{node[2]}" for node, data in G.nodes(data=True)}
    pos = {node: ((node[1]+(node[2]/4)), -int(node[0].split("_")[-1])) for node in G.nodes()}

    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Create labels for edges (optional)
    edge_labels = {(u, v): f"{data['weight']}%u" for u,v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Plot the paths on top of the base network
    legend_elements = []
    for i, (dc,path) in enumerate(paths):
        path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
        path_color = path_colors[path]
        nx.draw(G.subgraph(path), pos, with_labels=False, node_size=1000, node_color=path_color, edge_color=path_color, width=2, arrows=True, label=path_labels.get(path, f"Path {dc} o{path[0]}"))
        legend_elements.append(Patch(facecolor=path_color, edgecolor=path_color, 
                                     label=path_labels.get(path, f"Path {dc} o{path[0]}")))

    plt.legend(handles=legend_elements, loc='upper right')

    # Add path labels to the legend
#     legend_colors = [path_colors.get(path) for path in paths]
#     legend_labels = [path_labels.get(path, f"Path {i + 1}") for i, path in enumerate(paths)]

    plt.title("Base Network with Paths")
    plt.axis('off')
    
    if save_name is not None:
        # Save the plot as an image
        plt.savefig(fname=f"{save_name}.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_base_network_with_single_commodity(base_network, flowcom_arcs, alphaofnode, save_name = None, figsize = (16, 8)):
    # Create a copy of the base network for visualization
    G = nx.DiGraph()
    # Add nodes from the graph
    G.add_nodes_from(base_network.get_nodes())
    # Add edges to the graph
    for arc in flowcom_arcs:
        if (flowcom_arcs[arc]>1e-5):
            G.add_edge(arc[0], arc[1], weight= round(flowcom_arcs[arc],0))

    # # Generate random colors for paths
    # path_colors = {}
    # for dc,path in paths:
    #     color = random.choice(COLOR_PALETTE)  # Random hexadecimal color
    #     path_colors[path] = color

    # Create labels for paths (optional)
    # path_labels = {path: f"Path {dc} o{path[0]}" for i, path in enumerate(paths)}
    
    # Plot the base network
    # Create labels for nodes (optional)
    node_labels = {node: f"{node[0]}\n{node[1]},{node[2]}" for node, data in G.nodes(data=True)}
    pos = {node: ((node[1]+(node[2]/4)), -int(node[0].split("_")[-1])) for node in G.nodes()}

    alpha_labels = {node: f"alpha {alpha}" for (node, alpha) in alphaofnode.items()}
    pos_a = {node: ((node[1]+(node[2]/4)), -int(node[0].split("_")[-1]) + 0.5) for node in alpha_labels}

    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='lightblue', edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Create labels for edges (optional)
    edge_labels = {(u, v): f"{data['weight']}c" for u,v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # label for alpha
    nx.draw_networkx_labels(G, pos_a, labels=alpha_labels, font_size=10, font_color = "r" )
    
    # # Plot the paths on top of the base network
    # legend_elements = []
    # for i, (dc,path) in enumerate(paths):
    #     path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
    #     path_color = path_colors[path]
    #     nx.draw(G.subgraph(path), pos, with_labels=False, node_size=1000, node_color=path_color, edge_color=path_color, width=2, arrows=True, label=path_labels.get(path, f"Path {dc} o{path[0]}"))
    #     legend_elements.append(Patch(facecolor=path_color, edgecolor=path_color, 
    #                                  label=path_labels.get(path, f"Path {dc} o{path[0]}")))

    # plt.legend(handles=legend_elements, loc='upper right')

    # Add path labels to the legend
#     legend_colors = [path_colors.get(path) for path in paths]
#     legend_labels = [path_labels.get(path, f"Path {i + 1}") for i, path in enumerate(paths)]

    plt.title("Base Network with Paths")
    plt.axis('off')
    
    if save_name is not None:
        # Save the plot as an image
        plt.savefig(fname=f"{save_name}.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()



def plot_buildings_on_map(building_df, path_list):
    us_states = gpd.read_file('cb_2021_us_state_500k/cb_2021_us_state_500k.shp')
    us_states.to_file('us_states.geojson', driver='GeoJSON')
    # Create a map centered around the US
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=4, tiles='cartodb positron')
    # Add state borders to the map
    folium.GeoJson(
        'us_states.geojson',
        style_function=lambda feature: {
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0,
        }
    ).add_to(m)

    for p in path_list:
        for (o,d) in p:
            if (o not in building_df['SLIC'].values):
                print(f'no coor data of {o}, skip {o}-{d}')
                continue
            if (d not in building_df['SLIC'].values):
                print(f'no coor data of {d}, skip {o}-{d}')
                continue
            o_lat,o_lon = building_df.loc[building_df['SLIC']==o][['Lat','Lon']].values[0]
            d_lat,d_lon = building_df.loc[building_df['SLIC']==d][['Lat','Lon']].values[0]

            # Add an arrow marker at the end of the line
            folium.Marker(
                location=(d_lat,d_lon),
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html='<div style="font-size: 12px; color: blue;">&#x25BA;</div>'
                )
            ).add_to(m)

            folium.PolyLine(
            locations=[(o_lat,o_lon), (d_lat,d_lon)],
            color='orange',
            weight=1,
            ).add_to(m)

    # Iterate through the building data and add markers to the map
    for building_id, lat, lon in building_df.values:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            popup=f'Building ID: {building_id}'
        ).add_to(m)
        
    return m
