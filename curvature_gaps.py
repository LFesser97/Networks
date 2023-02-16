"""
curvature_gaps.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods used to compute curvature gaps.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


def compute_curvature_gap(Gr, curv_name, cmp_key="block"):
    """
    Get the mean and standard deviation of the curvature values of edges within and between communities.
    The curvature values are the ones stored in the graph.
    The graph must have the attribute "block" for each node.
    The graph must have the attributes "orc", "frc", "afrc", "afrc4", "afrc5" for each edge.

    Parameters
    ----------
    Gr : NetworkX graph
        An undirected graph.

    Returns
    -------
    res_diffs : dict
        A dictionary containing the mean and standard deviation of the curvature values of edges within and between communities.
    """
    
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        c_dict[k][curv_name] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                            
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:                    
            c_dict["withins"][curv_name]["data"].append(Gr.edges[u,v][curv_name])           
        else:                                                                         
            c_dict["betweens"][curv_name]["data"].append(Gr.edges[u,v][curv_name]) 

    for k in c_dict.keys():
        c_dict[k][curv_name]["mean"] = np.mean(c_dict[k][curv_name]["data"])      
        c_dict[k][curv_name]["std"] = np.std(c_dict[k][curv_name]["data"])        
            
    res_diffs = {}
    sum_std = np.sqrt(np.square(c_dict["withins"][curv_name]["std"]) + np.square(c_dict["betweens"][curv_name]["std"]))  
    res_diffs[curv_name] = np.abs((c_dict["withins"][curv_name]["mean"] - c_dict["betweens"][curv_name]["mean"]) / sum_std)   
    
    return res_diffs


def calculate_SBM_cycle_weight_var():
    """
    Calculate the curvature gaps for the SBM with different cycle weights.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    k_def = 20
    l_def = 3
    p_in_def = 0.7
    p_out_def = 0.05
    cycle_weights = [(0.4, 0.2), (0.8, 0.4), (1.2, 0.6), (1.6, 0.8), (2.0, 1.0)]
    for cw in cycle_weights :
        (q,p)  = cw
        s = "Variation of cycle weights / quad weight = " + str(q) + "pent weight = " + str(p) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l_def, p_in_def, p_out_def, s, 3, q, p)


def get_curvature_gap(Gr, cn="afrc", cmp_key="block"):
    """
    Get the curvature gap of the graph.
    The curvature values are the ones stored in the graph.
    The graph must have the attributes "orc", "frc", "afrc", "afrc4", "afrc5" for each edge.

    Parameters
    ----------
    Gr : NetworkX graph
        An undirected graph.

    Returns
    -------
    curv_gap : float
        The curvature gap of the graph.
    """
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        c_dict[k] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                            
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:    
            c_dict["withins"]["data"].append(Gr.edges[u,v][cn])            
        else:                                               
            c_dict["betweens"]["data"].append(Gr.edges[u,v][cn])      

    for k in c_dict.keys():
        c_dict[k]["mean"] = np.mean(c_dict[k]["data"])      
        c_dict[k]["std"] = np.std(c_dict[k]["data"])      
            
    sum_std = np.sqrt(np.square(c_dict["withins"]["std"]) + np.square(c_dict["betweens"]["std"]))   
    curv_gap = np.abs((c_dict["withins"]["mean"] - c_dict["betweens"]["mean"]) / sum_std)   
    
    return curv_gap


def optimization_func(a, G, cmp_key="value"):
    """
    Optimization function for the curvature gap.

    Parameters
    ----------
    a : array
        An array of initial values for the optimization.

    G : NetworkX graph
        An undirected graph.

    cmp_key : string
        The key for the community attribute in the graph.

    Returns
    -------
    c_gap : float
        The curvature gap of the graph.
    """
    t = a[0]
    if len(a) > 1:
        q = a[1]
    else:
        q = 0
    if len(a) > 2:
        p = a[2]
    else:
        p = 0
    get_edge_curvatures(G, t, q, p)
    c_gap = -1 * get_curvature_gap(G, "afrc5", cmp_key)

    return c_gap


def maximize_curvature_gap(G, a, cmp_key = "value"):
    """
    Maximize the curvature gap of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    a : array
        An array of initial values for the optimization.

    cmp_key : string
        The key for the community attribute in the graph.

    Returns
    -------
    results : array
        An array of the optimized curvature gap and the parameters.
    """
    results = []
    for i in range(len(a)):
        x0 = a[0:i+1]    # Initial values for t / t,q / t,q,p
        res = minimize(optimization_func, x0, method='nelder-mead', args = (G, cmp_key), options={'disp': False})
        results.extend([-1 * res.fun, *res.x])
    return results