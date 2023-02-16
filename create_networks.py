"""
create_networks.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods used to
create artificial and real-world networks.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


# Americal Football (AMF) read-in from gml   and Ground Truth

def colored_amf_graph(G):
    """
    Color nodes of a graph according to their community.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    G : NetworkX graph
        An undirected graph with colored nodes.
    """
    for n,d in G.nodes.data(): 
        d["color"] = d["value"]
    return G

def get_circles_in_circle_pos(G, sc, rd, cx, cy):
    """
    Get the positions of nodes in a graph in a circle in a circle.
    One small circle for each community. All communities are in one circle.
    Only used for AMF graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    sc : float
        Scale of the circle in the circle.

    rd : float
        Radius of the circle in the circle.

    cx : float
        x-coordinate of the center of the circle in the circle.

    cy : float
        y-coordinate of the center of the circle in the circle.

    Returns
    -------
    p : dict
    """
    p = {}
    values = set([d["value"]  for n,d in iter(G.nodes.items())])
    max_value = list(values)[-1]

    cnt = np.array([cx, cy])
    for v in values:
        temp = nx.circular_layout(
                    nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["value"] == v]),
                    scale = sc,
                    center = cnt + np.array([rd * np.cos(v/(max_value+1)*2*np.pi),
                                             rd * np.sin(v/(max_value+1)*2*np.pi)])
                    )
        p.update(temp)
    return p

def calculate_SBM(k, l, p_in, p_out, title_str, t_coeff, q_coeff, p_coeff):
    """
    Create a SBM graph for a given k, l, p_in, p_out.

    Parameters
    ----------
    k : int
        Number of nodes in each community.

    l : int
        Number of communities.

    p_in : float
        Probability of an edge within a community.

    p_out : float
        Probability of an edge between communities.

    title_str : str
        Title of the plot. # GET RID OF THIS

    t_coeff : float
        Coefficient for the threshold function. # GET RID OF THIS

    q_coeff : float
        Coefficient for the quadratic function. # GET RID OF THIS

    p_coeff : float
        Coefficient for the power function. # GET RID OF THIS

    Returns
    -------
    None.
    """
    print("k:",k," l:",l," p_in:",p_in," p_out:",p_out)
    sizes = build_size_list(k, l)
    probs = build_prob_list(l, p_in, p_out)
    
    G = nx.stochastic_block_model(sizes, probs, seed = 0)
    init_edge_attributes(G)
      
    H = G.to_directed()
    
    t0 = perf_counter()
    
    cycles = []
    for c in simple_cycles(H, 6):
        cycles.append(c) 

    t1 = perf_counter()
    print("Zyklen: ",len(cycles), " - Zeit: ", t1-t0)
    
    d = dict()
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]
        set_edge_attributes_2(G, d[i], i)
        
    get_orc_edge_curvatures (G)
    get_edge_curvatures (G, t_coeff, q_coeff, p_coeff)
    
    pos1 = nx.kamada_kawai_layout(H)
    blocks = [v["block"]  for u,v in H.nodes.data()]
    set_SBM_edge_colors(G)    
    edge_cols = [d["color"]  for u,v,d in G.edges.data()]
    plot_my_graph(G, pos1, node_col = blocks, edge_col = edge_cols)

    res_diffs = get_within_vs_between_curv(G)
    print("Resulting differences:") 
    for r in res_diffs.items():
        print(r)
        
    show_curv_data(G)
    

def calculate_SBMs():
    """
    Calculate SBM graphs for different k, l, p_in, p_out.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """
    ll_k = [5,10,15,20]
    k_def = 20
    ll_l = [2,3,4,5]
    l_def = 5
    ll_p_in = [0.6, 0.7, 0.8, 0.9]
    p_in_def = 0.7
    ll_p_out = [0.05, 0.03, 0.02, 0.01]
    p_out_def = 0.05
    for k in ll_k:
        s = "Variation of community size / k = " + str(k) + "\n" + \
            "k=" + str(k) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k, l_def, p_in_def, p_out_def, s, 3, 2, 1)
    for l in ll_l:
        s = "Variation of number of communities / l = " + str(l) + "\n" + \
            "k=" + str(k_def) + "  l=" + str(l) +  "  p_in=" + str(p_in_def) + "  p_out=" + str(p_out_def)
        calculate_SBM(k_def, l, p_in_def, p_out_def, s, 3, 2, 1)
    for p_in in ll_p_in:
        s = "Variation of p_in / p_in = " + str(p_in) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l_def, p_in, p_out_def, s, 3, 2, 1)
    for p_out in ll_p_out:
        s = "Variation of p_out / p_out = " + str(p_out) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in_def) + " p_out:" + str(p_out)
        calculate_SBM(k_def, l_def, p_in_def, p_out, s, 3, 2, 1)