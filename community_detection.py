"""
community_detection.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods for community detection.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


# community detection algorithm

'''
    For all e ∈ E, compute FA(e)
    while there exists an edge with negative AFRC in G do
        if there is a unique edge emin with minimal AFRC then
            remove emin from G
        else
            choose an edge emin from the edges with minimial AFRC uniformly at random
            remove emin from G
        end if
        Re-calculate the AFRC for all affected edges in G
    end while
    assign the same label l to each vertex v in a connected component of G
    return a list of tuples (v, l)
    '''

def detect_communities(G):
    """
    Sequential deletion algorithm for detecting communities in a graph G

    Parameters
    ----------
    G : graph
        A networkx graph

    Returns
    -------
    G : graph
        A networkx graph with node labels and node colors
    """

    # set graph attributes and calculate initial AFRC values
    set_edge_attributes(G)
    afrc_min, afrc_max = get_min_max_afrc_values(G)
    set_edge_colors(G,afrc_min)     

    # show histogram of curvature values
    show_histos(G)    # NEED TO IMPORT THIS
    print("min. AFRC value:", afrc_min, " / max. AFRC value:", afrc_max)
    afrc_threshold = int(input("Enter threshold value for AFRC to remove edges with a lower value: "))

    loop_counter = 0
    
    # # collect edges with minimal negative AFRC
    afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
    afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
    
    while len(afrc_min_list) > 0:      
        if len(afrc_min_list) == 1:
            # edge is the only element in the list
            a = afrc_min_list[0]
        else:
            # edge is randomly chosen from the list
            a = select_an_edge(afrc_min_list)[0]

        (u,v) = a[:2]
        afrc_below_list.remove(a)

        # remove chosen edge
        G.remove_edge(u,v)
        affecteds = list(G.edges([u,v]))
        below_edges = [(u,v)  for u,v,d in afrc_below_list]

        # update graph attributes and calculate new AFRC values
        set_edge_attributes(G, affecteds + below_edges)
        afrc_min, afrc_max = get_min_max_afrc_values(G, affecteds + below_edges)
        set_edge_colors(G, afrc_min, afrc_threshold)     
        loop_counter += 1
        
        # collect edges with minimal negative AFRC
        afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
        afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
        
    # determine connected components of graph of edges with positive ARFC
    C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    # Create list of tupels with node names and cluster labels, set node colors acc to cluster
    G = set_node_labels(G,C)    
    return G


# helper functions

cyc_names = {3:"triangles", 4:"quadrangles", 5:"pentagons"}     


def set_edge_attributes(G,ae=None): # NEED TO ALIGN THIS WITH get_edge_curvatures
    """
    Set edge attributes triangles and curvature

    Parameters
    ----------
    G : graph
        A networkx graph

    ae : list
        List of edges to be considered for calculation of edge attributes

    Returns
    -------
    None.
    """
    if ae == None: 
        ae = list(G.edges())
    for (u,v) in ae:
        G.edges[u,v]["triangles"] = m = len(all_triangles_of_edge(G,(u,v))) # NEED TO IMPORT THIS
        G.edges[u,v]["frc"] = fr_curvature(G, u, v) # NEED TO IMPORT THIS
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, m)  # NEED TO IMPORT THIS


def set_edge_attributes_2 (G, ll, i): # CHECK WHETHER THIS IS CORRECT
    """
    Set edge attributes triangles and curvature

    Parameters
    ----------
    G : graph
        A networkx graph
    
    ll : list
        List of lists of nodes

    i : int
        Number of nodes in each list

    Returns
    -------
    None.
    """
    for l in ll:     
        for e1 in range(0, i): 
            if e1 == i-1:
                e2 = 0
            else:
                e2 = e1 + 1
            u = l[e1]
            v = l[e2]
            G.edges[u,v][cyc_names[i]].append(l)


def init_edge_attributes(G):
    """
    Initialize edge attributes

    Parameters
    ----------
    G : graph
        A networkx graph

    Returns
    -------
    None.
    """
    curv_names = ["frc", "afrc", "afrc4", "afrc5"] 
    for (u,v) in list(G.edges()):
        for i in range(3,6):
            G.edges[u,v][cyc_names[i]] = []
        for cn in curv_names:
            G.edges[u,v][cn] = 0
        G.edges[u,v]["color"] = "lightgrey"       # default color for edges


def get_min_max_afrc_values(G, ae=None):
    """
    Get minimum and maximum values of AFRC

    Parameters
    ----------
    G : graph
        A networkx graph

    ae : list
        List of edges to be considered for calculation of edge attributes

    Returns
    -------
    a : int
        Minimum value of AFRC

    b : int
        Maximum value of AFRC
    """
    if ae == None: 
        ae = list(G.edges())
    a = len(G.nodes())
    b = -len(G.nodes())
    for (u,v) in ae:
        a = min(a, G.edges[u,v]["afrc"])
        b = max(b, G.edges[u,v]["afrc"])
    return a, b


def set_edge_colors(G,a,thr=0):
    """
    Set edge colors according to AFRC value

    Parameters
    ----------
    G : graph
        A networkx graph

    a : int
        Minimum value of AFRC

    thr : int
        Threshold value for AFRC

    Returns
    -------
    None.
    """
    for u,v,d in G.edges.data():
        if d["afrc"] >= thr:
            d["color"] = "green" 
        else:
            d["color"] = "darkred" 
        if (a < thr) and (d["afrc"] == a):
            d["color"] = "red"    


def select_an_edge(edge_list):
    """
    Select an edge from a list of edges with uniform probability distribution

    Parameters
    ----------
    edge_list : list
        List of edges

    Returns
    -------
    edge : tuple
        A randomly chosen edge from the list

    """
    def find_interval(x, partition):
        for i in range(0, len(partition)):
            if x < partition[i]:
                return i-1
        return -1
    
    def weighted_choice(sequence, weights):
        # random float between 0 and 1
        x = np.random.random()    
        # list of cumulated weights resp. probabilities
        cum_weights = [0] + list(np.cumsum(weights))   
        # determine index based on cumulated probabilities
        index = find_interval(x, cum_weights)
        # return element of sequence matching the index
        return sequence[index]          
    
    # use uniform probabiliity distribution to select one of the edges
    act_weights = [1.0 / len(edge_list)] * len(edge_list)  
    # return randomly chosen element of edge list 
    return [weighted_choice(edge_list, act_weights)]


def set_node_labels(G,C):
    """
    Set node labels according to connected component labels

    Parameters
    ----------
    G : graph
        A networkx graph

    C : list
        List of clusters

    Returns
    -------
    G : graph
        A networkx graph with node labels

    """
    for i,c in enumerate(C):
        for u in c:
            # G.nodes[u]["cluster"] = i
            G.nodes[u]["color"] = i
    return G


# Non-sequential community detection

'''
    remove all edges with afrc4 curvature above a given threshold
'''

def detect_communities_nonsequential(G, t_coeff = 3, q_coeff = 2):
    """
    Detect communities in a graph using non-sequential community detection,
    using the AFRC4 with expected weights. Only correct for an SBM.

    Parameters
    ----------
    G : graph
        A networkx graph

    t_coeff : int
        Coefficient for triangles, default = 3

    q_coeff : int
        Coefficient for quadrangles, default = 2

    Returns
    -------
    G : graph
        A networkx graph with node labels
    """    

    # get start values for curvature
    get_edge_curvatures(G, t_coeff, q_coeff)
    # get min,max,values for afrc4 curvature
    afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4")
    # show histogram of curvature values
    show_curv_data (G, title_str = "", cmp_key = "block")   # NEED TO IMPORT THIS

    afrc_threshold = int(input("Enter threshold value for AFRC4 to remove edges with a higher value: "))
    
    # collect edges with maximal AFRC4
    afrc_above_list = [(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc4"] > afrc_threshold)]
        
    # for all items in afrc_above_list
    for i,a in enumerate(afrc_above_list):
        # select edge from item
        (u,v) = a[:2]
        # remove edge from graph
        G.remove_edge(u,v)
        # print(i, " removed: ", (u,v))
        
    # determine connected components of graph of edges with positive ARFC
    C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # set node colors acc to cluster
    G = set_node_labels(G,C)
    return G


""" CODE BELW THIS LINE IS USED ONLY FOR LINEPLOTS OF THE ALGORITHM'S PERFORMANCE """

def detect_communities_sbm(G, afrc_thr_auto = False, afrc_thr_value = 0):
    # set graph attributes and calculate initial AFRC values
    set_edge_attributes(G)
    afrc_min, afrc_max = get_min_max_afrc_values(G)
    print("min. AFRC value:", afrc_min, " / max. AFRC value:", afrc_max)
    if afrc_thr_auto:
        afrc_threshold = afrc_thr_value
        print("afrc_threshold: ", afrc_threshold)
    else:
        afrc_threshold = int(input("Enter threshold value for AFRC to remove edges with a lower value: "))
    loop_counter = 0
    # # collect edges with minimal negative AFRC
    afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
    afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
    
    while len(afrc_min_list) > 0:      
        if len(afrc_min_list) == 1:
            a = afrc_min_list[0]
        else:
            a = select_an_edge(afrc_min_list)[0]
        (u,v) = a[:2]
        afrc_below_list.remove(a)
        G.remove_edge(u,v)
        affecteds = list(G.edges([u,v]))
        below_edges = [(u,v)  for u,v,d in afrc_below_list]
        # update graph attributes and calculate new AFRC values
        set_edge_attributes(G, affecteds + below_edges)
        afrc_min, afrc_max = get_min_max_afrc_values(G, affecteds + below_edges)
        loop_counter += 1        
        # collect edges with minimal negative AFRC
        afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
        afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
    # determine connected components of graph of edges with positive ARFC
    C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    set_node_blocks(G,C)
    L1, L2 = get_result_list_dict(G,C)        
    return G, L1, L2


# community detection on SBMs (line plots in miniproject)

def evaluate_out_blocks(di, do):
    eval_list = [v  for v in iter(di.values())]
    m = 0
    for i in range(len(do)):
        lo_s = sorted(do[i])
        if lo_s in eval_list: 
            m += 1
    return m/len(di)


# SBM 5 nodes  / 5 communities

sizes = [5, 5, 5, 5, 5]
probs = [[0.80, 0.05, 0.05, 0.05, 0.05], 
          [0.05, 0.80, 0.05, 0.05, 0.05], 
          [0.05, 0.05, 0.80, 0.05, 0.05],
          [0.05, 0.05, 0.05, 0.80, 0.05],
          [0.05, 0.05, 0.05, 0.05, 0.80]]

G = nx.stochastic_block_model(sizes, probs, seed=0)
set_edge_attributes(G)
afrc_min, afrc_max = get_min_max_afrc_values(G)

pos1 = nx.kamada_kawai_layout(G)
for (n,v) in iter(pos1.items()):
    if n in range(5,10): 
        v[0] = v[0]-0.1    
        v[1] = v[1]-0.5
    if n in range(10,15): 
        v[0] = v[0]-0.3    
        v[1] = v[1]+0.3

e1 = list(G.edges())
e2 = [d["afrc"]  for u,v,d in G.edges.data()]
elabels = dict(zip(e1, e2))

plot_my_graph(G, pos1, 
              node_col = [d["block"]  for n,d in G.nodes.data()], 
              edge_lab = elabels, 
              bbox = {"color": "white", "boxstyle": "round", 
                      "ec": (0.5, 0.5, 0.5), "fc": (1.0, 1.0, 0.9)}
              )

show_histos(G)


# Simulate 100 repetitions

steps_size_per_comm = list(range(5,10)) + list(range(10,45,5))
steps_num_of_comm = list(range(2,5)) + list(range(5,25,5))

afrc_thr_for_steps_size_per_com = [-2, -2, -2, -2, -2, -3, -4, -5, -5, -6, -6, -6]
afrc_thr_for_steps_num_of_com =   [-3, -2, -2, -3, -4, -5, -5]


def calculate_accuracy(steps_size, steps_num, steps_afrc_thr, afrc_thr_auto = False):
    accuracies = []
    p_in = 0.70
    p_out = 0.05
    i = 0
    for size_per_comm in steps_size:
        for num_of_comm in steps_num:         
            print("Size: ", size_per_comm, " - Num: ", num_of_comm)
            size = [size_per_comm] * num_of_comm
            prob = build_prob_list(num_of_comm, p_in, p_out)
            G = nx.stochastic_block_model(size, prob, seed=None)
            Dict_in = {bl : sorted([k2  for (k2,v2) in iter(G.nodes.items())  if v2["block"] == bl])  
                       for bl in range(num_of_comm)}
            set_edge_attributes(G)
            afrc_min, afrc_max = get_min_max_afrc_values(G)
            pos_sbm = get_spirals_in_circle_pos(G, 0.3, 1, 0.5, 0.5, num_of_comm, res = 0.5, eq = False)
            # save_pos_sbm(pos_sbm, size_per_comm, num_of_comm)
            # pos_sbm = read_pos_sbm(size_per_comm,num_of_comm)
            show_histos(G,bin_width=2)
            G, List_out, Dict_out = detect_communities_sbm(G, afrc_thr_auto, steps_afrc_thr[i])
            plot_my_graph(G, pos_sbm, node_col = [d["block"]  for n,d in G.nodes.data()],
                          color_map = "tab20", alpha = 0.7)
            accuracy = evaluate_out_blocks(Dict_in, Dict_out)
            print("Accuracy: ", accuracy, "\n")
            accuracies.append(accuracy)
            i += 1
    print("Accuracy: ", accuracies, "\n  Size: ", steps_size, 
          "\n  Num:  ", steps_num,  "\n  AFRC: ", steps_afrc_thr, "\n\n\n")
    return accuracies


l_size_accuracies = []
for j in range(100):
    l_size_accuracies.append(calculate_accuracy(steps_size_per_comm, [10], 
                                            afrc_thr_for_steps_size_per_com, afrc_thr_auto = True))
temp_l_size_acc = list(zip(*l_size_accuracies))   # transpose list of lists
mean_size_accuracies   = [np.mean(v) for v in temp_l_size_acc]
stddev_size_accuracies = [np.std(v)  for v in temp_l_size_acc]

l_num_accuracies = []
for j in range(100):
    l_num_accuracies.append(calculate_accuracy([10], steps_num_of_comm, 
                                           afrc_thr_for_steps_num_of_com, afrc_thr_auto = True))    
temp_l_num_acc = list(zip(*l_num_accuracies))   # transpose list of lists
mean_num_accuracies   = [np.mean(v) for v in temp_l_num_acc]
stddev_num_accuracies = [np.std(v)  for v in temp_l_num_acc]