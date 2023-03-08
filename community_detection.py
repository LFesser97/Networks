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
import random


# community detection algorithm

def detect_communities(G, curvature, threshold):
    """
    Sequential deletion algorithm for detecting communities in a graph G

    Parameters
    ----------
    G : graph
        A networkx graph

    curvature : str
        The curvature to be used for community detection.

    threshold : float
        The threshold value for the curvature to be used for community detection.

    Returns
    -------
    G : graph
        A networkx graph with the detected communities as node labels.
    """
    # create a copy of the graph
    G_copy = deepcopy(G)

    # set graph attributes and calculate initial AFRC values
    curv_min, curv_max = get_min_max_curv_values(G_copy, curvature)

    # collect edges with extremal negative curvature
    if curvature == "afrc":
        threshold_list = [edge for edge in G_copy.edges.data()  if (edge[2][curvature] > threshold)]
        val_list   = [edge for edge in threshold_list  if (edge[2][curvature] == curv_max)]

    else:
        threshold_list = [edge for edge in G_copy.edges.data()  if (edge[2][curvature] < threshold)]
        val_list   = [edge for edge in threshold_list  if (edge[2][curvature] == curv_min)]

    removed_edges = []
    
    while len(val_list) > 0:      
        if len(val_list) == 1:
            # edge is the only element in the list
            extremum = val_list[0]
        else:
            # edge is randomly chosen from the list
            extremum = select_an_edge(val_list)

        (u,v) = extremum[:2]
        threshold_list.remove(extremum)

        # remove chosen edge
        removed_edges.append((u,v))

        G_copy.remove_edge(u,v)
        affecteds = list(G_copy.edges([u,v]))
        threshold_edges = [(u,v)  for u,v,d in threshold_list]

        # update graph attributes and calculate new curvature values
        if curvature == "frc":
            G_copy.compute_frc(affected_edges = affecteds + threshold_edges)

        elif curvature == "afrc":
            G_copy.compute_afrc(affected_edges = affecteds + threshold_edges)

        elif curvature == "orc":
            G_copy.compute_orc(affected_edges = affecteds + threshold_edges)

        curv_min, curv_max = get_min_max_curv_values(G_copy, curvature, affecteds + threshold_edges)
        
        # collect edges with extremal negative curvature
        if curvature == "afrc":
            threshold_list = [edge for edge in G_copy.edges.data()  if (edge[2][curvature] > threshold)]
            val_list   = [edge for edge in threshold_list  if (edge[2][curvature] == curv_max)]    

        else:
            threshold_list = [edge for edge in G_copy.edges.data()  if (edge[2][curvature] < threshold)]
            val_list   = [edge for edge in threshold_list  if (edge[2][curvature] == curv_min)]

    # determine connected components of graph with edges whose curvature is above threshold
    C = [c for c in sorted(nx.connected_components(G_copy), key=len, reverse=True)]

    # Create list of tupels with node names and cluster labels, set node colors acc to cluster
    set_node_labels(G,C, curvature)
    print("removed edges: ", removed_edges)


# helper functions

cyc_names = {3:"triangles", 4:"quadrangles", 5:"pentagons"}     


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


def get_min_max_curv_values(G, curvature, affected_edges = None):
    """
    Get minimum and maximum values of the curvature

    Parameters
    ----------
    G : graph
        A networkx graph

    curvature : str
        The curvature to be used for community detection.

    affected_edges : list
        List of edges to be considered for calculation of edge attributes

    Returns
    -------
    minimum : int
        Minimum value of curvature

    maximum : int
        Maximum value of curvature
    """
    if affected_edges is None: 
        affected_edges = list(G.edges())

    affected_curvatures = [G.edges[edge][curvature] for edge in affected_edges]
    minimum = min(affected_curvatures)
    maximum = max(affected_curvatures)

    return minimum, maximum


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
    # randomly choose an edge from the list of edges
    edge = random.choice(edge_list)
    
    return edge


def set_node_labels(G, C, curvature):
    """
    Set node labels according to connected component labels

    Parameters
    ----------
    G : graph
        A networkx graph

    C : list
        List of clusters

    curvature : str
        The curvature to be used for community detection.

    Returns
    -------
    G : graph
        A networkx graph with node labels
    """
    for i,c in enumerate(C):
        for u in c:
            G.nodes[u][curvature + "_community"] = i
    #return G


# Non-sequential community detection

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
    #get_edge_curvatures(G, t_coeff, q_coeff)
    # get min,max,values for afrc4 curvature
    #afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4")
    # show histogram of curvature values
    #show_curv_data (G, title_str = "", cmp_key = "block")   # NEED TO IMPORT THIS

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