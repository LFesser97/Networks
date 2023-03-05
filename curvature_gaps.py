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
import sklearn
from sklearn.mixture import GaussianMixture


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
    sum_std = np.sqrt(0.5 * (np.square(c_dict["withins"][curv_name]["std"]) + np.square(c_dict["betweens"][curv_name]["std"])))  
    res_diffs[curv_name] = np.abs((c_dict["withins"][curv_name]["mean"] - c_dict["betweens"][curv_name]["mean"]) / sum_std)   
    
    return res_diffs


def find_threshold(G, curv_name, param, cmp_key):
    """
    Model the curvature distribution with a mixture of two Gaussians.
    Find the midpoint between the means of the two Gaussians.

    Parameters
    ----------
    G : NetworkX
        An undirected graph.

    curv_name : str
        The name of the curvature to be used.

    param : float
        The parameter that determines the threshold. Default is 0.5.

    cmp_key : str
        The key of the node attribute that contains the community assignment. Default is "block".

    Returns
    -------
    threshold : float
        The midpoint between the means of the two Gaussians
    """

    # get the curvature values of edges within and between communities
    # assume that each edge already has the attribute "group" for within/between communities
    c_dict = {"withins": [edge for edge in G.edges if G.edges[edge]["group"] == "within"], 
              "betweens": [edge for edge in G.edges if G.edges[edge]["group"] == "between"]}        
    
    # fit a mixture of two Gaussians to the curvature values of edges within and between communities
    # using GaussianMixture.fit() from sklearn.mixture
    for k in c_dict.keys():
        c_dict[k] = np.array([G.edges[edge][curv_name] for edge in c_dict[k]]).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=0).fit(np.concatenate((c_dict["withins"], c_dict["betweens"])))

    # find the midpoint between the means of the two Gaussians
    threshold = param * (gmm.means_[0] + gmm.means_[1])

    return threshold