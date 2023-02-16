"""
auxiliary_functions.py

Created on Feb 12 2023

@author: Lukas

This file contains all auxiliary methods used in the remainder of this repo.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


def save_data_to_json_file(d, filename):
    """
    Save data to json file. Used for saving positions of network nodes.
    Allows repeated identical visualizations of the same network.

    Parameters
    ----------
    d : dict
        Data to be saved.

    filename : str
        Name of the file to be saved.

    Returns
    -------
    None.
    """
    json_string = json.dumps(d, indent = 4)
    json_file = open(filename, "w")
    json_file.write(json_string)
    json_file.close() 
    return None

def read_data_from_json_file (fn):
    """
    Read data from json file. Used for reading positions of network nodes.

    Parameters
    ----------
    fn : str
        Name of the file to be read.

    Returns
    -------
    d : dict
        Data read from file.
    """
    f = open(fn)
    d = json.load(f)
    f.close()
    return d

def pos_array_as_list(p):
    """
    Convert pos dict to list.

    Parameters
    ----------
    p : dict
        Dictionary of node arrays.

    Returns
    -------
    d : dict
        Dictionary of positions as lists.
    """
    d = {k:list(a)  for k,a in iter(p.items())}
    return d

def pos_list_as_array(p):
    """
    Convert pos dict to array.

    Parameters
    ----------
    p : dict
        Dictionary of node lists.

    Returns
    -------
    d : dict
        Dictionary of positions as arrays.
    """
    d = {k:np.array(a)  for k,a in iter(p.items())}
    return d


def build_size_list (k, l):
    """
    Build list of number of nodes per community.

    Parameters
    ----------
    k : int
        Number of nodes.
    
    l : int
        Number of community.

    Returns
    -------
    ll : list
        List of number of nodes per community.
    """
    ll = [k  for i in range(l)]
    return ll

def build_prob_list (l, p_in, p_out):
    """
    Build list of probabilities for SBM.

    Parameters
    ----------
    l : int
        Number of communities.

    p_in : float
        Probability of edge within community.

    p_out : float
        Probability of edge between communities.

    Returns
    -------
    ll : list
        List of lists of probabilities for SBM.
        p_in on the main diagonal, p_out elsewhere.
    """
    ll = []
    for i in range(l):    
        temp_l = [p_out  for j in range(0,i)] + [p_in] + [p_out  for j in range(i+2,l+1)]
        ll.append(temp_l)
    return ll

def get_pos_layout (H, fn = ""):
    """
    Get positions of nodes for network layout.
    If fn is empty, create new Kamada-Kawai layout.
    Otherwise, read positions from file.

    Parameters
    ----------
    H : networkx graph
        Graph to be drawn.

    fn : str, optional
        Name of file containing positions of nodes.

    Returns
    -------
    pos : dict
        Dictionary of positions of nodes.
    """
    if fn == "":
        pos = nx.kamada_kawai_layout(H)
    else:
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        pos = pos_list_as_array(read_data_from_json_file(full_fn))
        pos = {int(k):v  for (k,v) in iter(pos.items())}
    return pos

def save_pos_layout(pos, fn = ""):
    """
    Save positions of nodes for network layout.
    If fn is empty, do nothing.

    Parameters
    ----------
    pos : dict
        Dictionary of positions of nodes.

    fn : str, optional
        Name of file containing positions of nodes.

    Returns
    -------
    None.
    """
    if fn != "":
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        save_data_to_json_file(pos_array_as_list(pos), full_fn)

def set_node_blocks(G,C):
    """
    Set node blocks and colors for network layout.
    Only used for SBM.

    Parameters
    ----------
    G : networkx graph
        Graph to be drawn.

    C : list
        List of communities.

    Returns
    -------
    None.
    """
    for i,c in enumerate(C):
        for u in c:
            G.nodes[u]["block"] = i
            G.nodes[u]["color"] = i

def save_pos_sbm(p,k,n):
    """
    Save positions of nodes for network layout.
    Only used for SBM.

    Parameters
    ----------
    p : dict
        Dictionary of positions of nodes.

    k : int
        Number of nodes in each community.

    n : int
        Number of communities.

    Returns
    -------
    None.
    """
    cwd = os.getcwd()
    fn = "pos_SBM_graph_" + str(k) + "_nodes_in_" +  str(n) + "_communities.json"
    full_fn = os.path.join(cwd, fn)
    save_data_to_json_file(pos_array_as_list(p), full_fn)
    

def read_pos_sbm(k,n):
    """
    Read positions of nodes for network layout.
    Only used for SBM.

    Parameters
    ----------
    k : int
        Number of nodes in each community.

    n : int
        Number of communities.

    Returns
    -------
    p : dict
        Dictionary of positions of nodes.
    """
    cwd = os.getcwd()
    fn = "pos_SBM_graph_" + str(k) + "_nodes_in_" +  str(n) + "_communities.json"
    full_fn = os.path.join(cwd, fn)
    p = pos_list_as_array(read_data_from_json_file(full_fn))
    p = {int(k):v  for (k,v) in iter(p.items())}
    return p