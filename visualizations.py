"""
visualizations.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods to visualize curvature results/ graphs.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


def plot_my_graph(G, pos, node_col, edge_lst, edge_col, edge_lab, bbox, color_map, alpha):
    """
    Plot a graph with the given node and edge colors.

    Parameters
    ----------
    G : networkx graph
        The graph to plot.

    pos : dict
        A dictionary with nodes as keys and positions as values.

    node_col : list, optional
        A list of node colors.

    edge_lst : list, optional
        A list of edges to draw.

    edge_col : list, optional
        A list of edge colors.

    edge_lab : dict, optional
        A dictionary with edges as keys and labels as values.

    bbox : dict, optional
        A dictionary with edge labels as keys and bounding boxes as values.

    color_map : str, optional
        The name of the color map to use.

    alpha : float, optional
        The alpha value to use for the nodes.

    Returns
    -------
    None.
    """
    node_options = {
        "font_size": 12, 
        "font_color": "black",
        "node_size": 300, 
        "cmap": plt.get_cmap(color_map),
        "alpha": alpha,
        "edgecolors": "black",
        "linewidths": 0.5,   
        "with_labels": True,
        "edgelist": None
        }
    edge_options = {
        "width": 0.5
        }
    fig = plt.figure(figsize=(15,15))
    # nx.draw_networkx (G, pos, **options)
    nx.draw_networkx (G, pos, node_color = node_col, edge_color = edge_col, **node_options)
    nx.draw_networkx_edges (G, pos, edge_lst, edge_color = edge_col, **edge_options)
    nx.draw_networkx_edge_labels(G, pos, label_pos = 0.5, 
                                 edge_labels = edge_lab, rotate=False,
                                 bbox = bbox)
    plt.gca().margins(0.20)
    plt.show()


def set_edge_colors(G,a,thr=0):
    """
    Set the edge colors for the given graph.

    Parameters
    ----------
    G : networkx graph
        The graph to set the edge colors for.

    a : float
        The value to compare the edge attributes to.

    thr : float, optional
        The threshold to use for the edge colors. The default is 0.

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

def set_SBM_node_colors(Gr, cmp_key="block"):                         
    """
    Set the node colors for the given graph.

    Parameters
    ----------
    Gr : networkx graph
        The graph to set the node colors for.

    cmp_key : str, optional
        The key to use for the node colors. The default is "block".

    Returns
    -------
    None.
    """
    for n,d in Gr.nodes.data():
        d["color"] = d[cmp_key] 
    return Gr

def set_SBM_edge_colors(Gr, cmp_key="block"):                         
    """
    Set the edge colors for the given graph.

    Parameters
    ----------
    Gr : networkx graph
        The graph to set the edge colors for.

    cmp_key : str, optional
        The key to use for the edge colors. The default is "block".

    Returns
    -------
    None.
    """
    for u,v,d in Gr.edges.data():
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:
            d["color"] = "grey" 
        else:
            d["color"] = "orange" 

def color_partition_nodes(G, part):
    """
    Color the nodes of the given graph according to the given partition.

    Parameters
    ----------
    G : networkx graph
        The graph to color the nodes of.

    part : list
        The partition to use for coloring the nodes.

    Returns
    -------
    G : networkx graph
        The graph with colored nodes.
    """
    for i,p in enumerate(part):
        for u in p:
            G.nodes[u]["color"] = i
    return G

def colored_karate_club_graph():
    """
    Create a karate club graph with colored nodes.

    Returns
    -------
    G : networkx graph
        The karate club graph with colored nodes.
    """
    G = nx.karate_club_graph()
    for n,d in G.nodes.data(): 
        if d["club"] == "Mr. Hi":
            d["color"] = 0
        else:
            d["color"] = 1
    return G

def get_bin_width (b_min, b_max, num_bin_lim):
    """
    Get the bin width for the given bin limits.

    Parameters
    ----------
    b_min : float
        The minimum bin value.

    b_max : float
        The maximum bin value.

    num_bin_lim : int
        The number of bins to use.

    Returns
    -------
    b_width : float
        The bin width.

    b_lo_lim : float
        The lower bin limit.

    b_hi_lim : float
        The upper bin limit.
    """
    scaling = 1
    multiplier = 10
    b_width = (b_max - b_min) // 40 + 1
    if abs(b_max) < 1 and abs(b_min) < 1:
        while (b_max - b_min)/scaling < num_bin_lim / 10:
            scaling /= multiplier    
        b_width = scaling
    if b_width < 1:   # for orc
        b_lo_lim = np.floor(b_min / b_width) * b_width
        b_hi_lim = np.floor(b_max / b_width) * b_width
        while (b_max - b_min) / b_width < num_bin_lim / 2:
            b_width /= 2
    else:     # for other curvatures
        b_lo_lim = b_min
        b_hi_lim = b_max             
    return b_lo_lim, b_hi_lim, b_width


# changed the name of this
# def show_histo(h_data, title_str, my_bin_num = 40):
def plot_curvature_hist_colors(h_data, title_str, 
                               x_axis_str, y_axis_str, my_bin_num = 40):
    """
    Show the histogram for the given data. 

    Parameters
    ----------
    h_data : dict
        The data to show the histogram for.

    title_str : str
        The title to use for the histogram.

    x_axis_str : str
        The label to use for the x-axis.

    y_axis_str : str
        The label to use for the y-axis.

    my_bin_num : int, optional
        The number of bins to use. The default is 40.

    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(figsize=(16,10))
    # bin_lo_lim, bin_hi_lim, bin_width = get_bin_width(h_data[k]["bin_min"], h_data[k]["bin_max"], my_bin_num)

    # get the smallest and largest values in the data
    min_0 = min(h_data[0])
    min_1 = min(h_data[1])
    min_val = min(min_0, min_1)

    max_0 = max(h_data[0])
    max_1 = max(h_data[1])
    max_val = max(max_0, max_1)

    print("min_val = ", min_val)
    print("max_val = ", max_val)

    bin_lo_lim, bin_hi_lim, bin_width = get_bin_width(min_val, max_val, my_bin_num)
    ax.hist(h_data,  # used to be h_data[k]["curv"]
                    bins = np.arange(bin_lo_lim, bin_hi_lim + bin_width, bin_width), 
                    edgecolor = "white", 
                    histtype='stepfilled', 
                    stacked=True)

    #ax.set_title(title_str)
    ax.title.set_size(16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(visible=True, axis="both")
    ax.set_xlabel(x_axis_str, fontsize=16)
    ax.set_ylabel(y_axis_str, fontsize=16)
    fig.suptitle(title_str, size=20)
    plt.show()  


def show_histos (h_data, title_str = "No Title", x_axis_str = "X-Axis", y_axis_str = "Y-Axis", 
                 my_nrows = 2, my_ncols = 3, bin_num_lim = 40, fixed_x_axis=False):
    """
    Show several histograms in a grid.

    Parameters
    ----------
    h_data : dict
        The data to show the histogram for.

    title_str : str
        The title to use for the histogram.
     
    x_axis_str : str
        The label to use for the x-axis.

    y_axis_str : str
        The label to use for the y-axis.

    my_nrows : int, optional
        The number of rows to use. The default is 2.

    my_ncols : int, optional
        The number of columns to use. The default is 3.
    
    bin_num_lim : int, optional
        The number of bins to use. The default is 40.

    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, sharey = True, figsize=(16,10))
    for i,k in enumerate(h_data.keys()):
        r = i // my_ncols
        c = i % my_ncols
        
        bin_lo_lim, bin_hi_lim, bin_width = get_bin_width(h_data[k]["bin_min"], h_data[k]["bin_max"], bin_num_lim)
        axes[r,c].hist(h_data[k]["curv"], 
                       bins = np.arange(bin_lo_lim, bin_hi_lim + bin_width, bin_width), 
                       edgecolor = "white", 
                       histtype='stepfilled', 
                       stacked=True,
                       alpha = 0.5)

        axes[r,c].set_title(h_data[k]["title"])
        axes[r,c].title.set_size(16)
        axes[r,c].tick_params(axis='both', labelsize=16)
        axes[r,c].grid(visible=True, axis="both")
        axes[r,c].set_xlabel(x_axis_str, fontsize=16)
        axes[r,c].set_ylabel(y_axis_str, fontsize=16)
        if fixed_x_axis:
            axes[r,c].set_xlim((-1, 1))
    fig.suptitle(title_str, size=20)
    plt.show()    
    

def show_histos (G, x_axis_str = "X-Axis", y_axis_str = "Y-Axis", bin_width = 1):
    """
    Show the histograms for the given graph.
    
    Parameters
    ----------
    G : networkx graph
        The graph to show the histograms for.
        
    x_axis_str : str
        The label to use for the x-axis.

    y_axis_str : str
        The label to use for the y-axis.

    bin_width : int, optional
        The bin width to use. The default is 1.
        
    Returns
    -------
    None.
    """
    l_frc =  [d["frc"]   for u,v,d in G.edges.data()]
    l_afrc = [d["afrc"]  for u,v,d in G.edges.data()]
    min_bin = min(min(l_frc), min(l_afrc))
    max_bin = max(max(l_frc), max(l_afrc))
    print("min_bin: ", min_bin, " - max_bin: ", max_bin)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex = True, sharey = True, figsize=(14,7))
    axes[0].hist(l_frc, 
                 bins = np.arange(min_bin, max_bin + bin_width, bin_width), 
                 edgecolor = "white")
    axes[0].set_title("FR curvature")
    axes[0].title.set_size(20)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].grid(visible=True, axis="both")
    axes[0].set_xlabel(x_axis_str, fontsize=16)
    axes[0].set_ylabel(y_axis_str, fontsize=16)
    axes[1].hist(l_afrc, 
                 bins = np.arange(min_bin, max_bin + bin_width, bin_width), 
                 edgecolor = "white")
    axes[1].set_title("Augmented FR curvature")
    axes[1].title.set_size(20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].grid(visible=True, axis="both")
    axes[1].set_xlabel(x_axis_str, fontsize=16)
    axes[1].set_ylabel(y_axis_str, fontsize=16)
    plt.show()


def show_curv_min_max_values (h_data): # NEED TO SPECIFY WHAT THIS FUNCTION DOES
    """
    Show the min/max curvature values for the given data.

    Parameters
    ----------
    h_data : dict
        The data to show the min/max values for.

    Returns
    -------
    None.
    """
    print("\nMin/Max Curvature values:")
    for k in h_data.keys():
        print(str(k).ljust(8), 
              "{0:<5s} {1:7.3f}".format("Min:", h_data[k]["bin_min"]), "  ",
              "{0:<5s} {1:7.3f}".format("Max:", h_data[k]["bin_max"])
              )
    print()


def plot_curvature_hist(curv_list, title = "No Title", x_axis_str = "X-Axis", y_axis_str = "Y-Axis"):
    """
    Plot histogram of curvature values

    Parameters
    ----------
    curv_list : list
        List of curvature values.
        
    title : str
        The title to use for the histogram.

    x_axis_str : str
        The label to use for the x-axis.

    y_axis_str : str
        The label to use for the y-axis.

    Returns
    -------
    None.
        Plots the histogram.
    """
    fig, ax = plt.subplots(figsize=(14,10))
    ax.hist(curv_list, bins=40, edgecolor="white")
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel(x_axis_str, fontsize=20)
    ax.set_ylabel(y_axis_str, fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(visible=True, axis="both")
    plt.show()