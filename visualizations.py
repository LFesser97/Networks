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


def plot_my_graph(G, pos, ax = None, node_col = "white", 
                  edge_lst = [], edge_col = "lightgrey", edge_lab = {},
                  bbox = None, color_map = "Set3", alpha = 1.0):
    """
    Plot a graph with the given node and edge colors.

    Parameters
    ----------
    G : networkx graph
        The graph to plot.

    pos : dict
        A dictionary with nodes as keys and positions as values.

    ax : matplotlib axis, optional
        The axis to draw the graph on.

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

def set_SBM_node_colors(Gr, cmp_key="block"):                         # Einfärben der Kanten innerhalb und außerhalb eines Blocks
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

def set_SBM_edge_colors(Gr, cmp_key="block"):                         # Einfärben der Kanten innerhalb und außerhalb eines Blocks
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

def show_histo (h_data, title_str, my_bin_num = 40): # NEED TO SPECIFY WHAT THIS FUNCTION IS USED FOR
    """
    Show the histogram for the given data. 

    Parameters
    ----------
    h_data : dict
        The data to show the histogram for.

    title_str : str
        The title to use for the histogram.

    my_bin_num : int, optional
        The number of bins to use. The default is 40.

    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(figsize=(16,10))
    for i,k in enumerate(h_data.keys()):

        bin_lo_lim, bin_hi_lim, bin_width = get_bin_width(h_data[k]["bin_min"], h_data[k]["bin_max"], my_bin_num)
        ax.hist(h_data[k]["curv"], 
                       bins = np.arange(bin_lo_lim, bin_hi_lim + bin_width, bin_width), 
                       edgecolor = "white", 
                       histtype='bar', 
                       stacked=True)

        ax.set_title(h_data[k]["title"])
        ax.title.set_size(16)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(visible=True, axis="both")
    fig.suptitle(title_str, size=16)
    plt.show()  


def show_histos (h_data, title_str, my_nrows = 2, my_ncols = 3, bin_num_lim = 40):
    """
    Show several histograms in a grid.

    Parameters
    ----------
    h_data : dict
        The data to show the histogram for.

    title_str : str
        The title to use for the histogram.

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
                       histtype='bar', 
                       stacked=True)

        axes[r,c].set_title(h_data[k]["title"])
        axes[r,c].title.set_size(16)
        axes[r,c].tick_params(axis='both', labelsize=16)
        axes[r,c].grid(visible=True, axis="both")
    fig.suptitle(title_str, size=16)
    plt.show()    

def show_histos (G, bin_width = 1):
    """
    Show the histograms for the given graph.
    
    Parameters
    ----------
    G : networkx graph
        The graph to show the histograms for.

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
    axes[0].hist(l_frc, bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
    axes[0].set_title("FR curvature")
    axes[0].title.set_size(20)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].grid(visible=True, axis="both")
    axes[1].hist(l_afrc, bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
    axes[1].set_title("Augmented FR curvature")
    axes[1].title.set_size(20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].grid(visible=True, axis="both")
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


def show_line_plots():
    """
    Show the line plots for the given data.
    Used for the community detection experiments to report the accuracy of the community detection algorithm.

    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey = True, figsize=(14,7))
    axes[0].plot(steps_size_per_comm, mean_size_accuracies, "b-o")
    size_std_lower = list(np.array(mean_size_accuracies) - np.array(stddev_size_accuracies))
    size_std_upper = list(np.array(mean_size_accuracies) + np.array(stddev_size_accuracies))
    axes[0].fill_between(steps_size_per_comm, size_std_lower, size_std_upper, color="blue", alpha=0.1)
    axes[0].set_title("AFRC-based algorithm\n for community detection (l=10)")
    axes[0].set_xlabel("size per community (k)")
    axes[0].set_ylabel("mean prediction accuracy")
    axes[0].title.set_size(20)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].xaxis.label.set_size(16)
    axes[0].yaxis.label.set_size(16)
    axes[0].xaxis.set_ticks(np.arange(0, 45, 5))
    axes[0].set(ylim=(0.0, 1.0))    
    axes[0].grid(visible=True, axis="both")
    axes[1].plot(steps_num_of_comm, mean_num_accuracies, "b-o")
    num_std_lower = list(np.array(mean_num_accuracies) - np.array(stddev_num_accuracies))
    num_std_upper = list(np.array(mean_num_accuracies) + np.array(stddev_num_accuracies))
    axes[1].fill_between(steps_num_of_comm, num_std_lower, num_std_upper, color="blue", alpha=0.1)
    axes[1].set_title("AFRC-based algorithm\n for community detection (k=10)")
    axes[1].set_xlabel("number of communities (l)")
    axes[1].set_ylabel("mean prediction accuracy")
    axes[1].title.set_size(20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].xaxis.label.set_size(16)
    axes[1].yaxis.label.set_size(16)
    axes[1].xaxis.set_ticks(np.arange(0, 22, 2))
    axes[1].set(ylim=(0.0, 1.0))    
    axes[1].grid(visible=True, axis="both")
    plt.show()

show_line_plots()


def show_histos (G, title_str, bin_width = 1): # HOW IS THIS DIFFERENT FROM THE 50 OTHER HISTOGRAM FUNCTIONS?
    """
    Show the histograms for the given graph.
    
    Parameters
    ----------
    G : networkx graph
        The graph to show the histograms for.

    title_str : str
        The title of the plot.

    bin_width : int, optional
        The width of the bins. The default is 1.

    Returns
    -------
    None.
    """
    my_nrows = 2
    my_ncols = 2
    l_frc =  [d["frc"]   for u,v,d in G.edges.data()]
    l_afrc = [d["afrc"]  for u,v,d in G.edges.data()]
    l_afrc4 = [d["afrc4"]  for u,v,d in G.edges.data()]
    l_afrc5 = [d["afrc5"]  for u,v,d in G.edges.data()]
    # l_orc = [d["orc"]   for u,v,d in G.edges.data()]
    # bin_data = [[l_frc, l_afrc, l_orc], [l_afrc, l_afrc4, l_afrc5]]
    bin_data = [[l_frc, l_afrc], [l_afrc4, l_afrc5]]
    titles = [["Forman Ricci (FR)", "Augm. FR curv. (triangles)"],
              ["AFR curv. (tri/quad)", "AFR curv. (tri/quad/pent)"]]
    # if MIN_BIN == None:
    #     MIN_BIN = min(min(l_frc), min(l_afrc), min(l_afrc4), min(l_afrc5))
    # if MAX_BIN == None:
    #     MAX_BIN = max(max(l_frc), max(l_afrc), max(l_afrc4), max(l_afrc5))
    min_bin = min(min(l_frc), min(l_afrc), min(l_afrc4), min(l_afrc5))
    max_bin = max(max(l_frc), max(l_afrc), max(l_afrc4), max(l_afrc5))
        
    # print("\n\nMIN_BIN: ", MIN_BIN, " - MAX_BIN: ", MAX_BIN)
    # print("\n\n min_bin: ", min_bin, " - max_bin: ", max_bin)
    # fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, sharex = True, sharey = True, figsize=(14,10))
    fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, sharey = True, figsize=(14,10))
    for r in range(my_nrows):
        for c in range(my_ncols):
            if titles[r][c] != "Ollivier Ricci (OR)":
                if (max_bin - min_bin) > 40:
                    bin_width = (max_bin - min_bin) // 40 + 1
                # axes[r,c].hist(bin_data[r][c], bins = np.arange(MIN_BIN, MAX_BIN + bin_width, bin_width), edgecolor = "white")
                axes[r,c].hist(bin_data[r][c], bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
                # axes[r,c].hist(bin_data[r][c], edgecolor = "white")
                    
            else:
                axes[r,c].hist(bin_data[r][c], bins = 40, edgecolor = "white")
            axes[r,c].set_title(titles[r][c])
            axes[r,c].title.set_size(16)
            axes[r,c].tick_params(axis='both', labelsize=16)
            axes[r,c].grid(visible=True, axis="both")
    fig.suptitle(title_str, size=16)
    # MIN_BIN = None
    # MAX_BIN = None
    plt.show()


""" NEED TO REVISIT THE CODE BELOW THIS LINE """

def calculate_karate_club ():
    # Karate-Club
    print("\n\nGround Truth")
    G = nx.karate_club_graph()
    
    # pos = nx.spring_layout(G)
    cwd = os.getcwd()
    full_fn = os.path.join(cwd, "pos_karate_club_graph.json")
    # save_data_to_json_file(pos_array_as_list(pos), full_fn)
    # pos einlesen und list in np.array konvertieren
    pos = pos_list_as_array(read_data_from_json_file(full_fn))
    # key von pos in integer konvertieren 
    pos = {int(k):v  for (k,v) in iter(pos.items())}
    # pos ist die Basis für alle Karate-Club-plots
    
    # Histos bauen
    set_edge_attributes(G)
    show_histos(G,1)
     
    # ----------------------------------
    # Ground truth für Karate-CLub
    
    G = colored_karate_club_graph()
    plot_my_graph(G, pos, 
                  node_col = [d["color"]  for n,d in G.nodes.data()])
    AFRC_N_edges_before = len(list(G.edges()))
    print("N edges before: ", AFRC_N_edges_before)
    
    # ---------------------------------------
    # AFRC algorithm
    
    print("\n\nAFRC Algorithm")
    G_in = deepcopy(G)
    print("Edges: ",len(list(G_in.edges())))
    C_in = [c for c in sorted(nx.connected_components(G_in), key=len, reverse=True)]
    List_in, Dict_in = get_result_list_dict(G_in,C_in)  
    
    set_edge_attributes(G)
    
    e1 = list(G.edges())
    e2 = [d["afrc"]  for u,v,d in G.edges.data()]
    elabels = dict(zip(e1, e2))
    
    plot_my_graph(G, pos, 
                  node_col = [d["color"]  for n,d in G.nodes.data()], 
                  edge_lab = elabels, 
                  bbox = {"color": "white", "boxstyle": "round", 
                          "ec": (0.5, 0.5, 0.5), "fc": (1.0, 1.0, 0.9)}
                  )
    
    G, List_out, Dict_out = detect_communities(G)
    
    plot_my_graph(G_in, pos, 
                  node_col = [d["color"]  for n,d in G.nodes.data()], 
                  edge_col = ["lightgrey"] * len(list(G_in.edges())))
    
    
    AFRC_N_edges_after = len(list(G.edges()))
    print("N edges before: ", AFRC_N_edges_before, "  N edges after: ", AFRC_N_edges_after)
    
    show_histos(G,1)

""" NEED TO REVISIT THE CODE ABOVE THIS LINE """