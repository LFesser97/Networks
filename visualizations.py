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


def plot_my_graph(G,
                  pos,
                  node_col,
                  edge_lst,
                  edge_col,
                  edge_lab,
                  bbox,
                  color_map,
                  alpha):
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
    fig = plt.figure(figsize=(15, 15))
    # nx.draw_networkx (G, pos, **options)
    nx.draw_networkx(G, pos, node_color=node_col,
                     edge_color=edge_col, **node_options)

    nx.draw_networkx_edges(G, pos, edge_lst,
                           edge_color=edge_col, **edge_options)

    nx.draw_networkx_edge_labels(G, pos, label_pos=0.5,
                                 edge_labels=edge_lab,
                                 rotate=False, bbox=bbox)
    plt.gca().margins(0.20)
    plt.show()


def get_bin_width(b_min, b_max, num_bin_lim):
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


def plot_curvature_hist_colors(h_data, title_str,
                               x_axis_str, y_axis_str, my_bin_num=40):
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
    fig, ax = plt.subplots(figsize=(16, 10))

    # get the smallest and largest values in the data
    min_0 = min(h_data[0])
    min_1 = min(h_data[1])
    min_val = min(min_0, min_1)

    max_0 = max(h_data[0])
    max_1 = max(h_data[1])
    max_val = max(max_0, max_1)

    bin_lo_lim, bin_hi_lim, bin_width = get_bin_width(
        min_val, max_val, my_bin_num)
    ax.hist(h_data,
            bins=np.arange(bin_lo_lim, bin_hi_lim + bin_width, bin_width),
            edgecolor="white",
            histtype='stepfilled',
            alpha=0.7)

    # ax.set_title(title_str)
    ax.title.set_size(16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(visible=True, axis="both")
    ax.set_xlabel(x_axis_str, fontsize=16)
    ax.set_ylabel(y_axis_str, fontsize=16)
    fig.suptitle(title_str, size=20)
    plt.show()


def plot_curvature_hist(curv_list, title, x_axis_str, y_axis_str):
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
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(curv_list, bins=40, edgecolor="white")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_axis_str, fontsize=20)
    ax.set_ylabel(y_axis_str, fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(visible=True, axis="both")
    plt.show()


def plot_curvature_differences(G, curvature_difference):
    """
    Plot the difference between two curvatures at an edge level.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The graph to show the curvature differences for.

    curvature_difference : str
        The curvature difference to show.

    Returns
    -------
    None.
        Plots the graph with the curvature differences colore-coded.
        Negative values are colored red and positive values are colored green,
        with the intensity of the color representing the magnitude of the difference.
    """
    try:
        # create a list of the curvature differences
        curv_diff_list = [edge[2][curvature_difference] for edge in G.edges.data()]
    
        # get the smallest and largest values in the data
        min_val = min(curv_diff_list)
        max_val = max(curv_diff_list)

        # create a colormap
        cmap = plt.cm.get_cmap('RdBu')
        norm = plt.Normalize(vmin=min_val, vmax=max_val)

        # create a list of colors for each edge
        colors = [cmap(norm(value)) for value in curv_diff_list]

        # plot the graph with the edges colored using plot_my_graph
        plot_my_graph(G, pos=G.pos, 
                      node_col="white", edge_lst=[],
                      edge_col=colors, edge_lab={},
                      bbox=None, color_map="Set3",
                      alpha=1.0)

    except KeyError:
        print("This curvature difference has not been calculated for this graph.")


def plot_clustering_accuracy(clustering_accuracy, x_axis, title=''):
    """
    Plot the clustering accuracy given a list of accuracy values.

    Parameters
    ----------
    clustering_accuracy : dict[float, list[float]]
        The clustering accuracy values to plot.

    x_axis : str
        The x-axis label.

    title : str, optional
        The title to use for the plot. The default is ''.

    Returns
    -------
    None.
        Plots the clustering accuracy.
    """
    # for each key in the dictionary, compute the mean and standard deviation
    # of the values in the list
    mean_list = []
    std_list = []

    for key in clustering_accuracy:
        mean_list.append(np.mean(clustering_accuracy[key]))
        std_list.append(np.std(clustering_accuracy[key]))

    # plot the mean as a line and the standard deviation as a shaded area
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(clustering_accuracy.keys(), mean_list, color="blue")
    ax.fill_between(clustering_accuracy.keys(),
                    np.array(mean_list) - np.array(std_list),
                    np.array(mean_list) + np.array(std_list),
                    color="blue", alpha=0.2)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_axis, fontsize=16)
    ax.set_ylabel("Mean Prediction Accuracy", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    plt.show()