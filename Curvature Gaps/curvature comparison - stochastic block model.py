# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:20:55 2022

@author: Ralf
"""

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
    # plt.axis("off")
    plt.show()


# save data as json file
def save_data_to_json_file(d, filename):
    json_string = json.dumps(d, indent = 4)
    json_file = open(filename, "w")
    json_file.write(json_string)
    json_file.close() 
    return None

# read data from json file
def read_data_from_json_file (fn):
    f = open(fn)
    d = json.load(f)
    f.close()
    return d

# convert pos ndarray to list
def pos_array_as_list(p):
    d = {k:list(a)  for k,a in iter(p.items())}
    return d

# convert list to pos ndarray
def pos_list_as_array(p):
    d = {k:np.array(a)  for k,a in iter(p.items())}
    return d



def all_triangles_of_edge(G, e):
    u,v = e
    tr_nodes = [n  for n in list(G[u])  if n in list(G[v])]
    return tr_nodes


def fr_curvature (G, ni, nj):
    return 4 - G.degree(ni) - G.degree(nj)


def afr_curvature (G, ni, nj, m):
    return 4 - G.degree(ni) - G.degree(nj) + 3*m


def set_edge_attributes(G,ae=None):
    if ae == None: 
        ae = list(G.edges())
    for (u,v) in ae:
        G.edges[u,v]["triangles"] = m = len(all_triangles_of_edge(G,(u,v)))
        G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, m)


def get_min_max_afrc_values(G, ae=None):
    if ae == None: 
        ae = list(G.edges())
    a = len(G.nodes())
    b = -len(G.nodes())
    for (u,v) in ae:
        a = min(a, G.edges[u,v]["afrc"])
        b = max(b, G.edges[u,v]["afrc"])
    return a, b


# def select_an_edge(edge_list):
#     def find_interval(x, partition):
#         for i in range(0, len(partition)):
#             if x < partition[i]:
#                 return i-1
#         return -1
    
#     def weighted_choice(sequence, weights):
#         # random float between 0 and 1
#         x = np.random.random()    
#         # list of cumulated weights resp. probabilities
#         cum_weights = [0] + list(np.cumsum(weights))   
#         # determine index based on cumulated probabilities
#         index = find_interval(x, cum_weights)
#         # return element of sequence matching the index
#         return sequence[index]          
    
#     # use uniform probabiliity distribution to select one of the edges
#     act_weights = [1.0 / len(edge_list)] * len(edge_list)  
#     # return randomly chosen element of edge list 
#     return [weighted_choice(edge_list, act_weights)]


def show_histos (G, bin_width = 1):
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


# def show_histo (G, bin_width = 1):
#     l_afrc = [d["afrc"]  for u,v,d in G.edges.data()]
#     min_bin = min(l_afrc)
#     max_bin = max(l_afrc)
#     print("min_bin: ", min_bin, " - max_bin: ", max_bin)
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
#     # ax.hist(l_afrc, bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
#     ax.hist(l_afrc, bins = np.arange(min_bin, max_bin + bin_width, bin_width))
#     # ax.hist(l_afrc)
#     ax.set_title("Histogram of augmented FR curvature")
#     ax.set_ylabel("Frequency")
#     # ax.title.set_size(20)
#     # ax.tick_params(axis='both', labelsize=16)
#     # ax.grid(visible=True, axis="both")
#     plt.show()


# def label_nodes(G, C):
#     for i,c in enumerate(C):
#         print(i,c)
#         for u in range(i*c, (i+1)*c):
#             G.nodes[u]["cluster"] = i
#     return G


# def label_edges(G):
#     pass


def build_prob_list(n, p_in, p_out):
    ll = []
    for i in range(n):    
        temp_l = [p_out  for j in range(0,i)] + [p_in] + [p_out  for j in range(i+2,n+1)]
        ll.append(temp_l)
    return ll


# def set_node_blocks(G,C):
#     for i,c in enumerate(C):
#         for u in c:
#             G.nodes[u]["block"] = i
#             G.nodes[u]["color"] = i
    

# def get_result_list_dict(G,C):
#     ll = []
#     ld = {}
#     for i,c in enumerate(C):
#         ld[i] = []
#         for u in sorted(c):
#             ll.append((u,i))
#             ld[i].append(u)
#     return ll, ld


# def get_circles_in_circle_pos(G, sc, rd, cx, cy, mv):
#     p = {}
#     cnt = np.array([cx, cy])
#     for v in range(mv):
#         temp = nx.circular_layout(
#                     nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["block"] == v]),
#                     scale = sc,
#                     center = cnt + np.array([rd * np.cos(v/mv*2*np.pi),
#                                              rd * np.sin(v/mv*2*np.pi)])
#                     )
#         p.update(temp)
#     return p
    

# def get_spirals_in_circle_pos(G, sc, rd, cx, cy, mv, res = 0.35, eq = False):
#     p = {}
#     cnt = np.array([cx, cy])
#     for v in range(mv):
#         temp = nx.spiral_layout(
#                     nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["block"] == v]),
#                     scale = sc,
#                     center = cnt + np.array([rd * np.cos(v/mv*2*np.pi),
#                                              rd * np.sin(v/mv*2*np.pi)]),
#                     resolution = res,
#                     equidistant = eq
#                     )
#         p.update(temp)
#     return p


# def detect_communities_sbm(G, afrc_thr_auto = False, afrc_thr_value = 0):
#     # set graph attributes and calculate initial AFRC values
#     set_edge_attributes(G)
#     afrc_min, afrc_max = get_min_max_afrc_values(G)
#     print("min. AFRC value:", afrc_min, " / max. AFRC value:", afrc_max)
#     if afrc_thr_auto:
#         afrc_threshold = afrc_thr_value
#         print("afrc_threshold: ", afrc_threshold)
#     else:
#         afrc_threshold = int(input("Enter threshold value for AFRC to remove edges with a lower value: "))
#     loop_counter = 0
#     # # collect edges with minimal negative AFRC
#     afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
#     afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
    
#     while len(afrc_min_list) > 0:      
#         if len(afrc_min_list) == 1:
#             a = afrc_min_list[0]
#         else:
#             a = select_an_edge(afrc_min_list)[0]
#         (u,v) = a[:2]
#         afrc_below_list.remove(a)
#         G.remove_edge(u,v)
#         affecteds = list(G.edges([u,v]))
#         below_edges = [(u,v)  for u,v,d in afrc_below_list]
#         # update graph attributes and calculate new AFRC values
#         set_edge_attributes(G, affecteds + below_edges)
#         afrc_min, afrc_max = get_min_max_afrc_values(G, affecteds + below_edges)
#         loop_counter += 1        
#         # collect edges with minimal negative AFRC
#         afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
#         afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
#     # determine connected components of graph of edges with positive ARFC
#     C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
#     set_node_blocks(G,C)
#     L1, L2 = get_result_list_dict(G,C)        
#     return G, L1, L2

 

# def save_pos_sbm(p,s,n):
#     cwd = os.getcwd()
#     fn = "pos_SBM_graph_" + str(s) + "_nodes_in_" +  str(n) + "_communities.json"
#     full_fn = os.path.join(cwd, fn)
#     save_data_to_json_file(pos_array_as_list(p), full_fn)
    

# def read_pos_sbm(s,n):
#     cwd = os.getcwd()
#     fn = "pos_SBM_graph_" + str(s) + "_nodes_in_" +  str(n) + "_communities.json"
#     full_fn = os.path.join(cwd, fn)
#     # pos einlesen und list in np.array konvertieren
#     p = pos_list_as_array(read_data_from_json_file(full_fn))
#     # key von pos in integer konvertieren 
#     p = {int(k):v  for (k,v) in iter(p.items())}
#     return p


# def evaluate_out_blocks(di, do):
#     eval_list = [v  for v in iter(di.values())]
#     m = 0
#     for i in range(len(do)):
#         lo_s = sorted(do[i])
#         if lo_s in eval_list: 
#             m += 1
#     return m/len(di)
   

#-----------------------------------------------------------

# TEST Graph
# sizes = [10, 10, 10]
# probs = [[0.5, 0.02, 0.03], [0.02, 0.5, 0.01], [0.03, 0.01, 0.50]]

# -----------------------------------------------------------

# Fig 1a  Fig 1c  auf Seite 2  im Sia-Paper reproduzieren  (bei 1c  sowohl FRC als auch AFRC plotten)

### DANACH   !!!
# FIg 2b und 2c  auf Seite 3  im Sia-Paper reproduzieren

### ZUERST !!!
# Fig 4 auf Seite 4 "Karate-Club"  reproduzieren :  aber hier GroundTruth  vs  AFRC
# Ground Truth visualisieren,, dann Ergebnis unseres Alg. (AFRC-Output) visualisieren, 
# dann Output der Louvain-Methode visualisieren, zusätzlich Histo der AFRC-Werte

# -----------------------------------------------

# SBM 5 Knoten / 5 Communities

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



# ---------------------------------------

# # Fig 2b / 2c

# steps_size_per_comm = list(range(5,10)) + list(range(10,45,5))
# steps_num_of_comm = list(range(2,5)) + list(range(5,25,5))

# afrc_thr_for_steps_size_per_com = [-2, -2, -2, -2, -2, -3, -4, -5, -5, -6, -6, -6]
# afrc_thr_for_steps_num_of_com =   [-3, -2, -2, -3, -4, -5, -5]


# def calculate_accuracy(steps_size, steps_num, steps_afrc_thr, afrc_thr_auto = False):
#     accuracies = []
#     p_in = 0.70
#     p_out = 0.05
#     i = 0
#     for size_per_comm in steps_size:
#         for num_of_comm in steps_num:         
#             print("Size: ", size_per_comm, " - Num: ", num_of_comm)
#             size = [size_per_comm] * num_of_comm
#             prob = build_prob_list(num_of_comm, p_in, p_out)
#             G = nx.stochastic_block_model(size, prob, seed=None)
#             Dict_in = {bl : sorted([k2  for (k2,v2) in iter(G.nodes.items())  if v2["block"] == bl])  
#                        for bl in range(num_of_comm)}
#             set_edge_attributes(G)
#             afrc_min, afrc_max = get_min_max_afrc_values(G)
#             pos_sbm = get_spirals_in_circle_pos(G, 0.3, 1, 0.5, 0.5, num_of_comm, res = 0.5, eq = False)
#             # save_pos_sbm(pos_sbm, size_per_comm, num_of_comm)
#             # pos_sbm = read_pos_sbm(size_per_comm,num_of_comm)
#             show_histos(G,bin_width=2)
#             G, List_out, Dict_out = detect_communities_sbm(G, afrc_thr_auto, steps_afrc_thr[i])
#             plot_my_graph(G, pos_sbm, node_col = [d["block"]  for n,d in G.nodes.data()],
#                           color_map = "tab20", alpha = 0.7)
#             accuracy = evaluate_out_blocks(Dict_in, Dict_out)
#             print("Accuracy: ", accuracy, "\n")
#             accuracies.append(accuracy)
#             i += 1
#     print("Accuracy: ", accuracies, "\n  Size: ", steps_size, 
#           "\n  Num:  ", steps_num,  "\n  AFRC: ", steps_afrc_thr, "\n\n\n")
#     return accuracies

# l_size_accuracies = []
# for j in range(100):
#     l_size_accuracies.append(calculate_accuracy(steps_size_per_comm[:8], [10], 
#                                             afrc_thr_for_steps_size_per_com[:8], afrc_thr_auto = True))
# temp_l_size_acc = list(zip(*l_size_accuracies))   # transpose list of lists
# mean_size_accuracies   = [np.mean(v) for v in temp_l_size_acc]
# stddev_size_accuracies = [np.std(v)  for v in temp_l_size_acc]

# l_num_accuracies = []
# for j in range(100):
#     l_num_accuracies.append(calculate_accuracy([10], steps_num_of_comm, 
#                                            afrc_thr_for_steps_num_of_com, afrc_thr_auto = True))    
# temp_l_num_acc = list(zip(*l_num_accuracies))   # transpose list of lists
# mean_num_accuracies   = [np.mean(v) for v in temp_l_num_acc]
# stddev_num_accuracies = [np.std(v)  for v in temp_l_num_acc]

# mean_size_accuracies_2 = mean_size_accuracies.copy()
# stddev_size_accuracies_2 = stddev_size_accuracies.copy()
# mean_num_accuracies_2 = mean_num_accuracies.copy()
# stddev_num_accuracies2 = stddev_num_accuracies.copy()

# mean_size_accuracies = mean_size_accuracies_2
# stddev_num_accuracies = stddev_num_accuracies[0:-1]

# print(len(steps_size_per_comm), len(mean_size_accuracies), len(stddev_size_accuracies), "\n", 
#       len(steps_num_of_comm), len(mean_num_accuracies), len(stddev_num_accuracies))

# mean_size_accuracies.append(1.0)
# stddev_size_accuracies.append(0.0)





# def show_line_plots():
#     accuracies_for_size_steps = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
#     accuracies_for_num_steps = [1.0, 1.0, 1.0, 1.0, 0.8, 0.5333333333333333, 0.35]


#     fig, axes = plt.subplots(nrows=1, ncols=2, sharey = True, figsize=(14,7))
    
#     axes[0].plot(steps_size_per_comm, mean_size_accuracies, "b-o")
#     size_std_lower = list(np.array(mean_size_accuracies) - np.array(stddev_size_accuracies))
#     size_std_upper = list(np.array(mean_size_accuracies) + np.array(stddev_size_accuracies))
#     axes[0].fill_between(steps_size_per_comm, size_std_lower, size_std_upper, color="blue", alpha=0.1)
    
#     axes[0].set_title("AFRC-based algorithm\n for community detection (l=10)")
#     axes[0].set_xlabel("size per community (k)")
#     axes[0].set_ylabel("mean prediction accuracy")
#     axes[0].title.set_size(20)
#     axes[0].tick_params(axis='both', labelsize=16)
#     axes[0].xaxis.label.set_size(16)
#     axes[0].yaxis.label.set_size(16)
#     axes[0].xaxis.set_ticks(np.arange(0, 45, 5))
#     axes[0].set(ylim=(0.0, 1.0))    
#     axes[0].grid(visible=True, axis="both")
    
#     axes[1].plot(steps_num_of_comm, mean_num_accuracies, "b-o")
#     num_std_lower = list(np.array(mean_num_accuracies) - np.array(stddev_num_accuracies))
#     num_std_upper = list(np.array(mean_num_accuracies) + np.array(stddev_num_accuracies))
#     axes[1].fill_between(steps_num_of_comm, num_std_lower, num_std_upper, color="blue", alpha=0.1)
    
#     axes[1].set_title("AFRC-based algorithm\n for community detection (k=10)")
#     axes[1].set_xlabel("number of communities (l)")
#     axes[1].set_ylabel("mean prediction accuracy")
#     axes[1].title.set_size(20)
#     axes[1].tick_params(axis='both', labelsize=16)
#     axes[1].xaxis.label.set_size(16)
#     axes[1].yaxis.label.set_size(16)
#     axes[1].xaxis.set_ticks(np.arange(0, 22, 2))
#     axes[1].set(ylim=(0.0, 1.0))    
#     axes[1].grid(visible=True, axis="both")
#     plt.show()


# show_line_plots()



def build_histos_param_variation():

    


# -----------------------------------------





