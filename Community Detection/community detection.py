# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:43:51 2022

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


def test_triangles(G):
    for e in G.edges(): 
        atr = all_triangles_of_edge(G,e)
        print(e, "  ", len(atr), "  ", atr)


def fr_curvature (G, ni, nj):
    '''
    computes the Forman-Ricci curvature of a given edge 
    
    Parameters
    ----------
    G : Graph
    ni : node i
    nj : node j

    Returns
    -------
    frc : int
        Forman Ricci curvature of the edge connecting nodes i and j

    '''
    frc = 4 - G.degree(ni) - G.degree(nj)
    return frc 


def afr_curvature (G, ni, nj, m):
    '''
    computes the Augmented Forman-Ricci curvature of a given edge 
    
    Parameters
    ----------
    G : Graph
    ni : node i
    nj : node j
    m : number of triangles containing the edge between node i and j

    Returns
    -------
    afrc : int
        Forman Ricci curvature of the edge connecting nodes i and j   
    '''
    afrc = 4 - G.degree(ni) - G.degree(nj) + 3*m
    return afrc


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


           
def select_an_edge(edge_list):
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


def set_edge_colors(G,a,thr=0):
    for u,v,d in G.edges.data():
        if d["afrc"] >= thr:
            d["color"] = "green" 
        else:
            d["color"] = "darkred" 
        if (a < thr) and (d["afrc"] == a):
            d["color"] = "red"    


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


def set_node_labels(G,C):
    for i,c in enumerate(C):
        for u in c:
            # G.nodes[u]["cluster"] = i
            G.nodes[u]["color"] = i
    return G
    

def get_result_list_dict(G,C):
    ll = []
    ld = {}
    for i,c in enumerate(C):
        ld[i] = []
        for u in c:
            ll.append((u,i))
            ld[i].append(u)
    return ll, ld


def color_partition_nodes(G, part):
    for i,p in enumerate(part):
        for u in p:
            G.nodes[u]["color"] = i
    return G
    

def colored_karate_club_graph():
    G = nx.karate_club_graph()
    for n,d in G.nodes.data(): 
        if d["club"] == "Mr. Hi":
            d["color"] = 0
        else:
            d["color"] = 1
    return G


def detect_communities(G):
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

    # set graph attributes and calculate initial AFRC values
    set_edge_attributes(G)
    afrc_min, afrc_max = get_min_max_afrc_values(G)
    set_edge_colors(G,afrc_min)     
    # show histogram of curvature values
    show_histos(G)    
    print("min. AFRC value:", afrc_min, " / max. AFRC value:", afrc_max)
    afrc_threshold = int(input("Enter threshold value for AFRC to remove edges with a lower value: "))
    # # plot graph and print minimum afrc value for control purposes
    # plot_my_graph(G, pos, edge_col = [d["color"]  for u,v,d in G.edges.data()])
    loop_counter = 0
    # print("loop_counter: ", loop_counter)    
    # print("new afrc_min value: ", afrc_min)
    
    # # collect edges with minimal negative AFRC
    afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
    afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
    
    # print("potential edges to remove: ")
    # for af in afrc_min_list: print(af)
    # # while there is at least one edge with negative AFRC
    while len(afrc_min_list) > 0:      
        if len(afrc_min_list) == 1:
            # edge is the only element in the list
            a = afrc_min_list[0]
        else:
            # edge is randomly chosen from the list
            a = select_an_edge(afrc_min_list)[0]
        # select first part of edge tuple 
        (u,v) = a[:2]
        # remove it from afrc_below_list, the remaining edges will be considered further
        afrc_below_list.remove(a)
        # print("chosen: ",a)
        # remove chosen edge
        G.remove_edge(u,v)
        # print("removed: ", (u,v))
        # determine those edges that either start or end at one of the removed edge's nodes
        affecteds = list(G.edges([u,v]))
        # print("affected edges: ", affecteds)
        # update attributes of these edges, include all edges to determine new afrc_min
        below_edges = [(u,v)  for u,v,d in afrc_below_list]
        # print("remaining egdes below threshold ", afrc_threshold, "\n", below_edges)     
        # update graph attributes and calculate new AFRC values
        set_edge_attributes(G, affecteds + below_edges)
        afrc_min, afrc_max = get_min_max_afrc_values(G, affecteds + below_edges)
        set_edge_colors(G, afrc_min, afrc_threshold)     
        # plot graph and print minimum afrc value for control purposes
        # plot_my_graph(G, pos, edge_col = [d["color"]  for u,v,d in G.edges.data()])
        loop_counter += 1
        # print("loop_counter: ", loop_counter)   
        # print("new afrc_min value: ", afrc_min)
        
        # collect edges with minimal negative AFRC
        afrc_below_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc"] < afrc_threshold)], key = lambda e: e[2]["afrc"])
        afrc_min_list   = [(u,v,d)  for u,v,d in afrc_below_list  if (d["afrc"]==afrc_min)]
        
        # print("potential edges to remove: ")
        # for af in afrc_min_list: print(af)
    # determine connected components of graph of edges with positive ARFC
    C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # Create list of tupels with node names and cluster labels, set node colors acc to cluster
    G = set_node_labels(G,C)
    L1, L2 = get_result_list_dict(G,C)        
    return G, L1, L2
        



'''
# Create example graph using Havel-Hakimi-Algorithm
k = [6, 4, 4, 4, 3, 3, 3, 3, 2, 2]
G = nx.havel_hakimi_graph(k)           
print("\nAdjacency list of example graph (Havel-Hakimi-Algorithm)")
print("k=",k, "\n")
for n in G: print("%2d" % n, " ", list(G[n])) 

'''

'''
# Simple graph 
G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(1, 4)
G.add_edge(4, 5)
G.add_edge(4, 6)
G.add_edge(5, 6)

'''

'''
# Stochastisches Blockmodell
sizes = [10, 10, 10]
probs = [[0.5, 0.02, 0.03], [0.02, 0.5, 0.01], [0.03, 0.01, 0.50]]
G = nx.stochastic_block_model(sizes, probs, seed=0)

'''


# --------------------------------------
# ------- Karate Club graph ------------
# --------------------------------------


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
print(len(list(G_in.edges())))
C_in = [c for c in sorted(nx.connected_components(G_in), key=len, reverse=True)]
List_in, Dict_in = get_result_list_dict(G_in,C_in)  


# pos1 = nx.kamada_kawai_layout(G)
# for (n,v) in iter(pos1.items()):
#     if n in range(5,10): 
#         v[0] = v[0]-0.1    
#         v[1] = v[1]-0.5
#     if n in range(10,15): 
#         v[0] = v[0]-0.3    
#         v[1] = v[1]+0.3

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
# plot_my_graph(G, pos, 
#               node_col = [d["color"]  for n,d in G.nodes.data()])

AFRC_N_edges_after = len(list(G.edges()))
print("N edges after: ", AFRC_N_edges_after)

print(AFRC_N_edges_before - AFRC_N_edges_after)




#-----------------------------------------------------

# Louvain - Methode

# print("\n\nLouvain")
# G = nx.karate_club_graph()
# lv = nx_comm.louvain_communities(G, seed=123)
# G = color_partition_nodes(G, lv)

# plot_my_graph(G, pos, 
#               node_col = [d["color"]  for n,d in G.nodes.data()])

# print(len(list(G.edges())))


# ------------------------------------------------------

# Girvan-Newman-Algorithmus   mit Anzahl entfernter Ecken aus AFRC-ALgorithmus
# Bestimmung mit ALgorithmus aus Package   --- DO NOT USE ! ---

# def girvan_newman_sets(gm, max_i):
#     i = 0 
#     max_i = max(1,max_i) 
#     while i < max_i:
#         try:
#             C1 = list(sorted(c)  for c in next(gm))
#             print(C1)
#             i += 1
#         except StopIteration:
#             break
#     return C1
    

# G = nx.karate_club_graph()
# gm = nx_comm.girvan_newman(G)
# # S = girvan_newman_sets(gm, (AFRC_N_edges_before - AFRC_N_edges_after) / 1)
# S = girvan_newman_sets(gm, 10)
# C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
# print("Karate-Club - C: ", len(C))
# G = color_partition_nodes(G, S)

# plot_my_graph(G, pos, 
#               node_col = [d["color"]  for n,d in G.nodes.data()])

# print(len(list(G.edges())))


# -----------------------------------------------------------

# Girvan-Newman-Algorithmus   mit Anzahl entfernter Ecken aus AFRC-ALgorithmus
# Bestimmung mit edge_betweenness_centrality

# print("\n\nGirvan-Newman")
# edges_to_remove = int((AFRC_N_edges_before - AFRC_N_edges_after) / 1)
# G_gn = nx.karate_club_graph()

# for k in range(edges_to_remove):
#     gn_edges = nx.edge_betweenness_centrality(G_gn)
#     gn_edges_max_val = max([v  for (k,v) in gn_edges.items()])
#     gn_edges_max = [k  for (k,v) in gn_edges.items()  if v == gn_edges_max_val]
#     if len(gn_edges_max) == 1:
#         gn_e = gn_edges_max[0]
#     else:
#         gn_e = select_an_edge(gn_edges_max)[0]
#     (u,v) = gn_e
#     G_gn.remove_edge(u,v)

# C_gn = [c for c in sorted(nx.connected_components(G_gn), key=len, reverse=True)]
# G_gn = set_node_labels(G_gn, C_gn)
# List_out_gn, Dict_out_gn = get_result_list_dict(G_gn, C_gn)  

# plot_my_graph(G_in, pos,    # Achtung, hier G_in, damit alle Kanten vorhanden sind !!!
#               node_col = [d["color"]  for n,d in G_gn.nodes.data()],
#               edge_col = ["lightgrey"] * len(list(G_in.edges()))) 

# print(edges_to_remove, "edges removed  ",len(list(G_gn.edges())), " edges remaining")

    



# --------------------------------------------------------
# --------  American Footbal graph -----------------------
# --------------------------------------------------------

# Americal Football (AMF) read-in from gml   and Ground Truth

# def colored_amf_graph(G):
#     for n,d in G.nodes.data(): 
#         d["color"] = d["value"]
#     return G

# def get_circles_in_circle_pos(G, sc, rd, cx, cy, mv):
#     p = {}
#     cnt = np.array([cx, cy])
#     for v in values:
#         temp = nx.circular_layout(
#                     nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["value"] == v]),
#                     scale = sc,
#                     center = cnt + np.array([rd * np.cos(v/(mv+1)*2*np.pi),
#                                              rd * np.sin(v/(mv+1)*2*np.pi)])
#                     )
#         p.update(temp)
#     return p


# print("\n\nGround Truth")
# G_amf = nx.read_gml("football.gml")
# # i = 0
# # for n,d in G_amf.nodes.data():
# #     print("%3d" % i,n,d)
# #     i += 1

# values = set([d["value"]  for n,d in iter(G_amf.nodes.items())])
# max_value = list(values)[-1]

# pos_amf = get_circles_in_circle_pos(G_amf, 0.1, 0.6, 0.5, 0.5, max_value)
# # cwd = os.getcwd()
# # full_fn = os.path.join(cwd, "pos_american_football_graph.json")
# # save_data_to_json_file(pos_array_as_list(pos_amf), full_fn)
# # pos_amf_backup = pos_list_as_array(read_data_from_json_file(full_fn))

# G_amf = colored_amf_graph(G_amf)
# plot_my_graph(G_amf, pos_amf, 
#               node_col = [d["color"]  for n,d in G_amf.nodes.data()],
#               color_map = "tab20", alpha = 0.7)

# AFRC_N_edges_before = len(list(G_amf.edges()))
# print("N edges before: ", AFRC_N_edges_before)


# --------------------------------------------------------

# AFRC-Algorithm for AMF 

# print("\n\nAFRC Algorithm")
# G_amf = nx.read_gml("football.gml")
# G_in_amf = deepcopy(G_amf)
# print(len(list(G_in_amf.edges())))

# G_amf, List_out_amf, Dict_out_amf = detect_communities(G_amf)

# plot_my_graph(G_in_amf, pos_amf, 
#               node_col = [d["color"]  for n,d in G_amf.nodes.data()], 
#               edge_col = ["lightgrey"] * len(list(G_in_amf.edges())),
#               color_map = "tab20", alpha = 0.7)

# AFRC_N_edges_after = len(list(G_amf.edges()))
# print("N edges after: ", AFRC_N_edges_after)
# print(AFRC_N_edges_before - AFRC_N_edges_after)




# -------------------------------------------------------------

# Louvain-Method for AMF

# print("\n\nLouvain")
# G_amf = nx.read_gml("football.gml")
# lv_amf = nx_comm.louvain_communities(G_amf, seed=123)
# G_amf = color_partition_nodes(G_amf, lv_amf)

# plot_my_graph(G_amf, pos_amf, 
#               node_col = [d["color"]  for n,d in G_amf.nodes.data()],
#               color_map = "tab20", alpha = 0.7)

# print(len(list(G_amf.edges())))


# -----------------------------------------------

# Girvan-Newman-Algorithm for AMF   mit Anzahl entfernter Ecken aus AFRC-ALgorithmus für AMF
# Bestimmung mit ALgorithmus aus Package   --- DO NOT USE ! ---

# def girvan_newman_sets(gm, max_i):
#     i = 0 
#     while i < max_i:
#         try:
#             C1 = list(sorted(c)  for c in next(gm))
#             i += 1
#         except StopIteration:
#             break
#     return C1

# print("Girvan-Newman")
# G_amf = nx.read_gml("football.gml")
# gm_amf = nx_comm.girvan_newman(G_amf)
# # S_amf = girvan_newman_sets(gm_amf, (AFRC_N_edges_before-AFRC_N_edges_after) / 1)
# S_amf = girvan_newman_sets(gm_amf, 200)
# C_amf = [c for c in sorted(nx.connected_components(G_amf), key=len, reverse=True)]
# print("American Football - C_amf: ", len(C_amf))

# G_amf = color_partition_nodes(G_amf, C_amf)

# plot_my_graph(G_amf, pos_amf, 
#               node_col = [d["color"]  for n,d in G_amf.nodes.data()],
#               color_map = "tab20", alpha = 0.7)

# print(len(list(G_amf.edges())))


# -------------------------------------------------------

# Girvan-Newman-Algorithmus   mit Anzahl entfernter Ecken aus AFRC-ALgorithmus
# Bestimmung mit edge_betweenness_centrality

# print("\n\nGirvan-Newman")
# edges_to_remove = int((AFRC_N_edges_before - AFRC_N_edges_after) / 1)
# G_gn_amf = nx.read_gml("football.gml")

# for k in range(edges_to_remove):
#     gn_edges = nx.edge_betweenness_centrality(G_gn_amf)
#     gn_edges_max_val = max([v  for (k,v) in gn_edges.items()])
#     gn_edges_max = [k  for (k,v) in gn_edges.items()  if v == gn_edges_max_val]
#     if len(gn_edges_max) == 1:
#         gn_e = gn_edges_max[0]
#     else:
#         gn_e = select_an_edge(gn_edges_max)[0]
#     (u,v) = gn_e
#     G_gn_amf.remove_edge(u,v)

# C_gn_amf = [c for c in sorted(nx.connected_components(G_gn_amf), key=len, reverse=True)]
# set_node_labels(G_gn_amf, C_gn_amf)
# List_out_gn_amf, Dict_out_gn_amf = get_result_list_dict(G_gn_amf, C_gn_amf)  

# plot_my_graph(G_in_amf, pos_amf, 
#               node_col = [d["color"]  for n,d in G_gn_amf.nodes.data()],
#               color_map = "tab20", alpha = 0.7)

# print(edges_to_remove, "edges removed  ",len(list(G_gn_amf.edges())), " edges remaining")


# ---------------------------------------------------------


