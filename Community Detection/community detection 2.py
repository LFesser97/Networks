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



cyc_names = {3:"triangles", 4:"quadrangles", 5:"pentagons"}   


def simple_cycles(G, limit):
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) < limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))
        

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


# def afr_curvature (G, ni, nj, m):
#     '''
#     computes the Augmented Forman-Ricci curvature of a given edge 
    
#     Parameters
#     ----------
#     G : Graph
#     ni : node i
#     nj : node j
#     m : number of triangles containing the edge between node i and j

#     Returns
#     -------
#     afrc : int
#         Forman Ricci curvature of the edge connecting nodes i and j   
#     '''
#     afrc = 4 - G.degree(ni) - G.degree(nj) + 3*m
#     return afrc


# def show_histos (G, bin_width = 1):
#     l_frc =  [d["frc"]   for u,v,d in G.edges.data()]
#     l_afrc = [d["afrc"]  for u,v,d in G.edges.data()]       # hier afrc4 verwenden !!!
#     min_bin = min(min(l_frc), min(l_afrc))
#     max_bin = max(max(l_frc), max(l_afrc))
#     print("Histos  -  min_bin: ", min_bin, " - max_bin: ", max_bin)
#     fig, axes = plt.subplots(nrows=1, ncols=2, sharex = True, sharey = True, figsize=(14,7))
#     axes[0].hist(l_frc, bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
#     axes[0].set_title("FR curvature")
#     axes[0].title.set_size(20)
#     axes[0].tick_params(axis='both', labelsize=16)
#     axes[0].grid(visible=True, axis="both")
#     axes[1].hist(l_afrc, bins = np.arange(min_bin, max_bin + bin_width, bin_width), edgecolor = "white")
#     axes[1].set_title("Augmented FR curvature")
#     axes[1].title.set_size(20)
#     axes[1].tick_params(axis='both', labelsize=16)
#     axes[1].grid(visible=True, axis="both")
#     plt.show()


           
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
            

# Nicht benutzen, hier werden Attribute und Curvature-Werte gemischt behandelt !!!
def set_edge_attributes(G,ae=None):
    if ae == None: 
        ae = list(G.edges())
    for (u,v) in ae:
        G.edges[u,v]["triangles"] = m = len(all_triangles_of_edge(G,(u,v)))
        G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, m)


def get_min_max_afrc_values(G, key = "afrc", ae=None):
    if ae == None: 
        ae = list(G.edges())
    a = len(G.nodes())
    b = -len(G.nodes())
    for (u,v) in ae:
        a = min(a, G.edges[u,v][key])     # hier jeweils afrc4 !!!
        b = max(b, G.edges[u,v][key])
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


def set_SBM_node_colors(Gr, cmp_key="block"):                         # Einfärben der Kanten innerhalb und außerhalb eines Blocks
    for n,d in Gr.nodes.data():
        d["color"] = d[cmp_key] 
    return Gr



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

    ### set graph attributes and calculate initial AFRC values
    ### set_edge_attributes(G)
    # get start values for curvature
    get_edge_curvatures(G, t_coeff = 3, q_coeff = 2)
    # get min,max,values for afrc4 curvature
    afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4")
    # set color of edges to distinguish those with extreme (min or max)curvature values 
    set_edge_colors(G,afrc_max)     
    # show histogram of curvature values
    # show_histos(G)    

    
    print("min. AFRC value:", afrc_min, " / max. AFRC value:", afrc_max)
    afrc_threshold = int(input("Enter threshold value for AFRC4 to remove edges with a higher value: "))
    # # plot graph and print minimum afrc value for control purposes
    # plot_my_graph(G, pos, edge_col = [d["color"]  for u,v,d in G.edges.data()])
    loop_counter = 0
    # print("loop_counter: ", loop_counter)    
    # print("new afrc_min value: ", afrc_min)
    
    # collect edges with maximal AFRC4, keep it in a sorted list    !!! change to max AFRC4 value !!!
    afrc_above_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc4"] > afrc_threshold)], key = lambda e: e[2]["afrc4"])
    # list of edges with max afrc4 curvature value
    afrc_max_list   = [(u,v,d)  for u,v,d in afrc_above_list  if (d["afrc4"]==afrc_max)]
    
    # print("potential edges to remove: ")
    # for af in afrc_min_list: print(af)
    # # while there is at least one edge with negative AFRC
    while len(afrc_max_list) > 0:      
        if len(afrc_max_list) == 1:
            # edge is the only element in the list
            a = afrc_max_list[0]
        else:
            # edge is randomly chosen from the list
            a = select_an_edge(afrc_max_list)[0]
        # select first part of edge tuple 
        (u,v) = a[:2]
        # remove it from afrc_below_list, the remaining edges will be considered further
        afrc_above_list.remove(a)
        # print("chosen: ",a)
        # remove chosen edge
        G.remove_edge(u,v)
        # print("removed: ", (u,v))
        
        # determine those edges that either start or end at one of the removed edge's nodes
        
        affecteds = list(G.edges([u,v]))
        
        ########################################
        # cycles updaten !  Wie ?
        ########################################
        
        # print("affected edges: ", affecteds)
        # update attributes of these edges, include all edges to determine new afrc_min
        above_edges = [(u,v)  for u,v,d in afrc_above_list]
        # print("remaining egdes below threshold ", afrc_threshold, "\n", below_edges)     
        ### update graph attributes and calculate new AFRC values
        ###set_edge_attributes(G, affecteds + below_edges)
        
        # get updated values for curvature
        get_edge_curvatures(G, t_coeff = 3, q_coeff = 2)
        
        afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4", affecteds + above_edges)
        # plot graph and print minimum afrc value for control purposes
        # set_edge_colors(G, afrc_max, afrc_threshold)     
        # plot_my_graph(G, pos, edge_col = [d["color"]  for u,v,d in G.edges.data()])
        loop_counter += 1
        # print("loop_counter: ", loop_counter)   
        # print("new afrc_min value: ", afrc_min)
        
        # collect edges with minimal negative AFRC
        afrc_above_list = sorted([(u,v,d)  for u,v,d in G.edges.data()  if (d["afrc4"] > afrc_threshold)], key = lambda e: e[2]["afrc"])
        afrc_max_list   = [(u,v,d)  for u,v,d in afrc_above_list  if (d["afrc4"]==afrc_max)]
        
        # print("potential edges to remove: ")
        # for af in afrc_min_list: print(af)
    # determine connected components of graph of edges with positive ARFC
    C = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # Create list of tupels with node names and cluster labels, set node colors acc to cluster
    G = set_node_labels(G,C)
    L1, L2 = get_result_list_dict(G,C)        
    return G, L1, L2
        


def detect_communities_nonsequential(G, t_coeff = 3, q_coeff = 2):
    '''
    remove all edges with afrc4 curvature above a given threshold
    '''

    # get start values for curvature
    get_edge_curvatures(G, t_coeff, q_coeff)
    # get min,max,values for afrc4 curvature
    afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4")
    # show histogram of curvature values
    show_curv_data (G, title_str = "", cmp_key = "block")    

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
    # create list of tupels with node names and cluster labels, 
    L1, L2 = get_result_list_dict(G,C)        
    return G, L1, L2
        
 



# --------------------------------------
# ------- Karate Club graph ------------
# --------------------------------------

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
    
    
# calculate_karate_club()


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


cyc_names = {3:"triangles", 4:"quadrangles"}   



def afr_curvature (G, ni, nj, t_coeff, t_num):
    '''
    computes the Augmented Forman-Ricci curvature of a given edge 
    includes 3-cycles in calculation 
    
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
    afrc = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num
    return afrc


def afr4_curvature (G, ni, nj, t_coeff, t_num, q_coeff, q_num):
    '''
    computes the Augmented Forman-Ricci curvature of a given edge, 
    includes 3- and 4-cycles in calculation 
    
    Parameters
    ----------
    G : Graph
    ni : node i
    nj : node j
    t : number of triangles containing the edge between node i and j
    q : number of quadrangles containing the edge between node i and j

    Returns
    -------
    afrc4 : int
        enhanced Forman Ricci curvature of the edge connecting nodes i and j   
    '''
    afrc4 = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num + q_coeff * q_num
    return afrc4


def init_edge_attributes(G):
    # für alle Kanten des Graphen: 
    # initialisiere die (leere) Liste von Zyklen, in denen die Kante enthalten ist 
    # initialisiere die Curvature-Werte mit 0
    
    # curv_names = ["frc", "afrc", "afrc4"] 
    curv_names = ["afrc4"] 
    for (u,v) in list(G.edges()):
        # for 3 and 4 only
        for i in range(3,5):
            G.edges[u,v][cyc_names[i]] = []
        for cn in curv_names:
            G.edges[u,v][cn] = 0
            

def allocate_cycles_to_edges (G, ll, i):
    # ordnet den Kanten eines Graphen alle Zyklen der Länge i zu, in denen sie enthalten isr
    for l in ll:     # für jeden Zyklus l in der Liste der Zyklen ll
        for e1 in range(0, i):    # für jeden Knoten e1 im Zyklus
            if e1 == i-1:             # bestimme den Nachfolger e2
                e2 = 0
            else:
                e2 = e1 + 1
            u = l[e1]         # Knoten u und v der Kante im Zyklus bestimmen
            v = l[e2]         
            # Zyklus l an Zyklenliste der Kante (u,v) anhängen 
            G.edges[u,v][cyc_names[i]].append(l)   # Zyklus l an Kante (u,v) anhängen
            
            
def get_edge_curvatures (G, t_coeff = 3, q_coeff = 2):           
    # calculates curvature values - here only for triangles and quadrangles, not for penatgons
    # for each edge
    for (u,v) in list(G.edges()):                 
        # divided by 2 due to directed graph and therefore always two permutations per cycle (1x forward / 1x reverse)
        tr = len(G.edges[u,v][cyc_names[3]]) / 2   # number of triangles the egde (u,v) is in
        qu = len(G.edges[u,v][cyc_names[4]]) / 2   # number of quadrangles the egde (u,v) is in
        # G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        # G.edges[u,v]["afrc"] = afr_curvature(G, u, v, t_coeff, tr)
        G.edges[u,v]["afrc4"] = afr4_curvature(G, u, v, t_coeff, tr, q_coeff, qu)


def build_size_list (k, l):
    ll = [k  for i in range(l)]
    return ll


def build_prob_list (n, p_in, p_out):
    ll = []
    for i in range(n):    
        temp_l = [p_out  for j in range(0,i)] + [p_in] + [p_out  for j in range(i+2,n+1)]
        ll.append(temp_l)
    return ll


def get_pos_layout (H, fn = ""):
    if fn == "":
        pos = nx.kamada_kawai_layout(H)
    else:
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        # pos einlesen und list in np.array konvertieren
        pos = pos_list_as_array(read_data_from_json_file(full_fn))
        # key von pos in integer konvertieren 
        pos = {int(k):v  for (k,v) in iter(pos.items())}
    return pos


def save_pos_layout(pos, fn = ""):
    if fn != "":
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        # np.array in list konvertieren und pos speichern
        save_data_to_json_file(pos_array_as_list(pos), full_fn)
        
      
def get_bin_width (b_min, b_max, num_bin_lim):
    scaling = 1
    multiplier = 10
    # print("b_min:", b_min, "b_max:", b_max, "num_bin_lim:", num_bin_lim, "scaling:", scaling, "multiplier:", multiplier)
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


def show_histo (h_data, title_str, my_bin_num = 40):
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


def show_curv_data (G, title_str = "", cmp_key = "block"):
    curv_names = ["afrc4"] 
    titles = {"afrc4": "AFR curv. (tri/quad)"}
    h_data = {}
    for cn in curv_names:
        h_data[cn] = {"curv": [[d[cn]  for u,v,d in G.edges.data()  if G.nodes[u][cmp_key] == G.nodes[v][cmp_key]],
                               [d[cn]  for u,v,d in G.edges.data()  if G.nodes[u][cmp_key] != G.nodes[v][cmp_key]]
                               ],
                      "bin_min":0, 
                      "bin_max":0, 
                      "title": titles[cn]
                      }
    for cn in curv_names:
        h_data[cn]["bin_min"] = min([min(h_data[cn]["curv"][i], default=0)  for i in range(len(h_data[cn]["curv"]))])
        h_data[cn]["bin_max"] = max([max(h_data[cn]["curv"][i], default=0)  for i in range(len(h_data[cn]["curv"]))])
    
    # show_curv_min_max_values (h_data)
    show_histo (h_data, title_str, my_bin_num = 40)
    # show_correlation_coeffs(h_data)

# --------------------------------
# ----------  simple SBM  --------
# --------------------------------


def simple_SBM():
    sbm = {"size_per_comm" : 40, "num_of_comm" : 5, "p_in" : 0.70, "p_out" : 0.05}
    sizes = build_size_list(sbm["size_per_comm"], sbm["num_of_comm"])
    probs = build_prob_list(sbm["num_of_comm"], sbm["p_in"], sbm["p_out"])
    
    G = nx.stochastic_block_model(sizes, probs, seed=0)   
    init_edge_attributes(G)
    
    G_in = deepcopy(G)
    
    fn = "pos_sbm_graph_test.json"
    pos1 = get_pos_layout(G, "")        # fn = "" :  random layout, else read in stored positions
    save_pos_layout(pos1, "")           # fn = "" :  don't save, else save layout using filename fn  
    
    G = set_SBM_node_colors(G)
    plot_my_graph(G, pos1,
                  node_col = [d["color"]  for n,d in G.nodes.data()])
    
    H = G.to_directed()
    
    cycles = []                         # hier werden die Zyklen bestimmt
    # for 3 and 4 only  => param = 5
    for c in simple_cycles(H, 5):       # siehe oben: Funktion simple_cycles
        cycles.append(c) 
    
    d = dict()        
    # for 3 and 4 only                  
    for i in range(3,5):
        d[i] = [c  for c in cycles  if len(c) == i]       # in d werden die Zyklen sortiert nach Länge
        allocate_cycles_to_edges(G, d[i], i)              # und für die Bestimmung der Curvature-Werte genutzt
            
    G, List_out, Dict_out = detect_communities_nonsequential(G, t_coeff = 3, q_coeff = -0.7411104936441378)
    
    plot_my_graph(G_in, pos1, 
                  node_col = [d["color"]  for n,d in G.nodes.data()], 
                  edge_col = ["lightgrey"] * len(list(G_in.edges())))
    
    plot_my_graph(G, pos1, 
                  node_col = [d["color"]  for n,d in G.nodes.data()], 
                  edge_col = ["lightgrey"] * len(list(G_in.edges())))
    
    
    return G, cycles

G, cycles = simple_SBM()
