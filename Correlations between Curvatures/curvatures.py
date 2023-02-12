# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:29:44 2022

@author: Ralf
"""

import networkx as nx
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from copy import deepcopy


# from GraphRicciCurvature.FormanRicci import FormanRicci
# from GraphRicciCurvature.OllivierRicci import OllivierRicci


def get_spirals_in_circle_pos(G, sc, rd, cx, cy, mv, res = 0.35, eq = False):
    p = {}
    cnt = np.array([cx, cy])
    for v in range(mv):
        temp = nx.spiral_layout(
                    nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["block"] == v]),
                    scale = sc,
                    center = cnt + np.array([rd * np.cos(v/mv*2*np.pi),
                                              rd * np.sin(v/mv*2*np.pi)]),
                    resolution = res,
                    equidistant = eq
                    )
        p.update(temp)
    return p


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
    nx.draw_networkx (G, pos, node_color = node_col, edge_color = edge_col, **node_options)
    nx.draw_networkx_edges (G, pos, edge_lst, edge_color = edge_col, **edge_options)
    nx.draw_networkx_edge_labels(G, pos, label_pos = 0.5, edge_labels = edge_lab, rotate=False, bbox = bbox)
    plt.gca().margins(0.20)
    plt.show()


# def remove_permutations(ll):
#     print("before:", len(ll))
#     i = 0
#     z = len(ll)
#     while i < z:
#         j = i + 1
#         a = sorted(ll[i])
#         while j < z:
#             # falls Permutation, dann entfernen.  Liste wird dadurch kürzer, daher jedesmal len(ll) überprüfen 
#             b = sorted(ll[j])
#             if a == b:
#                 ll.pop(j)
#                 z = len(ll)
#                 break
#             else:
#                 j += 1
#         i += 1
#     print("after: ", len(ll))
#     return ll


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


def afr5_curvature (G, ni, nj, t_coeff, t_num, q_coeff, q_num, p_coeff, p_num):
    '''
    computes the Augmented Forman-Ricci curvature of a given edge 
    includes 3-, 4- and 5-cycles in calculation 
    
    Parameters
    ----------
    G : Graph
    ni : node i
    nj : node j
    t : number of triangles containing the edge between node i and j
    q : number of quadrangles containing the edge between node i and j
    p : number of pentagons containing the edge between node i and j

    Returns
    -------
    afrc5 : int
        enhanced Forman Ricci curvature of the edge connecting nodes i and j   
    '''
    afrc5 = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num + q_coeff * q_num + p_coeff * p_num
    return afrc5


def init_edge_attributes(G):
    curv_names = ["frc", "afrc", "afrc4", "afrc5"] 
    for (u,v) in list(G.edges()):
        for i in range(3,6):
            G.edges[u,v][cyc_names[i]] = []
        for cn in curv_names:
            G.edges[u,v][cn] = 0
        G.edges[u,v]["color"] = "lightgrey"       # default Farbe
            

def set_edge_attributes_2 (G, ll, i):
    for l in ll:     # für jeden Zyklus in der Liste der Zyklen
        for e1 in range(0, i): 
            if e1 == i-1:
                e2 = 0
            else:
                e2 = e1 + 1
            u = l[e1]
            v = l[e2]
            G.edges[u,v][cyc_names[i]].append(l)
            

def set_SBM_edge_colors(Gr, cmp_key="block"):                         # Einfärben der Kanten innerhalb und außerhalb eines Blocks
    for u,v,d in Gr.edges.data():
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:
            d["color"] = "grey" 
        else:
            d["color"] = "orange" 
            
            
def get_within_vs_between_curv(Gr, cmp_key="block"):
    curv_names = ["orc", "frc", "afrc", "afrc4", "afrc5"] 
    
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        for cn in curv_names:
            c_dict[k][cn] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                           # für alle Kanten 
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:    # innerhalb eines Blocks
            for cn in curv_names:                          # für alle verschiedenen Curvartures
                c_dict["withins"][cn]["data"].append(Gr.edges[u,v][cn])           # hängt den Curvature-Wert der aktuellen Kante an die LIste an 
        else:                                               # zwischen Blöcken
            for cn in curv_names:                          # für alle verschiedenen Curvartures
                c_dict["betweens"][cn]["data"].append(Gr.edges[u,v][cn])           # hängt den Curvature-Wert der aktuellen Kante an die LIste an 

    for k in c_dict.keys():
        for cn in curv_names:
            c_dict[k][cn]["mean"] = np.mean(c_dict[k][cn]["data"])      # Mittelwerte berechnen
            c_dict[k][cn]["std"] = np.std(c_dict[k][cn]["data"])        # Std.-abw. berechnen
            
    res_diffs = {}
    for cn in curv_names:
        sum_std = np.sqrt(np.square(c_dict["withins"][cn]["std"]) + np.square(c_dict["betweens"][cn]["std"]))   # Gesamt-Stdabw berechnen
        res_diffs[cn] = np.abs((c_dict["withins"][cn]["mean"] - c_dict["betweens"][cn]["mean"]) / sum_std)     # Differenz der Mittelwerte bilden und normieren
    
    return res_diffs

            
def get_orc_edge_curvatures (G):          
    '''
    # compute the Ollivier-Ricci curvature of the given graph G
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    
    # transfer curvatire values from orc.G to G e"]
    for (u,v) in list(orc.G.edges()):               # für jede Kante
        G.edges[u,v]["orc"] = orc.G.edges[u,v]["ricciCurvature"]

    '''
    for (u,v) in list(G.edges()):               # für jede Kante
        G.edges[u,v]["orc"] = np.random.rand()
        
        

def get_edge_curvatures (G, t_coeff = 3, q_coeff = 2, p_coeff = 1):            
    for (u,v) in list(G.edges()):               # für jede Kante
        tr = len(G.edges[u,v][cyc_names[3]]) / 2
        qu = len(G.edges[u,v][cyc_names[4]]) / 2
        pe = len(G.edges[u,v][cyc_names[5]]) / 2
        G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, t_coeff, tr)
        G.edges[u,v]["afrc4"] = afr4_curvature(G, u, v, t_coeff, tr, q_coeff, qu)
        G.edges[u,v]["afrc5"] = afr5_curvature(G, u, v, t_coeff, tr, q_coeff, qu, p_coeff, pe)    
    
    
def show_curv_min_max_values (h_data):
    print("\nMin/Max Curvature values:")
    for k in h_data.keys():
        print(str(k).ljust(8), 
              "{0:<5s} {1:7.3f}".format("Min:", h_data[k]["bin_min"]), "  ",
              "{0:<5s} {1:7.3f}".format("Max:", h_data[k]["bin_max"])
              )
    print()
    
    
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
    else:     # for other curvatures
        b_lo_lim = b_min
        b_hi_lim = b_max             
    return b_lo_lim, b_hi_lim, b_width


def show_histos (h_data, title_str, my_nrows = 2, my_ncols = 3, bin_num_lim = 40):
    fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, sharey = True, figsize=(16,10))
    for i,k in enumerate(h_data.keys()):
        r = i // my_ncols
        c = i % my_ncols
        
        # n_bins = 40
        # axes[r,c].hist(h_data[k]["curv"], bins = n_bins, edgecolor = "white", histtype='bar', stacked=True)
        
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
    
    
def show_correlation_coeffs (h_data):
    def merged_list(ll):
        l = []
        for i in range(len(ll)):
            l.extend(ll[i])
        return l
    
    print("\nCorrelation coefficients:")
    curv_names = ["orc", "frc", "afrc", "afrc4", "afrc5"] 
    for i,cn in enumerate(curv_names):
        for j in range(i+1, len(curv_names)):
            s = h_data[cn]["title"] + " / " + h_data[curv_names[j]]["title"]
            c = np.corrcoef( merged_list(h_data[cn]["curv"]),  merged_list(h_data[curv_names[j]]["curv"]) ) [1][0]
            print(s.ljust(55,"."), f"{c:8.5f}")
        print()
        


def show_curv_data (G, title_str = "", cmp_key = "block"):
    curv_names = ["orc", "frc", "afrc", "afrc4", "afrc5"] 
    titles = {"orc": "Ollivier Ricci (OR)", "frc": "Forman Ricci (FR)", "afrc": "Augm. FR curv. (triangles)", 
              "afrc4": "AFR curv. (tri/quad)", "afrc5": "AFR curv. (tri/quad/pent)"}
    
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
        # h_data[cn]["bin_min"] = int(min([min(h_data[cn]["curv"][i])  for i in range(len(h_data[cn]["curv"]))]))
        # h_data[cn]["bin_max"] = int(max([max(h_data[cn]["curv"][i])  for i in range(len(h_data[cn]["curv"]))]))
        h_data[cn]["bin_min"] = min([min(h_data[cn]["curv"][i], default=0)  for i in range(len(h_data[cn]["curv"]))])
        h_data[cn]["bin_max"] = max([max(h_data[cn]["curv"][i], default=0)  for i in range(len(h_data[cn]["curv"]))])
        
    show_curv_min_max_values (h_data)
    show_histos (h_data, title_str, my_nrows = 2, my_ncols = 3, bin_num_lim = 40)
    show_correlation_coeffs(h_data)

    
def build_size_list (k, l):
    ll = [k  for i in range(l)]
    return ll
    

def build_prob_list (n, p_in, p_out):
    ll = []
    for i in range(n):    
        temp_l = [p_out  for j in range(0,i)] + [p_in] + [p_out  for j in range(i+2,n+1)]
        ll.append(temp_l)
    return ll


cyc_names = {3:"triangles", 4:"quadrangles", 5:"pentagons"}     
   


# ---------------------------------- 
# -------  simple example  ---------
# ---------------------------------- 

     
def simple_example():
    edges = [(0,3), (0,4), (3,1), (3,2), (3,4), (1,2), (2,4)]
    # edges = [(0,3), (0,4), (3,1), (3,2), (3,4), (1,2), (2,4), (0,5), (5,6), (6,4)]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    init_edge_attributes(G)
    
    H = G.to_directed()
    
    pos1 = nx.kamada_kawai_layout(H)
    plot_my_graph(H, pos1)
    
    cycles = []
    for c in simple_cycles(H, 6):
        cycles.append(c) 
        
    d = dict()
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]
        set_edge_attributes_2(G, d[i], i)
        
    get_edge_curvatures(G)

# simple_example()    


# --------------------------------
# ----------  simple SBM  --------
# --------------------------------


def simple_SBM():
    sbm = {"size_per_comm" : 15, "num_of_comm" : 4, "p_in" : 0.70, "p_out" : 0.05}
    sizes = build_size_list(sbm["size_per_comm"], sbm["num_of_comm"])
    probs = build_prob_list(sbm["num_of_comm"], sbm["p_in"], sbm["p_out"])
    
    G = nx.stochastic_block_model(sizes, probs, seed=0)   
    init_edge_attributes(G)
      
    H = G.to_directed()
    
    cycles = []                         # hier werden de Zyklen bestimmt
    for c in simple_cycles(H, 6):       # siehe oben: Funktion simple_cycles
        cycles.append(c) 
    
    d = dict()                          
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]       # in d werden die Zyklen sortiert nach Länge
        set_edge_attributes_2(G, d[i], i)                 # und für die Bestimmung der Curvature-Werte genutzt
        
    get_orc_edge_curvatures(G)
    get_edge_curvatures(G)
    
    pos1 = nx.kamada_kawai_layout(G)
    blocks = [v["block"]  for u,v in G.nodes.data()]
    set_SBM_edge_colors(G)    
    edge_cols = [d["color"]  for u,v,d in G.edges.data()]                              # Setzen der Kantenfarbe
    plot_my_graph(G, pos1, node_col = blocks, edge_col = edge_cols)
    
    res_diffs = get_within_vs_between_curv(G)
    print("Resulting differences:") 
    for r in res_diffs.items():
        print(r)
        
    show_curv_data(G)

    return G,H


G,H = simple_SBM()



# -------------------------------------------------------------------------
# ---------------  SBM with k / l / p_in / p_out variation  ---------------
# -------------------------------------------------------------------------


def calculate_SBM(k, l, p_in, p_out, title_str, t_coeff, q_coeff, p_coeff):
    print("k:",k," l:",l," p_in:",p_in," p_out:",p_out)
    sizes = build_size_list(k, l)
    probs = build_prob_list(l, p_in, p_out)
    
    G = nx.stochastic_block_model(sizes, probs, seed = 0)
    init_edge_attributes(G)
      
    H = G.to_directed()
    
    t0 = perf_counter()
    
    cycles = []
    for c in simple_cycles(H, 6):
        cycles.append(c) 

    t1 = perf_counter()
    print("Zyklen: ",len(cycles), " - Zeit: ", t1-t0)
    
    d = dict()
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]
        set_edge_attributes_2(G, d[i], i)
        
    get_orc_edge_curvatures (G)
    get_edge_curvatures (G, t_coeff, q_coeff, p_coeff)
    
    pos1 = nx.kamada_kawai_layout(H)
    blocks = [v["block"]  for u,v in H.nodes.data()]
    set_SBM_edge_colors(G)    
    edge_cols = [d["color"]  for u,v,d in G.edges.data()]                              # Setzen der Kantenfarbe
    plot_my_graph(G, pos1, node_col = blocks, edge_col = edge_cols)

    res_diffs = get_within_vs_between_curv(G)
    print("Resulting differences:") 
    for r in res_diffs.items():
        print(r)
        
    show_curv_data(G)
    

def calculate_SBMs():
    ll_k = [5,10,15,20]
    k_def = 20
    ll_l = [2,3,4,5]
    l_def = 5
    ll_p_in = [0.6, 0.7, 0.8, 0.9]
    p_in_def = 0.7
    ll_p_out = [0.05, 0.03, 0.02, 0.01]
    p_out_def = 0.05
    for k in ll_k:
        s = "Variation of community size / k = " + str(k) + "\n" + \
            "k=" + str(k) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k, l_def, p_in_def, p_out_def, s, 3, 2, 1)
    for l in ll_l:
        s = "Variation of number of communities / l = " + str(l) + "\n" + \
            "k=" + str(k_def) + "  l=" + str(l) +  "  p_in=" + str(p_in_def) + "  p_out=" + str(p_out_def)
        calculate_SBM(k_def, l, p_in_def, p_out_def, s, 3, 2, 1)
    for p_in in ll_p_in:
        s = "Variation of p_in / p_in = " + str(p_in) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l_def, p_in, p_out_def, s, 3, 2, 1)
    for p_out in ll_p_out:
        s = "Variation of p_out / p_out = " + str(p_out) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in_def) + " p_out:" + str(p_out)
        calculate_SBM(k_def, l_def, p_in_def, p_out, s, 3, 2, 1)

    
# calculate_SBMs()    



def calculate_sparse_SBMs():
    # ll_k = [5,10,15,20]
    ll_k = [20]
    k_def = 5
    # ll_l = [2,3,4,5]
    ll_l = [3]
    l_def = 2
    ll_ps = [(0.1, 0.7)] # , (0.1, 0.1)]
    for (p_in_def, p_out_def) in ll_ps:
        for k in ll_k:
            s = "Variation of community size / k = " + str(k) + "\n" + \
                "k=" + str(k) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
            calculate_SBM(k, l_def, p_in_def, p_out_def, s, 3, 0.02, 0.002)
        # for l in ll_l:
        #     s = "Variation of number of communities / l = " + str(l) + "\n" + \
        #         "k=" + str(k_def) + "  l=" + str(l) +  "  p_in=" + str(p_in_def) + "  p_out=" + str(p_out_def)
        #     calculate_SBM(k_def, l, p_in_def, p_out_def, s, 3, 0.02, 0.002)

# calculate_sparse_SBMs()  



# ----------------------------------------
# -------  Karate Club graph  ------------
# ----------------------------------------

def calculate_karate_club(title_str):

    G = nx.karate_club_graph()
    init_edge_attributes(G)
    H = G.to_directed()

    pos1 = nx.kamada_kawai_layout(H)
    colors= [0  if v["club"] == "Mr. Hi" else 1  for u,v in H.nodes.data()]
    set_SBM_edge_colors(G, cmp_key="club")    
    edge_cols = [d["color"]  for u,v,d in G.edges.data()]  
    
    plot_my_graph(H, pos1, node_col = colors, edge_col = edge_cols)
    
    cycles = []
    for c in simple_cycles(H, 6):
        cycles.append(c) 
    
    d = dict()
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]
        set_edge_attributes_2(G, d[i], i)
        
    get_orc_edge_curvatures (G)
    get_edge_curvatures (G)    
    show_curv_data (G, title_str, cmp_key = "club")
    
    
# calculate_karate_club("Karate Club")


    




# def get_spirals_in_circle_pos(G, sc, rd, cx, cy, mv, res = 0.35, eq = False):
#     p = {}
#     cnt = np.array([cx, cy])
#     for v in range(mv):
#         temp = nx.spiral_layout(
#                     nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["block"] == v]),
#                     scale = sc,
#                     center = cnt + np.array([rd * np.cos(v/mv*2*np.pi),
#                                               rd * np.sin(v/mv*2*np.pi)]),
#                     resolution = res,
#                     equidistant = eq
#                     )
#         p.update(temp)
#     return p





# --------------------------------------------------------
# --------  American Footbal graph -----------------------
# --------------------------------------------------------


def colored_amf_graph(G):
    for n,d in G.nodes.data(): 
        d["color"] = d["value"]
    return G


def get_circles_in_circle_pos(G, sc, rd, cx, cy, mv, values):
    p = {}
    cnt = np.array([cx, cy])
    for v in values:
        temp = nx.circular_layout(
                    nx.subgraph(G, [n  for n,d in iter(G.nodes.items())  if d["value"] == v]),
                    scale = sc,
                    center = cnt + np.array([rd * np.cos(v/(mv+1)*2*np.pi),
                                             rd * np.sin(v/(mv+1)*2*np.pi)])
                    )
        p.update(temp)
    return p


# Americal Football (AMF) read-in from football.gml
def calculate_AMF(title_str):
    G = nx.read_gml("football.gml")
    init_edge_attributes(G)
    H = G.to_directed()
    
    values = set([d["value"]  for n,d in iter(G.nodes.items())])
    max_value = list(values)[-1]
    pos_amf = get_circles_in_circle_pos(G, 0.1, 0.6, 0.5, 0.5, max_value, values)
    
    set_SBM_edge_colors(G, cmp_key="value")    
    edge_cols = [d["color"]  for u,v,d in G.edges.data()]  
    
    G = colored_amf_graph(G)
    plot_my_graph(G, pos_amf, 
                  node_col = [d["color"]  for n,d in G.nodes.data()],
                  edge_col = edge_cols,
                  color_map = "tab20", alpha = 0.7)

    cycles = []
    for c in simple_cycles(H, 6):
        cycles.append(c) 
    
    d = dict()
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]
        set_edge_attributes_2(G, d[i], i)
        
    get_orc_edge_curvatures (G)
    get_edge_curvatures (G)    
    show_curv_data (G, title_str, cmp_key = "value")
    return G
    
    
# G = calculate_AMF("American Football League")



# -----------------------------------------------------------------
# --------  Standard SBM with cycle weight variation  -------------
# -----------------------------------------------------------------

def calculate_SBM_cycle_weight_var():
    k_def = 20
    l_def = 3
    p_in_def = 0.7
    p_out_def = 0.05
    cycle_weights = [(0.4, 0.2), (0.8, 0.4), (1.2, 0.6), (1.6, 0.8), (2.0, 1.0)]
    for cw in cycle_weights :
        (q,p)  = cw
        s = "Variation of cycle weights / quad weight = " + str(q) + "pent weight = " + str(p) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l_def, p_in_def, p_out_def, s, 3, q, p)
        

# calculate_SBM_cycle_weight_var()


















