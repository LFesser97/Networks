# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 00:16:06 2022

@author: Ralf
"""

import networkx as nx
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
import pandas as pd
from time import perf_counter

from scipy.optimize import minimize

# from GraphRicciCurvature.FormanRicci import FormanRicci
# from GraphRicciCurvature.OllivierRicci import OllivierRicci


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
            
            
def get_edge_curvatures (G, t_coeff = 3, q_coeff = 2, p_coeff = 1):            
    for (u,v) in list(G.edges()):               # für jede Kante
        tr = len(G.edges[u,v][cyc_names[3]]) / 2  # geteilt durch 2 wegen gerichtetem Graph und daher immer zwei Permutationen pro Zyklus (1x vorwärts / 1x rückwärts)
        qu = len(G.edges[u,v][cyc_names[4]]) / 2
        pe = len(G.edges[u,v][cyc_names[5]]) / 2
        # G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, t_coeff, tr)
        G.edges[u,v]["afrc4"] = afr4_curvature(G, u, v, t_coeff, tr, q_coeff, qu)
        G.edges[u,v]["afrc5"] = afr5_curvature(G, u, v, t_coeff, tr, q_coeff, qu, p_coeff, pe)     
        
        
def get_curvature_gap(Gr, cn="afrc", cmp_key="block"):
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        c_dict[k] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                           # für alle Kanten 
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:    # innerhalb eines Blocks
            c_dict["withins"]["data"].append(Gr.edges[u,v][cn])           # hängt den Curvature-Wert der aktuellen Kante an die LIste an 
        else:                                               # zwischen Blöcken
            c_dict["betweens"]["data"].append(Gr.edges[u,v][cn])           # hängt den Curvature-Wert der aktuellen Kante an die LIste an 

    for k in c_dict.keys():
        c_dict[k]["mean"] = np.mean(c_dict[k]["data"])      # Mittelwerte berechnen
        c_dict[k]["std"] = np.std(c_dict[k]["data"])        # Std.-abw. berechnen
            
    sum_std = np.sqrt(np.square(c_dict["withins"]["std"]) + np.square(c_dict["betweens"]["std"]))   # Gesamt-Stdabw berechnen
    curv_gap = np.abs((c_dict["withins"]["mean"] - c_dict["betweens"]["mean"]) / sum_std)     # Differenz der Mittelwerte bilden und normieren
    
    return curv_gap


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


def optimization_func(a, G, cmp_key="value"):
    t = a[0]
    if len(a) > 1:
        q = a[1]
    else:
        q = 0
    if len(a) > 2:
        p = a[2]
    else:
        p = 0
    # print("t=", t, " q=", q, " p=", p)
    get_edge_curvatures(G, t, q, p)
    c_gap = -1 * get_curvature_gap(G, "afrc5", cmp_key)
    # Comment next line to suppress printing
    #print("t=",t," q=",q," p=",p," curv gap:",c_gap)

    return c_gap


def maximize_curvature_gap(G, a, cmp_key = "value"):
    results = []
    for i in range(len(a)):
        x0 = a[0:i+1]    # Initial values for t / t,q / t,q,p
        res = minimize(optimization_func, x0, method='nelder-mead', args = (G, cmp_key), options={'disp': False})
        # print("Parameters [t(,q)(,p)]: ",res.x, " - Optimized curvature gap: ",-1 * res.fun, "\n")
        results.extend([-1 * res.fun, *res.x])
    return results


# --------------------------------
# ----------  simple SBM  --------
# --------------------------------


def simple_SBM(p_in, p_out):
    # sbm = {"size_per_comm" : 20, "num_of_comm" : 4, "p_in" : 0.70, "p_out" : 0.05}
    size_per_comm = 20
    num_of_comm = 4
    sizes = build_size_list(size_per_comm, num_of_comm)
    probs = build_prob_list(num_of_comm, p_in, p_out)
    
    G = nx.stochastic_block_model(sizes, probs, seed=0)   
    init_edge_attributes(G)
      
    H = G.to_directed()
    
    cycles = []                         # hier werden die Zyklen bestimmt
    for c in simple_cycles(H, 6):       # siehe oben: Funktion simple_cycles
        cycles.append(c) 
    
    d = dict()                          
    for i in range(3,6):
        d[i] = [c  for c in cycles  if len(c) == i]       # in d werden die Zyklen sortiert nach Länge
        set_edge_attributes_2(G, d[i], i)                 # und für die Bestimmung der Curvature-Werte genutzt
        
    # get_orc_edge_curvatures(G)                            # Alle Curvatures bestimmen
    get_edge_curvatures(G, t_coeff = 3, q_coeff = 0.02, p_coeff = 0.002)

    res_list = maximize_curvature_gap(G, [3,2,1], cmp_key="block")

    return res_list


res = pd.DataFrame()

my_dict_keys = ["p_in", "p_out", "afrc_c_gap", "afrc_t", "afrc4_c_gap", "afrc4_t", "afrc4_q", "afrc5_c_gap", "afrc5_t", "afrc5_q", "afrc5_p"]

params = pd.DataFrame({"p_in":  np.array([i  for i in np.arange(0.5, 0.95, step=0.05)  for x in range(9)]), 
                        "p_out": np.array([x  for i in range(9)  for x in np.arange(0.02, 0.11, step=0.01)])})
# params = pd.DataFrame({"p_in":  np.array([i  for i in np.arange(0.5, 0.6, step=0.05)  for x in range(1,3)]), 
#                         "p_out": np.array([x  for i in range(1,3)  for x in np.arange(0.02, 0.04, step=0.01)])})

t00 = perf_counter()
for index, row in params.iterrows():
    t0 = perf_counter()
    print ("i:", f'{index:3d}', " p_in:", f'{row.loc["p_in"]:5.2f}', " p_out:", f'{row.loc["p_out"]:5.2f}', end="")
    res_list = simple_SBM(row.loc["p_in"], row.loc["p_out"])
    t1 = perf_counter()
    print("  time:",f'{t1-t0:7.2f}', "  total:", f'{t1-t00:9.2f}')
    res_list.insert(0, row.loc["p_in"])
    res_list.insert(1, row.loc["p_out"])
    ser = pd.Series(dict(zip(my_dict_keys, res_list)))
    res = pd.concat([res,ser.to_frame().T], ignore_index=True)

print("Total time:", t1-t00)
print(res)


# plt.style.use('_mpl-gallery')

# xs = params["p_in"]
# ys = params["p_out"]
# zs = res["afrc_t"]

# # Plot
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.scatter(xs, ys, zs)

# # ax.set(xticklabels=[],
# #        yticklabels=[],
# #        zticklabels=[])

# plt.show()

plot_list = ["afrc_t", "afrc4_t", "afrc4_q", "afrc5_t", "afrc5_q", "afrc5_p"]

fig, axes = plt.subplots(nrows = 6, ncols = 2, figsize = (8, 24), subplot_kw={"projection": "3d"}) # , tight_layout = True)

for i,pl in enumerate(plot_list):
    ax = axes[i,0]
    ax.set_title(pl, fontsize=16, loc="left")
    ax.set_xlabel("p_in",  fontsize=12)
    ax.set_ylabel("p_out", fontsize=12)
    ax.scatter(res["p_in"], res["p_out"], res[pl], c="blue", s=10)
    print(ax.get_xticks())
    ax = axes[i,1]
    ax.set_title(pl, fontsize=16, loc="left")
    ax.set_xlabel("p_out", fontsize=12)
    ax.set_ylabel("p_in",  fontsize=12)
    ax.scatter(res["p_out"], res["p_in"], res[pl], c="blue", s=10)
    print(ax.get_xticks())

    
    
plt.show()



    
    
    







