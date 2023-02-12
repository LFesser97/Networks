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

# from GraphRicciCurvature.FormanRicci import FormanRicci
# from GraphRicciCurvature.OllivierRicci import OllivierRicci



MIN_BIN, MAX_BIN = None, None


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
    afrc = 4 - G.degree(ni) - G.degree(nj) + 3*m
    return afrc


def afr4_curvature (G, ni, nj, t, q):
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
    afrc4 = 4 - G.degree(ni) - G.degree(nj) + 3*t + 2*q
    return afrc4


def afr5_curvature (G, ni, nj, t, q, p):
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
    afrc5 = 4 - G.degree(ni) - G.degree(nj) + 3*t + 2*q + 1*p
    return afrc5



def set_edge_attributes(G,orc_C=None,ae=None):
    if ae == None: 
        ae = list(G.edges())
        
    cycles = nx.cycle_basis(G)
    cycles_len = [len(c)  for c in cycles]     
    
    trmin,trmax = 999,0
    qumin,qumax = 999,0
    pemin,pemax = 999,0
    for (u,v) in ae:
        # print("u:", u, " v:", v, end="   ")
        G.edges[u,v]["triangles"] = [c  for c in cycles  if (u in c) and (v in c) and (len(c) == 3)]
        G.edges[u,v]["triangles_count"] = tr = len(G.edges[u,v]["triangles"])
        trmin = min(tr,trmin)
        trmax = max(tr,trmax)
        # print("tr: ", tr, end="  ")
        G.edges[u,v]["quadrangles"] = [c  for c in cycles  if (u in c) and (v in c) and (len(c) == 4)]
        G.edges[u,v]["quadrangles_count"] = qu = len(G.edges[u,v]["quadrangles"]) 
        qumin = min(qu,qumin)
        qumax = max(qu,qumax)
        # print("qu: ", qu, end="  ")
        G.edges[u,v]["pentagons"] = [c  for c in cycles  if (u in c) and (v in c) and (len(c) == 5)]
        G.edges[u,v]["pentagons_count"] = pe = len(G.edges[u,v]["pentagons"]) 
        pemin = min(pe,pemin)
        pemax = max(pe,pemax)
        # print("pe: ", pe, end="  ")

        
        G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, tr)
        G.edges[u,v]["afrc4"] = afr4_curvature(G, u, v, tr, qu)
        G.edges[u,v]["afrc5"] = afr5_curvature(G, u, v, tr, qu, pe)
        
    # print("\n(",trmin,"/",trmax,")  (",qumin,"/",qumax,")  (",pemin,"/",pemax,")")

    
    if orc_C != None:
        for i,(key,value) in enumerate(orc_C.items()):
            (u,v) = key
            G.edges[u,v]["orc"] = value
    else:   # Zufallszahlen, nur solange wie ORC nicht funktioniert !
        for (u,v) in ae:
            G.edges[u,v]["orc"] = np.random.rand()
        
        
   

def show_histos (G, title_str, bin_width = 1):
    # global MIN_BIN, MAX_BIN
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
    

def show_correlation_coeffs(G):
    print("\nCorrelation coefficients:")
    l_frc =  [d["frc"]   for u,v,d in G.edges.data()]
    l_afrc = [d["afrc"]  for u,v,d in G.edges.data()]
    l_afrc4 = [d["afrc4"]  for u,v,d in G.edges.data()]
    l_afrc5 = [d["afrc5"]  for u,v,d in G.edges.data()]
    # l_orc = [d["orc"]   for u,v,d in G.edges.data()]
    # corr_data = [l_afrc, l_afrc4, l_afrc5, l_orc]
    # corr_data = [l_afrc, l_afrc4, l_afrc5, l_orc, l_frc]
    corr_data = [l_afrc, l_afrc4, l_afrc5, l_frc]
    # titles = ["AFR curv. (tri)", "AFR curv. (tri/quad)", "AFR curv. (tri/quad/pent)", "Forman Ricci (FR)"]
    # titles = ["AFR curv. (tri)", "AFR curv. (tri/quad)", "AFR curv. (tri/quad/pent)", "Ollivier Ricci (OR)", "Forman Ricci (FR)"]
    titles = ["AFR curv. (tri)", "AFR curv. (tri/quad)", "AFR curv. (tri/quad/pent)", "Forman Ricci (FR)"]
    for i in range(len(corr_data)):
        for j in range(i,len(corr_data)):
            s = titles[i] + "/" + titles[j] + ":"
            c = np.corrcoef(corr_data[i],corr_data[j])[1][0]
            print(s.ljust(55,"."), f"{c:8.5f}")
        print()
    print("\n")
    
           
 


'''

# --------------------------------------
# ------- Karate Club graph ------------
# --------------------------------------

print("\n\nGround Truth - Karate Club")
G = nx.karate_club_graph()

orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
orc_Curv = nx.get_edge_attributes(orc.G,"ricciCurvature")


frc = FormanRicci(G)
frc.compute_ricci_curvature()
frc_Curv = nx.get_edge_attributes(frc.G,"formanCurvature")

# for i,(k,v) in enumerate(frc_Curv.items()):
#     print(i,": ",k,f"{v:8.5}")


set_edge_attributes(G, orc_Curv)
show_histos(G,1)
show_correlation_coeffs(G)

# for i,d in enumerate(G.edges.data()):
#     print(i,d)

'''


# --------------------------------------
# ------- Stochastic Block Model -------
# --------------------------------------

def build_prob_list(l, p_in, p_out):
    ll = []
    for i in range(l):    
        temp_l = [p_out  for j in range(0,i)] + [p_in] + [p_out  for j in range(i+2,l+1)]
        ll.append(temp_l)
    return ll


def build_size_list(k, l):
    ll = [k  for i in range(l)]
    return ll
        


def calculate_SBMs():
    ll_k = [10,20,30,40]
    # ll_k = [10]
    ll_l = [5,10,15,20]
    # ll_l = [5]
    ll_p_in = [0.6, 0.7, 0.8, 0.9]
    ll_p_out = [0.05, 0.03, 0.02, 0.01]
    k_def = 10
    # k_def = 5
    l_def = 10
    # l_def = 3
    p_in_def = 0.7
    p_out_def = 0.05
    for k in ll_k:
        s = "Variation of community size / k = " + str(k) + "\n" + \
            "k=" + str(k) + " l=" + str(l_def) + " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k, l_def, p_in_def, p_out_def, s)
    for l in ll_l:
        s = "Variation of number of communities / l = " + str(l) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l) +  " p_in:" + str(p_in_def) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l, p_in_def, p_out_def, s)
    for p_in in ll_p_in:
        s = "Variation of p_in / p_in = " + str(p_in) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in) + " p_out:" + str(p_out_def)
        calculate_SBM(k_def, l_def, p_in, p_out_def, s)
    for p_out in ll_p_out:
        s = "Variation of p_out / p_out = " + str(p_out) + "\n" + \
            "k=" + str(k_def) + " l=" + str(l_def) +  " p_in:" + str(p_in_def) + " p_out:" + str(p_out)
        calculate_SBM(k_def, l_def, p_in_def, p_out, s)
        
    
    

def calculate_SBM(k, l, p_in, p_out, title_str):
    print("k:",k," l:",l," p_in:",p_in," p_out:",p_out)
    sizes = build_size_list(k, l)
    # print(sizes)
    probs = build_prob_list(l, p_in, p_out)
    # print(probs)
    
    G = nx.stochastic_block_model(sizes, probs, seed=0)
    
    # orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    # orc.compute_ricci_curvature()
    # orc_Curv = nx.get_edge_attributes(orc.G,"ricciCurvature")
    
    # frc = FormanRicci(G)
    # frc.compute_ricci_curvature()
    # frc_Curv = nx.get_edge_attributes(frc.G,"formanCurvature")
    
    # set_edge_attributes(G, orc_Curv)
    set_edge_attributes(G)
    # show_histos(G, title_str, 1)
    show_histos(G, title_str)
    show_correlation_coeffs(G)
    
    

calculate_SBMs()


 
