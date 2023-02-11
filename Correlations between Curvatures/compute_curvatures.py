"""
compute_curvatures.ipynb

Created on Sat Nov 26 09:07:00 2022

@author: Lukas

This script contains all functions to compute the different curvatures of a given graph.
"""

# load packages using the setup.ipynb script

import setup


# define functions

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


# name cycles according to length
cyc_names = {3:"triangles", 4:"quadrangles", 5:"pentagons"}


def init_edge_attributes(G):
    """
    initialize edge attributes for all edges in G

    Parameters
    ----------
    G : Graph

    Returns
    -------
    None.
    """
    curv_names = ["frc", "afrc", "afrc4", "afrc5"] 
    for (u,v) in list(G.edges()):
        for i in range(3,6):
            G.edges[u,v][cyc_names[i]] = []
        for cn in curv_names:
            G.edges[u,v][cn] = 0


def set_edge_attributes_2 (G, ll, i):
    """
    set edge attributes for all edges in G

    Parameters
    ----------
    G : Graph
    ll : list of lists
        list of lists containing the cycles of length i
    i : int
        length of cycles

    Returns
    -------
    None.
    """
    for l in ll:     # for every cycle in the list of cycles
        for e1 in range(0, i): 
            if e1 == i-1:
                e2 = 0
            else:
                e2 = e1 + 1
            u = l[e1]
            v = l[e2]
            G.edges[u,v][cyc_names[i]].append(l)


def get_orc_edge_curvatures (G):          
    """
    compute the Ollivier-Ricci curvature of all edges in G

    Parameters
    ----------
    G : Graph

    Returns
    -------
    None.
    """
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    # transfer curvatire values from orc.G to G 
    for (u,v) in list(orc.G.edges()):               # for every edge
        G.edges[u,v]["orc"] = orc.G.edges[u,v]["ricciCurvature"]
        # print("ORC: ", orc.G.edges[u,v]["ricciCurvature"], ("  -  G: ",G.edges[u,v]["orc"])


def get_edge_curvatures (G, t_coeff, q_coeff, p_coeff):
    """
    compute the curvatures of all edges in G

    Parameters
    ----------
    G : Graph
    t_coeff : int
        coefficient for triangles
    q_coeff : int
        coefficient for quadrangles
    p_coeff : int
        coefficient for pentagons

    Returns
    -------
    None.
    """            
    for (u,v) in list(G.edges()):               # for every edge
        tr = len(G.edges[u,v][cyc_names[3]]) / 2  # divide by 2 because in a direct graph each cycle is counted twice
        qu = len(G.edges[u,v][cyc_names[4]]) / 2
        pe = len(G.edges[u,v][cyc_names[5]]) / 2
        G.edges[u,v]["frc"] = fr_curvature(G, u, v)        
        G.edges[u,v]["afrc"] = afr_curvature(G, u, v, t_coeff, tr)
        G.edges[u,v]["afrc4"] = afr4_curvature(G, u, v, t_coeff, tr, q_coeff, qu)
        G.edges[u,v]["afrc5"] = afr5_curvature(G, u, v, t_coeff, tr, q_coeff, qu, p_coeff, pe)