# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:11:43 2023

@author: Ralf
"""

import networkx as nx
import numpy as np
import scipy.sparse as sc
import time


def AugFormanSq_semisparse(e,G):
    E=np.zeros([len(G), len(G)]) #Matrix of edge contributions
    FR=0
    
    #Add a -1 to the contribution of all edges sharing a node with e
    for i in (set(G[e[0]]) - {e[1]}):
        E[min(e[0],i)][max(e[0],i)] = -1
    for i in (set(G[e[1]]) - {e[0]}):
        E[min(e[1],i)][max(e[1],i)] = -1
        
    #Count triangles, and add +1 to the contribution of edges contained in a triangle with e
    T=len(set(G[e[0]]) & set(G[e[1]]))
    for i in (set(G[e[0]]) & set(G[e[1]])):
        E[min(e[0],i)][max(e[0],i)] += 1
        E[min(e[1],i)][max(e[1],i)] += 1
      
    #Count squares,
    #Add +1 to each edge neighbour to e contained in a square with it
    #Add +1 or -1 for edges not touching e contained in a square with it (the matrix lets us keep track of both orientations separately)
    Sq=0
    neigh_0 = [i for i in G[e[0]] if i != e[1]]        
    for i in neigh_0:         # for all neighbors of first node                
        for j in set(G[i]) & set(G[e[1]]) - {e[0]}:   # for all nodes forming a square with first node, neighbors of first node, and second noce
            Sq +=1
            E[min(e[0],i)][max(e[0],i)] += 1      
            E[min(e[1],j)][max(e[1],j)] += 1      
            E[i][j] += 1                          
            
    # Convert to sparse matrix for matrix multiplications
    csr = sc.csr_matrix(E)
    csr.eliminate_zeros()   # eliminate zeros
              
    csr_t = csr.transpose()     # transpose sparse matrix 
    csr_res = csr - csr_t       # subtract tranposed from original matrix
    csr_sign = csr_res.sign()   # get sign for absolute value
    csr_res = csr_res.multiply(csr_sign)    # calculate absolute value by multiplication with sign of each cell
    
    FR -= csr_res.sum()         # summarize sparse matrix
    
    FR = int(FR / 2)            # divide by two, as both triangular matrices have been added above, we only need one
    FR += 2 + T + Sq
        
    return FR