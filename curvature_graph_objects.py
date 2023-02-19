"""
curvature_graph_objects.py

Created on Feb 13 2023

@author: Lukas

This file contains the classes for the curvature graph objects.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


# import functions from other scripts in the repository

import compute_curvatures as cc
import auxiliary_functions as af
import curvature_gaps as cg


# define abstract class for curvature graphs

class CurvatureGraph(nx.Graph):
    """
    An abstract class that inherits from the networkx Graph class and adds additional
    attributes and methods related to graph curvature. Used to create subclasses for
    artificial and real graphs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #for edge in self.edges:
        #    self.edges[edge]["triangles"] = 0
        #    self.edges[edge]["quadrangles"] = 0
        #    self.edges[edge]["pentagons"] = 0

        # self.cycles = {"triangles": [], "quadrangles": [], "pentagons": []}
        self.curvature_gap = {}
        for edge in self.edges:
            self.edges[edge]["weight"] = 1


        # more to come

    def get_cycles(self):
        """
        Get the cycles of the graph.
        """
        all_cycles = []
        for cycle in cc.simple_cycles(self.to_directed(), 6):
            all_cycles.append(cycle)

        self.cycles["triangles"] = [cycle for cycle in all_cycles if len(cycle) == 3]
        self.cycles["quadrangles"] = [cycle for cycle in all_cycles if len(cycle) == 4]
        self.cycles["pentagons"] = [cycle for cycle in all_cycles if len(cycle) == 5]

    def compute_frc(self):
        """
        Compute the Forman-Ricci curvature of the graph.
        """
        for edge in list(self.edges()):
            u, v = edge
            self.edges[edge]['frc'] = cc.fr_curvature(self, u, v)

    def compute_orc(self):
        """
        Compute the Ollivier-Ricci curvature of the graph.
        """
        # compute curvatures for all edges
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        
        # compute Ollivier-Ricci curvature
        cc.get_orc_edge_curvatures(G)
        for edge in self.edges:
            self.edges[edge]["orc"] = G.edges[edge]["orc"]

    def compute_afrc(self):
        """
        Compute the correct augmented Forman-Ricci curvature of the graph.
        """
        for edge in list(self.edges()):

            # compute curvature
            self.edges[edge]['afrc'] = cc.AugFormanSq(edge, self)

    def compute_afrc_3(self):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        try:
            for edge in list(self.edges()):
                u, v = edge

                # compute curvature
                self.edges[edge]['afrc_3'] = cc.afrc_3_curvature(
                    self, u, v, 
                    t_num = self.edges[edge]["triangles"]
                    )
        except KeyError:
            print("Need to compute the number of triangles first.")
            self.count_triangles()
            self.compute_afrc_3()

    def compute_afrc_4(self):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        for edge in list(self.edges()):
            u, v = edge

            # compute curvature
            self.edges[edge]['afrc_4'] = cc.afrc_4_curvature(
                self, u, v, 
                t_num = self.edges[edge]["triangles"], 
                q_num = self.edges[edge]["quadrangles"]
                )

    def compute_afrc_5(self):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        for edge in list(self.edges()):
            u, v = edge

            # compute curvature
            self.edges[edge]['afrc_5'] = cc.afrc_5_curvature(
                self, u, v, 
                t_num = self.edges[edge]["triangles"], 
                q_num = self.edges[edge]["quadrangles"], 
                p_num = self.edges[edge]["pentagons"]
                )

    def count_triangles(self): # reimplement this using allocate_cycles_to_edges
        """
        Count the number of triangles for each edge in the graph.
        """
        try:
            for edge in list(self.edges()):
                u, v = edge
                self.edges[edge]["triangles"] = len([cycle for cycle in self.cycles["triangles"] if u in cycle and v in cycle])/2

        except KeyError:
            print("Need to compute the cycles first.")
            self.get_cycles()
            self.count_triangles()
            
    def count_quadrangles(self):
        """
        Count the number of quadrangles for each edge in the graph.
        """
        for edge in list(self.edges()):
            u, v = edge
            self.edges[edge]["quadrangles"] = len([cycle for cycle in self.cycles["quadrangles"] if u in cycle and v in cycle])/2

    def count_pentagons(self):
        """
        Count the number of pentagons for each edge in the graph.
        """
        for edge in list(self.edges()):
            u, v = edge
            self.edges[edge]["pentagons"] = len([cycle for cycle in self.cycles["pentagons"] if u in cycle and v in cycle])/2

    def compute_correlation(self, curvature1, curvature2):
        """
        Compute the correlation between two curvatures.
        """
        corr_coeff = np.corrcoef(
            [self.edges[edge][curvature1] for edge in self.edges], 
            [self.edges[edge][curvature2] for edge in self.edges]
            )[0, 1]

        return corr_coeff




# define subclasses for artificial graphs

class CurvatureSBM(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for stochastic block models.
    """
    def __init__(self, l, k, p_in, p_out):
        sizes = af.build_size_list(k, l)
        probs = af.build_prob_list(l, p_in, p_out)

        super().__init__(nx.stochastic_block_model(sizes, probs, seed=0))

    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name)


class CurvatureER(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for Erdos-Renyi graphs.
    """
    def __init__(self, n, p):
        super().__init__(nx.erdos_renyi_graph(n, p))
    

class CurvatureBG(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for bipartite graphs.
    """
    def __init__(self, n, p):
        super().__init__(nx.bipartite.random_graph(n, n, p, seed=0))


# define subclasses for real graphs

class CurvatureKarate(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the karate club graph.
    """
    def __init__(self):
        super().__init__(nx.karate_club_graph())

    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "club")


class CurvatureAMF(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the American football graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/football.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)

    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "value")


class CurvatureDolphins(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the dolphins graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/dolphins.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


class CurvatureUSPowerGrid(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the US power grid graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/power.gml", label  = 'id'))


class CurvatureWordAdjacency(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the word adjacency graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/adjnoun.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)







    # needs an init method that takes the following arguments: SHOULD WE INHERIT THIS FROM networkx.Graph?!
    # - l : number of communities
    # - k : community sizes
    # - p_out : inter-community edge probabilities
    # - p_in : intra-community edge probabilities

    # needs an attribute that assigns each node to a community
    # needs an attribute that assigns each edge to inter- or intra-community
    # add attributes for the curvatures of the graph, initialized as empty lists, fill when computed?!

    # DONE: needs methods to compute the curvatures of the graph
    # needs a method to compute the correlation between 2 curvatures of the graph
    # needs a method to compute the curvature gap of the graph

    # needs a method to plot a histogram of a specific curvature
    # needs a method to return a plot with histograms of all curvatures
    # needs a method to visualize the graph
    # needs a method to save the graph as a json file
    # needs a method to load a graph from a json file
    # needs a method to run sequentual community detection on the graph
    # needs a method to run parallel community detection on the graph