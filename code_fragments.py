def show_line_plots():
    """
    Show the line plots for the given data.
    Used for the community detection experiments to report the accuracy of the community detection algorithm.

    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey = True, figsize=(14,7))
    axes[0].plot(steps_size_per_comm, mean_size_accuracies, "b-o")
    size_std_lower = list(np.array(mean_size_accuracies) - np.array(stddev_size_accuracies))
    size_std_upper = list(np.array(mean_size_accuracies) + np.array(stddev_size_accuracies))
    axes[0].fill_between(steps_size_per_comm, size_std_lower, size_std_upper, color="blue", alpha=0.1)
    axes[0].set_title("AFRC-based algorithm\n for community detection (l=10)")
    axes[0].set_xlabel("size per community (k)")
    axes[0].set_ylabel("mean prediction accuracy")
    axes[0].title.set_size(20)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].xaxis.label.set_size(16)
    axes[0].yaxis.label.set_size(16)
    axes[0].xaxis.set_ticks(np.arange(0, 45, 5))
    axes[0].set(ylim=(0.0, 1.0))    
    axes[0].grid(visible=True, axis="both")
    axes[1].plot(steps_num_of_comm, mean_num_accuracies, "b-o")
    num_std_lower = list(np.array(mean_num_accuracies) - np.array(stddev_num_accuracies))
    num_std_upper = list(np.array(mean_num_accuracies) + np.array(stddev_num_accuracies))
    axes[1].fill_between(steps_num_of_comm, num_std_lower, num_std_upper, color="blue", alpha=0.1)
    axes[1].set_title("AFRC-based algorithm\n for community detection (k=10)")
    axes[1].set_xlabel("number of communities (l)")
    axes[1].set_ylabel("mean prediction accuracy")
    axes[1].title.set_size(20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].xaxis.label.set_size(16)
    axes[1].yaxis.label.set_size(16)
    axes[1].xaxis.set_ticks(np.arange(0, 22, 2))
    axes[1].set(ylim=(0.0, 1.0))    
    axes[1].grid(visible=True, axis="both")
    plt.show()


def show_histos (G, title_str, bin_width = 1): # HOW IS THIS DIFFERENT FROM THE 50 OTHER HISTOGRAM FUNCTIONS?
    """
    Show the histograms for the given graph.
    
    Parameters
    ----------
    G : networkx graph
        The graph to show the histograms for.

    title_str : str
        The title of the plot.

    bin_width : int, optional
        The width of the bins. The default is 1.

    Returns
    -------
    None.
    """
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


""" NEED TO REVISIT THE CODE BELOW THIS LINE """

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

""" NEED TO REVISIT THE CODE ABOVE THIS LINE """


def compute_afrc_4(G, e):
    """
    Compute the Augmented Forman-Ricci curvature with 4-cycles for a given edge e.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    e : tuple
        The edge for which we compute the Augmented Forman-Ricci curvature.

    Returns
    -------
    curvature : float
        The Augmented Forman-Ricci curvature of the edge e.
    """

    # make sure the edge has the correct orientation
    edge = (min(e), max(e))

    # get the list of all 3-cycles containing the edge
    triangles = [(cycle) for cycle in G.cycles["triangles"] if edge[0] in cycle and edge[1] in cycle]

    # sort every cycle in triangles in ascending order and remove duplicates
    triangles = list(set([tuple(sorted(triangle)) for triangle in triangles]))

    # get the list of all 4-cycles containing the edge
    quadrangles = [(cycle) for cycle in G.cycles["quadrangles"] if edge[0] in cycle and edge[1] in cycle]

    # sort every cycle in quadrangles in ascending order and remove duplicates
    quadrangles = list(set([tuple(sorted(quadrangle)) for quadrangle in quadrangles]))

    # compute other contributions
        # need to rotate the cycles from (v_1, a, b, v_2) to (a, b, v_1, v_2)
        # i.e. so that the edge is the first two vertices

    
        


    # compute the curvature
    curvature = 2 + len(triangles) + len(quadrangles) - other_contributions


def calculate_SBM_cycle_weight_var():
    """
    Calculate the curvature gaps for the SBM with different cycle weights.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
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


def get_curvature_gap(Gr, curv_name="afrc", cmp_key="block"):
    """
    Get the curvature gap of the graph.
    The curvature values are the ones stored in the graph.
    The graph must have the attributes "orc", "frc", "afrc", "afrc4", "afrc5" for each edge.

    Parameters
    ----------
    Gr : NetworkX graph
        An undirected graph.

    Returns
    -------
    curv_gap : float
        The curvature gap of the graph.
    """
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        c_dict[k] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                            
        if Gr.nodes[u][cmp_key] == Gr.nodes[v][cmp_key]:    
            c_dict["withins"]["data"].append(Gr.edges[u,v][curv_name])            
        else:                                               
            c_dict["betweens"]["data"].append(Gr.edges[u,v][curv_name])      

    for k in c_dict.keys():
        c_dict[k]["mean"] = np.mean(c_dict[k]["data"])      
        c_dict[k]["std"] = np.std(c_dict[k]["data"])      
            
    sum_std = np.sqrt(np.square(c_dict["withins"]["std"]) + np.square(c_dict["betweens"]["std"]))   
    curv_gap = np.abs((c_dict["withins"]["mean"] - c_dict["betweens"]["mean"]) / sum_std)   
    
    return curv_gap


def optimization_func(a, G, cmp_key="value"):
    """
    Optimization function for the curvature gap.

    Parameters
    ----------
    a : array
        An array of initial values for the optimization.

    G : NetworkX graph
        An undirected graph.

    cmp_key : string
        The key for the community attribute in the graph.

    Returns
    -------
    c_gap : float
        The curvature gap of the graph.
    """
    t = a[0]
    if len(a) > 1:
        q = a[1]
    else:
        q = 0
    if len(a) > 2:
        p = a[2]
    else:
        p = 0
    get_edge_curvatures(G, t, q, p)
    c_gap = -1 * get_curvature_gap(G, "afrc5", cmp_key)

    return c_gap


def maximize_curvature_gap(G, a, cmp_key = "value"):
    """
    Maximize the curvature gap of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    a : array
        An array of initial values for the optimization.

    cmp_key : string
        The key for the community attribute in the graph.

    Returns
    -------
    results : array
        An array of the optimized curvature gap and the parameters.
    """
    results = []
    for i in range(len(a)):
        x0 = a[0:i+1]    # Initial values for t / t,q / t,q,p
        res = minimize(optimization_func, x0, method='nelder-mead', args = (G, cmp_key), options={'disp': False})
        results.extend([-1 * res.fun, *res.x])
    return results


def hbg_compute_curvature_gap(Gr, curv_name, cmp_key="prob"):
    
    c_dict = {"withins": {}, "betweens": {}}
    for k in c_dict.keys():
        c_dict[k][curv_name] = {"data": [], "mean": 0, "std": 0}
        
    for u,v,d in Gr.edges.data():                       
        if Gr.edges[u,v][cmp_key] == 1:                              
            c_dict["withins"][curv_name]["data"].append(Gr.edges[u,v][curv_name]) 
        else:                                               
            c_dict["betweens"][curv_name]["data"].append(Gr.edges[u,v][curv_name])       

    for k in c_dict.keys():
        c_dict[k][curv_name]["mean"] = np.mean(c_dict[k][curv_name]["data"])
        c_dict[k][curv_name]["std"] = np.std(c_dict[k][curv_name]["data"])
            
    res_diffs = {}

    sum_std = np.sqrt(np.square(c_dict["withins"][curv_name]["std"]) + np.square(c_dict["betweens"][curv_name]["std"]))   # Gesamt-Stdabw berechnen
    res_diffs[curv_name] = np.abs((c_dict["withins"][curv_name]["mean"] - c_dict["betweens"][curv_name]["mean"]) / sum_std)     # Differenz der Mittelwerte bilden und normieren
    
    return res_diffs




# Non-sequential community detection

def detect_communities_nonsequential(G, t_coeff=3, q_coeff=2):
    """
    Detect communities in a graph using non-sequential community detection,
    using the AFRC4 with expected weights. Only correct for an SBM.

    Parameters
    ----------
    G : graph
        A networkx graph

    t_coeff : int
        Coefficient for triangles, default = 3

    q_coeff : int
        Coefficient for quadrangles, default = 2

    Returns
    -------
    G : graph
        A networkx graph with node labels
    """    

    # get start values for curvature
    # get_edge_curvatures(G, t_coeff, q_coeff)
    # get min,max,values for afrc4 curvature
    # afrc_min, afrc_max = get_min_max_afrc_values(G, "afrc4")
    # show histogram of curvature values
    # show_curv_data (G, title_str = "", cmp_key = "block")

    afrc_threshold = int(input(
        "Enter threshold value for AFRC4 to remove edges with a higher value: "))
    
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
    return G


"""     NEED TO REPLACE THIS WITH THE IMPROVED ALGORITHM    """
def AugFormanSq(e,G):
    
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
    neigh_0= [i for i in G[e[0]] if i!=e[1]]
    for i in neigh_0:
        for j in (set(G[i]) & set(G[e[1]]) - {e[0]}):
            Sq +=1
            E[min(e[0],i)][max(e[0],i)] += 1
            E[min(e[1],j)][max(e[1],j)] += 1
            E[i][j] += 1
    
    
    FR += 2 + T + Sq

    for i in range(len(G)):
        for j in range(i,len(G)):
            FR += -abs(E[i][j]-E[j][i])
            
    return FR