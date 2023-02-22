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

show_line_plots()


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