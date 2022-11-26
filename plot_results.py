"""
plot_results.py

Created on Sat Nov 26 09:26:00 2022

@author: Lukas

This script contains functions to plot graphs and their curvatures.
"""

# load packages using the setup.py script

import setup


# define functions

def show_curv_min_max_values (h_data):
    """
    This function prints the minimum and maximum values of the curvatures.

    Parameters
    ----------
    h_data : dict
        Dictionary containing the curvatures.

    Returns
    -------
    None.
    """
    print("\nMin/Max Curvature values:")
    for k in h_data.keys():
        print(str(k).ljust(8), 
              "{0:<5s} {1:8.4f}".format("Min:", h_data[k]["bin_min"]), "  ",
              "{0:<5s} {1:8.4f}".format("Max:", h_data[k]["bin_max"])
              )
    print()


def get_bin_width (b_min, b_max, num_bin_lim):
    """
    This function calculates the bin width for a given range of values and a given number of bins.

    Parameters
    ----------
    b_min : float
        Minimum value of the range.
    b_max : float
        Maximum value of the range.
    num_bin_lim : int
        Number of bins.

    Returns
    -------
    bin_width : float
        Bin width.
    """
    scaling = 1
    multiplier = 10
    # print("b_min:", b_min, "b_max:", b_max, "num_bin_lim:", num_bin_lim, "scaling:", scaling, "multiplier:", multiplier)
    b_width = (b_max - b_min) // 40 + 1
    if abs(b_max) < 1 and abs(b_min) < 1:
        while (b_max - b_min)/scaling < num_bin_lim / 10:
            scaling /= multiplier    
        b_width = scaling
    return b_width


def show_histos (h_data, title_str, my_nrows = 2, my_ncols = 3, my_bin_num = 40):
    """
    This function plots histograms of the curvatures.

    Parameters
    ----------
    h_data : dict
        Dictionary containing the curvatures.
    title_str : str
        Title of the plot.
    my_nrows : int, optional
        Number of rows in the plot. The default is 2.
    my_ncols : int, optional
        Number of columns in the plot. The default is 3.
    my_bin_num : int, optional
        Number of bins in the histograms. The default is 40.

    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(nrows=my_nrows, ncols=my_ncols, sharey = True, figsize=(16,10))
    for i,k in enumerate(h_data.keys()):
        r = i // my_ncols
        c = i % my_ncols
        bin_width = get_bin_width(h_data[k]["bin_min"], h_data[k]["bin_max"], my_bin_num)
        axes[r,c].hist(h_data[k]["curv"], bins = np.arange(h_data[k]["bin_min"], h_data[k]["bin_max"] + bin_width, bin_width), edgecolor = "white")
        axes[r,c].set_title(h_data[k]["title"])
        axes[r,c].title.set_size(16)
        axes[r,c].tick_params(axis='both', labelsize=16)
        axes[r,c].grid(visible=True, axis="both")
    fig.suptitle(title_str, size=16)
    plt.show()


def show_correlation_coeffs (h_data):
    """
    This function prints the correlation coefficients between the curvatures.

    Parameters
    ----------
    h_data : dict
        Dictionary containing the curvatures.

    Returns
    -------
    None.
    """
    print("\nCorrelation coefficients:")
    ks = list(h_data.keys())
    for i,k in enumerate(ks):
        for l in ks[i+1:]:
            s = h_data[k]["title"] + " / " + h_data[l]["title"]
            c = np.corrcoef(h_data[k]["curv"], h_data[l]["curv"])[1][0]
            print(s.ljust(55,"."), f"{c:8.5f}")
        print()


def show_curv_data (G, title_str):
    """
    This function prints the minimum and maximum values of the curvatures and plots histograms of the curvatures.

    Parameters
    ----------
    G : Graph
    title_str : str
        Title of the plot.

    Returns
    -------
    None.
    """
    h_data = {"orc":  {"curv": [d["orc"]   for u,v,d in G.edges.data()], "bin_min":0, "bin_max":0, "title":"Ollivier Ricci (OR)"},
              "frc":  {"curv": [d["frc"]   for u,v,d in G.edges.data()], "bin_min":0, "bin_max":0, "title":"Forman Ricci (FR)"},
              "afrc": {"curv": [d["afrc"]  for u,v,d in G.edges.data()], "bin_min":0, "bin_max":0, "title":"Augm. FR curv. (triangles)"},
              "afrc4":{"curv": [d["afrc4"] for u,v,d in G.edges.data()], "bin_min":0, "bin_max":0, "title":"AFR curv. (tri/quad)"},
              "afrc5":{"curv": [d["afrc5"] for u,v,d in G.edges.data()], "bin_min":0, "bin_max":0, "title":"AFR curv. (tri/quad/pent)"}
              }
    
    for k in h_data.keys():
        # print("h_data.keys: ", k)
        # h_data[k]["bin_min"] = int(min(h_data[k]["curv"]))
        # h_data[k]["bin_max"] = int(max(h_data[k]["curv"]))
        h_data[k]["bin_min"] = min(h_data[k]["curv"])
        h_data[k]["bin_max"] = max(h_data[k]["curv"])
        
    show_curv_min_max_values (h_data)
    show_histos (h_data, title_str, my_nrows = 2, my_ncols = 3, my_bin_num = 40)
    show_correlation_coeffs(h_data)