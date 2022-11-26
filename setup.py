"""
setup.py

Created on Sat Nov 26 09:03:00 2022

@author: Lukas

This file installs the GraphRicciCurvature package and imports all other packages needed in the repository.
"""

# install GraphRicciCurvature package

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'GraphRicciCurvature'])

# import packages

import networkx as nx
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci