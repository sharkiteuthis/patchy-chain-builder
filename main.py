"""
Description:

Python 3.4.1
Tom Dodson
Date: Tue Jun  9 16:22:15 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore
from mpl_toolkits.mplot3d import Axes3D                #analysis:ignore

from generic_math import *                             #analysis:ignore
from patchy_chain import *                             #analysis:ignore

import matplotlib                                      #analysis:ignore
matplotlib.rcParams['svg.fonttype'] = 'none'

plt.close("all")

N_tot = 20

c = patchy_chain(0.05)  # 5% chance of creating a coordination 3 particle
while len(c) < N_tot:
    c.attempt_add_particle()

c.visualize()
#c.print_info()

#TODO: the pdb writer still isn't working very well - for some reason PyMol will
# randomly move atoms around - see the pdb_bug folder
#
#To make a nice looking PyMol chain:
#
# color red, name cq; show spheres; set sphere_scale, 0.75
#
c.write_pdb(open("chain.pdb",'w'))
