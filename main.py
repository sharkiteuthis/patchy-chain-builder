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

N_tot = 1000

c = patchy_chain(0.05)  # 5% chance of creating a coordination 3 particle
while len(c) < N_tot:
    c.attempt_add_particle()

#TODO put bounds on visualize?
#c.visualize()
#c.print_info()
c.write_pdb(open("chain.pdb",'w'))
