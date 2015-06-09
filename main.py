"""
Description:

Python 3.4.1
Tom Dodson
Date: Tue Jun  9 16:22:15 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore

from generic_math import *
from patchy_chain import *

import matplotlib                                      #analysis:ignore
matplotlib.rcParams['svg.fonttype'] = 'none'

plt.close("all")


N_tot = 10

c = patchy_chain()
while len(c) < N_tot:
    c.attempt_add_particle()

c.visualize()
c.print_info()
