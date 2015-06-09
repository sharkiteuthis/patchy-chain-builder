"""
Description:

Python 3.4.1
Tom Dodson
Date: Tue Jun  9 16:20:09 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore

from generic_math import *

################################################################################
################################################################################
#      particle class
################################################################################
################################################################################

#this is a shittty class that doesn't follow any of the "new" best practices such as
#    http://stackoverflow.com/questions/21584812/python-classes-best-practices
class particle():

    def __init__(self, radius, p_ternary):
        self.radius = radius
        self.pos = np.zeros(3)
        self.rotation_spherical = np.zeros(3)

        self.deviation_spherical = np.zeros(3)  #random deviation from neighbor patch,
                                                # only stored for later analysis

#TODO this is where we can allow coordination 3 particles
        assert p_ternary == 0.0
        self.coordination = 2
        self.open_patches = [0,1]

    def __repr__(self):
        return "Particle (" + str(self.coordination) + " patch): " + \
                "r={0:0.2f} pos={1:.2f},{2:.2f},{3:.2f}".format(self.radius,self.pos[0],self.pos[1],self.pos[2])

    #These are not *system* coordinates, they are local to the particle
    def get_local_patch_pos_spherical(self,ndx):
        if ndx == 0:
            theta = (np.pi)/2 + self.pos[1]
            phi = 0 + self.pos[2]
        elif ndx == 1:
            theta = (np.pi)/2 + self.pos[1]
            phi = 2*(np.pi)/3 + self.pos[2]
        else:
            theta = (np.pi)/2 + self.pos[1]
            phi = 4*(np.pi)/3 + self.pos[2]

        return np.asarray([self.radius,theta,phi])

    #This returns system coordinates
    def get_system_patch_pos(self,ndx):
        R = self.get_local_patch_pos_spherical(ndx) + self.rotation
        return self.pos + spherical_to_euclid(R)

#TODO: this is all fucked. and I forgot an entire rotational degree of freedom.
#TODO: AAADFDDDASHUKLASDFGKLSDAL:I!O!JKLsdfD#$$##%


    def _set_rotation(self,i_patch,deviation_spherical):
        self.rotation_spherical = self.get_local_patch_pos_spherical(i_patch) + \
                                    deviation_spherical
        self.rotation_spherical[0] = 0
        self.rotation_spherical[1] -= np.pi/2
        self.rotation_spherical[2] -= np.pi

    def set_position(self, x, deviation_spherical):
        i_patch = self.pick_random_patch()
        self.close_patch(i_patch)

        self._set_rotation(i_patch, deviation_spherical)

        self.pos = self.get_system_patch_pos(i_patch) + x + spherical_to_euclid(deviation_spherical)

        self.deviation_spherical = deviation_spherical

    def pick_random_patch(self):
        assert len(self.open_patches)
        return np.random.choice(self.open_patches)

    def close_patch(self,ndx):
        assert ndx in self.open_patches
        self.open_patches.remove(ndx)