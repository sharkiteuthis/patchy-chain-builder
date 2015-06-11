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
        self.rot_matrix = np.zeros((3,3))

#TODO this is where we can allow coordination 3 particles
        assert p_ternary == 0.0
        self.coordination = 2
        self.open_patches = [0,1]

    def __repr__(self):
        return "Particle (" + str(self.coordination) + " patch): " + \
                "r={0:0.2f} pos={1:.2f},{2:.2f},{3:.2f}".format(self.radius,self.pos[0],self.pos[1],self.pos[2])

    #These are not *system* coordinates, they are local to the particle. This is
    # where we define the particle's patchiness
    def _get_local_patch_pos_spherical(self,ndx):
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

    def _get_local_patch_pos_euclidean(self,ndx):
        return self._get_local_patch_pos_spherical(ndx)


    #beta is the rotation about the axis connecting the centers of the particles
    # i.e. the rotational degree of freedom in the patch
    def _set_rotation(self, i_patch, x_parent_patch, beta):
        # define a unit vector a which is normal to the patch vector and the
        # *negative* of the patch vector of the parent particle (since the axis
        # we are rotating about is pointing from the center of this particle)
        axis_patch = -1 * normalize(x_parent_patch)
        x_local_patch = normalize(_get_local_patch_pos_euclidean(i_patch))
        a = normalize(np.cross(x_local_patch, axis_patch))

        if np.abs(np.linalg.norm(a)) < EPSILON:
            R1 = np.eye(3)
        else:
            #angle between the two lies between zero and pi
            alpha = np.arccos(np.dot(x_local_patch, axis_patch))
            #R1 = get_rotation_matrix_two_vec(x_local_patch, axis_patch)
            R1 = get_rotation_matrix_axis_angle(a,alpha)

        R2 = get_rotation_matrix_axis_angle(axis_patch,beta)

        self.rot_matrix = np.dot(R2,R1)

    #This returns system coordinates
    def get_global_patch_pos(self,ndx):
        R = self.get_local_patch_pos_spherical(ndx)
        X = spherical_to_euclid(R)
        X_system = np.dot(self.rot_matrix,X)
        return X_system


    #place the particle so that the center of a random patch is at x_attachment
    # patch_axis_rotation is the rotational degree of freedom between the patches
    def set_position(self, x_center, x_patch, patch_axis_rotation):
        i_patch = self.pick_random_patch()
        self.close_patch(i_patch)

        self._set_rotation(i_patch, x_patch, patch_axis_rotation)

        self.pos = self.get_system_patch_pos(i_patch) + x_center + x_patch

    def pick_random_patch(self):
        assert len(self.open_patches)
        return np.random.choice(self.open_patches)

    def close_patch(self,ndx):
        assert ndx in self.open_patches
        self.open_patches.remove(ndx)