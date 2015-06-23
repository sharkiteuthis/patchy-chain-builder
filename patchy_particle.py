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


#average and stdev of particle radius
_gaussian_radii = False # controls monodispersity. If False monodisperse 
_r_avg = 1.0
_r_std =_r_avg/10

# average and stdev of particle separation
_gaussian_separation= False # If False, all particles are separated by the same distance
_dr_avg = _r_avg/10 # arbitrary 
_dr_std = _r_avg/(10**(3/2)) #arbitrary
_dr_same_separation=EPSILON 

# cutoff normal distributions to ensure correctness of cell lists
_normal_cutoff = 4          #cutoff at 4, units of std dev

def get_patchy_particle_interaction_cutoff():
    
    if _gaussian_radii:
        #TIP: I think this is safe and correct instead of worrying about the dr
        # offset which is by definition not part of the steric interaction
        return 2*(_r_avg + _normal_cutoff * _r_std)
    else:
        return 2*_r_avg

class particle():

    def __init__(self, p_ternary):
        self.radius = self._choose_radius()
        self.pos = np.zeros(3)
        self.rot_matrix = np.eye(3)
        self.deviation = np.zeros(3)

        if np.random.rand() < p_ternary:
            self.coordination = 3
        else:
            self.coordination = 2

        self.open_patches = list(range(self.coordination))

        # controls patch solid-angle width
        self.dphi = np.pi/12 
        self.dtheta = np.pi/12
        
    def _choose_radius(self):
        if _gaussian_radii:
            #bound this so we are guaranteed to have correct cell lists
            while True:
                r = np.random.normal(_r_avg,_r_std)
                if r > 0 and np.abs(_r_avg-r) < _normal_cutoff * _r_std:
                    break
        else:
            r=_r_avg

        return r

    def __repr__(self):
        return "Particle (" + str(self.coordination) + " patch): " + \
                "r={0:0.2f} pos={1:.2f},{2:.2f},{3:.2f}".format(self.radius,self.pos[0],self.pos[1],self.pos[2])

    #These are not *system* coordinates, they are local to the particle. This is
    # where we define the particle's patchiness
    def _get_local_patch_pos_spherical(self,ndx):
        if ndx == 0:
            theta = (np.pi)/2
            phi = 0
        elif ndx == 1:
            theta = (np.pi)/2
            phi = 2*(np.pi)/3
        else:
            theta = (np.pi)/2
            phi = 4*(np.pi)/3

        return np.asarray([self.radius,theta,phi])

    def _get_local_patch_pos_cartesian(self,ndx):
        return spherical_to_cartesian(self._get_local_patch_pos_spherical(ndx))
    
    # rotational degree of freedom between the two patches
    def _get_patch_axis_rotation_angle(self):
        return np.random.rand() * 2 * np.pi

    #beta is the rotation about the axis connecting the centers of the particles
    # i.e. the rotational degree of freedom in the patch
    def _set_rotation(self, i_patch, x_parent_patch, beta):
        # define a unit vector a which is normal to the patch vector and the
        # *negative* of the patch vector of the parent particle (since the axis
        # we are rotating about is pointing from the center of this particle)
        axis_patch = -1 * normalize(x_parent_patch)
        x_local_patch = normalize(self._get_local_patch_pos_cartesian(i_patch))
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
        
    # https://youtu.be/5iwf20t9J1k?t=34
    # "Introduce a little anarchy. Upset the established order, and
    # everything becomes chaos. I'm an agent of chaos. Oh, and you know the
    # thing about chaos? It's fair!"
    def get_patch_center_offset_spherical(self):
        if _gaussian_separation:
            while True:
                dr = np.random.normal(_dr_avg,_dr_std)
                if dr > 0:
                    break
        else:
            dr=_dr_same_separation

        dphi = np.random.normal(0,self.dphi)
        dtheta = np.random.normal(0,self.dtheta)

        return np.asarray([dr,dtheta,dphi])

    #This returns system coordinates
    def get_global_patch_pos(self,ndx):
        X = self._get_local_patch_pos_cartesian(ndx)
        X_system = np.dot(self.rot_matrix,X)
        return X_system


    #place the particle so that the center of a random patch is at x_attachment
    # patch_axis_rotation is the rotational degree of freedom between the patches
    def set_position(self, x_center, x_patch_center, r_deviation,beta_DEBUG= None):
        i_patch = self.pick_random_patch()
        self.close_patch(i_patch)

        self.r_deviation= r_deviation
        
        #this is the rotational degree of freedom that the patch has about the
        # axis connecting the centers of the particles
        if beta_DEBUG:
            patch_axis_rotation = beta_DEBUG
        else:
            patch_axis_rotation = self._get_patch_axis_rotation_angle()
        

        # have to add the offset from the center of the parent patch in spherical
        # coordinates to avoid doing a rotation.
        # TODO: check if rotation is faster
        r_patch_center = cartesian_to_spherical(x_patch_center)
        x_attach = spherical_to_cartesian(r_patch_center + r_deviation)

        self._set_rotation(i_patch, x_attach, patch_axis_rotation)

        self.pos = -1*self.get_global_patch_pos(i_patch) + x_center + x_attach

    def pick_random_patch(self):
        assert len(self.open_patches)
        return np.random.choice(self.open_patches)

    def close_patch(self,ndx):
        assert ndx in self.open_patches
        self.open_patches.remove(ndx)

