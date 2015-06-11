"""
Description: generic math stuffs i might want

Python 3.4.1
Tom Dodson
Date: Tue Jun  9 16:18:39 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore

#define "effectively zero" for e.g. checking norms of vectors
EPSILON = 1e-15

#theta is zenith measured from z axis (0,pi)
#phi is azimuthal measured from the x-axis (0,2*pi)
#i.e. physics convention
def spherical_to_euclid(s):
    assert s.shape == (3,)
    r = s[0]
    theta = s[1]
    phi = s[2]
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return r * np.asarray([x,y,z])

def euclid_to_spherical(e):
    assert e.shape == (3,)
    r = np.sqrt(np.sum(e**2))
    theta = np.arccos(e[2]/r)
    phi = np.arctan2(e[1],e[0])
    return np.asarray([r,theta,phi])

def dist_euclid(a,b):
    assert a.shape == b.shape
    assert a.shape == (3,)
    return np.linalg.norm(a-b)

def dist_spherical(s1,s2):
    assert s1.shape == (3,)
    assert s2.shape == (3,)

    #TODO you can save 2 sin/cos calls by computing the explicit expression
    # and using that here
    e1 = spherical_to_euclid(s1)
    e2 = spherical_to_euclid(s2)

    return dist_euclid(e1,e2)

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

#generates matrix to align a to b - both must be unit vectors
#def get_rotation_matrix_two_vec(q, u):
#    a = np.cross(q,u)
#    ax = a[0]
#    ay = a[1]
#    az = a[2]
#    s = np.linalg.norm(a)
#    c = np.dot(q,u)
#    C = 1-c
#
#    return np.asarray([[c + ax*ax*C, ax*ay*C - az*s, ax*az*C + ay*s],
#                       [ay*ax*C + az*s, c + ay*ay*C, ay*az*C - ax*s],
#                       [az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C]],dtype='float64')

def get_rotation_matrix_axis_angle(a,alpha):
    ax = a[0]
    ay = a[1]
    az = a[2]
    s = np.sin(alpha)
    c = np.cos(alpha)
    C = 1-c

    return np.asarray([[c + ax*ax*C, ax*ay*C - az*s, ax*az*C + ay*s],
                       [ay*ax*C + az*s, c + ay*ay*C, ay*az*C - ax*s],
                       [az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C]],dtype='float64')
