"""
Description:

Python 3.4.1
Tom Dodson
Date: Thu Jun 11 20:49:16 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore

from generic_math import *
from patchy_particle import *
from patchy_chain import *

################################################################################
################################################################################
#      generic math unit tests
################################################################################
################################################################################

import unittest

class TestMath(unittest.TestCase):

    def test_spherical_to_cartesian(self):
        x = spherical_to_cartesian(np.asarray([1,0,0]))
        self.assertAlmostEqual(x[0],0,delta=EPSILON)
        self.assertAlmostEqual(x[1],0,delta=EPSILON)
        self.assertAlmostEqual(x[2],1,delta=EPSILON)

        x = spherical_to_cartesian(np.asarray([1,2*np.pi/3,np.pi]))
        self.assertAlmostEqual(x[0],-1*np.sqrt(3)/2,delta=EPSILON)
        self.assertAlmostEqual(x[1],0,delta=EPSILON)
        self.assertAlmostEqual(x[2],-0.5,delta=EPSILON)

        x = spherical_to_cartesian(np.asarray([0.09, 0.71, 3.32]))
        self.assertAlmostEqual(x[0],-0.057734,delta=EPSILON)
        self.assertAlmostEqual(x[1],-0.010411,delta=EPSILON)
        self.assertAlmostEqual(x[2],0.068253,delta=EPSILON)

        x = spherical_to_cartesian(np.asarray([1, -1.636, -2.111]))
        self.assertAlmostEqual(x[0],0.513218,delta=EPSILON)
        self.assertAlmostEqual(x[1],0.855782,delta=EPSILON)
        self.assertAlmostEqual(x[2],-0.065157,delta=EPSILON)

        with self.assertRaises(AssertionError):
            spherical_to_cartesian(np.asarray([-1, 0, 0]))


    def test_cartesian_to_spherical(self):
        x = cartesian_to_spherical(np.asarray([1,0,0]))
        self.assertAlmostEqual(x[0],1,delta=EPSILON)
        self.assertAlmostEqual(x[1],np.pi/2,delta=EPSILON)
        self.assertAlmostEqual(x[2],0,delta=EPSILON)

        x = cartesian_to_spherical(np.asarray([0,0,1]))
        self.assertAlmostEqual(x[0],1,delta=EPSILON)
        self.assertAlmostEqual(x[1],0,delta=EPSILON)
        self.assertAlmostEqual(x[2],0,delta=EPSILON)

        x = cartesian_to_spherical(np.asarray([1,1,0]))
        self.assertAlmostEqual(x[0],np.sqrt(2),delta=EPSILON)
        self.assertAlmostEqual(x[1],np.pi/2,delta=EPSILON)
        self.assertAlmostEqual(x[2],np.pi/4,delta=EPSILON)

        x = cartesian_to_spherical(np.asarray([ 0.87,  0.45,  0.73]))
        self.assertAlmostEqual(x[0],1.221597,delta=EPSILON)
        self.assertAlmostEqual(x[1],0.930319,delta=EPSILON)
        self.assertAlmostEqual(x[2],0.477345,delta=EPSILON)

        x = cartesian_to_spherical(np.asarray([ -0.87,  -0.45,  0.73]))
        self.assertAlmostEqual(x[0],1.221597,delta=EPSILON)
        self.assertAlmostEqual(x[1],0.930319, delta=EPSILON)
        self.assertAlmostEqual(x[2],-2.66424727, delta=EPSILON)

    def test_rotation_matrix(self):
        R = get_rotation_matrix_axis_angle([0,0,1],np.pi/4)
        q = [1,0,0]
        x = np.dot(R,q)
        self.assertAlmostEqual(x[0],np.sqrt(2)/2,delta=EPSILON)
        self.assertAlmostEqual(x[1],np.sqrt(2)/2,delta=EPSILON)
        self.assertAlmostEqual(x[2],0,delta=EPSILON)

        q = [np.sqrt(2)/2,-np.sqrt(2)/2,0]
        x = np.dot(R,q)
        self.assertAlmostEqual(x[0],1,delta=EPSILON)
        self.assertAlmostEqual(x[1],0,delta=EPSILON)
        self.assertAlmostEqual(x[2],0,delta=EPSILON)

        R = get_rotation_matrix_axis_angle(normalize([1,1,0]),np.pi/2)
        q = [0,0,1]
        x = np.dot(R,q)
        self.assertAlmostEqual(x[0],np.sqrt(2)/2,delta=EPSILON)
        self.assertAlmostEqual(x[1],-np.sqrt(2)/2,delta=EPSILON)
        self.assertAlmostEqual(x[2],0,delta=EPSILON)


class TestParticle(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_set_postition(self):
        p = particle(1)
        s = np.asarray([1, np.pi/2, 2*np.pi/3])
        a = spherical_to_cartesian(s)
        p.set_position(np.asarray([0,0,0]), spherical_to_cartesian(s), np.asarray([0,0,0]),beta_DEBUG=0)

        #center position should be twice the connecting axis
        self.assertAlmostEqual(p.pos[0], 2*a[0], delta=EPSILON)
        self.assertAlmostEqual(p.pos[1], 2*a[1], delta=EPSILON)
        self.assertAlmostEqual(p.pos[2], 2*a[2], delta=EPSILON)

        #we know this is the patch that needs to lie along the axis because
        # with the random seed of 42 it will choose to attach to patch 0
        x1 = p.get_global_patch_pos(0)
        self.assertAlmostEqual(x1[0],-a[0],delta=EPSILON)
        self.assertAlmostEqual(x1[1],-a[1],delta=EPSILON)
        self.assertAlmostEqual(x1[2],-a[2],delta=EPSILON)
        #this should be rotated by d_phi = -pi/3 about z, so phi = pi/3
        x2 = p.get_global_patch_pos(1)
        self.assertAlmostEqual(x2[0],np.cos(np.pi/3),delta=EPSILON)
        self.assertAlmostEqual(x2[1],np.sin(np.pi/3),delta=EPSILON)
        self.assertAlmostEqual(x2[2],0,delta=EPSILON)
        #this should be rotated by d_phi = -pi, so phi=pi
        x3 = p.get_global_patch_pos(2)
        self.assertAlmostEqual(x3[0],np.cos(np.pi),delta=EPSILON)
        self.assertAlmostEqual(x3[1],np.sin(np.pi),delta=EPSILON)
        self.assertAlmostEqual(x3[2],0,delta=EPSILON)

        #now if we do it with a rotation about the patch axis by pi, it should
        # just flip the other two patches
        p = particle(1)
        s = np.asarray([1, np.pi/2, 2*np.pi/3])
        a = spherical_to_cartesian(s)
        p.set_position(np.asarray([0,0,0]), spherical_to_cartesian(s), np.asarray([0,0,0]), beta_DEBUG=np.pi)

        #center position should be twice the connecting axis
        self.assertAlmostEqual(p.pos[0], 2*a[0], delta=EPSILON)
        self.assertAlmostEqual(p.pos[1], 2*a[1], delta=EPSILON)
        self.assertAlmostEqual(p.pos[2], 2*a[2], delta=EPSILON)

        #we know this is the patch that needs to lie along the axis because
        # with the random seed of 42 it will choose to attach to patch 0
        x1 = p.get_global_patch_pos(0)
        self.assertAlmostEqual(x1[0],-a[0],delta=EPSILON)
        self.assertAlmostEqual(x1[1],-a[1],delta=EPSILON)
        self.assertAlmostEqual(x1[2],-a[2],delta=EPSILON)
        x2 = p.get_global_patch_pos(1)
        self.assertAlmostEqual(x2[0],np.cos(np.pi),delta=EPSILON)
        self.assertAlmostEqual(x2[1],np.sin(np.pi),delta=EPSILON)
        self.assertAlmostEqual(x2[2],0,delta=EPSILON)
        x3 = p.get_global_patch_pos(2)
        self.assertAlmostEqual(x3[0],np.cos(np.pi/3),delta=EPSILON)
        self.assertAlmostEqual(x3[1],np.sin(np.pi/3),delta=EPSILON)
        self.assertAlmostEqual(x3[2],0,delta=EPSILON)

class TestChain(unittest.TestCase):
    def setUp(self):
        x = np.random.randint(1<<10)
        np.random.seed(x)
        print("TestChain random seed:,",x)

    def test_cell_list(self):
        c = patchy_chain(1.0)
        x = list(c._generate_cells_to_check((5,5,5),1.0))
        self.assertTrue(len(x) == 27)
        #TODO: check all the cells to make sure we're getting the correct ones...
        self.assertTrue((6,6,6) in x)

    #make sure that when we make a new particle, it *never* conflicts with its parent
    def test_no_parent_reject(self):
        for i in range(10000):
            c = patchy_chain(1.0)
            self.assertTrue(len(c) == 1)
            c.attempt_add_particle()
            if len(c) != 2:
                print("TestChain test (test_no_parent_reject)",i,"failed.")
            self.assertTrue(len(c) == 2)

    def test_chain_for_ovelaps(self):
        for i in range(100):
            c = patchy_chain(0.1)
            self.assertTrue(len(c) == 1)

            N_tot = 100
            while(len(c) < N_tot):
                c.attempt_add_particle()

            self.assertTrue(len(c) == N_tot)
            overlap = c._check_all_particles_for_overlap_DEBUG()
            if overlap:
                print("TestChain test (test_chain_for_overlaps)",i,"failed.")
            self.assertFalse(overlap)

    #TODO: a lot more unit tests for this class....

unittest.main()
