"""
Description:

Python 3.4.1
Tom Dodson
Date: Wed May 27 16:01:47 2015
"""

import numpy as np                                     #analysis:ignore
import matplotlib.pyplot as plt                        #analysis:ignore
from collections import defaultdict
import itertools

from generic_math import *
from patchy_particle import *

#TODO make threadsafe so that we can spawn multiple worker threads
#   to grow the chain. I think the only two mutexes needed are for modifying
#   the two dicts in chain
#http://effbot.org/zone/thread-synchronization.htm
#https://docs.python.org/3/library/threading.html
#import threading

################################################################################
################################################################################
#      patchy_chain class
################################################################################
################################################################################

class patchy_chain(object):
    def __init__(self):
        self.open_patch_dict = {}
        self.chain_dict = {}
        self.N = 0

        self.__define_chain_properties()

        #this is guaranteed correct since we have bounded our normal distributions
        # with self.normal_cutoff
        self.interaction_cutoff = self.r_avg + self.dr_avg + \
                                self.normal_cutoff * (self.r_std + self.dr_std)
        self.cell_range = 1
        self.cell_width = self.interaction_cutoff/self.cell_range
        self.cell_dict = defaultdict(list)

        #add the first particle at the origin
        p = particle(self._choose_radius(), self.p_ternary_coordination)
        self.chain_dict[self.N] = p
        self.open_patch_dict[self.N] = len(p.open_patches)
        self.add_particle_to_cell_list(p)
        self.N += 1

        print("First particle:")
        print("\t",p)


    def __define_chain_properties(self):
        #p of making a coordination 3 node
        self.p_ternary_coordination = 0.0

        #average and stdev of particle radius
        self.r_avg = 1.0
        self.r_std = 0.1

        #average and stdev of particle separation
        #I know shit looks crazy, but the average dr is chosen to be 10 orders
        #of magnitude smaller than r, and the std deviation of dr is chosen to be
        # half an order of magnitude smaller than dr. Why? Just because.
        self.dr_avg = self.r_avg/10
        self.dr_std = self.r_avg/(10**(3/2))

        # cutoff normal distributions to ensure correctness of cell lists
        self.normal_cutoff = 4          #cutoff at 4, units of std dev

        #  *shrug*
        self.dphi = np.pi/6
        self.dtheta = np.pi/6

    # rotational degree of freedom between the two patches
    def _get_patch_axis_rotation_angle():
        return np.random.rand() * 2 * np.pi

    def _choose_radius(self):
        #bound this so we are guaranteed to have correct cell lists
        r = np.random.normal(self.r_avg,self.r_std)
        while r < 0 or np.abs(self.r_avg-r) > self.normal_cutoff * self.r_std:
            print("Produced inappropriate r in _choose_radius():",r)
            r = np.random.normal(self.r_avg,self.r_std)

        return r

    # https://youtu.be/5iwf20t9J1k?t=34
    # "Introduce a little anarchy. Upset the established order, and
    # everything becomes chaos. I'm an agent of chaos. Oh, and you know the
    # thing about chaos? It's fair!"
    def _get_random_offset(self):
        #bound this so that dr > 0 and we are guaranteed to have correct cell lists
        dr = np.random.normal(self.dr_avg,self.dr_std)
        while dr < 0 or np.abs(self.dr_avg-dr) > self.normal_cutoff * self.dr_std:
            print("Produced inappropriate dr in get_random_offset():",dr)
            dr = np.random.normal(self.dr_avg,self.dr_std)

        dphi = np.random.normal(0,self.dphi)
        dtheta = np.random.normal(0,self.dtheta)

        return spherical_to_euclid(np.asarray([dr,dtheta,dphi]))

    def _generate_cells_to_check(self,c):
        #TIP: if reducing cutoff range, implement the below check
        assert self.cell_range <= 2
        cell_range_to_check = lambda : itertools.chain(range(-self.cell_range,0),range(self.cell_range+1))
        #cell_dist = lambda c1,c2 : self.cell_width * dist_euclid(np.asarray(c1), np.asarray(c2))
        for i in cell_range_to_check():
            for j in cell_range_to_check():
                for k in cell_range_to_check():
                    #if cell_dist((c[0]+i,c[1]+j,c[2]+k), (i,j,k)) <= 2 * self.interaction_cutoff:
                    yield (i,j,k)

    # simple steric interaction for now
    def _calculate_accept(self,p_new):
        new_cell = self.get_cell_list_ndx(p_new)
        for c in self._generate_cells_to_check(new_cell):
            for p in self.cell_dict[c]:
                if dist_euclid(p_new.pos,p.pos) < (p_new.radius + p.radius):
                    print("\tRejecting new particle")
                    print("\tdist = ",dist_euclid(p_new.pos,p.pos))
                    print("\tradii = ", p_new.radius + p.radius)
                    print("\trejecting due to particle:")
                    print("\t",p)
                    return False

        return True

    def __len__(self):
        return self.N

    #needs to return a tuple - ndarray is not hashable
    def get_cell_list_ndx(self, p):
        return (int(np.mod(p.pos[0],self.cell_width)), \
                int(np.mod(p.pos[1],self.cell_width)), \
                int(np.mod(p.pos[2],self.cell_width)))

    def add_particle_to_cell_list(self,p):
        ndx = self.get_cell_list_ndx(p)
        self.cell_dict[ndx].append(p)


    def make_new_random_particle(self, p_orig, ndx_patch):
        p_new = particle(self._choose_radius(), self.p_ternary_coordination)

        #this is the rotational degree of freedom that the patch has about the
        # axis connecting the centers of the particles
        beta = self._get_patch_axis_rotation_angle()

        attach_pos = p_orig.get_system_patch_pos(ndx_patch) + self._get_random_offset()
        p_new.set_position(p_orig.pos, attach_pos, beta)

        return p_new

    def attempt_add_particle(self):
        assert len(self.open_patch_dict)

        #pick a random open end of the chain
        i_part = np.random.choice(list(self.open_patch_dict.keys()))
        p_orig = self.chain_dict[i_part]

        #choose a random patch and create a new particle there
        i_patch = p_orig.pick_random_patch()
        p_new = self.make_new_random_particle(p_orig, i_patch)

        print("Attempting to add a particle at particle",i_part,"patch",i_patch)
        print("\t",p_new)

        accept = self._calculate_accept(p_new)

        #the new particle is acceptable. Bookkeepping time!
        if accept:
            print("\tAccepting new particle.")
            self.do_add_particle(p_new, p_orig, i_part, i_patch)

    def do_add_particle(self, p_new, p_orig, orig_part, orig_patch):
        self.chain_dict[self.N] = p_new     #make sure we can access the particle
        self.open_patch_dict[self.N] = len(p_new.open_patches)
        self.N +=1

        #check if we have closed off the chain at the particle where we added
        #the new particle
        p_orig.close_patch(orig_patch)
        n_open = len(p_orig.open_patches)
        if n_open == 0:
            self.open_patch_dict.pop(orig_part, None)
        else:
            self.open_patch_dict[orig_part] = n_open

        self.add_particle_to_cell_list(p_new)

    def visualize(self):
        #stole this kludge from here:
        # http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
        def axisEqual3D(ax):
            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        def draw_sphere(r,orig,ax):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = orig[0] + r * np.cos(u)*np.sin(v)
            y = orig[1] + r * np.sin(u)*np.sin(v)
            z = orig[2] + r * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in self.chain_dict:
            radius = self.chain_dict[i].radius
            x = self.chain_dict[i].pos
            #TODO: Color by open patches?
            #TODO: size particles appropriately
            ax.scatter(x[0],x[1],x[2])
            #draw_sphere(radius,x,ax)
            ax.text(x[0],x[1],x[2],str(i))

        axisEqual3D(ax)
        plt.show()

    def print_info(self):
        for i in self.chain_dict:
            print(self.chain_dict[i].pos)
        for i in self.open_patch_dict:
            print("Partcle",i,"has",self.open_patch_dict[i],"open patches.")

