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
    def __init__(self,p_ternary,verbose_level=0):
        #p of making a coordination 3 node
        self.p_ternary_coordination = p_ternary

        self.verbose_level = verbose_level

        self.open_patch_dict = {}
        self.chain_dict = {}
        self.N = 0
        self.N3 = 0

        self.interaction_cutoff = get_patchy_particle_interaction_cutoff()

        #TODO cell-list-optimize: I think the optimal value here is probably
        # cell_range = 2, since that reduces spurious comparisons, but it also
        # introduces additional checks in _generate_cells_to_check, so that
        # tradeoff needs to be profiled
        self.cell_range = 1
        self.cell_width = self.interaction_cutoff/self.cell_range
        self.cell_dict = defaultdict(list)

        #add the first particle at the origin
        p = particle(self.p_ternary_coordination)
        self.chain_dict[self.N] = p
        self.open_patch_dict[self.N] = len(p.open_patches)
        self.add_particle_to_cell_list(p)
        self.N += 1

        if self.verbose_level >= 1:
            print("First particle:")
            print("\t",p)



    # totally dumb O(N^2) check for particle overlaps, for debugging purposes
    def _check_all_particles_for_overlap_DEBUG(self):
        for i in self.chain_dict:
            p1 = self.chain_dict[i]
            for j in self.chain_dict:
                p2 = self.chain_dict[j]

                #self-checks are bad, also eliminate duplicate checks
                if i < j:
                    if dist_cartesian(p1.pos,p2.pos) < (p1.radius + p2.radius):
                        return True
        return False

    def _generate_cells_to_check(self,c,this_r):
        #TODO cell-list-optimize: if this assert starts triggering, search for
        # cell-list-optimize and make sure that we aren't doing spurious cell list
        # checks, etc
        # NONE of the cell-list-optimize code has been tested, unit-tested, etc
        assert self.cell_range == 1

        #TODO cell-list-optimize: avoid calling into the one in generic_math since
        # it uses np.asarray() which is somewhat expensive and not needed here
        # (that's worth checking with the profiler, but is probably true)
        #
        #def tuple_cartesian_dist(a,b):
        #    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
        #
        #cell_dist = lambda c1,c2 : self.cell_width * tuple_cartesian_dist(c1,c2)

        cell_range_to_check = lambda : itertools.chain(range(-self.cell_range,0),range(self.cell_range+1))

        for i in cell_range_to_check():
            for j in cell_range_to_check():
                for k in cell_range_to_check():
                    new_cell = (c[0]+i,c[1]+j,c[2]+k)
                    #TODO cell-list-optimize: if reducing cutoff range, implement the below check
                    #       and comment out the other yield
                    #if cell_dist(new_cell, c) - (this_r + self.interaction_cutoff) > EPSILON:
                    #   yield new_cell
                    yield new_cell

    # simple steric interaction for now
    def _calculate_accept(self,p_new):
        new_cell = self.get_cell_list_ndx(p_new)
        for c in self._generate_cells_to_check(new_cell, p_new.radius):
            for p in self.cell_dict[c]:
                if dist_cartesian(p_new.pos,p.pos) < (p_new.radius + p.radius):
                    if self.verbose_level == 2:
                        print("\tRejecting new particle")
                        print("\tdist = ",dist_cartesian(p_new.pos,p.pos))
                        print("\tradii = ", p_new.radius + p.radius)
                        print("\trejecting due to particle:")
                        print("\t",p)
                    return False

        return True

    def __len__(self):
        return self.N

    def __repr__(self):
        return "Chain of {} particles, {} of coordination 3.".format(self.N,self.N3)

    #level 0 - default. Nothing.
    #level 1 - basic info
    #level 2 - complete info
    def set_verbose_level(self,level):
        self.verbose_level = level

    #needs to return a tuple - ndarray is not hashable
    def get_cell_list_ndx(self, p):
        #int() is effectively floor(), which is necessary (as opposed to round())
        return (int(p.pos[0]/self.cell_width), \
                int(p.pos[1]/self.cell_width), \
                int(p.pos[2]/self.cell_width))

    def add_particle_to_cell_list(self,p):
        ndx = self.get_cell_list_ndx(p)
        self.cell_dict[ndx].append(p)


    def make_new_random_particle(self, p_orig, ndx_patch):
        p_new = particle(self.p_ternary_coordination)


        x_patch_center = p_orig.get_global_patch_pos(ndx_patch)
        r_deviation = p_orig.get_patch_center_offset_spherical()

        p_new.set_position(p_orig.pos, x_patch_center, r_deviation)

        return p_new

    def attempt_add_particle(self):
        assert len(self.open_patch_dict)

        #pick a random open end of the chain
        i_part = np.random.choice(list(self.open_patch_dict.keys()))
        p_orig = self.chain_dict[i_part]

        #choose a random patch and create a new particle there
        i_patch = p_orig.pick_random_patch()

        while True:
            p_new = self.make_new_random_particle(p_orig,i_patch)
            if p_new != None:
                break

        if self.verbose_level == 2:
            print("Attempting to add a particle at particle",i_part,"patch",i_patch)
            print("\t",p_new)

        accept = self._calculate_accept(p_new)

        #the new particle is acceptable. Bookkeepping time!
        if accept:
            if self.verbose_level >= 1 and np.mod(N,10) == 0:
                print("\tAccepting new particle ({}).".format(self.N))
            self.do_add_particle(p_new, p_orig, i_part, i_patch)

    def do_add_particle(self, p_new, p_orig, orig_part, orig_patch):
        #make sure we can access the particle
        self.chain_dict[self.N] = p_new
        self.open_patch_dict[self.N] = len(p_new.open_patches)
        self.N +=1
        if p_new.coordination == 3:
            self.N3 += 1

        #check if we have closed off the chain at the particle where we added
        #the new particle
        p_orig.close_patch(orig_patch)
        n_open = len(p_orig.open_patches)
        if n_open == 0:
            self.open_patch_dict.pop(orig_part, None)
        else:
            self.open_patch_dict[orig_part] = n_open

        self.add_particle_to_cell_list(p_new)

    def visualize(self,spheres=True,numbers=False):
        #stole this kludge from here:
        # http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
        def axis_equal_3D(ax):
            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        def bounding_box(ax,ax_min,ax_max):
            for x in (ax_min[0],ax_min[0]):
                for y in (ax_min[1],ax_min[1]):
                    for z in (ax_min[2],ax_min[2]):
                        ax.plot([x,y,z], 'w')

        def draw_sphere(r,orig,ax,color):
            #stolen from here, which has more good stuff about plotting arrows, etc.
            #  http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = orig[0] + r * np.cos(u)*np.sin(v)
            y = orig[1] + r * np.sin(u)*np.sin(v)
            z = orig[2] + r * np.cos(v)
            ax.plot_wireframe(x, y, z, color=color)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax_min = np.zeros(3)
        ax_max = np.zeros(3)

        for i in self.chain_dict:
            radius = self.chain_dict[i].radius
            x = self.chain_dict[i].pos
            #TODO: Color by open patches?

            if self.chain_dict[i].coordination == 3:
                c = 'b'
            else:
                c = 'r'

            if spheres:
                draw_sphere(radius,x,ax,c)
            else:
                ax.scatter(x[0],x[1],x[2],color=c)

            if numbers:
                ax.text(x[0],x[1],x[2],str(i))

            ax_min = np.min(np.vstack((ax_min,x)),axis=0)
            ax_max = np.min(np.vstack((ax_min,x)),axis=0)

        #bounding_box(ax, ax_min, ax_max)
        axis_equal_3D(ax)
        plt.show()

    def print_info(self, verbose_level=None):
        print(self)

        verbose_level_save = None
        if verbose_level:
            verbose_level_save = self.verbose_level
            self.verbose_level = verbose_level

        if self.verbose_level == 2:
            for i in self.chain_dict:
                print(self.chain_dict[i].pos)

        if self.verbose_level >= 1:
            for i in self.open_patch_dict:
                print("Partcle",i,"has",self.open_patch_dict[i],"open patches.")

        if verbose_level:
            self.verbose_level = verbose_level_save

    def write_pdb(self, f):
        for i in self.chain_dict:
            x = self.chain_dict[i].pos
            name = 'CS'
            if self.chain_dict[i].coordination == 3:
                name = 'CQ'

            line = "ATOM  {:>5d} {:>3}  ASP A    1 {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00      01   C  \n".format(\
                        i+1, name, x[0], x[1], x[2])
            f.write(line)
        f.write("END\n")
        f.close()
