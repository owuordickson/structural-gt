# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Create a graph skeleton from an image binary
"""

import numpy as np
from scipy import ndimage
from ypstruct import struct
from cv2.typing import MatLike
from skimage.morphology import binary_dilation as dilate
from skimage.morphology import disk, skeletonize, remove_small_objects


class GraphSkeleton:

    def __init__(self, img_bin: MatLike, configs: struct):
        """
        A class that builds a skeleton graph from an image.

        :param img_bin: OpenCV image in binary format.
        :param configs: options and parameters.

        >>> import cv2
        >>> import numpy
        >>> from ypstruct import struct
        >>> opt_gte = struct()
        >>> opt_gte.merge_nearby_nodes = 1
        >>> opt_gte.remove_disconnected_segments = 1
        >>> opt_gte.remove_object_size = 500
        >>> opt_gte.prune_dangling_edges = 1
        >>> dummy_image = 127 * numpy.ones((40, 40), dtype = np.uint8)
        >>> img = cv2.threshold(dummy_image, 127, 255, cv2.THRESH_BINARY)[1]
        >>> graph_skel = GraphSkeleton(img, opt_gte)

        """
        self.img_bin = img_bin
        self.configs = configs
        # clean_skel, skel_int, Bp_coord_x, Bp_coord_y, Ep_coord_x, Ep_coord_y
        self.skeleton = None
        self.skel_int = None
        self.bp_coord_x = None
        self.bp_coord_y = None
        self.ep_coord_x = None
        self.ep_coord_y = None
        self.make_skeleton()

    def make_skeleton(self):
        """
        Creates a graph skeleton of the image.

        :return:
        """

        # rebuilding the binary image as a boolean for skeletonizing
        img_bin = (self.img_bin * (1 / 255)).astype(bool)

        # making the initial skeleton image, then getting x and y co-ords of all branch points and endpoints
        skeleton = skeletonize(img_bin)
        skel_int = 1 * skeleton
        b_points = GraphSkeleton.branched_points(skel_int)
        e_points = GraphSkeleton.end_points(skel_int)

        bp_coord_y, bp_coord_x = np.where(b_points == 1)
        ep_coord_y, ep_coord_x = np.where(e_points == 1)

        # calling the three functions for merging nodes, pruning edges, and removing disconnected segments
        if self.configs.merge_nearby_nodes == 1:
            skeleton = GraphSkeleton.merge_nodes(skeleton)

        if self.configs.remove_disconnected_segments == 1:
            skeleton = remove_small_objects(skeleton, self.configs.remove_object_size, connectivity=2)

        skel_int = 1 * skeleton

        if self.configs.prune_dangling_edges == 1:
            b_points_1 = GraphSkeleton.branched_points(skel_int)
            skeleton = GraphSkeleton.pruning(skeleton, 500, b_points_1)

        self.skeleton = skeleton
        self.skel_int = skel_int
        self.bp_coord_x = bp_coord_x
        self.bp_coord_y = bp_coord_y
        self.ep_coord_x = ep_coord_x
        self.ep_coord_y = ep_coord_y

    def assign_weights_by_width(self, ge):
        # Inputs:
        # ge: a list of pts that trace along a graph edge
        # img_bin: the binary image that the graph is derived from

        # check to see if ge is an empty or unity list, if so, set wt to 1
        if len(ge) < 2:
            pix_width = 10
            wt = 1
        # if ge exists, find the midpoint of the trace, and orthogonal unit vector
        else:
            end_index = len(ge) - 1
            mid_index = int(len(ge) / 2)
            pt1 = ge[0]
            pt2 = ge[end_index]
            m = ge[mid_index]
            mid_pt, ortho = GraphSkeleton.find_orthogonal(pt1, pt2)
            m[0] = int(m[0])
            m[1] = int(m[1])
            pix_width = int(self.length_to_edge(m, ortho))
            wt = self.length_to_edge(m, ortho) / 10

        # returns the width in pixels; the weight which is the width normalized by 10
        return pix_width, wt

    def length_to_edge(self, m, ortho):
        # Inputs:
        # m: the midpoint of a trace of an edge
        # ortho: an orthogonal unit vector
        # img_bin: the binary image that the graph is derived from

        img_bin = self.img_bin
        w, h = img_bin.shape  # finds dimensions of img_bin for boundary check
        check = 0  # initializing boolean check
        i = 0  # initializing iterative variable
        l1 = np.nan
        l2 = np.nan
        while check == 0:  # iteratively check along orthogonal vector to see if the coordinate is either...
            pt_check = m + i * ortho  # ... out of bounds, or no longer within the fiber in img_bin
            pt_check[0], pt_check[1] = int(pt_check[0]), int(pt_check[1])
            oob, pt_check = GraphSkeleton.boundary_check(pt_check, w, h)
            if img_bin[int(pt_check[0])][int(pt_check[1])] == 0 or oob == 1:
                edge = m + (i - 1) * ortho
                edge[0], edge[1] = int(edge[0]), int(edge[1])
                l1 = edge  # When the check indicates oob or black space, assign width to l1
                check = 1
            else:
                i += 1
        check = 0
        i = 0
        while check == 0:  # Repeat, but following the negative orthogonal vector
            pt_check = m - i * ortho
            pt_check[0], pt_check[1] = int(pt_check[0]), int(pt_check[1])
            oob, pt_check = GraphSkeleton.boundary_check(pt_check, w, h)
            if img_bin[int(pt_check[0])][int(pt_check[1])] == 0 or oob == 1:
                edge = m - (i - 1) * ortho
                edge[0], edge[1] = int(edge[0]), int(edge[1])
                l2 = edge  # When the check indicates oob or black space, assign width to l1
                check = 1
            else:
                i += 1

        # returns the length between l1 and l2, which is the width of the fiber associated with an edge, at its midpoint
        return np.linalg.norm(l1 - l2)

    @staticmethod
    def branched_points(skeleton):

        # defining branch shapes to locate nodes
        # over-explained this section a bit
        x_branch_0 = np.array([[1, 0, 1],
                               [0, 1, 0],
                               [1, 0, 1]])

        x_branch_1 = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

        t_branch_0 = np.array([[0, 0, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

        # flipud is flipping them up-down
        # t_branch_2 is t_branch_0 transposed, which permutes it in all directions (might not be using that word right)
        # t_branch_3 is t_branch_2 flipped left right
        # those 3 functions are used to create all possible branches with just a few starting arrays below

        t_branch_1 = np.flipud(t_branch_0)
        t_branch_2 = t_branch_0.T
        t_branch_3 = np.fliplr(t_branch_2)

        t_branch_4 = np.array([[1, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0]])
        t_branch_5 = np.flipud(t_branch_4)
        t_branch_6 = np.fliplr(t_branch_4)
        t_branch_7 = np.fliplr(t_branch_5)

        y_branch_0 = np.array([[1, 0, 1],
                               [0, 1, 0],
                               [0, 1, 0]])

        y_branch_1 = np.flipud(y_branch_0)
        y_branch_2 = y_branch_0.T
        y_branch_3 = np.fliplr(y_branch_2)

        y_branch_4 = np.array([[0, 1, 0],
                               [1, 1, 0],
                               [0, 0, 1]])

        y_branch_5 = np.flipud(y_branch_4)
        y_branch_6 = np.fliplr(y_branch_4)
        y_branch_7 = np.fliplr(y_branch_5)

        off_branch_0 = np.array([[0, 1, 0],
                                 [1, 1, 0],
                                 [1, 0, 1]])

        off_branch_1 = np.flipud(off_branch_0)
        off_branch_2 = np.fliplr(off_branch_0)
        off_branch_3 = np.fliplr(off_branch_1)
        off_branch_4 = off_branch_0.T
        off_branch_5 = np.flipud(off_branch_4)
        off_branch_6 = np.fliplr(off_branch_4)
        off_branch_7 = np.fliplr(off_branch_5)

        clust_branch_0 = np.array([[0, 1, 1],
                                   [0, 1, 1],
                                   [1, 0, 0]])

        clust_branch_1 = np.flipud(clust_branch_0)
        clust_branch_2 = np.fliplr(clust_branch_0)
        clust_branch_3 = np.fliplr(clust_branch_1)

        clust_branch_4 = np.array([[1, 1, 1],
                                   [0, 1, 1],
                                   [1, 0, 0]])

        clust_branch_5 = np.flipud(clust_branch_4)
        clust_branch_6 = np.fliplr(clust_branch_4)
        clust_branch_7 = np.fliplr(clust_branch_5)

        clust_branch_8 = np.array([[1, 1, 1],
                                   [0, 1, 1],
                                   [1, 0, 1]])

        clust_branch_9 = np.flipud(clust_branch_8)
        clust_branch_10 = np.fliplr(clust_branch_8)
        clust_branch_11 = np.fliplr(clust_branch_9)

        cross_branch_0 = np.array([[1, 0, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]])

        cross_branch_1 = np.flipud(cross_branch_0)
        cross_branch_2 = np.fliplr(cross_branch_0)
        cross_branch_3 = np.fliplr(cross_branch_1)
        cross_branch_4 = cross_branch_0.T
        cross_branch_5 = np.flipud(cross_branch_4)
        cross_branch_6 = np.fliplr(cross_branch_4)
        cross_branch_7 = np.fliplr(cross_branch_5)

        # finding the location of all the branch points based on the arrays above
        br1 = ndimage.binary_hit_or_miss(skeleton, x_branch_0)
        br2 = ndimage.binary_hit_or_miss(skeleton, x_branch_1)
        br3 = ndimage.binary_hit_or_miss(skeleton, t_branch_0)
        br4 = ndimage.binary_hit_or_miss(skeleton, t_branch_1)
        br5 = ndimage.binary_hit_or_miss(skeleton, t_branch_2)
        br6 = ndimage.binary_hit_or_miss(skeleton, t_branch_3)
        br7 = ndimage.binary_hit_or_miss(skeleton, t_branch_4)
        br8 = ndimage.binary_hit_or_miss(skeleton, t_branch_5)
        br9 = ndimage.binary_hit_or_miss(skeleton, t_branch_6)
        br10 = ndimage.binary_hit_or_miss(skeleton, t_branch_7)
        br11 = ndimage.binary_hit_or_miss(skeleton, y_branch_0)
        br12 = ndimage.binary_hit_or_miss(skeleton, y_branch_1)
        br13 = ndimage.binary_hit_or_miss(skeleton, y_branch_2)
        br14 = ndimage.binary_hit_or_miss(skeleton, y_branch_3)
        br15 = ndimage.binary_hit_or_miss(skeleton, y_branch_4)
        br16 = ndimage.binary_hit_or_miss(skeleton, y_branch_5)
        br17 = ndimage.binary_hit_or_miss(skeleton, y_branch_6)
        br18 = ndimage.binary_hit_or_miss(skeleton, y_branch_7)
        br19 = ndimage.binary_hit_or_miss(skeleton, off_branch_0)
        br20 = ndimage.binary_hit_or_miss(skeleton, off_branch_1)
        br21 = ndimage.binary_hit_or_miss(skeleton, off_branch_2)
        br22 = ndimage.binary_hit_or_miss(skeleton, off_branch_3)
        br23 = ndimage.binary_hit_or_miss(skeleton, off_branch_4)
        br24 = ndimage.binary_hit_or_miss(skeleton, off_branch_5)
        br25 = ndimage.binary_hit_or_miss(skeleton, off_branch_6)
        br26 = ndimage.binary_hit_or_miss(skeleton, off_branch_7)
        br27 = ndimage.binary_hit_or_miss(skeleton, clust_branch_0)
        br28 = ndimage.binary_hit_or_miss(skeleton, clust_branch_1)
        br29 = ndimage.binary_hit_or_miss(skeleton, clust_branch_2)
        br30 = ndimage.binary_hit_or_miss(skeleton, clust_branch_3)
        br31 = ndimage.binary_hit_or_miss(skeleton, clust_branch_4)
        br32 = ndimage.binary_hit_or_miss(skeleton, clust_branch_5)
        br33 = ndimage.binary_hit_or_miss(skeleton, clust_branch_6)
        br34 = ndimage.binary_hit_or_miss(skeleton, clust_branch_7)
        br35 = ndimage.binary_hit_or_miss(skeleton, clust_branch_8)
        br36 = ndimage.binary_hit_or_miss(skeleton, clust_branch_9)
        br37 = ndimage.binary_hit_or_miss(skeleton, clust_branch_10)
        br38 = ndimage.binary_hit_or_miss(skeleton, clust_branch_11)
        br39 = ndimage.binary_hit_or_miss(skeleton, cross_branch_0)
        br40 = ndimage.binary_hit_or_miss(skeleton, cross_branch_1)
        br41 = ndimage.binary_hit_or_miss(skeleton, cross_branch_2)
        br42 = ndimage.binary_hit_or_miss(skeleton, cross_branch_3)
        br43 = ndimage.binary_hit_or_miss(skeleton, cross_branch_4)
        br44 = ndimage.binary_hit_or_miss(skeleton, cross_branch_5)
        br45 = ndimage.binary_hit_or_miss(skeleton, cross_branch_6)
        br46 = ndimage.binary_hit_or_miss(skeleton, cross_branch_7)

        br = (br1 + br2 + br3 + br4 + br5 + br6 + br7 + br8 + br9 + br10 + br11 + br12 + br13 + br14 + br15 + br16 +
              br17 + br18 + br19 + br20 + br21 + br22 + br23 + br24 + br25 + br26 + br27 + br28 + br29 + br30 + br31 +
              br32 + br33 + br34 + br35 + br36 + br37 + br38 + br39 + br40 + br41 + br42 + br43 + br44 + br45 + br46)
        return br

    @staticmethod
    def end_points(skeleton):

        # defining different types of endpoints
        end_point_1 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0]])

        end_point_2 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

        end_point_3 = np.array([[0, 0, 0],
                                [0, 1, 1],
                                [0, 0, 0]])

        end_point_4 = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [0, 0, 0]])

        end_point_5 = np.array([[0, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

        end_point_6 = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

        end_point_7 = np.array([[0, 0, 0],
                                [1, 1, 0],
                                [0, 0, 0]])

        end_point_8 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [1, 0, 0]])

        end_point_9 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

        # finding all the locations of the endpoints
        ep1 = ndimage.binary_hit_or_miss(skeleton, end_point_1)
        ep2 = ndimage.binary_hit_or_miss(skeleton, end_point_2)
        ep3 = ndimage.binary_hit_or_miss(skeleton, end_point_3)
        ep4 = ndimage.binary_hit_or_miss(skeleton, end_point_4)
        ep5 = ndimage.binary_hit_or_miss(skeleton, end_point_5)
        ep6 = ndimage.binary_hit_or_miss(skeleton, end_point_6)
        ep7 = ndimage.binary_hit_or_miss(skeleton, end_point_7)
        ep8 = ndimage.binary_hit_or_miss(skeleton, end_point_8)
        ep9 = ndimage.binary_hit_or_miss(skeleton, end_point_9)
        ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8 + ep9
        return ep

    @staticmethod
    def pruning(skeleton, size, b_points):
        branch_points = b_points
        # remove iteratively end points "size" times from the skeleton
        for i in range(0, size):
            end_points = GraphSkeleton.end_points(skeleton)
            points = np.logical_and(end_points, branch_points)
            end_points = np.logical_xor(end_points, points)
            end_points = np.logical_not(end_points)
            skeleton = np.logical_and(skeleton, end_points)
        return skeleton

    @staticmethod
    def merge_nodes(skeleton):

        # overlay a disk over each branch point and find the overlaps to combine nodes
        skeleton_integer = 1 * skeleton
        radius = 2
        mask_elem = disk(radius)
        bp_skel = GraphSkeleton.branched_points(skeleton_integer)
        bp_skel = 1 * (dilate(bp_skel, mask_elem))

        # wide-nodes is initially an empty image the same size as the skeleton image
        sh = skeleton_integer.shape
        wide_nodes = np.zeros(sh, dtype='int')

        # this overlays the two skeletons
        # skeleton_integer is the full map, bp_skel is just the branch points blown up to a larger size
        for x in range(sh[0]):
            for y in range(sh[1]):
                if skeleton_integer[x, y] == 0 and bp_skel[x, y] == 0:
                    wide_nodes[x, y] = 0
                else:
                    wide_nodes[x, y] = 1

        # re-skeletonizing wide-nodes and returning it, nearby nodes in radius 2 of each other should have been merged
        new_skel = skeletonize(wide_nodes)
        return new_skel

    @staticmethod
    def find_orthogonal(u, v):
        # Inputs:
        # u, v: two coordinates (x, y) or (x, y, z)
        vec = u - v  # find the vector between u and v

        if np.count_nonzero(vec) == 0:  # prevents divide by zero
            n = vec
        else:
            n = vec / np.linalg.norm(vec)  # make n a unit vector along u,v
        if np.isnan(n[0]) or np.isnan(n[1]):
            n[0], n[1] = float(0), float(0)
        hl = np.linalg.norm(vec) / 2  # find the half-length of the vector u,v
        ortho = np.random.randn(2)  # take a random vector
        ortho -= ortho.dot(n) * n  # make it orthogonal to vector u,v
        ortho /= np.linalg.norm(ortho)  # make it a unit vector

        # Returns the coordinates of the midpoint of vector u,v; the orthogonal unit vector
        return (v + n * hl), ortho

    @staticmethod
    def boundary_check(coord, w, h):
        # Inputs:
        # coord: the coordinate (x,y) to check; no (x,y,z) compatibility yet
        # w,h: the width and height of the image to set the boundaries

        oob = 0  # Generate a boolean check for out-of-boundary
        # Check if coordinate is within the boundary
        if coord[0] < 0 or coord[1] < 0 or coord[0] > (w - 1) or coord[1] > (h - 1):
            oob = 1
            coord[0], coord[1] = 1, 1

        # returns the boolean oob (1 if boundary error); coordinates (reset to (1,1) if boundary error)
        return oob, coord
