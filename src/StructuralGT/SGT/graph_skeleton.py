# SPDX-License-Identifier: GNU GPL v3

"""
Create a graph skeleton from an image binary
"""

import numpy as np
import math
from scipy import ndimage
from cv2.typing import MatLike
from skimage.morphology import binary_dilation as dilate
from skimage.morphology import disk, skeletonize, remove_small_objects


class GraphSkeleton:

    def __init__(self, img_bin: MatLike, configs: dict):
        """
        A class that builds a skeleton graph from an image.

        :param img_bin: OpenCV image in binary format.
        :param configs: options and parameters.

        >>> import cv2
        >>> import numpy
        >>> opt_gte = {}
        >>> opt_gte["merge_nearby_nodes"]["value"] = 1
        >>> opt_gte["remove_disconnected_segments"]["value"] = 1
        >>> opt_gte["remove_object_size"]["value"] = 500
        >>> opt_gte["prune_dangling_edges"]["value"] = 1
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

        # calling the three functions for merging nodes, pruning edges, and removing disconnected segments
        if self.configs["merge_nearby_nodes"]["value"] == 1:
            skeleton = GraphSkeleton.merge_nodes(skeleton)

        if self.configs["remove_disconnected_segments"]["value"] == 1:
            skeleton = remove_small_objects(skeleton, int(self.configs["remove_disconnected_segments"]["items"][0]["value"]) , connectivity=2)

        skel_int = 1 * skeleton
        if self.configs["prune_dangling_edges"]["value"] == 1:
            b_points_1 = GraphSkeleton.branched_points(skel_int)
            skeleton = GraphSkeleton.pruning(skeleton, 500, b_points_1)

        b_points = GraphSkeleton.branched_points(skel_int)
        e_points = GraphSkeleton.end_points(skel_int)
        self.bp_coord_y, self.bp_coord_x = np.where(b_points == 1)
        self.ep_coord_y, self.ep_coord_x = np.where(e_points == 1)
        self.skeleton = skeleton
        self.skel_int = 1 * skeleton

    def assign_weights(self, edge_pts: MatLike, weight_type: str = None, weight_options: dict = None,
                       pixel_dim: float = 1, rho_dim: float = 1):
        """
        Compute and assign weights to a line edge between 2 nodes.

        :param edge_pts: a list of pts that trace along a graph edge.
        :param weight_type: basis of computation for the weight (i.e., length, width, resistance, conductance etc.)
        :param weight_options: weight types to be used in computation of weights.
        :param pixel_dim: physical size of width of a single pixel in nanometers.
        :param rho_dim: the resistivity value of the material.
        :return: width pixel count of edge, computed weight.
        """

        # Initialize parameters
        # Idea copied from 'sknw' library
        pix_length = np.linalg.norm(edge_pts[1:] - edge_pts[:-1], axis=1).sum()
        epsilon = 0.001  # to avoid division by zero
        pix_length += epsilon
        # wt = 1 * (10 ** -9)  # Smallest possible

        if len(edge_pts) < 2:
            # check to see if ge is an empty or unity list, if so, set pixel count to 0
            # Assume only 1/2 pixel exists between edge points
            pix_width = 0.5
            pix_angle = None
        else:
            # if ge exists, find the midpoint of the trace, and orthogonal unit vector
            pix_width, pix_angle = self.estimate_edge_width(edge_pts)
            pix_width += 0.5  # (normalization) to make it larger than empty widths

        if weight_type is None:
            wt = pix_width / 10
        elif weight_options.get(weight_type) == weight_options.get('DIA'):
            wt = pix_width * pixel_dim
        elif weight_options.get(weight_type) == weight_options.get('AREA'):
            wt = math.pi * (pix_width * pixel_dim * 0.5) ** 2
        elif weight_options.get(weight_type) == weight_options.get('LEN') or weight_options.get(weight_type) == weight_options.get('INV_LEN'):
            wt = pix_length * pixel_dim
            if weight_options.get(weight_type) == weight_options.get('INV_LEN'):
                wt = wt + epsilon if wt == 0 else wt
                wt = wt ** -1
        elif weight_options.get(weight_type) == weight_options.get('ANGLE'):
            """
            Edge angle centrality" in graph theory refers to a measure of an edge's importance within a network, 
            based on the angles formed between the edges connected to its endpoints, essentially assessing how "central" 
            an edge is in terms of its connection to other edges within the network, with edges forming more acute 
            angles generally considered more central. 
            To calculate edge angle centrality, you would typically:
               1. For each edge, identify the connected edges at its endpoints.
               2. Calculate the angles between these connected edges.
               3. Assign a higher centrality score to edges with smaller angles, indicating a more central position in the network structure.
            """
            sym_angle = np.minimum(pix_angle, (360 - pix_angle))
            wt = (sym_angle + epsilon) ** -1
        elif weight_options.get(weight_type) == weight_options.get('FIX_CON') or weight_options.get(weight_type) == weight_options.get('VAR_CON') or weight_options.get(weight_type) == weight_options.get('RES'):
            # Varies with width
            length = pix_length * pixel_dim
            area = math.pi * (pix_width * pixel_dim * 0.5) ** 2
            if weight_options.get(weight_type) == weight_options.get('FIX_CON'):
                area = math.pi * (1 * pixel_dim) ** 2
            num = length * rho_dim
            area = area + epsilon if area == 0 else area
            num =  num + epsilon if num == 0 else num
            wt = (num / area)  # Resistance
            if weight_options.get(weight_type) == weight_options.get('VAR_CON') or weight_options.get(weight_type) == weight_options.get('FIX_CON'):
                wt = wt ** -1  # Conductance is inverse of resistance
        else:
            raise TypeError('Invalid weight type')
        return pix_width, pix_angle, wt

    def assign_weights_by_width(self, ge):
        # Inputs:
        # ge: a list of pts that trace along a graph edge
        # img_bin: the binary image that the graph is derived from

        if len(ge) < 2:
            # check to see if ge is an empty or unity list, if so, set pixel count to 0
            # Assume only 1/2 pixel exists between edge points
            pix_width = 0
            wt = 0.0001  # Smallest possible
        else:
            # if ge exists, find the midpoint of the trace, and orthogonal unit vector
            pix_width, pix_angle = self.estimate_edge_width(ge)
            wt = pix_width / 10

        # returns the width in pixels; the weight which is the width normalized by 10
        return pix_width, wt

    def estimate_edge_width(self, graph_edge_coords):
        """Estimates the edge width of a graph edge."""

        # 1. Estimate orthogonal and mid-point
        end_index = len(graph_edge_coords) - 1
        mid_index = int(len(graph_edge_coords) / 2)
        pt1 = graph_edge_coords[0]
        pt2 = graph_edge_coords[end_index]
        m = graph_edge_coords[mid_index]
        mid_pt, ortho = GraphSkeleton.find_orthogonal(pt1, pt2)
        m[0] = int(m[0])
        m[1] = int(m[1])
        # m: the midpoint of a trace of an edge
        # ortho: an orthogonal unit vector
        # img_bin: the binary image that the graph is derived from

        # 2. Compute angle in Radians
        # Delta X and Y: Compute the  difference in x and y coordinates:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        # Angle Calculation: Use the arc-tangent function to get the angle in radians:
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        # print(f"Edge Pts: {graph_edge_coords}, Angle Deg: {angle_deg}\n")

        # 3. Estimate width
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
        edge_width = np.linalg.norm(l1 - l2)
        return edge_width, angle_deg

    @staticmethod
    def branched_points(skeleton):

        # Define base patterns
        base_patterns = [
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],  # x_branch
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # x_branch variant
            [[0, 0, 0], [1, 1, 1], [0, 1, 0]],  # t_branch
            [[1, 0, 1], [0, 1, 0], [1, 0, 0]],  # t_branch variant
            [[1, 0, 1], [0, 1, 0], [0, 1, 0]],  # y_branch
            [[0, 1, 0], [1, 1, 0], [0, 0, 1]],  # y_branch variant
            [[0, 1, 0], [1, 1, 0], [1, 0, 1]],  # off_branch
            [[0, 1, 1], [0, 1, 1], [1, 0, 0]],  # clust_branch
            [[1, 1, 1], [0, 1, 1], [1, 0, 0]],  # clust_branch variant
            [[1, 1, 1], [0, 1, 1], [1, 0, 1]],  # clust_branch variant
            [[1, 0, 0], [1, 1, 1], [0, 1, 0]]  # cross_branch
        ]

        # Generate all transformations
        all_patterns = []
        for pattern in base_patterns:
            all_patterns.extend(GraphSkeleton.generate_transformations(np.array(pattern)))

        # Remove duplicate patterns (if any)
        unique_patterns = []
        for pattern in all_patterns:
            if not any(np.array_equal(pattern, existing) for existing in unique_patterns):
                unique_patterns.append(pattern)

        # Apply binary hit-or-miss for all unique patterns
        br = sum(ndimage.binary_hit_or_miss(skeleton, pattern) for pattern in unique_patterns)
        return br

    @staticmethod
    def generate_transformations(pattern):
        """Generate common transformations for a pattern."""
        # flipud is flipping them up-down
        # t_branch_2 is t_branch_0 transposed, which permutes it in all directions (might not be using that word right)
        # t_branch_3 is t_branch_2 flipped left right
        # those 3 functions are used to create all possible branches with just a few starting arrays below
        return [
            pattern,
            np.flipud(pattern),
            np.fliplr(pattern),
            np.fliplr(np.flipud(pattern)),
            pattern.T,
            np.flipud(pattern.T),
            np.fliplr(pattern.T),
            np.fliplr(np.flipud(pattern.T))
        ]

    @staticmethod
    def end_points(skeleton):

        # List of endpoint patterns
        endpoints = [
            [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]

        # Apply binary hit-or-miss for each pattern and sum results
        ep = sum(ndimage.binary_hit_or_miss(skeleton, np.array(pattern)) for pattern in endpoints)
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
