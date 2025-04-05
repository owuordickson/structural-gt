# SPDX-License-Identifier: GNU GPL v3

"""
Create a graph skeleton from an image binary
"""

import numpy as np
import math
from scipy import ndimage
from cv2.typing import MatLike
from skimage.morphology import binary_dilation as dilate, binary_closing
from skimage.morphology import disk, skeletonize, remove_small_objects

from src.StructuralGT.SGT.sgt_utils import write_gsd_file


class GraphSkeleton:
    """A class that is used to get estimate the width of edges and compute their weights using binerized 2D/3D images."""

    temp_skeleton = None

    # TO DELETE
    g_2d = True
    gsd_name = ""

    def __init__(self, img_bin: MatLike, configs: dict = None, is_2d: bool = True):
        """
        A class that builds a skeleton graph from an image.
        The skeleton will be 3D so that it can be analyzed with OVITO

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
        self.is_2d = is_2d
        self.skeleton = None        # will always be a 3D skeleton
        self.skel_int = None
        self.bp_coord_x = None
        self.bp_coord_y = None
        self.ep_coord_x = None
        self.ep_coord_y = None
        if configs is not None:
            self._build_skeleton()

    def _build_skeleton(self):
        """
        Creates a graph skeleton of the image.

        :return:
        """

        # rebuilding the binary image as a boolean for skeletonizing
        img_bin = (self.img_bin * (1 / 255)).astype(bool)

        # making the initial skeleton image, then getting x and y co-ords of all branch points and endpoints
        GraphSkeleton.temp_skeleton = skeletonize(img_bin)

        # if self.configs["remove_bubbles"]["value"] == 1:
        #    GraphSkeleton.remove_bubbles(self.img_bin, mask_elements)

        if self.configs["merge_nearby_nodes"]["value"] == 1:
            GraphSkeleton.merge_nodes()

        if self.configs["remove_disconnected_segments"]["value"] == 1:
            min_size = int(self.configs["remove_disconnected_segments"]["items"][0]["value"])
            GraphSkeleton.temp_skeleton = remove_small_objects(GraphSkeleton.temp_skeleton, min_size=min_size, connectivity=2)

        if self.configs["prune_dangling_edges"]["value"] == 1:
            b_points = GraphSkeleton.get_branched_points()
            GraphSkeleton.prune_edges(500, b_points)

        b_points = GraphSkeleton.get_branched_points()
        e_points = GraphSkeleton.get_end_points()

        self.bp_coord_y, self.bp_coord_x = np.where(b_points == 1)
        self.ep_coord_y, self.ep_coord_x = np.where(e_points == 1)
        clean_skel = GraphSkeleton.temp_skeleton
        self.skel_int = 1 * clean_skel
        self.skeleton = np.asarray([clean_skel]) if self.is_2d else np.asarray(clean_skel)

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
            pix_width, pix_angle = self._estimate_edge_width(edge_pts)
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
            pix_width, pix_angle = self._estimate_edge_width(ge)
            wt = pix_width / 10

        # returns the width in pixels; the weight which is the width normalized by 10
        return pix_width, wt

    def _estimate_edge_width(self, graph_edge_coords):
        """Estimates the edge width of a graph edge."""

        # 1. Estimate orthogonal and mid-point
        end_index = len(graph_edge_coords) - 1
        mid_index = int(len(graph_edge_coords) / 2)
        pt1 = graph_edge_coords[0]
        pt2 = graph_edge_coords[end_index]
        m = graph_edge_coords[mid_index]
        m = m.astype(int)

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
        check = 0  # initializing boolean check
        i = 0      # initializing iterative variable
        l1 = np.nan
        l2 = np.nan
        while check == 0:             # iteratively check along orthogonal vector to see if the coordinate is either...
            pt_check = m + i * ortho  # ... out of bounds, or no longer within the fiber in img_bin
            pt_check = pt_check.astype(int)
            is_in_edge = GraphSkeleton.point_check(img_bin, pt_check)

            if is_in_edge:
                edge = m + (i - 1) * ortho
                edge = edge.astype(int)
                l1 = edge  # When the check indicates oob or black space, assign width to l1
                check = 1
            else:
                i += 1

        check = 0
        i = 0
        while check == 0:  # Repeat, but following the negative orthogonal vector
            pt_check = m - i * ortho
            pt_check = pt_check.astype(int)
            is_in_edge = GraphSkeleton.point_check(img_bin, pt_check)

            if is_in_edge:
                edge = m - (i - 1) * ortho
                edge = edge.astype(int)
                l2 = edge  # When the check indicates oob or black space, assign width to l2
                check = 1
            else:
                i += 1

        # returns the length between l1 and l2, which is the width of the fiber associated with an edge, at its midpoint
        edge_width = np.linalg.norm(l1 - l2)
        return edge_width, angle_deg

    @classmethod
    def _generate_transformations(cls, pattern):
        """
        Generate common transformations for a pattern.

         * flipud is flipping them up-down
         * t_branch_2 is t_branch_0 transposed, which permutes it in all directions (might not be using that word right)
         * t_branch_3 is t_branch_2 flipped left right
         * those 3 functions are used to create all possible branches with just a few starting arrays below

        :param pattern: pattern of box as a numpy array.

        """
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

    @classmethod
    def get_branched_points(cls):
        """Identify and retrieve the branched points from graph skeleton."""
        skel_int = cls.temp_skeleton * 1

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
            all_patterns.extend(cls._generate_transformations(np.array(pattern)))

        # Remove duplicate patterns (if any)
        unique_patterns = []
        for pattern in all_patterns:
            if not any(np.array_equal(pattern, existing) for existing in unique_patterns):
                unique_patterns.append(pattern)

        # Apply binary hit-or-miss for all unique patterns
        br = sum(ndimage.binary_hit_or_miss(skel_int, pattern) for pattern in unique_patterns)
        return br

    @classmethod
    def get_end_points(cls):
        """Identify and retrieve the end points from graph skeleton."""
        skel_int = cls.temp_skeleton * 1

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
        ep = sum(ndimage.binary_hit_or_miss(skel_int, np.array(pattern)) for pattern in endpoints)
        return ep

    @classmethod
    def prune_edges(cls, size, branch_points):
        """Prune dangling edges around b_points. Remove iteratively end points 'size' times from the skeleton"""

        for i in range(0, size):
            end_points = GraphSkeleton.get_end_points()
            points = np.logical_and(end_points, branch_points)
            end_points = np.logical_xor(end_points, points)
            end_points = np.logical_not(end_points)
            cls.temp_skeleton = np.logical_and(cls.temp_skeleton, end_points)

        # TO BE DELETED
        skeleton_3d = np.asarray(cls.temp_skeleton)
        if cls.g_2d:
            skeleton_3d = np.asarray([cls.temp_skeleton])
        pos_count = int(sum(skeleton_3d.ravel()))
        pos_arr = np.asarray(np.where(skeleton_3d != 0)).T
        write_gsd_file(cls.gsd_name, pos_count, pos_arr)
        print(f"Ran prune for an image with shape {skeleton_3d.shape}")
        return skeleton_3d

    @classmethod
    def merge_nodes(cls):
        """Merge nearby nodes in the graph skeleton."""
        # overlay a disk over each branch point and find the overlaps to combine nodes
        skeleton_int = 1 * cls.temp_skeleton
        radius = 2
        mask_elem = disk(radius)
        bp_skel = GraphSkeleton.get_branched_points()
        bp_skel = 1 * (dilate(bp_skel, mask_elem))

        # wide-nodes is initially an empty image the same size as the skeleton image
        skel_shape = skeleton_int.shape
        wide_nodes = np.zeros(skel_shape, dtype='int')

        # this overlays the two skeletons
        # skeleton_integer is the full map, bp_skel is just the branch points blown up to a larger size
        for x in range(skel_shape[0]):
            for y in range(skel_shape[1]):
                if skeleton_int[x, y] == 0 and bp_skel[x, y] == 0:
                    wide_nodes[x, y] = 0
                else:
                    wide_nodes[x, y] = 1

        # re-skeletonizing wide-nodes and returning it, nearby nodes in radius 2 of each other should have been merged
        cls.temp_skeleton = skeletonize(wide_nodes)

        # TO BE DELETED
        skeleton_3d = np.asarray(cls.temp_skeleton)
        if cls.g_2d:
            skeleton_3d = np.asarray([cls.temp_skeleton])
        pos_count = int(sum(skeleton_3d.ravel()))
        pos_arr = np.asarray(np.where(skeleton_3d != 0)).T
        write_gsd_file(cls.gsd_name, pos_count, pos_arr)
        print(f"Ran merge for an image with shape {skeleton_3d.shape}")
        return skeleton_3d

    @classmethod
    def remove_bubbles(cls, img_bin, mask_elements: list):
        """Remove bubbles from graph skeleton."""
        if not isinstance(mask_elements, list):
            return

        canvas = img_bin.copy()
        for mask_elem in mask_elements:
            canvas = skeletonize(mask_elem)
            canvas = binary_closing(canvas, footprint=mask_elem)

        cls.temp_skeleton = skeletonize(canvas)

        # TO BE DELETED
        skeleton_3d = np.asarray(cls.temp_skeleton)
        if cls.g_2d:
            skeleton_3d = np.asarray([cls.temp_skeleton])
        pos_count = int(sum(skeleton_3d.ravel()))
        pos_arr = np.asarray(np.where(skeleton_3d != 0)).T
        write_gsd_file(cls.gsd_name, pos_count, pos_arr)
        print(f"Ran de-bubble for an image with shape {skeleton_3d.shape}")
        return skeleton_3d

    @staticmethod
    def find_orthogonal(u, v):
        # Inputs:
        # u, v: two coordinates (x, y) or (x, y, z)
        vec = u - v  # find the vector between u and v

        if np.count_nonzero(vec) == 0:  # prevents divide by zero
            # n = vec
            n = np.array([0,] * len(u), dtype=np.float16)
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
    def boundary_check(coord, w, h, d=None):
        """

        Args:
            coord: the coordinate (x,y) to check; no (x,y,z) compatibility yet.
            w: width of the image to set the boundaries.
            h: the height of the image to set the boundaries.
            d: the depth of the image to set the boundaries.
        Returns:

        """

        # Check if image is 2D
        is_2d = len(coord) == 2

        oob = 0  # Generate a boolean check for out-of-boundary
        # Check if coordinate is within the boundary
        if is_2d:
            if coord[0] < 0 or coord[1] < 0 or coord[0] > (w - 1) or coord[1] > (h - 1):
                oob = 1
                coord[0], coord[1] = 1, 1
        else:
            if sum(coord < 0) > 0 or sum(coord > [w - 1, h - 1, d - 1]) > 0:
                oob = 1
                coord = np.array([1, 1, 1])

        # returns the boolean oob (1 if boundary error); coordinates (reset to (1,1) if boundary error)
        return oob, coord.astype(int)

    @staticmethod
    def point_check(img_bin, pt_check):
        """Checks and verifies that a point is on a graph edge."""

        # Check if the image is 2D
        if len(img_bin.shape) == 2:
            is_2d = True
            w, h = img_bin.shape  # finds dimensions of img_bin for boundary check
            d = 0
        else:
            is_2d = False
            w, h, d = img_bin.shape

        if is_2d:
            oob, pt_check = GraphSkeleton.boundary_check(pt_check, w, h)
            is_in_edge = (img_bin[pt_check[0], pt_check[1]] == 0 or oob == 1)  # Checks if point in fiber
        else:
            oob, pt_check = GraphSkeleton.boundary_check(pt_check, w, h, d=d)
            is_in_edge = (img_bin[pt_check[0], pt_check[1], pt_check[2]] == 0 or oob == 1)

        return is_in_edge
