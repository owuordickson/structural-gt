# Copyright (c) 2023-2024 The Regents of the University of Michigan.
# This file is from the StructuralGT project, released under the BSD 3-Clause
# License.

import gsd.hoomd
import numpy as np

from . import sknwEdits



def shift(points, _2d=False, _shift=None):
    """Translates all points such that the minimum coordinate in points is
    the origin.

    Args:
        points (:class:`numpy.ndarray`):
            The points to shift.
        _2d (bool):
            Whether the points are 2D coordinates.
        _shift (:class:`numpy.ndarray`):
            The shift to apply

    Returns:
        :class:`numpy.ndarray`: The shifted points.
        :class:`numpy.ndarray`: The applied shift.
    """
    if _shift is None:
        if _2d:
            _shift = np.full(
                (np.shape(points)[0], 2),
                [np.min(points.T[0]), np.min(points.T[1])],
            )
        else:
            _shift = np.full(
                (np.shape(points)[0], 3),
                [
                    np.min(points.T[0]),
                    np.min(points.T[1]),
                    np.min(points.T[2]),
                ],
            )

    points = points - _shift

    return points, _shift


def oshift(points, _2d=False, _shift=None):
    """Translates all points such that the points become approximately centred
    at the origin.

    Args:
        points (:class:`numpy.ndarray`):
            The points to shift.
        _2d (bool):
            Whether the points are 2D coordinates.
        _shift (:class:`numpy.ndarray`):
            The shift to apply.

    Returns:
        :class:`numpy.ndarray`: The shifted points.
        :class:`numpy.ndarray`: The applied shift.
    """
    if _shift is None:
        if _2d:
            _shift = np.full(
                (np.shape(points)[0], 2),
                [np.max(points.T[0]) / 2, np.max(points.T[1]) / 2],
            )
            _shift = np.full(
                (np.shape(points)[0], 3),
                [
                    np.max(points.T[0]) / 2,
                    np.max(points.T[1]) / 2,
                    np.max(points.T[2]) / 2,
                ],
            )

    points = points - _shift

    return points


def isinside(points, crop):
    """Determines whether the given points are all within the given crop.

    Args:
        points (:class:`numpy.ndarray`):
            The points to check.
        crop (list):
            The x, y, and (optionally) z coordinates of the space to check
            for membership.

    Returns:
        bool: Whether all the points are within the crop region.
    """

    if points.T.shape[0] == 2:
        for point in points:
            if (
                point[0] < crop[0]
                or point[0] > crop[1]
                or point[1] < crop[2]
                or point[1] > crop[3]
            ):
                return False
            return True
    else:
        for point in points:
            if (
                point[0] < crop[0]
                or point[0] > crop[1]
                or point[1] < crop[2]
                or point[1] > crop[3]
                or point[2] < crop[4]
                or point[2] > crop[5]
            ):
                return False
            return True


def gsd_to_G(gsd_name, sub=False, _2d=False, crop=None):
    """Function takes gsd rendering of a skeleton and returns the list of
    nodes and edges, as calculated by sknw.

    Args:
        gsd_name (str):
            The file name to write.
        sub (optional, bool):
            Whether to return only to largest connected component. If True, it
            will reduce the returned graph to the largest connected induced
            subgraph, resetting node numbers to consecutive integers,
            starting from 0.
        _2d (optional, bool):
            Whether the skeleton is 2D. If True it only ensures additional
            redundant axes from the position array is removed. It does not
            guarantee a 3d graph.
        crop (list):
            The x, y and (optionally) z coordinates of the cuboid/shape
            enclosing the skeleton from which a :class:`igraph.Graph` object
            should be extracted.

    Returns:
        (:class:`igraph.Graph`): The extracted :class:`igraph.Graph` object.
    """
    frame = gsd.hoomd.open(name=gsd_name, mode="r")[0]
    positions = shift(frame.particles.position.astype(int))[0]
    if crop is not None:
        from numpy import logical_and as a

        p = positions.T
        positions = p.T[
            a(
                a(a(p[1] >= crop[0], p[1] <= crop[1]), p[2] >= crop[2]),
                p[2] <= crop[3],
            )
        ]
        positions = shift(positions)[0]

    if sum((positions < 0).ravel()) != 0:
        positions = shift(positions)[0]

    if _2d:
        # positions = dim_red(positions)
        # For lists of positions where all elements along one axis have the same value, this returns the same list of
        # positions but with the redundant dimension(s) removed.
        unique_positions = [len(np.unique(positions.T[i])) for i in range(len(positions.T))]
        unique_positions = np.asarray(unique_positions)
        redundant = unique_positions == 1
        positions = positions.T[~redundant].T

        new_pos = np.zeros(positions.T.shape)
        new_pos[0] = positions.T[0]
        new_pos[1] = positions.T[1]
        positions = new_pos.T.astype(int)

    canvas = np.zeros(
        list((max(positions.T[i]) + 1) for i in list(
            range(min(positions.shape))))
    )
    canvas[tuple(list(positions.T))] = 1
    canvas = canvas.astype(int)

    G = sknwEdits.build_sknw(canvas)

    if sub:
        # G = sub_G(G)
        # Function generates the largest connected induced subgraph. Node and edge numbers are reset such that they
        # are consecutive integers, starting from 0.
        print(f"Before removing smaller components, graph has {G.vcount()}  nodes")
        components = G.connected_components()
        G = components.giant()
        print(f"After removing smaller components, graph has {G.vcount()}  nodes")

    return G


