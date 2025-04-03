# Copyright (c) 2023-2024 The Regents of the University of Michigan.
# This file is from the StructuralGT project, released under the BSD 3-Clause
# License.

import os
import numpy as np

from . import base, exceptions




class _cropper:
    """Cropper class contains methods to deal with images of different
    dimensions and their geometric modificaitons. Generally there is no need
    for the user to instantiate this directly.

    Args:
        Network (:class:`Network`:):
            The :class:`Network` object to which the cropper is associated
            with
        domain (list):
            The corners of the cuboid/rectangle which enclose the network's
            region of interest
    """

    def __init__(self, Network, domain=None):
        self.dim = Network.dim
        if Network._2d:
            self.surface = 0
        elif domain is None:
            slice_name = next(iter(Network.image_stack.keys()))  # Only first item
            self.surface = int(
                # _fname(Network.dir + "/" + Network.image_stack[0][1]).num
                _fname(Network.dir + "/" + slice_name).num
            )  # Strip file type and 'slice' then convert to int
        else:
            self.surface = domain[4]
        if Network._2d:
            depth = 1
        else:
            if domain is None:
                # depth = len(Network.image_stack)
                depth = len(Network.image_stack.items())
            else:
                depth = domain[5] - domain[4]
            if depth == 0:
                raise exceptions.ImageDirectoryError(Network.stack_dir)
        self.depths = Network.depth

        if domain is None:
            self.domain = None
            self.crop = slice(None)
            img_arr = next(iter(Network.image_stack.values()))  # Only first item
            # planar_dims = Network.image_stack[0][0].shape[0:2]
            planar_dims = img_arr.shape[0:2]
            if self.dim == 2:
                self.dims = (1,) + planar_dims
            else:
                self.dims = planar_dims + (depth,)

        else:
            if self.dim == 2:
                self.crop = (slice(domain[2], domain[3]),
                             slice(domain[0], domain[1]))
                self.dims = (1, domain[3] - domain[2], domain[1] - domain[0])
                self.domain = (domain[2], domain[3], domain[0], domain[1])

            else:
                self.crop = (
                    slice(domain[0], domain[1]),
                    slice(domain[2], domain[3]),
                    slice(domain[4], domain[5]),
                )
                self.dims = (
                    domain[1] - domain[0],
                    domain[3] - domain[2],
                    domain[5] - domain[4],
                )
                self.domain = (
                    domain[0],
                    domain[1],
                    domain[2],
                    domain[3],
                    domain[4],
                    domain[5],
                )

    @classmethod
    def from_string(cls, Network, domain):
        if domain == "None":
            return cls(Network, domain=None)
        else:
            domain = domain.split(",")
        if len(domain) == 4:
            _0 = int(domain[0][1:])
            _1 = int(domain[1])
            _2 = int(domain[2])
            _3 = int(domain[3][:-1])
            return cls(Network, domain=[_0, _1, _2, _3])

    def intergerise(self):
        """Method casts decimal values in the _croppers crop attribute to
        integers such that the new crop contains at least all the space
        enclosed by the old crop
        """
        first_x = np.floor(self.crop[0].start).astype(int)
        last_x = np.ceil(self.crop[0].stop).astype(int)

        first_y = np.floor(self.crop[1].start).astype(int)
        last_y = np.ceil(self.crop[1].stop).astype(int)

        if self.dim == 2:
            self.crop = slice(first_x, last_x), slice(first_y, last_y)
            self.dims = (1, last_x - first_x, last_y - first_y)
        else:
            first_z = np.floor(self.crop[2].start).astype(int)
            last_z = np.ceil(self.crop[2].stop).astype(int)
            self.crop = (
                slice(first_x, last_x),
                slice(first_y, last_y),
                slice(first_z, last_z),
            )

    def __str__(self):
        return str(self.domain)

    @property
    def _3d(self):
        if self.dim == 2:
            return None
        elif self.crop == slice(None):
            return self.depths
        else:
            return [self.crop[2].start, self.crop[2].stop]

    @property
    def _2d(self):
        """list: If a crop is associated with the object, return the component
        which crops the rectangle associated with the :class:`Network` space.
        """
        if self.crop == slice(None):
            return slice(None)
        else:
            return self.crop[0:2]

    @property
    def _outer_crop(self):
        """Method supports square 2D crops only. It calculates the crop which
        could contain any rotation about the origin of the _cropper's crop
        attribute.

        Returns:
            (list): The outer crop
        """

        if self.dim != 2:
            raise ValueError("Only 2D crops are supported")
        if self.crop == slice(None):
            raise ValueError("No crop associated with this _cropper")

        centre = (
            self.crop[0].start + 0.5 *
            (self.crop[0].stop - self.crop[0].start),
            self.crop[1].start + 0.5 *
            (self.crop[1].stop - self.crop[1].start),
        )

        diagonal = (
            (self.crop[0].stop - self.crop[0].start) ** 2
            + (self.crop[1].stop - self.crop[1].start) ** 2
        ) ** 0.5

        outer_crop = np.array(
            [
                centre[0] - diagonal * 0.5,
                centre[0] + diagonal * 0.5,
                centre[1] - diagonal * 0.5,
                centre[1] + diagonal * 0.5,
            ],
            dtype=int,
        )

        return outer_crop



class _fname:
    """Class to represent file names of 2D image slices, with helper
    functions.

    Assumes each file has 4 character number, e.g. 0053, followed by 3 or 4
    character extension, e.g. .tif or .tiff.

    Args:
        name (str):
            The name of the file.
        depth :
            The spatial dimensions of the associated Network object.
    """

    def __init__(self, name, depth=None, _2d=False):
        if not os.path.exists(name):
            raise ValueError("File does not exist.")
        self.name = name

        # self.domain is an infinitely large space when depth is None
        self.domain = [-np.inf, np.inf] if depth is None else depth
        self._2d = _2d

        if self._2d:
            self.num = "0000"
        else:
            base_name = os.path.splitext(os.path.split(self.name)[1])[0]
            if len(base_name) < 4:
                raise ValueError(
                    "For 3D networks, filenames must end in 4 digits, indicating the depth of the slice.")
            self.num = base_name[-4::]

            if not self.num.isnumeric():
                raise ValueError(
                    "For 3D networks, filenames must end in 4 digits, indicating the depth of the slice.")

    @property
    def isinrange(self):
        """bool: Returns true iff the filename is numeric and within the
        spatial dimensions of the associated :class:`_domain` object.
        """
        if not self.isimg:
            return False

        if self._2d:
            return True
        else:
            return (int(self.num) > self.domain[0] and int(self.num) < self.domain[1])

    @property
    def isimg(self):
        """bool: Returns true iff the filename suffix is a supported image
        file type.
        """
        return base.Q_img(self.name)

    def __contains__(self, item):
        if item is None:
            return True
        else:
            """str: Returns leading string of the filename."""
            return item in os.path.splitext(os.path.basename(self.name))[0]
