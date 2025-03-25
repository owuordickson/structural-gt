# SPDX-License-Identifier: GNU GPL v3

"""
Processes 2D or 3D images.
"""

import re
import os
import sys
import cv2
import pydicom
import logging
import numpy as np
import nibabel as nib
from PIL import Image

from .image_base import ImageBase

logger = logging.getLogger("SGT App")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

Image.MAX_IMAGE_PIXELS = None  # Disable limit on maximum image size
ALLOWED_IMG_EXTENSIONS = ('*.jpg', '*.png', '*.jpeg', '*.tif', '*.tiff', '*.qptiff')
class ImageProcessor:
    """
    A class for processing and preparing microscopy images for graph theory analysis.

    Args:
        img_path (str): input image path.
        out_dir (str): directory path for storing results.
    """

    def __init__(self, img_path, out_dir, auto_scale=True):
        """
        A class for processing and preparing microscopy images for graph theory analysis.

        Args:
            img_path (str): input image path.
            out_dir (str): directory path for storing results.

        >>>
        >>> i_path = "path/to/image"
        >>> o_dir = ""
        >>>
        >>> imp_obj = ImageProcessor(i_path, o_dir)
        >>> imp_obj.images[0].apply_img_filters()
        """
        self.img_path = img_path
        self.output_dir = out_dir
        self.auto_scale = auto_scale
        img_raw, self.scaling_options, self.scale_factor = self._load_img_from_file(img_path)
        self.props = []
        self.images = []
        self.selected_img = -1
        self._initialize_members(img_raw)

    def _load_img_from_file(self, file: str):
        """
        Read image and save it as an OpenCV object.

        Most 3D images are like layers of multiple image frames layered on-top of each other. The image frames may be
        images of the same object/item through time or through space (i.e., from different angles). Our approach is to
        separate these frames, extract GT graphs from them. and then layer back the extracted graphs in the same order.

        Our software will display all the frames retrieved from the 3D image (automatically downsample large ones
        depending on the user-selected re-scaling options); and allows the user to select which frames to run
        GT computations on (some frames are just too noisy to be used).

        Again, our software provides a button that allows the user to select which frames are used to reconstruct the
        layered GT graphs in the same order as their respective frames.

        :param file: file path.
        :return:
        """

        ext = os.path.splitext(file)[1].lower()
        try:
            if ext in ['.png', '.jpg', '.jpeg']:
                # Load standard 2D images with OpenCV
                image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Failed to load {file}")

                scale_factor = 1
                scaling_opts = []
                img_px_size = max(image.shape[0], image.shape[1])
                if img_px_size > 0 and self.auto_scale:
                    scaling_opts = ImageProcessor.get_scaling_options(img_px_size, self.auto_scale)
                    image, scale_factor = ImageProcessor.rescale_img(image, scaling_opts)

                return image, scaling_opts, scale_factor
            elif ext in ['.tif', '.tiff', '.qptiff']:
                # Try load multi-page TIFF using PIL
                img = Image.open(file)

                images = []
                img_px_size = 0
                scale_factor = 1
                scaling_opts = []
                while True:
                    frame = np.array(img)  # Convert the current frame to numpy array
                    images.append(frame)
                    temp_px = max(frame.shape[0], frame.shape[1])
                    img_px_size = temp_px if temp_px > img_px_size else img_px_size
                    try:
                        # Move to next frame
                        img.seek(img.tell() + 1)
                    except EOFError:
                        # Stop when all frames are read
                        break

                images_small = []
                if img_px_size > 0 and self.auto_scale:
                    scaling_opts = ImageProcessor.get_scaling_options(img_px_size, self.auto_scale)
                    images_small, scale_factor = ImageProcessor.rescale_img(images, scaling_opts)

                # Convert back to numpy arrays
                images = images_small if len(images_small) > 0 else images
                images_lst = [np.array(f) for f in images]

                """
                # Plot all frames
                num_frames = len(images)
                                cols = min(5, num_frames)  # Limit to 5 columns
                                rows = (num_frames // cols) + (num_frames % cols > 0)  # Auto-calculate rows

                                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                                axes = np.array(axes).flatten()  # Flatten in case of single row

                                for ax, frame, idx in zip(axes, images, range(num_frames)):
                                    ax.imshow(frame, cmap="gray" if len(frame.shape) == 2 else None)
                                    ax.set_title(f"Frame {idx + 1}")
                                    ax.axis("off")

                                # Hide extra subplots if any
                                for ax in axes[num_frames:]:
                                    ax.axis("off")
                                plt.tight_layout()
                                plt.show()"""
                return images_lst, scaling_opts, scale_factor
            elif ext in ['.nii', '.nii.gz']:
                # Load NIfTI image using nibabel
                img_nib = nib.load(file)
                data = img_nib.get_fdata()
                # Normalize and convert to uint8 for OpenCV compatibility
                data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return data
            elif ext == '.dcm':
                # Load DICOM image using pydicom
                dcm = pydicom.dcmread(file)
                data = dcm.pixel_array
                # Normalize and convert to uint8 if needed
                data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return data
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as err:
            logging.exception(f"Error loading {file}:", err, extra={'user': 'SGT Logs'})
            return None

    def _initialize_members(self, img_data):
        """

        Reads all slices of a 3D image and loads them as separate images if it is 3D, or image itself if 2D.

        Args:
            img_data: image data.

        Returns:

        """
        if img_data is None:
            return

        if type(img_data) is list:
            self.images = [ImageBase(img, self.scale_factor) for img in img_data]
            logging.info("Image is 3D.", extra={'user': 'SGT Logs'})
        else:
            self.images.append(ImageBase(img_data, self.scale_factor))
        self.selected_img = 0  if len(self.images) > 0 else -1
        self.props = self.get_img_props()

    def apply_img_scaling(self):
        """Re-scale (downsample or up-sample) a 2D image or 3D images to a specified size"""

        scale_factor = 1
        if len(self.images) <= 0:
            return None, scale_factor

        scale_size = 0
        for scale_item in self.scaling_options:
            try:
                scale_size = scale_item["dataValue"] if scale_item["value"] == 1 else scale_size
            except KeyError:
                continue

        if scale_size <= 0:
            return None, scale_factor

        img_px_size = 1
        for img_obj in self.images:
            img = img_obj.img_raw
            temp_px = max(img.shape[0], img.shape[1])
            img_px_size = temp_px if temp_px > img_px_size else img_px_size
        scale_factor = scale_size / img_px_size

        # Resize (Downsample) all frames to smaller pixel size while maintaining aspect ratio
        for img_obj in self.images:
            img = img_obj.img_raw.copy()
            scale_size = scale_factor * max(img.shape[0], img.shape[1])
            img_small, _ = ImageBase.resize_img(scale_size, img)
            if img_small is None:
                # raise Exception("Unable to Rescale Image")
                return
            img_obj.img_2d = img_small
            img_obj.scale_factor = scale_factor
        self.props = self.get_img_props()

    def crop_image(self, x: float, y: float, width: float, height: float):
        """
        A function that crops images into a new box dimension.

        :param x: left coordinate of cropping box.
        :param y: top coordinate of cropping box.
        :param width: width of cropping box.
        :param height: height of cropping box.
        """

        if self.selected_img >= 0:
            self.images[self.selected_img].apply_img_crop(x, y, width, height)
        self.props = self.get_img_props()

    def undo_cropping(self):
        """
        A function that restores image to its original size.
        """
        if self.selected_img >= 0:
            self.images[self.selected_img].init_image()
        self.props = self.get_img_props()

    def create_filenames(self, image_path: str = None):
        """
        Splits image path into file name and image directory.

        :param image_path: image directory path.

        Returns:
            filename (str): image file name., output_dir (str): image directory path.
        """

        if image_path is None:
            img_dir, filename = os.path.split(self.img_path)
        else:
            img_dir, filename = os.path.split(image_path)
        if self.output_dir == '':
            output_dir = img_dir
        else:
            output_dir = self.output_dir

        for ext in ALLOWED_IMG_EXTENSIONS:
            ext = ext.replace('*', '')
            pattern = re.escape(ext) + r'$'
            filename = re.sub(pattern, '', filename)
        return filename, output_dir

    def get_img_props(self):
        """
        A method that retrieves image properties and stores them in a list-array.

        Returns: list of image properties

        """

        f_name, _ = self.create_filenames()
        if len(self.images) > 1:
            # (Depth, Height, Width, Channels)
            alpha_channel = self.images[self.selected_img].has_alpha_channel
            fmt = "Multi + Alpha" if alpha_channel else "Multi"
            num_dim = 3
        else:
            _, fmt = ImageBase.check_alpha_channel(self.images[self.selected_img].img_raw)
            num_dim = 2

        slices = 0
        height, width = self.images[self.selected_img].img_2d.shape[:2]
        if num_dim >= 3:
            slices = len(self.images)

        props = [
            ["Name", f_name],
            ["Height x Width", f"({width} x {height}) pixels"] if slices == 0 else ["Depth x H x W",
                                                                                    f"({slices} x {width} x {height}) pixels"],
            ["Dimensions", f"{num_dim}D"],
            ["Format", f"{fmt}"],
            # ["Pixel Size", "2nm x 2nm"]
        ]
        return props

    @staticmethod
    def get_scaling_options(orig_size: float, auto_scale: bool):
        """"""
        orig_size = int(orig_size)
        if orig_size > 2048:
            recommended_size = 1024
            scaling_options = [1024, 2048, int(orig_size * 0.25), int(orig_size * 0.5), int(orig_size * 0.75),
                               orig_size]
        elif orig_size > 1024:
            recommended_size = 1024
            scaling_options = [1024, int(orig_size * 0.25), int(orig_size * 0.5), int(orig_size * 0.75), orig_size]
        else:
            recommended_size = orig_size
            scaling_options = [int(orig_size * 0.25), int(orig_size * 0.5), int(orig_size * 0.75), orig_size]

        # Remove duplicates and arrange in ascending order
        scaling_options = sorted(list(set(scaling_options)))
        scaling_data = []
        for val in scaling_options:
            data = {"text": f"{val} px", "value": 0, "dataValue": val}
            if val == orig_size:
                data["text"] = f"{data['text']}*"

            if val == recommended_size:
                data["text"] = f"{data['text']} (recommended)"
                data["value"] = 1 if auto_scale else 0
            scaling_data.append(data)
        return scaling_data

    @staticmethod
    def rescale_img(image_data, scale_options):
        """Downsample or up-sample image to specified pixel size."""

        scale_factor = 1
        img_2d, img_3d = None, None

        if image_data is None:
            return None, scale_factor

        scale_size = 0
        for scale_item in scale_options:
            try:
                scale_size = scale_item["dataValue"] if scale_item["value"] == 1 else scale_size
            except KeyError:
                continue

        if scale_size <= 0:
            return None, scale_factor

        if type(image_data) is np.ndarray:
            img_2d, scale_factor = ImageBase.resize_img(scale_size, image_data)
            return img_2d, scale_factor

        if type(image_data) is list:
            img_px_size = 1
            images = image_data
            for img in images:
                temp_px = max(img.shape[0], img.shape[1])
                img_px_size = temp_px if temp_px > img_px_size else img_px_size
            scale_factor = scale_size / img_px_size

            # Resize (Downsample) all frames to smaller pixel size while maintaining aspect ratio
            img_3d = []
            for img in images:
                scale_size = scale_factor * max(img.shape[0], img.shape[1])
                img_small, _ = ImageBase.resize_img(scale_size, img)
                img_3d.append(img_small)
        return img_3d, scale_factor
