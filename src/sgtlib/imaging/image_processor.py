# SPDX-License-Identifier: GNU GPL v3

"""
Processes 2D or 3D images and generate a fiber graph network.
"""

import re
import os
import cv2
# import pydicom
import logging
import numpy as np
# import nibabel as nib
from PIL import Image
from cv2.typing import MatLike
from dataclasses import dataclass
from collections import defaultdict

from ..imaging.base_image import BaseImage
from ..search.filter_env import FilterSearchSpace
from ..networks.fiber_network import FiberNetworkBuilder
from ..utils.config_loader import load_ai_configs
from ..utils.sgt_utils import plot_to_opencv, AbortException, ProgressUpdate, ProgressData
from ..search.filter_env import sgt_genetic_algorithm, sgt_hill_climbing_algorithm

logger = logging.getLogger("SGT App")

Image.MAX_IMAGE_PIXELS = None  # Disable limit on maximum image size
ALLOWED_IMG_EXTENSIONS = ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.qptiff')
ALLOWED_3D_IMG_EXTENSIONS = ('*.qptiff', '*.nii', '*.nii.gz', '*.dcm')
ALLOWED_GRAPH_FILE_EXTENSIONS = ('*.csv', '*.gsd')


class ImageProcessor(ProgressUpdate):
    """
    A class for processing and preparing 2D or 3D microscopy images for building a fiber graph network.

    Args:
        img_path (str): input image path
        out_dir (str): directory path for storing results.
    """

    @dataclass
    class ImageBatch:
        """A class for storing image batch data."""
        numpy_image: np.ndarray
        images: list[BaseImage]
        graph_obj: FiberNetworkBuilder
        shape: tuple
        props: list
        is_2d: bool
        is_graph_only: bool
        scale_factor: float
        scaling_options: list[dict]
        selected_images_positions: set
        selected_frame_pos: int
        view_options: list[dict]

    def __init__(self, img_path, out_dir, cfg_file="", graph_file="", auto_scale=True):
        """
        A class for processing and preparing microscopy images for building a fiber graph network.

        Args:
            img_path (str | list): input image path
            out_dir (str): directory path for storing results
            cfg_file (str): configuration file path
            graph_file (str): graph file path (when creating the graph from CSV/GSD data)
            auto_scale (bool): whether to automatically scale the image

        >>>
        >>> i_path = "path/to/image"
        >>> cfg_path = "path/to/sgt_configs.ini"
        >>>
        >>> ntwk_p, img_file = ImageProcessor.from_image_file(i_path, config_file=cfg_path)
        >>> ntwk_p.apply_img_filters()
        """
        super(ImageProcessor, self).__init__()
        self._configs = load_ai_configs(cfg_file)
        self._img_path: str = img_path if type(img_path) is str else img_path[0]
        self._output_dir: str = out_dir
        self._config_file: str = cfg_file
        self._graph_file: str = graph_file
        self._auto_scale: bool = auto_scale
        self._image_batches: list[ImageProcessor.ImageBatch] = []
        self._selected_batch_index: int = 0
        self._filter_space: FilterSearchSpace.SearchSpace|None = None
        # self._initialize_image_batches(self._load_img_from_file(img_path))

    @property
    def configs(self):
        """Returns the configuration (ML model) settings for the image processor."""
        return self._configs

    @configs.setter
    def configs(self, configs):
        """Sets the configuration (ML model) settings for the image processor."""
        self._configs = configs

    @property
    def img_path(self) -> str:
        """Returns the input image path."""
        return self._img_path

    @property
    def output_dir(self) -> str:
        """Returns the output directory path for storing results."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, folder: str):
        """Sets the output directory path for storing results."""
        self._output_dir = folder

    @property
    def config_file(self) -> str:
        """Returns the configuration file path (usually sgt_configs.ini)."""
        return self._config_file

    @property
    def graph_file(self) -> str:
        """Returns the graph file path."""
        return self._graph_file

    @property
    def auto_scale(self) -> bool:
        """Returns whether to automatically scale the image."""
        return self._auto_scale

    @auto_scale.setter
    def auto_scale(self, value: bool):
        self._auto_scale = value

    @property
    def image_batches(self) -> list["ImageProcessor.ImageBatch"]:
        """Returns a list of ImageBatch objects."""
        return self._image_batches

    @property
    def selected_batch_index(self) -> int:
        """Returns the selected batch index."""
        return self._selected_batch_index

    @property
    def filter_space(self) -> FilterSearchSpace|None:
        """Returns the filter space."""
        return self._filter_space

    @property
    def selected_batch(self):
        """
        Retrieved data of the current selected batch.
        """
        return self._image_batches[self._selected_batch_index]

    @property
    def selected_batch_view(self):
        """Gets the current image batch view."""
        sel_img_batch = self.selected_batch
        for view_dict in sel_img_batch.view_options:
            if view_dict["value"] == 1:
                return view_dict["dataValue"]
        return sel_img_batch.view_options[0]["dataValue"]

    @selected_batch_view.setter
    def selected_batch_view(self, value):
        """Sets the current image batch view."""
        sel_img_batch = self.selected_batch
        for view_dict in sel_img_batch.view_options:
            view_dict["value"] = 1 if value == view_dict["dataValue"] else 0

    @property
    def selected_images(self) -> list[BaseImage]:
        """Returns a list of selected images."""
        sel_img_batch = self.selected_batch
        if sel_img_batch.is_graph_only:
            return []
        sel_images = [sel_img_batch.images[i] for i in sel_img_batch.selected_images_positions]
        return sel_images

    @property
    def image_obj(self) -> BaseImage:
        """Returns the first image (2D) object/instance in the batch."""
        sel_img_batch = self.selected_batch
        first_index = next(iter(sel_img_batch.selected_images_positions), None)  # 1st selected image
        first_index = first_index if first_index is not None else 0  # first image if None
        return sel_img_batch.images[first_index]

    @property
    def image_obj_3d(self) -> list[BaseImage]:
        """Returns the full image list (3D) BaseImage objects/instances in the batch."""
        return self.selected_batch.images

    @property
    def graph_obj(self):
        """Returns the NetworkX graph extracted from the image."""
        return self.selected_batch.graph_obj

    @property
    def image_2d(self) -> MatLike:
        """Returns OpenCV 2D version of the image (first slice/frame/image in the batch)."""
        return self.image_obj.img_2d

    @property
    def image_3d(self) -> list[MatLike]:
        """Returns the 3D version of the image as a list of OpenCV arrays."""
        images = [obj.img_2d for obj in self.image_obj_3d]
        return images

    @property
    def binary_image_2d(self) -> MatLike:
        """Returns OpenCV version of the binary image (first slice/frame/image in the batch)."""
        # img_bin_rgb = cv2.cvtColor(self.image_obj.img_bin, cv2.COLOR_BGR2RGB)
        return self.image_obj.img_bin

    @property
    def binary_image_3d(self) -> list[MatLike]:
        """Returns the 3D version of the binary image as a list of OpenCV arrays."""
        bin_images = [obj.img_bin for obj in self.image_obj_3d]
        return bin_images

    @property
    def processed_image_3d(self) -> list[MatLike]:
        """Returns the 3D version of the modified image as a list of OpenCV arrays."""
        mod_images = [obj.img_mod for obj in self.image_obj_3d]
        return mod_images

    @property
    def mutated_image_3d(self) -> list[MatLike]:
        """Returns the 3D version of the mutated image as a list of OpenCV arrays."""
        mut_images = [obj.img_mut for obj in self.image_obj_3d]
        return mut_images

    def _load_img_from_file(self, file: list | str):
        """
        Read the image and save it as an OpenCV object.

        Most 3D images are like layers of multiple image frames layered on-top of each other. The image frames may be
        images of the same object/item through time or through space (i.e., from different angles). Our approach is to
        separate these frames, extract GT graphs from them, and then the layer back from the extracted graphs in the same order.

        Our software will display all the frames retrieved from the 3D image (automatically downsample large ones
        depending on the user-selected re-scaling options), and allows the user to select which frames to run
        GT computations on. (Some frames are just too noisy to be used.)

        Again, our software provides a button that allows the user to select which frames are used to reconstruct the
        layered GT graphs in the same order as their respective frames.

        :param file: The file path.
        :return: list[ImageProcessor.ImageBatch]
        """

        # First file if it's a list
        ext = os.path.splitext(file[0])[1].lower() if (type(file) is list) else os.path.splitext(file)[1].lower()
        try:
            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_groups = defaultdict(list)
                if type(file) is list:
                    for img_file in file:
                        # Create clusters/groups of similar size images
                        frame = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame_rgb.shape[:2]
                        image_groups[(h, w)].append(frame_rgb)
                else:
                    # Load standard 2D images with OpenCV
                    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError(f"Failed to load {file}")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Cluster the images into batches based on (h, w) size
                    h, w = image_rgb.shape[:2]
                    image_groups[(h, w)].append(image_rgb)
                img_batch_groups = ImageProcessor.create_img_batch_groups(image_groups, self._config_file,
                                                                          self._auto_scale)
                return img_batch_groups
            elif ext in ['.tif', '.tiff', '.qptiff']:
                image_groups = defaultdict(list)
                if type(file) is list:
                    for img_file in file:
                        # Create clusters/groups of similar size images
                        frame = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame_rgb.shape[:2]
                        image_groups[(h, w)].append(frame_rgb)
                else:
                    # Try load multi-page TIFF using PIL
                    img = Image.open(file)
                    while True:
                        # Create clusters/groups of similar size images
                        frame = np.array(img)  # Convert the current frame to the numpy array
                        # Cluster the images into batches based on (h, w) size
                        h, w = frame.shape[:2]
                        image_groups[(h, w)].append(frame)
                        try:
                            # Move to the next frame
                            img.seek(img.tell() + 1)
                        except EOFError:
                            # Stop when all frames are read
                            break
                img_batch_groups = ImageProcessor.create_img_batch_groups(image_groups, self._config_file,
                                                                          self._auto_scale)
                return img_batch_groups
            elif ext in ['.nii', '.nii.gz']:
                """# Load NIfTI image using nibabel
                img_nib = nib.load(file)
                data = img_nib.get_fdata()
                # Normalize and convert to uint8 for OpenCV compatibility
                data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return data"""
                return []
            elif ext == '.dcm':
                """# Load DICOM image using pydicom
                dcm = pydicom.dcmread(file)
                data = dcm.pixel_array
                # Normalize and convert to uint8 if needed
                data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return data"""
                return []
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as err:
            logging.exception(f"Error loading {file}:", err, extra={'user': 'SGT Logs'})
            # self.update_status(ProgressData(sender="GT", type="error", message=f"Failed to load {file}: {err}"))
            return None

    def _initialize_image_batches(self, img_batches: list[ImageBatch]):
        """
        Retrieve all image slices of the selected image batch. If the image is 2D, only one slice exists
        if it is 3D, then multiple slices exist.
        """

        # Check if image batches exist
        if len(img_batches) == 0:
            raise ValueError("No images available! Please add at least one image.")

        for i, img_batch in enumerate(img_batches):
            img_data = img_batch.numpy_image
            scale_factor = img_batch.scale_factor

            # Load images for processing
            if img_data is None:
                raise ValueError(f"Problem with images in batch {i}!")

            _, fmt_2d = BaseImage.check_alpha_channel(img_data)
            image_list = []
            if (len(img_data.shape) >= 3) and (fmt_2d is None):
                # If the image has shape (d, h, w) and does not an alpha channel which is less than 4 - (h, w, a)
                image_list = [BaseImage(img, self._config_file, scale_factor) for img in img_data]
            else:
                img_obj = BaseImage(img_data, self._config_file, scale_factor)
                image_list.append(img_obj)

            is_2d = True
            if len(image_list) == 1:
                if len(image_list[0].img_2d.shape) == 3 and image_list[0].has_alpha_channel:
                    logging.info("Image is 2D with Alpha Channel.", extra={'user': 'SGT Logs'})
                    # self.update_status(ProgressData(sender="GT", type="warning", message=f"Image is 2D with Alpha Channel"))
                else:
                    logging.info("Image is 2D.", extra={'user': 'SGT Logs'})
                    # self.update_status(ProgressData(sender="GT", type="warning", message=f"Image is 2D"))
            elif len(image_list) > 1:
                is_2d = False
                logging.info("Image is 3D.", extra={'user': 'SGT Logs'})
                # self.update_status(ProgressData(sender="GT", type="warning", message=f"Image is 3D"))

            img_batch.images = image_list
            img_batch.is_2d = is_2d
            self.update_image_props(img_batch)
        self._image_batches = img_batches

    def select_image_batch(self, sel_batch_idx: int, selected_images: set = None):
        """
        Update the selected image batch and the selected image slices.

        Args:
            sel_batch_idx: index of the selected image batch
            selected_images: indices of the selected image slices.

        Returns:

        """

        if sel_batch_idx >= len(self._image_batches):
            raise ValueError(
                f"Selected image batch {sel_batch_idx} out of range! Select in range 0-{len(self._image_batches)}")

        self._selected_batch_index = sel_batch_idx
        self.update_image_props(self._image_batches[sel_batch_idx])
        self.reset_img_filters()

        if selected_images is None:
            return

        if type(selected_images) is set:
            self._image_batches[sel_batch_idx].selected_images_positions = selected_images

    def track_progress(self, status_data: ProgressData):
        self.update_status(status_data)

    def apply_img_filters(self, filter_type=2):
        """
        Executes function for processing image filters and converting the resulting image into a binary.

        Filter Types:
        1 - Just Image Filters
        2 - Both Image and Binary (1 and 2) Filters

        :return: None
        """

        sel_batch = self.selected_batch
        if sel_batch.is_graph_only:
            return

        self.update_status(ProgressData(percent=10, sender="GT", message=f"Processing image..."))
        if filter_type == 2:
            self.reset_img_filters()

        progress = 10
        incr = 90 / len(sel_batch.images) - 1
        for i in range(len(sel_batch.images)):
            img_obj = sel_batch.images[i]
            if i not in sel_batch.selected_images_positions:
                img_obj.img_mod, img_obj.img_bin = None, None
                continue

            if progress < 100:
                progress += incr
                self.update_status(ProgressData(percent=int(progress), sender="GT", message=f"Image processing in progress..."))

            img_data = img_obj.img_mut.copy()
            img_obj.img_mod = img_obj.process_img(image=img_data)

            if filter_type == 2:
                img_mod = img_obj.img_mod.copy()
                img_obj.img_bin = img_obj.binarize_img(img_mod)
                img_obj.img_mod = img_mod
            img_obj.get_pixel_width()
        self.update_status(ProgressData(percent=100, sender="GT", message=f"Image processing complete..."))

    def undo_img_changes(self, img_pos: int = -1):
        """
        A function that restores the image to its original size and colors.
        """
        try:
            sel_batch = self.selected_batch
            if img_pos >= 0:
                sel_batch.images[img_pos].init_image()
                self.update_image_props(sel_batch)
                return

            if len(sel_batch.selected_images_positions) > 0:
                [sel_batch.images[i].init_image() for i in sel_batch.selected_images_positions]
            self.update_image_props(sel_batch)
        except Exception as err:
            logging.exception(f"Undo Error: {err}", extra={'user': 'GT'})
            return

    def eliminate_selected_img_colors(self, img_pos: int, swap_color: int):
        """
        Removes user-selected dominant colors from an image by replacing their pixels with
        white or black. This preprocessing step helps refine the binary image and improve
        graph structure extraction.

        Args:
            img_pos (int): Index of the image in the selected batch,
            swap_color (int): Choose 1 for white, 0 for black.

        Returns:
            None
        """
        sel_batch = self.selected_batch
        if sel_batch.is_graph_only:
            return

        self.update_status(ProgressData(percent=10, sender="GT", message=f"Eliminating colors..."))
        img_obj = sel_batch.images[img_pos]
        if len(img_obj.dominant_colors) == 0:
            self.update_status(ProgressData(percent=100, sender="GT", message="No dominant colors found!"))
            return

        img = img_obj.img_mut.copy()
        swap_to_white = True if swap_color == 1 else False
        for sel_color in img_obj.dominant_colors:
            if sel_color.is_selected:
                hex_code = sel_color.hex_code
                pixels = sel_color.pixel_positions
                img = BaseImage.eliminate_img_colors(image=img, hex_color=hex_code, pixel_pos=pixels, is_white=swap_to_white)

        img_obj.img_mut = img.copy()
        self.update_status(ProgressData(percent=100, sender="GT", message="Color elimination complete..."))

    def reset_img_filters(self):
        """Delete existing filters that have been applied on the image."""
        sel_batch = self.selected_batch
        for img_obj in sel_batch.images:
            img_obj.img_mod, img_obj.img_bin = None, None
            sel_batch.graph_obj.reset_graph()

    def reset_metaheuristic_search(self):
        """
        Reset the search results for the metaheuristic algorithm by moving the 'best candidate' to the 'ignore list'
        """
        if self._filter_space is None:
            return

        opt_model = self._configs
        if opt_model["find_filter_selections"]["value"] == 1:
            if self._filter_space.best_candidate is not None:
                self._filter_space.ignore_candidates.append(self._filter_space.best_candidate.position)
                self.image_obj.reset_img_configs(self._config_file)

        if opt_model["find_filter_values"]["value"] == 1:
            val_space = self._filter_space.best_candidate.value_space
            if val_space.best_candidate is not None:
                val_space.ignore_candidates.append(val_space.best_candidate.position)

        if opt_model["find_brightness_contrast"]["value"] == 1:
            bright_space = self._filter_space.best_candidate.brightness_space
            if bright_space.best_candidate is not None:
                bright_space.ignore_candidates.append(bright_space.best_candidate.position)

    def apply_img_scaling(self):
        """Re-scale (downsample or up-sample) a 2D image or 3D images to a specified size"""

        # scale_factor = 1
        sel_batch = self.selected_batch
        if len(sel_batch.images) <= 0:
            return

        scale_size = 0
        for scale_item in sel_batch.scaling_options:
            try:
                scale_size = scale_item["dataValue"] if scale_item["value"] == 1 else scale_size
            except KeyError:
                continue

        if scale_size <= 0:
            return

        img_px_size = 1
        for img_obj in sel_batch.images:
            img = img_obj.img_raw
            temp_px = max(img.shape[0], img.shape[1])
            img_px_size = temp_px if temp_px > img_px_size else img_px_size
        scale_factor = scale_size / img_px_size

        # Resize (Downsample) all frames to the smaller pixel size while maintaining the aspect ratio
        for img_obj in sel_batch.images:
            img = img_obj.img_raw.copy()
            scale_size = scale_factor * max(img.shape[0], img.shape[1])
            img_small, _ = BaseImage.resize_img(scale_size, img)
            if img_small is None:
                # raise Exception("Unable to Rescale Image")
                return
            img_obj.img_2d = img_small
            img_obj.scale_factor = scale_factor
        self.update_image_props(sel_batch)

    def crop_image(self, x: int, y: int, crop_w: int, crop_h: int, actual_w: int, actual_h: int):
        """
        A function that crops images into a new box dimension.

        :param x: Left coordinate of cropping box.
        :param y: Top coordinate of cropping box.
        :param crop_w: Width of cropping box.
        :param crop_h: Height of cropping box.
        :param actual_w: Width of actual image.
        :param actual_h: Height of actual image.
        """
        sel_batch = self.selected_batch
        if len(sel_batch.selected_images_positions) > 0:
            [sel_batch.images[i].apply_img_crop(x, y, crop_w, crop_h, actual_w, actual_h) for i in
             sel_batch.selected_images_positions]
        self.update_image_props(sel_batch)
        self.selected_batch_view = 'processed'

    def compute_img_histograms(self, img_pos: int) -> None | list:
        """
        Compute the histograms (original, binary, processes, mutated) of an image at the position img_pos

        :param img_pos: position index of the image to compute histograms for
        :return: list of the Matplotlib histogram objects
        """
        sel_batch = self.selected_batch
        if sel_batch.is_graph_only:
            return None

        self.update_status(ProgressData(percent=10, sender="GT", message="Starting histogram computation..."))
        lst_histograms = []
        # Get BaseImage object
        img_obj = sel_batch.images[img_pos]

        # Computations
        self.update_status(ProgressData(percent=20, sender="GT", message="Computing histogram of original image..."))
        img_hist = plot_to_opencv(img_obj.plot_img_histogram(curr_view="original"))
        lst_histograms.append(img_hist.copy())

        self.update_status(ProgressData(percent=40, sender="GT", message="Computing histogram of binary image..."))
        img_hist = plot_to_opencv(img_obj.plot_img_histogram(curr_view="binary"))
        lst_histograms.append(img_hist.copy())

        self.update_status(ProgressData(percent=50, sender="GT", message="Computing histogram of processed image..."))
        img_hist = plot_to_opencv(img_obj.plot_img_histogram(curr_view="processed"))
        lst_histograms.append(img_hist.copy())

        self.update_status(ProgressData(percent=80, sender="GT", message="Computing histogram of mutated image..."))
        img_hist = plot_to_opencv(img_obj.plot_img_histogram(curr_view="mutated"))
        lst_histograms.append(img_hist.copy())
        return lst_histograms

    def retrieve_dominant_img_colors(self, img_pos: int, top_k: int = 6) -> None | list:
        """
        Search and get the top k dominant colors of the image.
        Args:
            img_pos: position index of the image-object in the selected batch.
            top_k: maximum number of top colors to search.

        Returns:
            True if dominant colors are found, False otherwise.
        """
        sel_batch = self.selected_batch
        if sel_batch.is_graph_only:
            return None

        self.update_status(ProgressData(percent=10, sender="GT", message=f"Retrieving dominant colors..."))
        # Get BaseImage object
        img_obj = sel_batch.images[img_pos]
        if len(img_obj.dominant_colors) == top_k:
            self.update_status(ProgressData(percent=100, sender="GT", message="Dominant already colors retrieved!"))
            return img_obj.dominant_colors

        top_colors = img_obj.get_dominant_img_colors(top_k=top_k)
        if top_colors is None:
            self.update_status(ProgressData(percent=100, sender="GT", message="No dominant colors found!"))
            return None

        self.update_status(ProgressData(percent=100, sender="GT", message="Dominant colors retrieved!"))
        return top_colors

    def metaheuristic_image_configs(self) -> dict | None:
        """
        A function that runs metaheuristic algorithms (Genetic Algorithm and Hill-climbing Algorithm) to find the best
        image configurations for extracting accurate graphs from SEM images.

        :return: A dictionary containing the best candidate's image configuration settings.
        """

        def _run_genetic_algorithm(search_space):
            """Runs the Genetic Algorithm to find the best candidate image configuration."""
            new_img_configs = sgt_genetic_algorithm(search_space, self.image_obj, generations=max_iters,
                                                    pop_size=ga_init_pop)
            filter_space.best_candidate.std_cost = search_space.best_candidate.std_cost
            filter_space.img_configs = new_img_configs

        self.update_status(ProgressData(percent=0, sender="AI", message=f"Starting filter search..."))
        opt_model = self._configs
        max_iters = opt_model["max_iterations"]["value"]
        ga_init_pop = opt_model["genetic_alg_initial_pop"]["value"]

        if self.abort:
            self.update_status(ProgressData(type="error", sender="AI", message=f"Task stopped!"))
            return None

        # 1. Create a search space
        if self._filter_space is None:
            self.update_status(ProgressData(percent=20, sender="AI", message=f"Creating search environment..."))
            self._filter_space = FilterSearchSpace.build_search_space(self.image_obj, initial_pop=ga_init_pop)
        filter_space = self._filter_space

        if opt_model["find_filter_selections"]["value"] == 1:
            # 2. Run the Hill-climbing algorithm to find the best "image config combination"
            self.update_status(ProgressData(percent=50, sender="AI", message=f"Searching for filter selections..."))
            try:
                # filter_space.ignore_candidates.append(filter_space.best_candidate.position)
                sgt_hill_climbing_algorithm(filter_space, self.image_obj, max_iters=max_iters)
            except AbortException as err:
                self.abort = True
                logging.exception(f"Error finding best apply selections:", err, extra={'user': 'SGT Logs'})
                self.update_status(ProgressData(type="error", sender="AI", message=f"{err}"))
                return None

        if opt_model["find_filter_values"]["value"] == 1:
            # 3. Run the Genetic Algorithm to find the best "image filter values"
            self.update_status(ProgressData(percent=65, sender="AI", message=f"Searching for filter values..."))
            try:
                _run_genetic_algorithm(filter_space.best_candidate.value_space)
            except AbortException as err:
                self.abort = True
                self.update_status(ProgressData(type="error", sender="AI", message=f"{err}"))
                return None

        if opt_model["find_brightness_contrast"]["value"] == 1:
            # 4. Run the Genetic Algorithm to find the best "brightness/contrast values" (only if 'val_search_space' fxn fails)
            self.update_status(ProgressData(percent=80, sender="AI", message=f"Searching for brightness/contrast values..."))
            try:
                _run_genetic_algorithm(filter_space.best_candidate.brightness_space)
            except AbortException as err:
                self.abort = True
                self.update_status(ProgressData(type="error", sender="AI", message=f"{err}"))
                return None
        return filter_space.best_candidate.img_configs

    def build_graph_network(self):
        """Generates or extracts graphs of selected images."""

        self.update_status(ProgressData(percent=0, sender="GT", message=f"Starting graph extraction..."))
        try:
            # Get the selected batch
            self.selected_batch_view= 'graph'
            sel_batch = self.selected_batch
            sel_images = self.get_batch_images(sel_batch)

            if sel_batch.is_graph_only:
                self.update_status(ProgressData(percent=20, sender="GT", message=f"Fetching graph file..."))
                f_name, out_dir = self.get_filenames(file_path=self._graph_file)

                sel_batch.graph_obj.abort = False
                sel_batch.graph_obj.add_listener(self.track_progress)
                sel_batch.graph_obj.fit_graph(out_dir, input_data=self._graph_file, file_name=f_name)
                sel_batch.graph_obj.remove_listener(self.track_progress)
            else:
                # Get binary image
                self.update_status(ProgressData(percent=20, sender="GT", message=f"Getting binary image..."))
                img_bin = [img.img_bin for img in sel_images]
                img_bin = np.asarray(img_bin)

                # Check if filters have been applied
                if img_bin[0] is None:
                    self.update_status(ProgressData(type="warning", sender="GT", message=f"No filters applied! Please wait, applying image filters."))
                    self.apply_img_filters()
                    self.build_graph_network()
                    return

                # Get the selected batch's graph object and generate the graph
                px_size = float(sel_batch.images[0].configs["pixel_width"]["value"])  # First BaseImage in batch
                rho_val = float(sel_batch.images[0].configs["resistivity"]["value"])  # First BaseImage in batch
                f_name, out_dir = self.get_filenames()

                sel_batch.graph_obj.abort = False
                sel_batch.graph_obj.add_listener(self.track_progress)
                sel_batch.graph_obj.fit_graph(out_dir, img_bin, sel_batch.is_2d, px_size, rho_val, file_name=f_name)

            self.update_status(ProgressData(percent=95, sender="GT", message=f"Plotting graph network..."))
            self.draw_graph_image(sel_batch)

            sel_batch.graph_obj.remove_listener(self.track_progress)
            self.abort = sel_batch.graph_obj.abort
            if self.abort:
                self.selected_batch_view = 'processed'
                return
        except Exception as err:
            self.abort = True
            logging.exception("Graph Extraction Error: %s", err, extra={'user': 'SGT Logs'})
            self.update_status(ProgressData(type="error", sender="GT", message=f"Graph Extraction Error: {err}"))
            return

    def build_graph_from_patches(self, num_kernels: int, patch_count_per_kernel: int, img_padding: tuple = (0, 0),
                                 compute_avg: bool = False):
        """
        Extracts graphs from smaller square patches of selected images.

        Given `num_square_filters` (k), the method generates k square filters/windows, each of sizes NxN—where N is
        a distinct value computed or estimated for each filter.

        For every NxN window, it randomly selects `patch_count_per_filter` (m) patches (aligned with the window)
        from across the entire image.

        :param num_kernels: Number of square kernels/filters to generate.
        :param patch_count_per_kernel: Number of patches per filter.
        :param img_padding: Padding around the image.
        :param compute_avg: If True, allows for computing of GT params from 95% of the original image sampled at different
        locations (by extracting the graphs at these locations).

        """
        # Get the selected batch
        patch_count_per_kernel = patch_count_per_kernel if patch_count_per_kernel > 5 else 5
        graph_configs = self.graph_obj.configs
        img_obj = self.image_obj  # ONLY works for 2D

        def extract_cropped_image_patches() -> list[MatLike]:
            """A method that extracts 4 filters from the original binary image. Each filter is the of size approximately
            90% of the original image height and width. This method ensures exactly four patches are extracted from
            the corners. Compute GT descriptors of 90% original image at different locations (to get their averages)
            """
            # Create a kernel that is 95% the size of the image
            img_bin = self.binary_image_2d
            h, w = img_bin.shape
            k_h, k_w = int(0.95 * h), int(0.95 * w)

            # Coordinates for the 4 positions (top-left, top-right, bottom-left, bottom-right)
            slide_positions = [
                (0, 0),  # top-left
                (0, w - k_w),  # top-right
                (h - k_h, 0),  # bottom-left
                (h - k_h, w - k_w),  # bottom-right
            ]

            # Retrieve kernel patches and compute GT descriptors
            lst_img_90pct = []
            for y, x in slide_positions:
                img_90pct = img_bin[y:y + k_h, x:x + k_w]
                lst_img_90pct.append(img_90pct)
            return lst_img_90pct

        def retrieve_kernel_patches(img: MatLike, num_filters: int, num_patches: int, padding: tuple) -> list[BaseImage.ScalingKernel]:
            """
            Perform an incomplete convolution operation that breaks down an image into smaller square mini-images.
            Extract all patches from the image based on filter size, stride, and padding, similar to
            CNN convolution but without applying the multiplication and addition operations. The kernel patches
            are retrieved from random/deterministic locations in the image.

            :param img: OpenCV image.
            :param num_filters: Number of convolution kernels/filters.
            :param num_patches: Number of patches to extract per filter window size.
            :param padding: Padding value (pad_y, pad_x).
            :return: List of convolved images.
            """

            def estimate_kernel_size(parent_width, num) -> int:
                """
                Applies a non-linear function to compute the width-size of a filter based on its index location.
                :param parent_width: Width of parent image.
                :param num: Index of filter.
                """
                # return int(parent_width / ((2*num) + 4))
                # est_w = int((parent_width * np.exp(-0.3 * num) / 4))  # Exponential decay
                est_w = int((parent_width - 10) * (1 - (num / num_kernels)))
                return max(50, est_w)  # Avoid too small sizes

            def extract_random_patches(kernel_dim) -> list[MatLike]:
                """
                Retrieve kernel patches at random locations in the image.
                Args:
                    kernel_dim: dimension of kernel.

                Returns:
                    list of extracted patches each of size kernel_dim.
                """

                lst_patches = []
                img_h, img_w = img.shape[:2]
                k_h, k_w = kernel_dim, kernel_dim
                for _ in range(num_patches):
                    # Random top-left corner
                    x = np.random.randint(0, img_w - k_w)
                    y = np.random.randint(0, img_h - k_h)

                    patch = img_padded[y:y + k_h, x:x + k_w].copy()
                    lst_patches.append(patch)
                    # print(f"Filter Shape: {patch.shape} at strides: x={x}, y={y}")
                return lst_patches

            if img is None:
                return []

            # Initialize Parameters
            lst_img_filter = []

            # Pad the image
            pad_h, pad_w = padding
            img_padded = np.pad(img.copy(), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

            # Get largest image dimension (height or width)
            h, w = img.shape[:2]
            max_dim = h if h < w else w

            # For each filter-window: (1) estimate HxW, (2) stride size and (3) sliding window patch retrieval
            for k in range(num_filters):
                # (1) Estimate HxW dimensions of filter
                kernel_size = estimate_kernel_size(max_dim, k)

                # (2) Retrieve multiple patches of size (k_h, k_w)
                lst_kernel_patches = extract_random_patches(kernel_size)

                # Save filter parameters in dict
                img_filter = BaseImage.ScalingKernel(
                    image_patches=lst_kernel_patches,
                    kernel_shape=(kernel_size, kernel_size),
                )
                lst_img_filter.append(img_filter)

                # Stop loop if filter size is too small
                if kernel_size <= 50:
                    break
            return lst_img_filter

        if len(img_obj.square_segments) <= 0:
            # Scaling patches (square kernel)
            lst_filters = retrieve_kernel_patches(img_obj.img_bin, num_kernels, patch_count_per_kernel, img_padding)

            # Average patches with sizes 90% of the image (rectangular kernel)
            if compute_avg:
                self.update_status(ProgressData(percent=66, sender="GT", message=f"Computing GT parameters on 95% of image at 4 locations..."))
                lst_img_filters = extract_cropped_image_patches()
                c_h, c_w = lst_img_filters[0].shape[:2]
                crop_filter = BaseImage.ScalingKernel(
                    image_patches=lst_img_filters,
                    kernel_shape=(c_h, c_w),
                )
                lst_filters.append(crop_filter)
            img_obj.square_segments = lst_filters

        filter_count = len(img_obj.square_segments)
        graph_groups = defaultdict(list)
        for i, scale_filter in enumerate(img_obj.square_segments):
            self.update_status(ProgressData(type="warning", sender="GT", message=f"Extracting random graphs using image filter {i + 1}/{filter_count}..."))
            # num_img_patches = len(scale_filter.image_patches)
            for bin_img_patch in scale_filter.image_patches:
                graph_patch = FiberNetworkBuilder(cfg_file=self._config_file)
                graph_patch.configs = graph_configs
                nx_graph = graph_patch.extract_graph(bin_img_patch, is_img_2d=True)
                success = graph_patch.verify_graph(nx_graph)
                # success = graph_patch.extract_graph(bin_img_patch, is_img_2d=True)
                if success:
                    height, width = bin_img_patch.shape
                    graph_groups[(height, width)].append(graph_patch.nx_giant_graph)
                else:
                    self.update_status(ProgressData(type="warning", sender="GT", message=f"Filter {bin_img_patch.shape} graph extraction failed!"))
        return graph_groups

    def update_graph_rating(self, score: float) -> str|None:
        """Updates the score rating of the extracted graph."""
        if score is None:
            self.update_status(ProgressData(type="warning", sender="AI", message=f"The score cannot be None!"))
            return None

        if score < 0  or score > 100:
            self.update_status(ProgressData(type="warning", sender="AI", message=f"The score rating is out of range! Please try 0-100."))
            return None

        # 1. Update rating
        self.graph_obj.score_rating = score

        # 2. Update in the filter search space
        if self._filter_space is not None:
            self._filter_space.best_candidate.graph_accuracy = score

        # 3. Save the graph image as JPG
        if self.graph_obj.img_ntwk is None:
            self.update_status(ProgressData(type="warning", sender="AI", message=f"No graph extracted! Please extract a graph first."))
            return None
        img_file_name, out_dir = self.get_filenames()
        graph_filename = img_file_name + f"_rated_graph-{int(score)}percent.jpg"
        graph_file = os.path.join(out_dir, graph_filename)
        cv2.imwrite(graph_file, self.graph_obj.img_ntwk)
        self.update_status(ProgressData(type="warning", sender="AI", message=f"Graph image downloaded!"))
        return graph_file

    def get_filenames(self, file_path: str = None):
        """
        Splits the image path into file name and image directory.

        :param file_path: Path to the file of interest.

        Returns:
            filename (str): image file name., output_dir (str): image directory path.
        """

        img_dir, filename = os.path.split(self._img_path) if file_path is None else os.path.split(file_path)
        output_dir = img_dir if self._output_dir == '' else self._output_dir

        for ext in ALLOWED_IMG_EXTENSIONS:
            ext = ext.replace('*', '')
            pattern = re.escape(ext) + r'$'
            filename = re.sub(pattern, '', filename)

        for ext in ALLOWED_GRAPH_FILE_EXTENSIONS:
            ext = ext.replace('*', '')
            pattern = re.escape(ext) + r'$'
            filename = re.sub(pattern, '', filename)

        for ext in ALLOWED_3D_IMG_EXTENSIONS:
            ext = ext.replace('*', '')
            pattern = re.escape(ext) + r'$'
            filename = re.sub(pattern, '', filename)
        return filename, output_dir

    def get_batch_images(self, selected_batch: ImageBatch):
        """
        Get indices of selected images.
        :param selected_batch: The selected batch ImageBatch object.
        """
        if selected_batch is None:
            selected_batch = self.selected_batch

        sel_images = [selected_batch.images[i] for i in selected_batch.selected_images_positions]
        return sel_images

    def update_image_props(self, selected_batch: ImageBatch = None):
        """
        A method that retrieves image properties and stores them in a list-array.

        :param selected_batch: ImageBatch data object.

        Returns: list of image properties

        """

        if selected_batch is None:
            return

        f_name, _ = self.get_filenames()
        if len(selected_batch.images) > 1:
            # (Depth, Height, Width, Channels)
            alpha_channel = selected_batch.images[0].has_alpha_channel  # first image
            fmt = "Multi + Alpha" if alpha_channel else "Multi"
            num_dim = 3
        elif len(selected_batch.images) == 1:
            if selected_batch.is_graph_only:
                return
            else:
                # (Height, Width, Channels)
                _, fmt = BaseImage.check_alpha_channel(selected_batch.images[0].img_raw)  # first image
                num_dim = 2
        else:
            # No Image Found
            return

        slices = 0
        height, width = selected_batch.images[0].img_2d.shape[:2]  # first image
        if num_dim >= 3:
            slices = len(selected_batch.images)

        props = [
            ["Name", f_name],
            ["Height x Width", f"({height} x {width}) pixels"] if slices == 0
            else ["Depth x H x W", f"({slices} x {height} x {width}) pixels"],
            ["Dimensions", f"{num_dim}D"],
            ["Format", f"{fmt}"],
            # ["Pixel Size", "2nm x 2nm"]
        ]
        selected_batch.props = props

    def save_images_to_file(self, img_pos):
        """
        Write images to a file.
        """
        img_file_name, out_dir = self.get_filenames()
        sel_batch = self.selected_batch

        if img_pos is not None:
            if type(img_pos) is int:
                img_obj = sel_batch.images[img_pos]
                crop_filename = f"{img_file_name}_cropped.jpg"
                crop_file = os.path.join(out_dir, crop_filename)
                cv2.imwrite(crop_file, img_obj.img_2d)
                self.update_status(ProgressData(type="warning", sender="GT", message=f"Cropped image saved!"))
            return

        sel_images = self.get_batch_images(sel_batch)
        is_3d = True if len(sel_images) > 1 else False

        for i, img in enumerate(sel_images):
            if img.configs["save_images"]["value"] == 0:
                continue

            filename = f"{img_file_name}_Frame{i}" if is_3d else img_file_name
            pr_filename = filename + "_processed.jpg"
            bin_filename = filename + "_binary.jpg"
            img_file = os.path.join(out_dir, pr_filename)
            bin_file = os.path.join(out_dir, bin_filename)

            if img.img_mod is not None:
                cv2.imwrite(str(img_file), img.img_mod)

            if img.img_bin is not None:
                cv2.imwrite(str(bin_file), img.img_bin)

    def draw_graph_image(self, sel_batch: ImageBatch, show_giant_only: bool = False):
        """
        Use Matplotlib to draw the extracted graph which is superimposed on the processed image.

        :param sel_batch: ImageBatch data object.
        :param show_giant_only: If True, only draw the largest/giant graph on the processed image.
        """
        sel_images = self.get_batch_images(sel_batch)

        if len(sel_images) > 0:
            img_3d = [img.img_2d for img in sel_images]
            img_3d = np.asarray(img_3d)
        else:
            img_3d = None

        if sel_batch.graph_obj is None:
            return

        plt_fig = sel_batch.graph_obj.plot_graph_network(image_arr=img_3d, giant_only=show_giant_only)
        if plt_fig is not None:
            self.update_status(ProgressData(percent=98, sender="GT", message=f"Graph image created."))
            sel_batch.graph_obj.img_ntwk = plot_to_opencv(plt_fig)

    # MODIFIED TO EXCLUDE 3D IMAGES (TO BE REVISITED LATER)
    # Problems:
    # 1. Merge Nodes
    # 2. Prune dangling edges
    # 3. Matplotlib plot nodes and edges
    @staticmethod
    def create_img_batch_groups(img_groups: defaultdict, cfg_file: str, auto_scale: bool):
        """"""

        def get_scaling_options(orig_size: float):
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

        def rescale_img(image_data, scale_options):
            """Downsample or up-sample image to a specified pixel size."""

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

            # if type(image_data) is np.ndarray:
            has_alpha, fmt_2d = BaseImage.check_alpha_channel(image_data)
            if (len(image_data.shape) == 2) or has_alpha:
                # If the image has shape (h, w) or shape (h, w, a), where 'a' - alpha channel which is less than 4
                img_2d, scale_factor = BaseImage.resize_img(scale_size, image_data)
                return img_2d, scale_factor

            # if type(image_data) is list:
            if (len(image_data.shape) >= 3) and (fmt_2d is None):
                # If the image has shape (d, h, w) and third is not alpha channel
                img_3d = []
                for img in image_data:
                    img_small, scale_factor = BaseImage.resize_img(scale_size, img)
                    img_3d.append(img_small)
            return np.array(img_3d), scale_factor

        img_info_list = []
        for (h, w), images in img_groups.items():
            images_small = []
            scaling_factor = 1
            scaling_opts = []
            images = np.array(images)
            max_size = max(h, w)
            if max_size > 0 and auto_scale:
                scaling_opts = get_scaling_options(max_size)
                images_small, scaling_factor = rescale_img(images, scaling_opts)

            # Convert back to numpy arrays
            images = images_small if len(images_small) > 0 else images
            #images = np.array([images[0]])  # REMOVE TO ALLOW 3D
            views  = [
                {"text": "Original Image", "dataValue": "original", "value": 1, "visible": 1 },
                {"text": "Binary Image", "dataValue": "binary", "value": 0, "visible": 1 },
                {"text": "Processed Image", "dataValue": "processed", "value": 0, "visible": 1 },
                {"text": "Extracted Graph", "dataValue": "graph", "value": 0, "visible": 1}
            ]
            img_batch = ImageProcessor.ImageBatch(
                numpy_image=images,
                images=[],
                graph_obj=FiberNetworkBuilder(cfg_file=cfg_file),
                shape=(h, w),
                props=[],
                is_2d=True,
                is_graph_only=False,
                scale_factor=scaling_factor,
                scaling_options=scaling_opts,
                selected_images_positions=set(range(len(images))),
                selected_frame_pos= 0,
                view_options=views,
            )
            img_info_list.append(img_batch)
            #break  # REMOVE TO ALLOW 3D
        return img_info_list

    @classmethod
    def from_image_file(cls, img_path: str, out_folder: str = "", config_file: str = "", allow_auto_scale: bool = True) -> tuple["ImageProcessor", str]:
        """
        Creates an ImageProcessor object. Make sure the image path exists, is verified, and points to an image.
        :param img_path: Path to the image to be processed
        :param out_folder: Path to the output directory
        :param config_file: Path to the config file
        :param allow_auto_scale: Allows automatic scaling of the image
        :return: ImageProcessor object.
        """

        # Get the image path and folder
        img_files = []
        img_dir, img_file = os.path.split(str(img_path))
        img_file_ext = os.path.splitext(img_file)[1].lower()

        is_prefix = True
        # Regex pattern to extract the prefix (non-digit characters at the beginning of the file name)
        img_name_pattern = re.match(r'^([a-zA-Z_]+)(\d+)(?=\.[a-zA-Z]+$)', img_file)
        if img_name_pattern is None:
            # Regex pattern to extract the suffix (non-digit characters at the end of the file name)
            is_prefix = False
            img_name_pattern = re.match(r'^\d+([a-zA-Z_]+)(?=\.[a-zA-Z]+$)', img_file)

        # If 3D file (ignore multiple input files)
        """allowed_3d_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_3D_IMG_EXTENSIONS)
        if img_path.endswith(allowed_3d_extensions):
            img_name_pattern = None
        """

        if img_name_pattern:
            img_files.append(img_path)
            f_name = img_name_pattern.group(1)
            name_pattern = re.compile(rf'^{f_name}\d+{re.escape(img_file_ext)}$', re.IGNORECASE) \
                if is_prefix else re.compile(rf'^\d+{f_name}{re.escape(img_file_ext)}$', re.IGNORECASE)

            # Check if 3D image slices exist in the image folder. Same file name but different number
            files = sorted(os.listdir(img_dir))
            for a_file in files:
                if a_file.endswith(img_file_ext):
                    if name_pattern.match(a_file):
                        img_files.append(os.path.join(img_dir, a_file))

        # Create the Output folder if it does not exist
        if out_folder != "":
            default_out_dir = out_folder
        else:
            out_dir_name = "sgt_files"
            default_out_dir = os.path.join(img_dir, out_dir_name)
        out_dir = os.path.normpath(default_out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Create the StructuralGT object
        input_file = img_files if len(img_files) > 1 else str(img_path)
        imp_obj = cls(input_file, out_dir, config_file, auto_scale=allow_auto_scale)
        imp_obj._initialize_image_batches(imp_obj._load_img_from_file(img_path))
        return imp_obj, img_file

    @classmethod
    def from_graph_file(cls, file_path: str, out_folder: str = "") -> tuple["ImageProcessor", str]:
        """
        Creates an ImageProcessor object. Make sure the graph file path exists, is verified, and points to a
        CSV file or GSD/HOOMD file.
        :param file_path: Path to the graph file
        :param out_folder: Path to the output directory
        :return: ImageProcessor object.
        """
        # Separate graph path and folder
        file_dir, graph_file = os.path.split(str(file_path))
        file_ext = os.path.splitext(graph_file)[1].lower()
        allowed_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_GRAPH_FILE_EXTENSIONS)
        if not graph_file.endswith(allowed_extensions):
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Create the Output folder if it does not exist
        if out_folder != "":
            default_out_dir = out_folder
        else:
            out_dir_name = "sgt_files"
            default_out_dir = os.path.join(file_dir, out_dir_name)
        out_dir = os.path.normpath(default_out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Create the StructuralGT object
        imp_obj = cls("", out_dir, graph_file=str(file_path))

        # Create Graph Object from the added file_path
        graph_obj = FiberNetworkBuilder(cfg_file="")

        # Create an Image Batch data object and add it to the ImageProcessor object
        views = [
            {"text": "Original Image", "dataValue": "original", "value": 0, "visible": 0},
            {"text": "Binary Image", "dataValue": "binary", "value": 0, "visible": 0},
            {"text": "Processed Image", "dataValue": "processed", "value": 0, "visible": 0},
            {"text": "Extracted Graph", "dataValue": "graph", "value": 1, "visible": 1}
        ]
        img_batch = ImageProcessor.ImageBatch(
            numpy_image=np.array([None]),
            images=[BaseImage(None)],
            graph_obj=graph_obj,
            shape=(0,0),
            props=[],
            is_2d=True,
            is_graph_only=True,
            scale_factor=0.0,
            scaling_options=[],
            selected_images_positions=set(),
            selected_frame_pos= 0,
            view_options=views,
        )
        imp_obj._image_batches = [img_batch]
        return imp_obj, graph_file
