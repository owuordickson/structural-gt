# SPDX-License-Identifier: GNU GPL v3

"""
Processes of an image by applying filters to it and converting it to binary version.
"""

import re
import os
import io
import sys
import cv2
import pydicom
import logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from cv2.typing import MatLike
from skimage.morphology import disk
from skimage.filters.rank import autolevel, median

from ..configs.config_loader import load_img_configs
logger = logging.getLogger("SGT App")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)


ALLOWED_IMG_EXTENSIONS = ('*.jpg', '*.png', '*.jpeg', '*.tif', '*.tiff', '*.qptiff')
class ImageProcessor:
    """
    A class for processing and preparing microscopy images for graph theory analysis.

    Args:
        img_path (str): input image path.
        out_dir (str): directory path for storing results.
    """

    def __init__(self, img_path, out_dir):
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
        >>> imp_obj.apply_filters()
        """
        self.configs = load_img_configs()  # image processing configuration parameters and options.
        self.img_path = img_path
        self.output_dir = out_dir
        self.img_raw = ImageProcessor.load_img_from_file(img_path)
        self.props = []
        self.img_3d = None
        self.img_2d = None
        self.img_bin = None
        self.img_mod = None
        self.img_net = None
        self.scale_factor = 1
        self.has_alpha_channel = False
        if self.img_raw is not None:
            self._initialize_members()

    def _initialize_members(self):
        """"""
        self.has_alpha_channel, _ = ImageProcessor.check_alpha_channel(self.img_raw)
        self.img_2d = self._convert_to_grayscale()
        self.img_3d = self._convert_to_3d()
        if self.img_2d is not None:
            self.props = self.get_img_props()

    def _convert_to_3d(self):
        """
        A functions that converts an image into 2 dimensions (depth/slices x height x width)

        """
        img_data = self.img_raw.copy()
        if img_data is None or self.has_alpha_channel or len(img_data.shape) < 3:
            return None

        # Extract slices
        front = img_data[:, :, 0]  # First slice along depth (Z-axis)
        back = img_data[:, :, -1]  # Last slice along depth (Z-axis)
        top = img_data[0, :, :]  # First slice along height (Y-axis)
        bottom = img_data[-1, :, :]  # Last slice along height (Y-axis)
        side_left = img_data[:, 0, :]  # First slice along width (X-axis)
        side_right = img_data[:, -1, :]  # Last slice along width (X-axis)

        img_views = {
            "front": front,
            "back": back,
            "top": top,
            "bottom": bottom,
            "side-left": side_left,
            "side-right": side_right
        }

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, view, title in zip(axes.flat, img_views.values(), img_views.keys()):
            ax.imshow(view, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()

        # Convert plot to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Convert PNG buffer to OpenCV image
        file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        img_3d = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as OpenCV image

        logging.info("Image is 3D.", extra={'user': 'SGT Logs'})
        return img_3d

    def _convert_to_grayscale(self):
        """
            Reads the first slice of a 3D image if it is indeed 3D, image itself if 2D.

            Returns:
                numpy.ndarray or None: The first slice of the image if it's 3D, image itself if 2D, None otherwise.
        """
        img_data = self.img_raw.copy()

        if img_data is None:
            return None

        if len(img_data.shape) == 3 and self.has_alpha_channel:
            logging.info("Image is 2D with Alpha Channel (RGBA).", extra={'user': 'SGT Logs'})
            img_2d = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            return img_2d

        if len(img_data.shape) >= 3:
            # Get first slice and check if it is RGB or RGBA?
            img_2d = img_data[0, :, :]  # or image[:, :, 0] depending on the axis
            if self.has_alpha_channel:
                img_2d = cv2.cvtColor(img_2d, cv2.COLOR_RGB2GRAY)
            return img_2d
        else:
            logging.info("Image is 2D.", extra={'user': 'SGT Logs'})
            return img_data

    def apply_filters(self):
        """
        Executes function for processing image filters and converting the resulting image into a binary.

        :return: None
        """
        self.img_mod = self.process_img(self.img_2d.copy())
        self.img_bin = self.binarize_img(self.img_mod.copy())

        # Compute pixel dimension in nanometers
        opt_img = self.configs
        pixel_count = int(opt_img["scalebar_pixel_count"]["value"])
        scale_val = float(opt_img["scale_value_nanometers"]["value"])
        if (scale_val > 0) and (pixel_count > 0):
            px_width = ImageProcessor.compute_pixel_width(scale_val, pixel_count)
            opt_img["pixel_width"]["value"] = px_width/self.scale_factor

    def crop_img(self, x: float, y: float, width: float, height: float):
        """
        A function that crops images into a new box dimension.
        :param x: left coordinate of cropping box.
        :param y: top coordinate of cropping box.
        :param width: width of cropping box.
        :param height: height of cropping box.
        """

        # Verify bounds of cropping box
        h, w = self.img_2d.shape
        x = max(0.0, min(x, w))
        y = max(0.0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)

        # Crop image
        self.img_2d = self.img_2d[y:y + height, x:x + width]
        self.img_3d = self.img_3d[y:y + height, x:x + width]

    def undo_cropping(self):
        """
        A function that restores image to its original size.
        """
        # self.img, self.scale_factor = ImageProcessor.resize_img(512, self.img_raw.copy())
        self.img_2d = self._convert_to_grayscale()
        self.img_3d = self._convert_to_3d()

    def process_img(self, image: MatLike):
        """
        Apply filters to image.

        :param image: OpenCV image.
        :return: None
        """

        if image is None:
            return None

        opt_img = self.configs
        filtered_img = ImageProcessor.control_brightness(image, opt_img["brightness_level"]["value"], opt_img["contrast_level"]["value"])

        if float(opt_img["apply_gamma"]["dataValue"]) != 1.00:
            inv_gamma = 1.00 / float(opt_img["apply_gamma"]["dataValue"])
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype('uint8')
            filtered_img = cv2.LUT(filtered_img, table)

        # applies a low-pass filter
        if opt_img["apply_lowpass_filter"]["value"] == 1:
            w, h = filtered_img.shape
            ham1x = np.hamming(w)[:, None]  # 1D hamming
            ham1y = np.hamming(h)[:, None]  # 1D hamming
            ham2d = np.sqrt(np.dot(ham1x, ham1y.T)) ** int(opt_img["apply_lowpass_filter"]["dataValue"]) # expand to 2D hamming
            f = cv2.dft(filtered_img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
            f_filtered = ham2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img * 255 / filtered_img.max()
            filtered_img = filtered_img.astype(np.uint8)

        # applying median filter
        if opt_img["apply_median_filter"]["value"] == 1:
            # making a 5x5 array of all 1's for median filter
            med_disk = disk(5)
            filtered_img = median(filtered_img, med_disk)

        # applying gaussian blur
        if opt_img["apply_gaussian_blur"]["value"] == 1:
            b_size = int(opt_img["apply_gaussian_blur"]["dataValue"])
            filtered_img = cv2.GaussianBlur(filtered_img, (b_size, b_size), 0)

        # applying auto-level filter
        if opt_img["apply_autolevel"]["value"] == 1:
            # making a disk for the auto-level filter
            auto_lvl_disk = disk(int(opt_img["apply_autolevel"]["dataValue"]))
            filtered_img = autolevel(filtered_img, footprint=auto_lvl_disk)

        # applying a scharr filter,
        if opt_img["apply_scharr_gradient"]["value"] == 1:
            # applying a scharr filter, and then taking that image and weighting it 25% with the original
            # this should bring out the edges without separating each "edge" into two separate parallel ones
            d_depth = cv2.CV_16S
            grad_x = cv2.Scharr(filtered_img, d_depth, 1, 0)
            grad_y = cv2.Scharr(filtered_img, d_depth, 0, 1)
            filtered_img = ImageProcessor.apply_filter('scharr', filtered_img, grad_x, grad_y)

        # applying sobel filter
        if opt_img["apply_sobel_gradient"]["value"] == 1:
            scale = 1
            delta = 0
            d_depth = cv2.CV_16S
            grad_x = cv2.Sobel(filtered_img, d_depth, 1, 0, ksize=int(opt_img["apply_sobel_gradient"]["dataValue"]), scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(filtered_img, d_depth, 0, 1, ksize=int(opt_img["apply_sobel_gradient"]["dataValue"]), scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            filtered_img = ImageProcessor.apply_filter('sobel', filtered_img, grad_x, grad_y)

        # applying laplacian filter
        if opt_img["apply_laplacian_gradient"]["value"] == 1:
            d_depth = cv2.CV_16S
            dst = cv2.Laplacian(filtered_img, d_depth, ksize=int(opt_img["apply_laplacian_gradient"]["dataValue"]))
            # dst = cv2.Canny(img_filtered, 100, 200); # canny edge detection test
            abs_dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, abs_dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        return filtered_img

    def binarize_img(self, image: MatLike):
        """
        Convert image to binary.

        :param image:
        :return: None
        """

        img_bin = None
        opt_img = self.configs
        # only needed for OTSU threshold
        otsu_res = 0

        if image is None:
            return None, None

        # applying universal threshold, checking if it should be inverted (dark foreground)
        if opt_img["threshold_type"]["value"] == 0:
            if opt_img["apply_dark_foreground"]["value"] == 1:
                img_bin = cv2.threshold(image, int(opt_img["global_threshold_value"]["value"]), 255, cv2.THRESH_BINARY_INV)[1]
            else:
                img_bin = cv2.threshold(image, int(opt_img["global_threshold_value"]["value"]), 255, cv2.THRESH_BINARY)[1]

        # adaptive threshold generation
        elif opt_img["threshold_type"]["value"] == 1:
            if self.configs["adaptive_local_threshold_value"]["value"] <= 1:
                # Bug fix (crushes app)
                self.configs["adaptive_local_threshold_value"]["value"] = 3

            if opt_img["apply_dark_foreground"]["value"] == 1:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, int(opt_img["adaptive_local_threshold_value"]["value"]), 2)
            else:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, int(opt_img["adaptive_local_threshold_value"]["value"]), 2)

        # OTSU threshold generation
        elif opt_img["threshold_type"]["value"] == 2:
            if opt_img["apply_dark_foreground"]["value"] == 1:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]
            else:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]
        opt_img["otsu"]["value"] = otsu_res
        return img_bin

    def compute_pixel_length(self, img_path: str, img_path_w_bar: str = None, img_length: float = None):
        """
        Compute the length of a single pixel in meters.

        :param img_path: directory path to image including the scale bar.
        :param img_path_w_bar: directory path to cropped image of scale bar.
        :param img_length: length of the image in nanometers.
        :return: None
        """

        # img_orig = ImageProcessor.load_img_from_file(img_path_w_bar)
        # scale_bar_length = 0  # or Use image length in meters
        # scale_bar_pixel_count = 0
        # img_orig_length_pixel_count = 0
        # self.pixel_length = 0
        pass

    def compute_fractal_dimension(self):
        """
        Compute the fractal dimension of the processed image using a C module.

        :return:
        """
        pass

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

    def get_config_info(self):
        """
        Get the user selected parameters and options information.
        :return:
        """

        opt_img = self.configs
        
        run_info = "***Image Filter Configurations***\n"
        if opt_img["threshold_type"]["value"] == 0:
            run_info += "Global Threshold (" + str(opt_img["global_threshold_value"]["value"]) + ")"
        elif opt_img["threshold_type"]["value"] == 1:
            run_info += "Adaptive Threshold, " + str(opt_img["adaptive_local_threshold_value"]["value"]) + " bit kernel"
        elif opt_img["threshold_type"]["value"] == 2:
            run_info += "OTSU Threshold"
        if opt_img["apply_gamma"]["value"] == 1:
            run_info += f" || Gamma = {opt_img["apply_gamma"]["dataValue"]}"
        run_info += "\n"
        if opt_img["apply_median_filter"]["value"]:
            run_info += "Median Filter ||"
        if opt_img["apply_gaussian_blur"]["value"]:
            run_info += "Gaussian Blur, " + str(opt_img["apply_gaussian_blur"]["dataValue"]) + " bit kernel || "
        if opt_img["apply_autolevel"]["value"]:
            run_info += "Autolevel, " + str(opt_img["apply_autolevel"]["dataValue"]) + " bit kernel || "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info
        run_info += "\n"
        if opt_img["apply_dark_foreground"]["value"]:
            run_info += "Dark Foreground || "
        if opt_img["apply_laplacian_gradient"]["value"]:
            run_info += "Laplacian Gradient || "
        if opt_img["apply_scharr_gradient"]["value"]:
            run_info += "Scharr Gradient || "
        if opt_img["apply_sobel_gradient"]["value"]:
            run_info += "Sobel Gradient || "
        if opt_img["apply_lowpass_filter"]["value"]:
            run_info += "Low-pass filter, " + str(opt_img["apply_lowpass_filter"]["dataValue"]) + " window size || "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info
        run_info += "\n\n"
        
        run_info += "***Microscopy Parameters***\n"
        run_info += f"Scalebar Value = {opt_img["scale_value_nanometers"]["value"]} nm"
        run_info += f" || Scalebar Pixel Count = {opt_img["scalebar_pixel_count"]["value"]}\n"
        run_info += f"Resistivity = {opt_img["resistivity"]["value"]}" + r"$\Omega$m"

        return run_info

    def get_img_props(self):
        """
        A method that retrieves image properties and stores them in a list-array.

        Returns: list of image properties

        """
        f_name, _ = self.create_filenames()
        _, fmt = ImageProcessor.check_alpha_channel(self.img_raw)
        num_dim = 2 if self.img_3d is None else len(self.img_raw.shape)
        if num_dim == 2:
            slices = 0
            height, width = self.img_2d.shape
        else:
            slices, height, width = self.img_raw.shape
        props = [
            ["Name", f_name],
            ["Height x Width", f"({width} x {height}) pixels"] if slices==0 else ["Slices x H x W", f"({slices} x {width} x {height}) pixels"],
            ["Dimensions", f"{num_dim}D"],
            ["Format", f"{fmt}"],
            # ["Pixel Size", "2nm x 2nm"]
        ]
        return props

    def display_images(self):
        """
        Create plot figures of original, processed, and binary image.

        :return:
        """
        opt_img = self.configs
        raw_img = self.img_2d
        filtered_img = self.img_mod
        img_bin = self.img_bin

        img_histogram = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=400)
        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        ax_3 = fig.add_subplot(2, 2, 3)
        ax_4 = fig.add_subplot(2, 2, 4)

        ax_1.set_title("Original Image")
        ax_1.set_axis_off()
        ax_1.imshow(raw_img, cmap='gray')

        ax_2.set_title("Processed Image")
        ax_2.set_axis_off()
        ax_2.imshow(filtered_img, cmap='gray')

        ax_3.set_title("Binary Image")
        ax_3.set_axis_off()
        ax_3.imshow(img_bin, cmap='gray')

        ax_4.set_title("Histogram of Processed Image")
        ax_4.set(yticks=[], xlabel='Pixel values', ylabel='Counts')
        ax_4.plot(img_histogram)
        if opt_img["threshold_type"]["value"] == 0:
            thresh_arr = np.array([[int(opt_img["global_threshold_value"]["value"]), int(opt_img["global_threshold_value"]["value"])],
                                   [0, max(img_histogram)]], dtype='object')
            ax_4.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
        elif opt_img["threshold_type"]["value"] == 2:
            otsu_val = opt_img["otsu"]["value"]
            thresh_arr = np.array([[otsu_val, otsu_val],
                                   [0, max(img_histogram)]], dtype='object')
            ax_4.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
        return fig

    @staticmethod
    def apply_filter(filter_type: str, img: MatLike, grad_x, grad_y):
        """"""
        if filter_type == 'scharr' or filter_type == 'sobel':
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            abs_dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(img, 0.75, abs_dst, 0.25, 0)
            return cv2.convertScaleAbs(filtered_img)

    @staticmethod
    def control_brightness(img: MatLike, brightness_val: int = 0, contrast_val: int = 0):
        """
        Apply contrast and brightness filters to image.

        :param img: OpenCV image.
        :param brightness_val: brightness value.
        :param contrast_val: contrast value.
        :return:
        """

        brightness = ((brightness_val / 100) * 127)
        contrast = ((contrast_val / 100) * 127)

        # img = np.int16(img)
        # img = img * (contrast / 127 + 1) - contrast + brightness
        # img = np.clip(img, 0, 255)
        # img = np.uint8(img)

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max_val = 255
            else:
                shadow = 0
                max_val = 255 + brightness
            alpha_b = (max_val - shadow) / 255
            gamma_b = shadow
            img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        if contrast != 0:
            alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma_c = 127 * (1 - alpha_c)
            img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

        # text string in the image.
        # cv2.putText(new_img, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        # 1, (0, 0, 255), 2)
        return img

    @staticmethod
    def load_img_from_file_v1(file: str):
        """
        Read image and save it as an OpenCV object.

        :param file: file path.
        :return:
        """
        if file == "":
            return None
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        return img

    @staticmethod
    def load_img_from_file(file: str):
        """
        Read image and save it as an OpenCV object.

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
                return image
            elif ext in ['.tif', '.tiff', '.qptiff']:
                # Try load multi-page TIFF using PIL
                img = Image.open(file)
                images = []
                while True:
                    images.append(np.array(img))
                    try:
                        img.seek(img.tell() + 1)
                    except EOFError:
                        break
                if len(images) > 1:
                    return np.stack(images, axis=0)
                else:
                    return images[0]
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
            logging.exception(f"Error loading {file}", err, extra={'user': 'SGT Logs'})
            return None

    @staticmethod
    def load_img_from_pil(img_pil: MatLike|Image.Image):
        """
        Read image from PIL.

        :param img_pil:
        :return:
        """
        img_arr = np.array(img_pil)
        cv2_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return cv2_image

    @staticmethod
    def plot_to_img(fig: plt.Figure):
        """
        Convert a Matplotlib figure to a PIL Image and return it

        :param fig: Matplotlib figure.
        """
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img

    @staticmethod
    def check_alpha_channel(img: MatLike):
        """
        A function that checks if an image has an Alpha channel or not. Only works for images with upto 4-Dimensions.

        :param img: OpenCV image.
        """

        if img is None:
            return False, None

        if len(img.shape) > 4:
            # Unknown Format
            return False, None

        if len(img.shape) == 2:
            return False, "Grayscale"

        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 4:
                return True, "RGBA"
            elif channels == 3:
                return True, "RGB"
            elif channels == 2:
                return True, "Grayscale + Alpha"
            elif channels == 1:
                return True, "Grayscale"
            else:
                return False, "Multi"

        if len(img.shape) == 4:  # (Depth, Height, Width, Channels)
            return True, "Multi + Alpha"

    @staticmethod
    def resize_img(size: int, image: MatLike):
        """
        Resizes image to specified size.

        :param size: new image pixel size.
        :param image: OpenCV image.
        :return: rescaled image
        """
        if image is None:
            return None, None
        w, h = image.shape  # what about 3D?
        if h > w:
            scale_factor = size / h
        else:
            scale_factor = size / w
        std_width = int(scale_factor * w)
        std_height = int(scale_factor * h)
        std_size = (std_height, std_width)
        std_img = cv2.resize(image, std_size)
        return std_img, scale_factor

    @staticmethod
    def rescale_to_square(image: MatLike):
        """
        Rescales image so that it is equal in length and width.

        :param image: OpenCV image.
        :return: rescaled image.
        """
        w, h = image.shape
        length = h if h > w else w
        return cv2.resize(image, (length, length))

    @staticmethod
    def compute_pixel_width(scale_val: float, scalebar_pixel_count: int):
        """
        Compute the width of a single pixel in nanometers.

        :param scale_val: unit value of the scale in nanometers.
        :param scalebar_pixel_count: pixel count of the width of the scalebar.
        :return: width of a single pixel in nanometers.
        """

        val_in_meters = scale_val / 1e9
        pixel_width = val_in_meters/scalebar_pixel_count
        return pixel_width

