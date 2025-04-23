# SPDX-License-Identifier: GNU GPL v3

"""
Processes of an image by applying filters to it and converting it to a binary version.
"""

import cv2
import numpy as np
from cv2.typing import MatLike
from skimage.morphology import disk
from skimage.filters.rank import autolevel, median

from ..configs.config_loader import load_img_configs
from .sgt_utils import safe_uint8_image



class BaseImage:
    """
    A class that is used to binarize an image by applying filters to it and converting it to a binary version.

    Args:
        raw_img (MatLike): Raw image in OpenCV format
        scale_factor (float): Scale factor used to downsample/up-sample the image.
    """

    def __init__(self, raw_img: MatLike, scale_factor: float = 1.0):
        """
        A class that is used to binarize an image by applying filters to it and converting it to a binary version.

        Args:
            raw_img: Raw image in OpenCV format.
            scale_factor: Scale factor used to downsample/up-sample the image.
        """
        self.configs: dict = load_img_configs()  # image processing configuration parameters and options.
        self.img_raw: MatLike | None = safe_uint8_image(raw_img)
        self.img_2d: MatLike | None = None
        self.img_bin: MatLike | None = None
        self.img_mod: MatLike | None = None
        self.has_alpha_channel: bool = False
        self.scale_factor: float = scale_factor
        self.init_image()

    def init_image(self):
        """
        Initialize the class member variables (or attributes).
        Returns:

        """
        img_data = self.img_raw.copy()
        if img_data is None:
            return

        self.has_alpha_channel, _ = BaseImage.check_alpha_channel(self.img_raw)
        self.img_2d = img_data

    def get_pixel_width(self):
        """Compute pixel dimension in nanometers to estimate and update the width of graph edges."""
        opt_img = self.configs
        pixel_count = int(opt_img["scalebar_pixel_count"]["value"])
        scale_val = float(opt_img["scale_value_nanometers"]["value"])
        if (scale_val > 0) and (pixel_count > 0):
            px_width = BaseImage.compute_pixel_width(scale_val, pixel_count)
            opt_img["pixel_width"]["value"] = px_width / self.scale_factor

    def apply_img_crop(self, x: float, y: float, width: float, height: float):
        """
        A function that crops images into a new box dimension.

        :param x: Left coordinate of cropping box.
        :param y: Top coordinate of cropping box.
        :param width: Width of cropping box.
        :param height: Height of cropping box.
        """

        # Verify the bounds of the cropping box
        h, w = self.img_2d.shape[:2]
        x = max(0.0, min(x, w))
        y = max(0.0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)

        # Convert to integer pixel indices
        x = int(round(x))
        y = int(round(y))
        width = int(round(width))
        height = int(round(height))

        # Crop image
        self.img_2d = self.img_2d[y:y + height, x:x + width]

    def process_img(self, image: MatLike):
        """
        Apply filters to image.

        :param image: OpenCV image.
        :return: None
        """

        if image is None:
            return None

        alpha_channel, _ = BaseImage.check_alpha_channel(image)
        if alpha_channel:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        opt_img = self.configs
        filtered_img = BaseImage.control_brightness(image, opt_img["brightness_level"]["value"],
                                                    opt_img["contrast_level"]["value"])

        if float(opt_img["apply_gamma"]["dataValue"]) != 1.00:
            inv_gamma = 1.00 / float(opt_img["apply_gamma"]["dataValue"])
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype('uint8')
            filtered_img = cv2.LUT(filtered_img, table)

        # applies a low-pass filter
        if opt_img["apply_lowpass_filter"]["value"] == 1:
            h, w = filtered_img.shape
            ham1x = np.hamming(w)[:, None]  # 1D hamming
            ham1y = np.hamming(h)[:, None]  # 1D hamming
            ham2d = np.sqrt(np.dot(ham1x, ham1y.T)) ** int(
                opt_img["apply_lowpass_filter"]["dataValue"])  # expand to 2D hamming
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
            filtered_img = BaseImage.apply_filter('scharr', filtered_img, grad_x, grad_y)

        # applying sobel filter
        if opt_img["apply_sobel_gradient"]["value"] == 1:
            scale = 1
            delta = 0
            d_depth = cv2.CV_16S
            grad_x = cv2.Sobel(filtered_img, d_depth, 1, 0, ksize=int(opt_img["apply_sobel_gradient"]["dataValue"]),
                               scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(filtered_img, d_depth, 0, 1, ksize=int(opt_img["apply_sobel_gradient"]["dataValue"]),
                               scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            filtered_img = BaseImage.apply_filter('sobel', filtered_img, grad_x, grad_y)

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

        if image is None:
            return None

        img_bin = None
        opt_img = self.configs
        otsu_res = 0  # only needed for the OTSU threshold

        # Applying the universal threshold, checking if it should be inverted (dark foreground)
        if opt_img["threshold_type"]["value"] == 0:
            if opt_img["apply_dark_foreground"]["value"] == 1:
                img_bin = \
                cv2.threshold(image, int(opt_img["global_threshold_value"]["value"]), 255, cv2.THRESH_BINARY_INV)[1]
            else:
                img_bin = cv2.threshold(image, int(opt_img["global_threshold_value"]["value"]), 255, cv2.THRESH_BINARY)[
                    1]

        # adaptive threshold generation
        elif opt_img["threshold_type"]["value"] == 1:
            if self.configs["adaptive_local_threshold_value"]["value"] <= 1:
                # Bug fix (crushes app)
                self.configs["adaptive_local_threshold_value"]["value"] = 3

            if opt_img["apply_dark_foreground"]["value"] == 1:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV,
                                                int(opt_img["adaptive_local_threshold_value"]["value"]), 2)
            else:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,
                                                int(opt_img["adaptive_local_threshold_value"]["value"]), 2)

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
        self.configs["otsu"]["value"] = otsu_res
        return img_bin

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
        run_info += "\n\n"

        run_info += "***Image Scale***\n"
        run_info += f"Size = {self.img_2d.shape[0]} x {self.img_2d.shape[1]} px"
        run_info += f" || Scale Factor = {self.scale_factor}"

        return run_info

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
        return img

    @staticmethod
    def control_brightness(img: MatLike, brightness_val: int = 0, contrast_val: int = 0):
        """
        Apply contrast and brightness filters to image.

        :param img: OpenCV image.
        :param brightness_val: Brightness value.
        :param contrast_val: Contrast value.
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
    def check_alpha_channel(img: MatLike):
        """
        A function that checks if an image has an Alpha channel or not. Only works for images with up to 4-Dimensions.

        :param img: OpenCV image.
        """

        if img is None:
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

        # Unknown Format
        return False, None

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
        w, h = image.shape[:2]
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
    def compute_pixel_width(scale_val: float, scalebar_pixel_count: int):
        """
        Compute the width of a single pixel in nanometers.

        :param scale_val: Unit value of the scale in nanometers.
        :param scalebar_pixel_count: Pixel count of the scalebar width.
        :return: Width of a single pixel in nanometers.
        """

        val_in_meters = scale_val / 1e9
        pixel_width = val_in_meters / scalebar_pixel_count
        return pixel_width
