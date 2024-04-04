# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Processing of an image by applying filters to it and converting it to binary version.
"""

import re
import os
import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import autolevel, median


class ImageProcessor:

    def __init__(self, img_path, out_path, options_img=None, img=None):
        self.configs_img = options_img
        self.img_path = img_path
        self.output_path = out_path
        self.img_raw = ImageProcessor.load_img_from_file(img_path)
        if img is None:
            self.img = ImageProcessor.resize_img(512, self.img_raw.copy())
        else:
            self.img = img
        self.img_bin = None
        self.img_mod = None
        self.img_net = None
        self.otsu_val = None

    def apply_filters(self):
        self.img_mod = self.process_img(self.img.copy())
        self.img_bin, self.otsu_val = self.binarize_img(self.img_mod.copy())

    def process_img(self, image):
        """

        :return:
        """

        options = self.configs_img
        filtered_img = ImageProcessor.control_brightness(image, options.brightness_level, options.contrast_level)

        if options.gamma != 1.00:
            inv_gamma = 1.00 / options.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype('uint8')
            filtered_img = cv2.LUT(filtered_img, table)

        # applies a low-pass filter
        if options.apply_lowpass == 1:
            w, h = filtered_img.shape
            ham1x = np.hamming(w)[:, None]  # 1D hamming
            ham1y = np.hamming(h)[:, None]  # 1D hamming
            ham2d = np.sqrt(np.dot(ham1x, ham1y.T)) ** options.lowpass_window_size  # expand to 2D hamming
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
        if options.apply_median == 1:
            # making a 5x5 array of all 1's for median filter
            med_disk = disk(5)
            filtered_img = median(filtered_img, med_disk)

        # applying gaussian blur
        if options.apply_gaussian == 1:
            b_size = options.gaussian_blurring_size
            filtered_img = cv2.GaussianBlur(filtered_img, (b_size, b_size), 0)

        # applying auto-level filter
        if options.apply_autolevel == 1:
            # making a disk for the auto-level filter
            auto_lvl_disk = disk(options.autolevel_blurring_size)
            filtered_img = autolevel(filtered_img, footprint=auto_lvl_disk)

        # applying a scharr filter, and then taking that image and weighting it 25% with the original
        # this should bring out the edges without separating each "edge" into two separate parallel ones
        if options.apply_scharr == 1:
            d_depth = cv2.CV_16S
            grad_x = cv2.Scharr(filtered_img, d_depth, 1, 0)
            grad_y = cv2.Scharr(filtered_img, d_depth, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        # applying sobel filter
        if options.apply_sobel == 1:
            scale = 1
            delta = 0
            d_depth = cv2.CV_16S
            grad_x = cv2.Sobel(filtered_img, d_depth, 1, 0, ksize=options.sobel_kernel_size, scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(filtered_img, d_depth, 0, 1, ksize=options.sobel_kernel_size, scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        # applying laplacian filter
        if options.apply_laplacian == 1:
            d_depth = cv2.CV_16S
            dst = cv2.Laplacian(filtered_img, d_depth, ksize=options.laplacian_kernel_size)
            # dst = cv2.Canny(img_filtered, 100, 200); # canny edge detection test
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        return filtered_img

    def binarize_img(self, image):
        """

        :return:
        """
        # image = self.img_filtered.copy()
        img_bin = None
        options = self.configs_img
        # only needed for OTSU threshold
        otsu_res = 0

        if image is None:
            return None

        # applying universal threshold, checking if it should be inverted (dark foreground)
        if options.threshold_type == 0:
            if options.apply_dark_foreground == 1:
                img_bin = cv2.threshold(image, options.threshold_global, 255, cv2.THRESH_BINARY_INV)[1]
            else:
                img_bin = cv2.threshold(image, options.threshold_global, 255, cv2.THRESH_BINARY)[1]

        # adaptive threshold generation
        elif options.threshold_type == 1:
            if options.apply_dark_foreground == 1:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, options.threshold_adaptive, 2)
            else:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, options.threshold_adaptive, 2)

        # OTSU threshold generation
        elif options.threshold_type == 2:
            if options.apply_dark_foreground == 1:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]
            else:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]

        return img_bin, otsu_res

    def create_filenames(self, image_path=None):
        """
            Making the new filenames
        :return:
        """
        if image_path is None:
            img_dir, filename = os.path.split(self.img_path)
        else:
            img_dir, filename = os.path.split(image_path)
        if self.output_path == '':
            output_location = img_dir
        else:
            output_location = self.output_path

        filename = re.sub('.png', '', filename)
        filename = re.sub('.tif', '', filename)
        filename = re.sub('.jpg', '', filename)
        filename = re.sub('.jpeg', '', filename)

        return filename, output_location

    @staticmethod
    def load_img_from_file(file):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        return img

    @staticmethod
    def load_img_from_pil(img_pil):
        img_arr = np.array(img_pil)
        cv2_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return cv2_image

    @staticmethod
    def control_brightness(img, brightness_val=0, contrast_val=0):
        """

        :param contrast_val:
        :param brightness_val:
        :param img:
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
    def resize_img(size, image):
        w, h = image.shape
        if h > w:
            scale_factor = size / h
        else:
            scale_factor = size / w
        std_width = int(scale_factor * w)
        std_height = int(scale_factor * h)
        std_size = (std_height, std_width)
        std_img = cv2.resize(image, std_size)
        return std_img
