# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Processing of images and chaos engineering
"""

import cv2


class ChaoticStruct:

    def __init__(self, img_path):
        self.img = ChaoticStruct.load_img_from_file(img_path)

    def resize_img(self, size):
        w, h = self.img.shape
        if h > w:
            scale_factor = size / h
        else:
            scale_factor = size / w
        std_width = int(scale_factor * w)
        std_height = int(scale_factor * h)
        std_size = (std_height, std_width)
        std_img = cv2.resize(self.img, std_size)
        return std_img

    @staticmethod
    def load_img_from_file(file):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        return img
