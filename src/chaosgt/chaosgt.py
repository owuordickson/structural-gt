# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Processing of images and chaos engineering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps


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

    def compute_fractal_dimension(self):
        # sierpinski_im = ps.generators.sierpinski_foam(4, 5)
        fd_metrics = ps.metrics.boxcount(self.img)
        print(fd_metrics.slope)
        x = np.log(np.array(fd_metrics.size))
        y = np.log(np.array(fd_metrics.count))
        fD = np.polyfit(x, y, 1)[0]  # fD = lim r -> 0 log(Nr)/log(1/r)
        print(fD)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('box size')
        ax1.set_ylabel('box count')
        ax2.set_xlabel('box size')
        ax2.set_ylabel('slope')
        ax2.set_xscale('log')
        ax1.plot(fd_metrics.size, fd_metrics.count, '-o')
        ax2.plot(fd_metrics.size, fd_metrics.slope, '-o')
        plt.show()

    @staticmethod
    def load_img_from_file(file):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        return img
