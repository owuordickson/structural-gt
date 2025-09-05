# SPDX-License-Identifier: GNU GPL v3
"""
A class for building an MDP environment of image filters for graph generation.
"""

import math
import numpy as np
from dataclasses import dataclass
from sgtlib.modules import BaseImage


class SGTGraphEnv:

    def __init__(self):
        pass

    def _build_action_space(self):
        pass

    def _build_observation_space(self):
        pass



class FilterSearchSpace:

    @dataclass
    class Candidate:
        position: int|None = None
        std_cost: float|None = None
        img_configs: dict|None = None

    @dataclass
    class SearchSpace:
        candidates: list["FilterSearchSpace.Candidate"] = None
        ignore_candidates: list["FilterSearchSpace.Candidate"] = None
        best_candidate: "FilterSearchSpace.Candidate" = None

    def __init__(self):
        pass

    @staticmethod
    def build_search_space(img_obj: BaseImage) -> "FilterSearchSpace.SearchSpace":
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.

        :param img_obj: The image object.
        :return: The search space.
        """
        # Set ranges for each parameter (Discrete action space)
        threshold_types = [0, 1, 2]                         # global / adaptive / OTSU
        global_thresh_range = list(range(1, 256))           # 1–255
        adaptive_local_range = list(range(1, 100, 2))       # 1–99 (odd)
        brightness_levels = list(range(-100, 101))          # -100–100
        contrast_levels = list(range(-100, 101))            # -100–100
        gamma_range = np.arange(0.01, 5.01, 0.01)           # 0.01–5.0
        blurring_window_sizes = list(range(2, 8, 2))        # 2, 4, 6 (even)
        filter_window_sizes = list(range(1, 101))           # 1–100
        """
        toggle_filters = [
            {"apply_dark_foreground": 1, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 1, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0,  "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 1, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 1, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 1, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 1, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 1, "apply_median_filter": 0, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 1, "apply_scharr_gradient": 0},
            {"apply_dark_foreground": 0, "apply_gamma": 0, "apply_autolevel": 0, "apply_laplacian_gradient": 0, "apply_gaussian_blur": 0, "apply_lowpass_filter": 0, "apply_sobel_gradient": 0, "apply_median_filter": 0, "apply_scharr_gradient": 1}
        ]
        """

        # Initialize search space
        init_configs = img_obj.configs
        search_space = FilterSearchSpace.SearchSpace(candidates=[], ignore_candidates=[])

        for tt in threshold_types:
            global_range = global_thresh_range if tt == 0 else [128]
            adaptive_range = adaptive_local_range if tt == 1 else [11]
            for global_thresh in global_range:
                for adaptive_thresh in adaptive_range:
                    for brightness in brightness_levels:
                        for contrast in contrast_levels:
                            for gamma_val in gamma_range:
                                for blur_size in blurring_window_sizes:
                                    for filter_size in filter_window_sizes:
                                        pass



        return search_space

    @staticmethod
    def cost_function(candidate: "FilterSearchSpace.Candidate", img_obj: BaseImage) -> None:
        """Calculate and apply the cost of a candidate. Given the image filter configurations, apply them to get a
        binary image and find the number of white pixels in the image. Retrieve the corresponding pixel values from the
        original image and calculate the Standard Deviation (SD) of the pixel values.

        :param candidate: A candidate in the search space.
        :param img_obj: The image object.
        """

        if img_obj is None:
            return

        if candidate.img_configs is None:
            candidate.std_cost = math.inf
            return

        # Copy image filter configurations to the image object
        img_obj.configs = candidate.img_configs
        # Reset image filters
        img_obj.img_mod, img_obj.img_bin = None, None
        # Apply image filters
        img_data = img_obj.img_2d.copy()
        img_obj.img_mod = img_obj.process_img(image=img_data)
        img_mod = img_obj.img_mod.copy()
        img_obj.img_bin = img_obj.binarize_img(img_mod)
        img_obj.img_mod = img_mod
        # Compute SD as cost
        eval_std, eval_hist = img_obj.evaluate_img_binary()
        candidate.std_cost = eval_std

    @staticmethod
    def evaluate_candidate(search_space: "FilterSearchSpace.SearchSpace", candidate: "FilterSearchSpace.Candidate") -> None:
        """
        Evaluate a candidate in the search space, check if it is better than the best candidate.

        :param search_space: The search space.
        :param candidate: The candidate to evaluate.
        """
        if candidate.std_cost is None:
            return

        if search_space.best_candidate is None:
            search_space.best_candidate = FilterSearchSpace.Candidate(
                position=candidate.position,
                std_cost=candidate.std_cost,
                img_configs=candidate.img_configs)

        if candidate.position in search_space.ignore_candidates:
            return

        elif candidate.std_cost < search_space.best_candidate.std_cost:
            search_space.best_candidate = FilterSearchSpace.Candidate(
                position=candidate.position,
                std_cost=candidate.std_cost,
                img_configs=candidate.img_configs
            )


