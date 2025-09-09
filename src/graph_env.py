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
    """
    Class for building a discrete search space of image filters. This search space is huge and irregular
    (over 11k Trillion candidates) and does not have the structure of (Markov Decision Process) MDP states. For example,
    if the current state S1 is a combination of image filter configurations, then the decision to select the
    next or future states S2, S3, ... does not depend on the previous state S1 (No Markov Property). For this reason,
    we use Genetic Algorithm (GA) to find the best combination of image filter configurations. GA is a global optimizer
    (or global optimization method/algorithm) that finds the best solution in a given search space.
    """

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
    def build_search_space(img_obj: BaseImage, total_pop: int = 1000) -> "FilterSearchSpace.SearchSpace":
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.
        The actual search space has over 118k Trillion candidates.

        :param img_obj: The image object.
        :param total_pop: The total population size.
        :return: The search space.
        """
        # Set ranges for each parameter (Discrete action space)
        threshold_types = [0, 1, 2]                         # global / adaptive / OTSU
        global_thresh_range = list(range(1, 256))           # 1–255
        adaptive_local_range = list(range(1, 100, 2))       # 1–99 (odd)
        brightness_levels = list(range(-100, 101))          # -100–100
        contrast_levels = list(range(-100, 101))            # -100–100
        gamma_range = np.arange(0.01, 5.01, 0.01)           # 0.01–5.0
        blurring_window_sizes = list(range(1, 8, 2))        # 1, 3, 7 (odd)
        filter_window_sizes = list(range(1, 101))           # 1–100

        # Initialize search space
        pos = 0
        init_configs = img_obj.configs.copy()
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
                                        init_configs["threshold_type"]["value"] = tt
                                        init_configs["global_threshold_value"]["value"] = global_thresh
                                        init_configs["adaptive_local_threshold_value"]["value"] = adaptive_thresh
                                        init_configs["brightness_level"]["value"] = brightness
                                        init_configs["contrast_level"]["value"] = contrast
                                        init_configs["apply_gamma"]["dataValue"] = gamma_val
                                        init_configs["apply_autolevel"]["dataValue"] = blur_size
                                        init_configs["apply_gaussian_blur"]["dataValue"] = blur_size
                                        init_configs["apply_lowpass_filter"]["dataValue"]  = filter_size
                                        # init_configs["apply_laplacian_gradient"]["dataValue"] = 3
                                        # init_configs["apply_sobel_gradient"]["dataValue"] = 3
                                        for apply_dark_fg in [0, 1]:
                                            for apply_gamma in [0, 1]:
                                                for apply_auto_lvl in [0, 1]:
                                                    for apply_laplacian in [0, 1]:
                                                        for apply_gaussian in [0, 1]:
                                                            for apply_lowpass in [0, 1]:
                                                                for apply_sobel in [0, 1]:
                                                                    for apply_median in [0, 1]:
                                                                        for apply_scharr in [0, 1]:
                                                                            init_configs["apply_dark_foreground"]["value"] = apply_dark_fg
                                                                            init_configs["apply_gamma"]["value"] = apply_gamma
                                                                            init_configs["apply_autolevel"]["value"] = apply_auto_lvl
                                                                            init_configs["apply_laplacian_gradient"]["value"] = apply_laplacian
                                                                            init_configs["apply_gaussian_blur"]["value"] = apply_gaussian
                                                                            init_configs["apply_lowpass_filter"]["value"] = apply_lowpass
                                                                            init_configs["apply_sobel_gradient"]["value"] = apply_sobel
                                                                            init_configs["apply_median_filter"]["value"] = apply_median
                                                                            init_configs["apply_scharr_gradient"]["value"] = apply_scharr
                                                                            # candidate = FilterSearchSpace.Candidate(
                                                                            #     position=pos,
                                                                            #     std_cost=None,
                                                                            #     img_configs=init_configs.copy()
                                                                            # )
                                                                            # search_space.candidates.append(candidate)
                                                                            print(f"Candidate {pos} added to search space.")
                                                                            pos += 1

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




if __name__ == "__main__":
    s_space = FilterSearchSpace.build_search_space(BaseImage(np.ones((256, 256))))
    print(f"Search space size: {len(s_space.candidates)}")
