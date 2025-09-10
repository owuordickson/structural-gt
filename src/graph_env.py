# SPDX-License-Identifier: GNU GPL v3
"""
A class for building an MDP environment of image filters for graph generation.
"""

import math
import numpy as np
from dataclasses import dataclass
from sgtlib.modules import BaseImage


class SGTGraphEnv:
    """
    Class for building a (Markov Decision Process) MDP environment for manipulating the Genetic Algorithm (GA) optimizer
    into navigating the search space of image filters.
    """

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
    next or future states S2, S3, ... does not depend on the previous state S1 (No Markov Property).

    No Markov Property: picking the next candidate (a combination of image filter configurations) does not depend on the
    previous candidate. No future candidate/state/action is prohibited (or determined) by the
    current candidate/state/action.

    Markov Property: current candidate determines the next future candidates.

    For this reason, we use Genetic Algorithm (GA) to find the best combination of image filter configurations. GA is
    a global optimizer (or global optimization method/algorithm) that finds the best solution in a given search space.
    """

    @dataclass
    class Candidate:
        """A candidate in the search space. It contains a position in the search space and, the Standard Deviation (SD)
        of the pixel values."""
        position: int | None = None
        std_cost: float | None = None

    @dataclass
    class SearchSpace:
        """Discrete search space of image filters; where, each candidate is a combination of image filter
        configurations. We use this template to build 3 search spaces: apply filters, value filters, and brightness
        filters."""
        candidates: list["FilterSearchSpace.Candidate"] = None
        ignore_candidates: list["FilterSearchSpace.Candidate"] = None
        best_candidate: "FilterSearchSpace.Candidate" = None

    @dataclass
    class FilterCandidate:
        """A filter candidate in the search space. It contains
        a position in the search space (which encodes a binary number), the position determines the value range of the
        value search space. It also has the brightness search space, and the cost of applying the filter is calculated by
        evaluating the binary image and finding the number of white pixels in the image. Retrieve the corresponding pixel
        values from the original image and calculate the Standard Deviation (SD) of the pixel values. Finally, it has
        the combination of image filter configurations."""
        apply_position: int | None = None       # 11 bits long (approx. 2k candidates)
        value_range: list[int] | None = None    # [min, max] values -- 0bits-20bits
        # value_candidate: "FilterSearchSpace.Candidate" = None
        # brightness_candidate: "FilterSearchSpace.Candidate" = None
        value_space: "FilterSearchSpace.SearchSpace" = None         # approx. 268M candidates
        brightness_space: "FilterSearchSpace.SearchSpace" = None    # approx. 256 candidates
        std_cost: float | None = None
        img_configs: dict | None = None

    def __init__(self):
        pass

    @staticmethod
    def _build_full_search_space(img_obj: BaseImage) -> SearchSpace | None:
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.
        The actual search space has over 118k Trillion candidates. This method is used for debugging purposes -- the
        search space is too large to be used in production.

        :param img_obj: The image object.
        :return: The search space.
        """
        if img_obj is None:
            return None

        # Set ranges for each parameter (Discrete action space)
        threshold_types = [0, 1, 2]  # global / adaptive / OTSU
        global_thresh_range = list(range(1, 256))  # 1–255
        adaptive_local_range = list(range(1, 100, 2))  # 1–99 (odd)
        brightness_levels = list(range(-100, 101))  # -100–100
        contrast_levels = list(range(-100, 101))  # -100–100
        gamma_range = np.arange(0.01, 5.01, 0.01)  # 0.01–5.0
        blurring_window_sizes = list(range(1, 8, 2))  # 1, 3, 7 (odd)
        filter_window_sizes = list(range(1, 101))  # 1–100

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
                                        init_configs["apply_lowpass_filter"]["dataValue"] = filter_size
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
                                                                            init_configs["apply_dark_foreground"][
                                                                                "value"] = apply_dark_fg
                                                                            init_configs["apply_gamma"][
                                                                                "value"] = apply_gamma
                                                                            init_configs["apply_autolevel"][
                                                                                "value"] = apply_auto_lvl
                                                                            init_configs["apply_laplacian_gradient"][
                                                                                "value"] = apply_laplacian
                                                                            init_configs["apply_gaussian_blur"][
                                                                                "value"] = apply_gaussian
                                                                            init_configs["apply_lowpass_filter"][
                                                                                "value"] = apply_lowpass
                                                                            init_configs["apply_sobel_gradient"][
                                                                                "value"] = apply_sobel
                                                                            init_configs["apply_median_filter"][
                                                                                "value"] = apply_median
                                                                            init_configs["apply_scharr_gradient"][
                                                                                "value"] = apply_scharr
                                                                            # candidate = FilterSearchSpace.Candidate(
                                                                            #     position=pos,
                                                                            #     std_cost=None,
                                                                            #     img_configs=init_configs.copy()
                                                                            # )
                                                                            # search_space.candidates.append(candidate)
                                                                            print(
                                                                                f"Candidate {pos} added to search space.")
                                                                            pos += 1
        return search_space

    @staticmethod
    def decode_candidate_position(pos_data: dict, img_configs: dict) -> dict:
        """
        Decode the position of a candidate in the search space into a dictionary of image filter configurations.

        :param pos_data: The dictionary of position information.
        :param img_configs: The dictionary of image filter configurations.
        :return: The dictionary of image filter configurations.
        """

        return img_configs

    @staticmethod
    def build_search_space(img_obj: BaseImage, total_pop: int = 1000) -> SearchSpace | None:
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.
        Encodes a combination of image filter configurations as a binary number, then this number is converted into an
        integer position in the search space.

        :param img_obj: The image object.
        :param total_pop: The total population size.
        :return: The search space.
        """

        def encode_filter_combination(
                threshold_type=1,  # 0, 1, or 2 → needs 2 bits
                apply_dark_foreground=0,
                apply_gamma=1,
                apply_auto_level=0,
                apply_laplacian_gradient=0,
                apply_gaussian_blur=0,
                apply_lowpass_filter=0,
                apply_sobel_gradient=0,
                apply_median_filter=0,
                apply_scharr_gradient=0,
        )-> tuple[str, int]:
            """
            Encode 10 image filter configurations as an 11-bit binary string (2 bits for the threshold type,
            9 bits for filters). The total number of filter combinations is 2^11 = 2048.
            :returns: Both the binary string and integer representation.
            """

            # --- Step 1: Encode threshold_type into 2 bits ---
            if threshold_type not in [0, 1, 2]:
                raise ValueError("threshold_type must be 0, 1, or 2")
            threshold_bits = format(threshold_type, "02b")  # 2-bit binary

            # --- Step 2: Encode 9 filters into 1 bit each ---
            filters = [
                apply_dark_foreground,
                apply_gamma,
                apply_auto_level,
                apply_laplacian_gradient,
                apply_gaussian_blur,
                apply_lowpass_filter,
                apply_sobel_gradient,
                apply_median_filter,
                apply_scharr_gradient,
            ]

            filter_bits = "".join(str(int(f)) for f in filters)

            # --- Step 3: Concatenate ---
            bitstring = threshold_bits + filter_bits

            # --- Step 4: Convert to integer ---
            bit_int = int(bitstring, 2)
            return bitstring, bit_int

        # def encode_filter_

        if img_obj is None:
            return None

        # Empty candidate template
        init_configs = img_obj.configs.copy()
        pos_data = {"apply": encode_filter_combination()[1]}
        empty_candidate = FilterSearchSpace.Candidate(
            position_data=pos_data,
            std_cost=None,
            img_configs=init_configs
        )

        # Initialize search space
        candidate_pop = [empty_candidate] * total_pop

        search_space = FilterSearchSpace.SearchSpace(candidates=[], ignore_candidates=[])

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
    def evaluate_candidate(search_space: "FilterSearchSpace.SearchSpace",
                           candidate: "FilterSearchSpace.Candidate") -> None:
        """
        Evaluate a candidate in the search space, check if it is better than the best candidate.

        :param search_space: The search space.
        :param candidate: The candidate to evaluate.
        """
        if candidate.std_cost is None:
            return

        if search_space.best_candidate is None:
            search_space.best_candidate = FilterSearchSpace.Candidate(
                position_data=candidate.position_data,
                std_cost=candidate.std_cost,
                img_configs=candidate.img_configs)

        if candidate.position_data in search_space.ignore_candidates:
            return

        elif candidate.std_cost < search_space.best_candidate.std_cost:
            search_space.best_candidate = FilterSearchSpace.Candidate(
                position_data=candidate.position_data,
                std_cost=candidate.std_cost,
                img_configs=candidate.img_configs
            )


if __name__ == "__main__":
    s_space = FilterSearchSpace.build_search_space(BaseImage(np.ones((256, 256))))
    print(f"Search space size: {len(s_space.candidates)}")
