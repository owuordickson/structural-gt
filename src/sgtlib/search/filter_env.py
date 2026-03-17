# SPDX-License-Identifier: GNU GPL v3
"""
A class for building an image filter search space for graph generation.
"""

import copy
import random
import numpy as np
from dataclasses import dataclass
from ..imaging.base_image import BaseImage
from ..utils.sgt_utils import AbortException
from ..utils.config_loader import load_img_configs


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
        min_pos: int = 0
        max_pos: int = 0
        candidates: list = None
        ignore_candidates: set = None
        loser_candidates: set = None
        best_candidate = None

    @dataclass
    class FilterCandidate:
        """A filter candidate in the search space. It contains
        a position in the search space (which encodes a binary number), the position determines the value range of the
        value search space. It also has the brightness search space, and the cost of applying the filter is calculated by
        evaluating the binary image and finding the number of white pixels in the image. Retrieve the corresponding pixel
        values from the original image and calculate the Standard Deviation (SD) of the pixel values. Finally, it has
        the combination of image filter configurations."""
        position: int | None = None                  # 11 bits long (approx. 2k candidates)
        std_cost: float | None = None
        value_range: tuple[int, int] | None = None                      # [min, max] values -- 0bits-20bits
        value_space: "FilterSearchSpace.SearchSpace" = None                      # approx. 1B candidates
        brightness_space: "FilterSearchSpace.SearchSpace" = None                 # approx. 256 candidates
        graph_accuracy: float | None = None          # CNN model prediction accuracy of the generated graph
        img_configs: dict | None = None

    def __init__(self):
        pass

    @classmethod
    def _build_full_search_space(cls, img_obj: BaseImage) -> SearchSpace | None:
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
        init_configs = copy.deepcopy(img_obj.configs)
        search_space = FilterSearchSpace.SearchSpace(candidates=[], ignore_candidates=set())

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
                                        init_configs["apply_laplacian_gradient"]["dataValue"] = blur_size
                                        init_configs["apply_sobel_gradient"]["dataValue"] = blur_size
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
                                                                            #     img_configs=init_configs
                                                                            # )
                                                                            # search_space.candidates.append(candidate)
                                                                            print(
                                                                                f"Candidate {pos} added to search space.")
                                                                            pos += 1
        return search_space

    @staticmethod
    def get_initial_candidate(search_space: "FilterSearchSpace.SearchSpace"):
        """
        Get the initial candidate.

        :param search_space: The search space.
        """
        idx = random.randint(0, len(search_space.candidates) - 1)
        if search_space.best_candidate is None:
            init_candidate = search_space.candidates[idx]
        elif search_space.best_candidate.position in search_space.ignore_candidates or search_space.best_candidate.position in search_space.loser_candidates:
            init_candidate = search_space.candidates[idx]
        else:
            init_candidate = search_space.best_candidate
        return init_candidate

    @staticmethod
    def encode_filter_selections(img_configs: dict) -> int:
        """
        Encode the image filter configurations into a position in the search space.

        :param img_configs: The dictionary of image filter configurations.
        :return: The position in the search space.
        """
        if img_configs is None:
            return -1

        threshold_type = img_configs["threshold_type"]["value"]
        apply_dark_fg = img_configs["apply_dark_foreground"]["value"]
        apply_gamma = img_configs["apply_gamma"]["value"]
        apply_autolevel = img_configs["apply_autolevel"]["value"]
        apply_laplacian_gradient = img_configs["apply_laplacian_gradient"]["value"]
        apply_gaussian_blur = img_configs["apply_gaussian_blur"]["value"]
        apply_lowpass_filter = img_configs["apply_lowpass_filter"]["value"]
        apply_sobel_gradient = img_configs["apply_sobel_gradient"]["value"]
        apply_median_filter = img_configs["apply_median_filter"]["value"]
        apply_scharr_gradient = img_configs["apply_scharr_gradient"]["value"]

        bitstring = (format(threshold_type, "02b") +
                     format(apply_dark_fg, "01b") +
                     format(apply_gamma, "01b") +
                     format(apply_autolevel, "01b") +
                     format(apply_laplacian_gradient, "01b") +
                     format(apply_gaussian_blur, "01b") +
                     format(apply_lowpass_filter, "01b") +
                     format(apply_sobel_gradient, "01b") +
                     format(apply_median_filter, "01b") +
                     format(apply_scharr_gradient, "01b"))

        return int(bitstring, 2)

    @staticmethod
    def decode_filter_selections(encoded_pos: int) -> None | dict:
        """
        Decode the position of a candidate in the search space into a dictionary of selected image filter configurations.

        :param encoded_pos: The position of the candidate in the search space.
        :return: The dictionary of image filter configurations.
        """

        try:
            encoded_pos = int(encoded_pos)
        except (TypeError, ValueError):
            raise TypeError("Value must be convertible to an integer")

        if not (0 <= encoded_pos < 2048):
            raise ValueError("Value must be between 0 and 2048")

        if encoded_pos is None:
            return None

        # Step 1: Convert integer to 11-bit binary string
        bitstring = format(encoded_pos, "011b")  # always 11 bits

        # Step 2: Extract threshold type (first 2 bits)
        threshold_bits = bitstring[:2]
        threshold_type = int(threshold_bits, 2)

        # Step 3: Extract 9 filter bits (order consistent with the encoding)
        filter_bits = bitstring[2:]
        filters = [int(b) for b in filter_bits]

        # Step 4: Map to variable names
        img_configs = load_img_configs("")
        img_configs["threshold_type"]["value"] = threshold_type
        img_configs["apply_dark_foreground"]["value"] = filters[0]
        img_configs["apply_gamma"]["value"] = filters[1]
        img_configs["apply_autolevel"]["value"] = filters[2]
        img_configs["apply_laplacian_gradient"]["value"] = filters[3]
        img_configs["apply_gaussian_blur"]["value"] = filters[4]
        img_configs["apply_lowpass_filter"]["value"] = filters[5]
        img_configs["apply_sobel_gradient"]["value"] = filters[6]
        img_configs["apply_median_filter"]["value"] = filters[7]
        img_configs["apply_scharr_gradient"]["value"] = filters[8]

        # print(f"Bits: {bitstring}, Default Candidate: {img_configs}")
        # print(f"Position: {encoded_pos} -> Bits: {bitstring}")
        return img_configs

    @staticmethod
    def decode_filter_values(img_configs: dict, value_candidate: "FilterSearchSpace.Candidate"=None, bright_candidate: "FilterSearchSpace.Candidate"=None) -> dict|None:
        """
        Decode the image filter configurations of a candidate into a dictionary.

        :param img_configs: The dictionary of image filter configurations.
        :param value_candidate: The filter-value candidate for updating image filter value configurations.
        :param bright_candidate: The brightness-value candidate for updating image brightness/contrast configurations.
        :return: The dictionary of image filter configurations.
        """
        if img_configs is None:
            return None

        new_img_configs = copy.deepcopy(img_configs)
        if value_candidate is not None:
            encoded_val_pos = value_candidate.position
            if encoded_val_pos is None:
                return None
            bit_str = format(encoded_val_pos, "030b")
            threshold_val = int(bit_str[:8], 2)
            gamma_val = int(bit_str[8:17], 2)
            autolevel_val = int(bit_str[17:19], 2)
            gaussian_val = int(bit_str[19:21], 2)
            laplacian_val = int(bit_str[21:23], 2)
            lowpass_val = int(bit_str[23:30], 2)
            blur_window_size = [1, 3, 5, 7]

            if new_img_configs["threshold_type"]["value"] == 0:
                new_img_configs["global_threshold_value"]["value"] = threshold_val

            if new_img_configs["threshold_type"]["value"] == 1:
                # Should be an Odd number
                threshold_val = threshold_val+1 if threshold_val%2==0 else threshold_val
                new_img_configs["adaptive_local_threshold_value"]["value"] = threshold_val

            if new_img_configs["apply_gamma"]["value"] == 1:
                new_img_configs["apply_gamma"]["dataValue"] = round(gamma_val / 100.0, 2) if gamma_val > 0 else 0.01

            if new_img_configs["apply_lowpass_filter"]["value"] == 1:
                new_img_configs["apply_lowpass_filter"]["dataValue"] = lowpass_val

            if new_img_configs["apply_autolevel"]["value"] == 1:
                new_img_configs["apply_autolevel"]["dataValue"] = blur_window_size[autolevel_val]

            if new_img_configs["apply_gaussian_blur"]["value"] == 1:
                new_img_configs["apply_gaussian_blur"]["dataValue"] = blur_window_size[gaussian_val]

            if new_img_configs["apply_laplacian_gradient"]["value"] == 1:
                new_img_configs["apply_laplacian_gradient"]["dataValue"] = blur_window_size[laplacian_val]

            if new_img_configs["apply_sobel_gradient"]["value"] == 1:
                # To Be Updated (we need to keep bit_str less than 30 bits)
                new_img_configs["apply_sobel_gradient"]["dataValue"] = blur_window_size[laplacian_val] # re-use laplacian gradient

        if bright_candidate is not None:
            encoded_brightness_pos = bright_candidate.position
            if encoded_brightness_pos is None:
                return None
            bit_str = format(encoded_brightness_pos, "016b")
            is_brightness_neg = bit_str[0]
            brightness_val = int(bit_str[1:8], 2)
            if is_brightness_neg == "1":
                brightness_val = -brightness_val

            is_contrast_neg = bit_str[8]
            contrast_val = int(bit_str[9:16], 2)
            if is_contrast_neg == "1":
                contrast_val = -contrast_val

            new_img_configs["contrast_level"]["value"] = contrast_val
            new_img_configs["brightness_level"]["value"] = brightness_val
        return new_img_configs

    @staticmethod
    def build_search_space(default_img_configs: dict, initial_pop: int = 256) -> SearchSpace | None:
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.
        Encodes a combination of image filter configurations as a binary number, then this number is converted into an
        integer position in the search space.

        :param default_img_configs: The original image configs.
        :param initial_pop: The total population size for Genetic Algorithm search space. Default is 256.
        :return: The search space.
        """

        if default_img_configs is None:
            return None

        # Parameters
        apply_pop = 2**11
        val_range = (2**22, 2**30)  # minimum, maximum value range for search space
        bri_range = (0, 2**16)

        # Initialize search space
        initial_position = FilterSearchSpace.encode_filter_selections(default_img_configs)
        initial_position = initial_position if (0 <= initial_position < 2048) else 128
        search_space = FilterSearchSpace.SearchSpace(candidates=[], ignore_candidates=set(), loser_candidates=set(), min_pos=0, max_pos=apply_pop-1)
        for pos in range(apply_pop):
            img_configs = FilterSearchSpace.decode_filter_selections(pos)
            if img_configs is not None:
                # Create an empty candidate
                val_pop = [FilterSearchSpace.Candidate(position=random.randrange(val_range[0], val_range[1]), std_cost=None) for _ in range(initial_pop)]
                b_pop = [FilterSearchSpace.Candidate(position=random.randrange(bri_range[0], bri_range[1]), std_cost=None) for _ in range(initial_pop)]
                value_space = FilterSearchSpace.SearchSpace(candidates=val_pop, ignore_candidates=set(), loser_candidates=set(), min_pos=val_range[0], max_pos=val_range[1])
                brightness_space = FilterSearchSpace.SearchSpace(candidates=b_pop, ignore_candidates=set(), loser_candidates=set(), min_pos=bri_range[0], max_pos=bri_range[1])

                if pos == initial_position:
                    search_space.best_candidate = FilterSearchSpace.FilterCandidate(
                        position=pos,
                        value_range=(0, initial_pop),
                        value_space=value_space,
                        brightness_space=brightness_space,
                        std_cost=None,
                        graph_accuracy=None,
                        img_configs=img_configs,
                    )
                else:
                    filter_candidate = FilterSearchSpace.FilterCandidate(
                        position=pos,
                        value_range=(0, initial_pop),
                        value_space=value_space,
                        brightness_space=brightness_space,
                        std_cost=None,
                        graph_accuracy=None,
                        img_configs=img_configs,
                    )
                    search_space.candidates.append(filter_candidate)
        return search_space

    @staticmethod
    def cost_function(img_configs: dict, img_2d: np.ndarray) -> float:
        """Calculate and apply the cost of a candidate. Given the image filter configurations, apply them to get a
        binary image and compute the histogram of the binary image. Then, compute the cost of the candidate based on the
        histogram statistics.

        :param img_configs: The dictionary of image filter configurations.
        :param img_2d: The OpenCV or NumPy array representing the image. Can be either grayscale or color.

        :return: The cost of the candidate as a float. If the candidate is invalid, return np.inf.
        """

        def compute_num_filters():
            """Compute the number of filters in the image filter configurations."""
            # 1. Get the selected
            encoded_pos = FilterSearchSpace.encode_filter_selections(img_configs)
            binary_str = format(encoded_pos, "011b")
            applied_filters = [int(b) for b in binary_str]
            total_filter_cost = sum(applied_filters)

            # Add cost to the selected threshold type
            total_filter_cost += img_configs["threshold_type"]["value"]
            if img_configs["threshold_type"]["value"] == 0:
                total_filter_cost += abs(img_configs["global_threshold_value"]["value"] - 128)

            # 3. Add cost to brightness and contrast
            if img_configs["brightness_level"]["value"] != 0:
                total_filter_cost += 1

            if img_configs["contrast_level"]["value"] != 0:
                total_filter_cost += 1
            return total_filter_cost

        if img_2d is None:
            return np.inf

        if img_configs is None:
            return np.inf

        # Create an image object from the image data
        img_obj = BaseImage(raw_img=img_2d)
        # Copy image filter configurations to the image object
        img_obj.configs = img_configs
        # Apply image filters
        img_data = img_obj.img_2d.copy()
        img_obj.img_grayscale = img_obj.process_img(image=img_data)
        img_obj.img_bin = img_obj.binarize_img(image=img_obj.img_grayscale.copy())

        # Compute cost
        try:
            start_bin = 110
            end_bin = 190
            w1 = 10.0
            w2 = 0.2
            w3 = 0.6

            cover_ratio = img_obj.evaluate_histogram_window(start_bin, end_bin)
            # actual_hist = img_obj.evaluate_img_binary()
            pix_vals = img_obj.img_bin.flatten()
            mean_val = np.mean(pix_vals)
            p5 = np.percentile(pix_vals, 10).astype(int)
            p95 = np.percentile(pix_vals, 90).astype(int)

            target_center = (start_bin + end_bin) // 2
            coverage = (1/(cover_ratio+1e-12))  # Add a small number to avoid division by zero
            num_filters = compute_num_filters()
            spread_err = abs(p95 - p5)
            mean_err = abs(mean_val - target_center)

            eval_cost = coverage + (w1*num_filters) + (w2*mean_err) + (w3*spread_err)
            print(f"Filter-Fxn (cost) -> Cover: {cover_ratio}, Cost: {eval_cost} ")
        except Exception as e:
            print(f"Filter-Fxn (cost) -> Error in cost function: {e}")
            eval_cost = np.inf
        # eval_cost = np.inf if eval_cost is None else eval_cost
        return eval_cost



def sgt_genetic_algorithm(s_space: FilterSearchSpace.SearchSpace, img_data: np.ndarray, img_configs: dict, generations: int = 4, pop_size: int = 8, gamma: float = 1.0, mu: float = 0.9, sigma: float = 0.9) -> dict|None:
    """
    Executes the genetic algorithm to find the best candidate from a huge search space.

    :param s_space: Search space object.
    :param img_data: Image as a NumPy array or OpenCV image.
    :param img_configs: Dictionary of selected image filter configurations.
    :param generations: Number of family generations to run the algorithm for.
    :param pop_size: Initial size of the population.
    :param gamma: Crossover probability.
    :param mu: Mutation probability.
    :param sigma: Standard deviation of the Gaussian mutation.

    :return: A dictionary containing the best candidate's image configuration settings.
    """

    def _select_parents():
        """Select parents for crossover."""

        # Select a random parent population (1/3 of the population)
        q = np.random.permutation(pop_size)
        parent_pop = []
        for i in range(pop_size//3):
            parent_pop.append(s_space.candidates[q[i]])
        return parent_pop

    def _crossover(parent_1, parent_2):
        """Cross over two parents to generate two children."""
        if isinstance(parent_1, FilterSearchSpace.Candidate):
            alpha = random.uniform(0, gamma)
            child_1 = FilterSearchSpace.Candidate()
            child_2 = FilterSearchSpace.Candidate()
            # Apply crossover and ensure positions are within bounds
            child_1.position = int(max(s_space.min_pos, min(parent_1.position * alpha + parent_2.position * (1 - alpha), s_space.max_pos)))
            child_2.position = int(max(s_space.min_pos, min(parent_2.position * alpha + parent_1.position * (1 - alpha), s_space.max_pos)))
            return child_1, child_2
        else:
            return parent_1, parent_2

    def _mutate(x):
        """Mutate an individual x to generate a new individual y."""
        if isinstance(x, FilterSearchSpace.Candidate):
            y = FilterSearchSpace.Candidate()
            # Apply Gaussian mutation with mean mu and standard deviation sigma
            mutation_value = np.random.normal(mu, sigma)
            # Mutate the position and ensure it stays within bounds
            y.position = int(np.clip(x.position + mutation_value, s_space.min_pos, s_space.max_pos))
            return y
        else:
            return x

    def _compute_fitness(sol):
        """Compute fitness for an individual."""
        if isinstance(sol, FilterSearchSpace.Candidate):
            if s_space.max_pos >= 2 ** 30:
                new_img_configs = FilterSearchSpace.decode_filter_values(img_configs, value_candidate=sol)
            else:
                new_img_configs = FilterSearchSpace.decode_filter_values(img_configs, bright_candidate=sol)
            std_cost = FilterSearchSpace.cost_function(new_img_configs, img_data)
        else:
            std_cost, new_img_configs = np.inf, None
        return std_cost, new_img_configs

    if s_space is None:
        raise AbortException("Search space cannot be None")

    # Initialize search space
    s_space.loser_candidates = set()

    # Initialize best candidate
    best_sol = FilterSearchSpace.get_initial_candidate(s_space)
    best_sol.std_cost, best_configs = _compute_fitness(best_sol)
    print(f"GA-Alg (initial) -> position: {best_sol.position}, cost: {best_sol.std_cost}")
    for _ in range(generations):
        best_individual = None
        temp_configs = None

        # 1. Compute fitness for each individual in the population/search space
        for individual in s_space.candidates:
            if isinstance(individual, FilterSearchSpace.Candidate):
                individual.std_cost, new_configs = _compute_fitness(individual)

                if individual.std_cost == np.inf:
                    s_space.loser_candidates.add(individual.position)
                    continue

                if best_individual is None or best_individual.std_cost is None or individual.std_cost < best_individual.std_cost:
                    print(f"GA-Alg (winner) -> position: {individual.position}, cost: {individual.std_cost}")
                    best_individual = individual
                    temp_configs = new_configs

        # 1.1. Update the current best candidate
        if best_individual is None:
            break

        # 1.2. Check if fitness is valid
        if best_individual.std_cost is None:
            break

        # 1.3. Update the current best candidate
        if best_individual.std_cost < best_sol.std_cost:
            best_sol = best_individual
            best_configs = temp_configs

        # 2. Select parents
        parents = _select_parents()

        # 3. Create offspring through crossover and mutation
        new_population = []
        for _ in range(pop_size // 2):
            p_1, p_2 = np.random.choice(parents, size=2, replace=False)

            # 3.1. Crossover parents to generate two children
            c_1, c_2 = _crossover(p_1, p_2)

            # 3.1. Mutate children to generate new candidates
            x_1 = _mutate(c_1)
            x_2 = _mutate(c_2)

            # 3.3. Add children to the new population
            new_population.append(x_1) if (x_1.position not in s_space.ignore_candidates) or (x_1.position not in s_space.loser_candidates) else None
            new_population.append(x_2) if (x_2.position not in s_space.ignore_candidates) or (x_2.position not in s_space.loser_candidates)  else None
        # 4. Apply replacement/elitism if desired
        s_space.candidates = new_population
    s_space.best_candidate = best_sol
    print(f"GA-Alg (best) -> position:: {best_sol.position}, cost: {best_sol.std_cost}\n")
    return best_configs



def sgt_hill_climbing_algorithm(s_space: FilterSearchSpace.SearchSpace, img_data: np.ndarray, max_iters: int = 5, step_size: int = 1) -> None:
    """
    Executes the hill climbing algorithm to find the best candidate from a small search space.

    :param s_space: Search space object.
    :param img_data: Image as a NumPy array or OpenCV image.
    :param max_iters: Maximum number of iterations to run the algorithm for.
    :param step_size: Step size to move the current candidate.

    :return: None
    """

    def _generate_neighbors(iter_val: int):
        """Generate neighbors by slightly modifying the current candidate."""
        lst_neighbor = []
        for i in range(20):
            conf_val = float(random.randint(0, 100)/100)
            if conf_val >= 0.95:
                # with random probability, move the center position to a random position in the search space
                center_pos = random.randint(s_space.min_pos, s_space.max_pos)
            else:
                # otherwise, use the best candidate as the center position
                center_pos = best_sol.position
            left_pos = max(s_space.min_pos, center_pos - step_size - i - iter_val)
            right_pos = min(s_space.max_pos, center_pos + step_size + i + iter_val)
            if isinstance(best_sol, (FilterSearchSpace.Candidate, FilterSearchSpace.FilterCandidate)):
                for item in s_space.candidates:
                    if (item.position in (left_pos, center_pos, right_pos)) and ((item.position not in s_space.ignore_candidates) or (item.position not in s_space.loser_candidates)):
                        lst_neighbor.append(item)
        return lst_neighbor

    def _compute_fitness(sol):
        """Compute fitness for an individual."""
        if isinstance(sol, FilterSearchSpace.FilterCandidate):
            # val_sol = sol.value_space.best_candidate
            # bri_sol = sol.brightness_space.best_candidate
            # _ = FilterSearchSpace.decode_filter_values(sol.img_configs, val_sol, bri_sol)
            std_cost = FilterSearchSpace.cost_function(sol.img_configs, img_data)
        else:
            std_cost = np.inf
        return std_cost

    if s_space is None or img_data is None:
        raise AbortException("Search space or ImageObject cannot be None")

    # Initialize search space
    s_space.loser_candidates = set()

    # 1a. Initialize the current best candidate
    best_sol = FilterSearchSpace.get_initial_candidate(s_space)
    # 1b. Reset image configs to default values
    best_sol.img_configs = FilterSearchSpace.decode_filter_selections(best_sol.position)
    best_sol.std_cost = _compute_fitness(best_sol)
    print(f"HC-Alg (initial) -> position: {best_sol.position}, cost: {best_sol.std_cost}")
    # 2. Run the hill climbing algorithm
    for it in range(max_iters):
        # Get neighbors to the current best candidate
        neighbors = _generate_neighbors(it)
        best_neighbor = None

        # Find the best neighbor among the neighbors
        for neighbor in neighbors:
            neighbor.std_cost = _compute_fitness(neighbor)

            if neighbor.std_cost == np.inf:
                s_space.loser_candidates.add(neighbor.position)
                continue

            if best_neighbor is None or best_neighbor.std_cost is None or neighbor.std_cost < best_neighbor.std_cost:
                print(f"HC-Alg (winner) -> position: {neighbor.position}, cost: {neighbor.std_cost}")
                best_neighbor = neighbor

        # Update the current best candidate
        if best_neighbor is None:
            break

        if best_neighbor.std_cost is None:
            break

        if best_neighbor.std_cost < best_sol.std_cost:
            print(f"HC-Alg (new best) -> position: {best_neighbor.position}, cost: {best_neighbor.std_cost}")
            best_sol = best_neighbor
        else:
            # No improvement found, reached a local optimum
            break

    # 3. Update the current best candidate
    print(f"HC-Alg (best) -> position: {best_sol.position}, cost: {best_sol.std_cost}\n")
    s_space.best_candidate = best_sol
    return None
