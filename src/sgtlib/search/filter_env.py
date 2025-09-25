# SPDX-License-Identifier: GNU GPL v3
"""
A class for building an image filter search space for graph generation.
"""

import random
import numpy as np
from dataclasses import dataclass
from ..imaging.base_image import BaseImage
from ..utils.sgt_utils import AbortException


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
        ignore_candidates: list= None
        best_candidate = None

    @dataclass
    class FilterCandidate:
        """A filter candidate in the search space. It contains
        a position in the search space (which encodes a binary number), the position determines the value range of the
        value search space. It also has the brightness search space, and the cost of applying the filter is calculated by
        evaluating the binary image and finding the number of white pixels in the image. Retrieve the corresponding pixel
        values from the original image and calculate the Standard Deviation (SD) of the pixel values. Finally, it has
        the combination of image filter configurations."""
        position: int | None = None               # 11 bits long (approx. 2k candidates)
        value_range: tuple[int, int] | None = None                  # [min, max] values -- 0bits-20bits
        # value_candidate: "FilterSearchSpace.Candidate" = None
        # brightness_candidate: "FilterSearchSpace.Candidate" = None
        value_space: "FilterSearchSpace.SearchSpace" = None         # approx. 1B candidates
        brightness_space: "FilterSearchSpace.SearchSpace" = None    # approx. 256 candidates
        std_cost: float | None = None
        graph_accuracy: float | None = None     # CNN model prediction accuracy of the generated graph
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
                                                                            #     img_configs=init_configs.copy()
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
        elif search_space.best_candidate.position in search_space.ignore_candidates:
            init_candidate = search_space.candidates[idx]
        else:
            init_candidate = search_space.best_candidate
        init_candidate.std_cost = np.inf
        return init_candidate

    @staticmethod
    def decode_candidate_position(encoded_pos: int, img_configs: dict) -> None|dict:
        """
        Decode the position of a candidate in the search space into a dictionary of image filter configurations.

        :param encoded_pos: The position of the candidate in the search space.
        :param img_configs: The dictionary of image filter configurations.
        :return: The dictionary of image filter configurations.
        """
        if encoded_pos is None or img_configs is None:
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

        """
        # Step 5: Build value_bits for extra parameter ranges
        value_bits = ""
        if img_configs["apply_gamma"]["value"] == 1:
            value_bits += format(500, "09b")  # gamma parameter
        if img_configs["apply_autolevel"]["value"] == 1:
            value_bits += format(1, "03b")  # autolevel param
        if img_configs["apply_gaussian_blur"]["value"] == 1:
            value_bits += format(1, "03b")  # blur strength
        if img_configs["apply_lowpass_filter"]["value"] == 1:
            value_bits += format(3, "07b")  # filter kernel

        # Step 6: Convert extra bits to integer (if any)
        max_val = int(value_bits, 2) if value_bits else 0
        """
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

            if img_configs["threshold_type"]["value"] == 0:
                img_configs["global_threshold_value"]["value"] = threshold_val

            if img_configs["threshold_type"]["value"] == 1:
                # Should be an Odd number
                threshold_val = threshold_val+1 if threshold_val%2==0 else threshold_val
                img_configs["adaptive_local_threshold_value"]["value"] = threshold_val

            if img_configs["apply_gamma"]["value"] == 1:
                img_configs["apply_gamma"]["dataValue"] = round(gamma_val / 100.0, 2) if gamma_val > 0 else 0.01

            if img_configs["apply_lowpass_filter"]["value"] == 1:
                img_configs["apply_lowpass_filter"]["dataValue"] = lowpass_val

            if img_configs["apply_autolevel"]["value"] == 1:
                img_configs["apply_autolevel"]["dataValue"] = blur_window_size[autolevel_val]

            if img_configs["apply_gaussian_blur"]["value"] == 1:
                img_configs["apply_gaussian_blur"]["dataValue"] = blur_window_size[gaussian_val]

            if img_configs["apply_laplacian_gradient"]["value"] == 1:
                img_configs["apply_laplacian_gradient"]["dataValue"] = blur_window_size[laplacian_val]

            if img_configs["apply_sobel_gradient"]["value"] == 1:
                # To Be Updated (we need to keep bit_str less than 30 bits)
                img_configs["apply_sobel_gradient"]["dataValue"] = blur_window_size[laplacian_val] # re-use laplacian gradient

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

            img_configs["contrast_level"]["value"] = contrast_val
            img_configs["brightness_level"]["value"] = brightness_val
        return img_configs

    @staticmethod
    def build_search_space(img_obj: BaseImage, initial_pop: int = 256) -> SearchSpace | None:
        """
        Create a discrete search space where each candidate is a combination of image filter configurations.
        Encodes a combination of image filter configurations as a binary number, then this number is converted into an
        integer position in the search space.

        :param img_obj: The image object.
        :param initial_pop: The total population size for Genetic Algorithm search space. Default is 256.
        :return: The search space.
        """

        if img_obj is None:
            return None

        # Parameters
        apply_pop = 2**11
        val_range = (2**22, 2**30)  # minimum, maximum value range for search space
        bri_range = (0, 2**16)

        # Initialize search space
        init_configs = img_obj.configs.copy()
        search_space = FilterSearchSpace.SearchSpace(candidates=[], ignore_candidates=[], min_pos=0, max_pos=apply_pop-1)
        for pos in range(apply_pop):
            img_configs = FilterSearchSpace.decode_candidate_position(pos, init_configs)
            if img_configs is not None:
                # Create an empty candidate
                val_pop = [FilterSearchSpace.Candidate(position=random.randrange(val_range[0], val_range[1]), std_cost=None) for _ in range(initial_pop)]
                b_pop = [FilterSearchSpace.Candidate(position=random.randrange(bri_range[0], bri_range[1]), std_cost=None) for _ in range(initial_pop)]

                filter_candidate = FilterSearchSpace.FilterCandidate(
                    position=pos,
                    value_range=(0, initial_pop),
                    value_space=FilterSearchSpace.SearchSpace(candidates=val_pop, ignore_candidates=[], min_pos=val_range[0], max_pos=val_range[1]),
                    brightness_space=FilterSearchSpace.SearchSpace(candidates=b_pop, ignore_candidates=[], min_pos=bri_range[0], max_pos=bri_range[1]),
                    std_cost=None,
                    graph_accuracy=None,
                    img_configs=img_configs.copy(),
                )
                search_space.candidates.append(filter_candidate)
                if pos == 640:
                    # default candidate
                    search_space.best_candidate = filter_candidate
                    search_space.best_candidate.value_space.best_candidate = FilterSearchSpace.Candidate(position=2**22, std_cost=None)
                    search_space.best_candidate.brightness_space.best_candidate = FilterSearchSpace.Candidate(position=0, std_cost=None)
        return search_space

    @staticmethod
    def cost_function(new_img_configs: dict, img_obj: BaseImage) -> float:
        """Calculate and apply the cost of a candidate. Given the image filter configurations, apply them to get a
        binary image and find the number of white pixels in the image. Retrieve the corresponding pixel values from the
        original image and calculate the Standard Deviation (SD) of the pixel values.

        :param new_img_configs: The dictionary of image filter configurations.
        :param img_obj: The image object.
        """

        if img_obj is None:
            return np.inf

        if new_img_configs is None:
            return np.inf

        # Copy image filter configurations to the image object
        img_obj.configs = new_img_configs.copy()
        # Reset image filters
        img_obj.img_mod, img_obj.img_bin = None, None
        # Apply image filters
        img_data = img_obj.img_2d.copy()
        img_obj.img_mod = img_obj.process_img(image=img_data)
        img_mod = img_obj.img_mod.copy()
        img_obj.img_bin = img_obj.binarize_img(img_mod)
        img_obj.img_mod = img_mod
        # Compute SD as cost
        try:
            eval_std, eval_hist = img_obj.evaluate_img_binary()
        except Exception as e:
            print(f"Error in cost function: {e}")
            eval_std = np.inf
        eval_std = np.inf if eval_std is None else eval_std
        return eval_std

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
                position=candidate.position,
                std_cost=candidate.std_cost
            )

        if candidate.position in search_space.ignore_candidates:
            return

        elif candidate.std_cost < search_space.best_candidate.std_cost:
            search_space.best_candidate = FilterSearchSpace.Candidate(
                position=candidate.position,
                std_cost=candidate.std_cost,
            )


def sgt_genetic_algorithm(s_space: FilterSearchSpace.SearchSpace, img_obj: BaseImage, generations: int = 4, pop_size: int = 8, gamma: float = 1.0, mu: float = 0.9, sigma: float = 0.9) -> dict|None:
    """
    Executes the genetic algorithm to find the best candidate from a huge search space.

    :param s_space: Search space object.
    :param img_obj: BaseImage object which contains the image itself and the image configurations.
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

    if s_space is None:
        raise AbortException("Search space cannot be None")

    best_sol = FilterSearchSpace.get_initial_candidate(s_space)
    best_configs = img_obj.configs.copy()
    for _ in range(generations):
        best_individual = None
        temp_configs = None

        # 1. Compute fitness for each individual in the population/search space
        for individual in s_space.candidates:
            if isinstance(individual, FilterSearchSpace.Candidate):
                if s_space.max_pos >= 2**30:
                    new_configs = FilterSearchSpace.decode_filter_values(img_obj.configs.copy(), value_candidate=individual)
                else:
                    new_configs = FilterSearchSpace.decode_filter_values(img_obj.configs.copy(), bright_candidate=individual)
                individual.std_cost = FilterSearchSpace.cost_function(new_configs, img_obj)
                if best_individual is None or best_individual.std_cost is None or individual.std_cost < best_individual.std_cost:
                    best_individual = individual
                    temp_configs = new_configs.copy()

        # 1.1. Update the current best candidate
        if best_individual is None:
            break

        # 1.2. Check if fitness is valid
        if best_individual.std_cost is None:
            break

        # 1.3. Update the current best candidate
        if best_individual.std_cost < best_sol.std_cost:
            best_sol = best_individual
            best_configs = temp_configs.copy()

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
            new_population.append(x_1) if x_1.position not in s_space.ignore_candidates else None
            new_population.append(x_2) if x_2.position not in s_space.ignore_candidates else None
        # 4. Apply replacement/elitism if desired
        s_space.candidates = new_population
    s_space.best_candidate = best_sol
    return best_configs



def sgt_hill_climbing_algorithm(s_space: FilterSearchSpace.SearchSpace, img_obj: BaseImage, max_iters: int = 5, step_size: int = 1) -> None:
    """
    Executes the hill climbing algorithm to find the best candidate from a small search space.

    :param s_space: Search space object.
    :param img_obj: BaseImage object which contains the image itself and the image configurations.
    :param max_iters: Maximum number of iterations to run the algorithm for.
    :param step_size: Step size to move the current candidate.

    :return: None
    """

    def _generate_neighbors():
        """Generate neighbors by slightly modifying the current candidate."""
        lst_neighbor = []
        for i in range(5):
            center_pos = best_sol.position
            left_pos = max(s_space.min_pos, center_pos - step_size)
            right_pos = min(s_space.max_pos, center_pos + step_size)
            if isinstance(best_sol, (FilterSearchSpace.Candidate, FilterSearchSpace.FilterCandidate)):
                for item in s_space.candidates:
                    if (item.position in (left_pos, center_pos, right_pos)) and (item.position not in s_space.ignore_candidates):
                        lst_neighbor.append(item)
        return lst_neighbor

    if s_space is None or img_obj is None:
        raise AbortException("Search space or ImageObject cannot be None")

    # 1. Initialize the current best candidate
    init_sol = FilterSearchSpace.get_initial_candidate(s_space)
    if isinstance(s_space.best_candidate, FilterSearchSpace.FilterCandidate):
        best_sol = FilterSearchSpace.FilterCandidate(
            position=init_sol.position,
            value_range=init_sol.value_range,
            value_space=init_sol.value_space,
            brightness_space=init_sol.brightness_space,
            std_cost=np.inf,
            graph_accuracy=0,
            img_configs=init_sol.img_configs,
        )
    else:
        best_sol = FilterSearchSpace.Candidate(position=init_sol.position, std_cost=np.inf)

    # 2. Run the hill climbing algorithm
    for _ in range(max_iters):
        # Get neighbors to the current best candidate
        neighbors = _generate_neighbors()
        best_neighbor = None

        # Find the best neighbor among the neighbors
        for neighbor in neighbors:
            if isinstance(neighbor, FilterSearchSpace.FilterCandidate):
                FilterSearchSpace.decode_candidate_position(neighbor.position, neighbor.img_configs)
                val_sol = neighbor.value_space.best_candidate
                bri_sol = neighbor.brightness_space.best_candidate
                FilterSearchSpace.decode_filter_values(neighbor.img_configs, val_sol, bri_sol)
                neighbor.std_cost = FilterSearchSpace.cost_function(neighbor.img_configs, img_obj)

                if best_neighbor is None:
                    best_neighbor = neighbor
                if best_neighbor.std_cost is None:
                    best_neighbor = neighbor
                if neighbor.std_cost < best_neighbor.std_cost:
                    best_neighbor = neighbor
            elif isinstance(neighbor, FilterSearchSpace.Candidate):
                 new_configs = FilterSearchSpace.decode_filter_values(img_obj.configs.copy(), bright_candidate=neighbor)
                 neighbor.std_cost = FilterSearchSpace.cost_function(new_configs, img_obj)

                 if best_neighbor is None or neighbor.std_cost < best_neighbor.std_cost:
                     best_neighbor = neighbor

        # Update the current best candidate
        if best_neighbor is None:
            break

        if best_neighbor.std_cost is None:
            break

        if best_neighbor.std_cost < best_sol.std_cost:
            best_sol = best_neighbor
        else:
            # No improvement found, reached a local optimum
            break

    # 3. Update the current best candidate
    s_space.best_candidate = best_sol
    return None
