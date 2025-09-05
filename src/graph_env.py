# SPDX-License-Identifier: GNU GPL v3
"""
A class for building an MDP environment of image filters for graph generation.
"""

import math
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
    def build_search_space():
        """Create a search space where each candidate is a combination of image filter configurations."""
        pass

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


