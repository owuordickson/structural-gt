# SPDX-License-Identifier: GNU GPL v3
"""
Function to generate graph images after applying a combination of random image filters.
"""

import os, random, uuid
import numpy as np
import pandas as pd
from sgtlib.modules import ALLOWED_IMG_EXTENSIONS, ImageProcessor, BaseImage
from matplotlib import pyplot as plt
from graph_env import FilterSearchSpace


def print_updates(progress_val, progress_msg):
    """Function that prints out progress updates."""
    if progress_val < 0:
        print(f"{progress_val}: {progress_msg}")


def auto_graph_generator(images_dir: str, out_dir: str, loops: int = 1000, num_tries: int = 5) -> None:
    """
    Function to generate graph images after applying a combination of random image filters. Steps:
        A. Identifies an image folder
        B. Run 1, 2, 3 in a loop (if we want 10k, then the loop should repeat itself 10k times):
            1. Randomly come with image filters
            2. Apply them to all images
            3. Save the graph images (with unique names) and the randomly selected filters
            (used to generate that graph - in CVS file with the file name of the graph image)
    Args:
        images_dir: the folder containing the images to process
        out_dir: the folder where to save the graph images
        loops: the number of times to run the loop
        num_tries: the number of times to re-try generating random image filters per loop cycle

    Returns:

    """

    if images_dir=="" or out_dir=="":
        print("Image or Output folder cannot be empty")
        return

    # 1. Fetch image paths
    files = os.listdir(images_dir)
    files = sorted(files)
    img_paths = []
    for a_file in files:
        allowed_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_IMG_EXTENSIONS)
        if a_file.endswith(allowed_extensions):
            img_path = os.path.join(str(images_dir), a_file)
            img_paths.append(img_path)

    if not img_paths:
        print(f"[automated_graph_generator] No images found in {images_dir}")
        return

    # 2. Make output directory and empty CSV file to store the generated filters
    os.makedirs(out_dir, exist_ok=True)
    filter_file_path = os.path.join(str(out_dir), "auto_filter.csv")

    # 3. Create column names
    filter_columns = [
        "file_name","Adaptive Kernel","Global Threshold","OTSU","Dark FG","Autolevel",
        "Gaussian Kernel","Laplacian","Sobel","Median","Scharr","Lowpass Window","Gamma","result"
    ]

    # 4. Generate rows of random filters
    for run_idx in range(loops):
        lst_filters = []

        for img_path in img_paths:
            attempt = 0
            success = False
            img_file = None
            cfgs = {}
            _, img_file_name = os.path.split(img_path)

            while attempt < num_tries and not success:
                # build SGT object and apply config
                ntwk_obj, _ = ImageProcessor.from_image_file(str(img_path))
                cfgs = ntwk_obj.image_obj.configs

                # Generate a new random config each retry
                cfgs["threshold_type"]["value"] = random.choice([0, 1, 2])
                cfgs["global_threshold_value"]["value"] = random.randint(1, 255)
                cfgs["adaptive_local_threshold_value"]["value"] = random.randrange(1, 1000, 2)

                cfgs["apply_gamma"]["value"] = random.randint(0, 1)
                cfgs["apply_gamma"]["dataValue"] = round(random.randint(1, 500) / 100.0, 2)

                cfgs["apply_autolevel"]["value"] = random.randint(0, 1)
                cfgs["apply_autolevel"]["dataValue"] = random.choice([1, 3, 5, 7])

                cfgs["apply_gaussian_blur"]["value"] = random.randint(0, 1)
                cfgs["apply_gaussian_blur"]["dataValue"] = random.choice([1, 3, 5, 7])

                cfgs["apply_lowpass_filter"]["value"] = random.randint(0, 1)
                cfgs["apply_lowpass_filter"]["dataValue"] = random.randint(0, 1000)

                cfgs["apply_laplacian_gradient"]["value"] = random.randint(0, 1)
                cfgs["apply_laplacian_gradient"]["dataValue"] = random.choice([1, 3, 5, 7])

                cfgs["apply_sobel_gradient"]["value"] = random.randint(0, 1)
                cfgs["apply_sobel_gradient"]["dataValue"] = random.choice([1, 3, 5, 7])

                cfgs["apply_median_filter"]["value"] = random.randint(0, 1)
                cfgs["apply_scharr_gradient"]["value"] = random.randint(0, 1)
                cfgs["apply_dark_foreground"]["value"] = random.randint(0, 1)

                try:
                    ntwk_obj.add_listener(print_updates)
                    ntwk_obj.apply_img_filters()
                    ntwk_obj.build_graph_network()
                    ntwk_obj.remove_listener(print_updates)

                    if getattr(ntwk_obj, "graph_image", None) is None:
                        attempt += 1
                        continue

                    uid = uuid.uuid4().hex[:8]
                    img_file = f"{img_file_name}__run{run_idx:05d}__{uid}.png"
                    out_file = os.path.join(str(out_dir), img_file)

                    plt.figure()
                    plt.imshow(ntwk_obj.graph_image)
                    plt.axis("off")
                    plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    success = True
                except Exception as e:
                    print(f"[automated_graph_generator] Exception encountered: {e}")
                    attempt += 1

            if not success:
                print(f"Skipping {img_file_name} after {num_tries} failed attempts.")
                continue

            lst_filters.append({
                "file_name": img_file,
                "Adaptive Kernel": int(cfgs["adaptive_local_threshold_value"]["value"]) if cfgs["threshold_type"]["value"] == 1 else "",
                "Global Threshold": int(cfgs["global_threshold_value"]["value"]) if cfgs["threshold_type"]["value"] == 0 else "",
                "OTSU": "TRUE" if cfgs["threshold_type"]["value"] == 2 else "",
                "Dark FG": "TRUE" if cfgs["apply_dark_foreground"]["value"] == 1 else "",
                "Autolevel": int(cfgs["apply_gaussian_blur"]["dataValue"]) if cfgs["apply_autolevel"]["value"] == 1 else "",
                "Gaussian Kernel": int(cfgs["apply_gaussian_blur"]["dataValue"]) if cfgs["apply_gaussian_blur"]["value"] == 1 else "",
                "Laplacian": int(cfgs["apply_laplacian_gradient"]["dataValue"]) if cfgs["apply_laplacian_gradient"]["value"] == 1 else "",
                "Sobel": int(cfgs["apply_sobel_gradient"]["dataValue"]) if cfgs["apply_sobel_gradient"]["value"] == 1 else "",
                "Median": "TRUE" if cfgs["apply_median_filter"]["value"] == 1 else "",
                "Scharr": "TRUE" if cfgs["apply_scharr_gradient"]["value"] == 1 else "",
                "Lowpass Window": int(cfgs["apply_lowpass_filter"]["dataValue"] ) if cfgs["apply_lowpass_filter"]["value"] == 1 else "",
                "Gamma": float(cfgs["apply_gamma"]["dataValue"]) if cfgs["apply_gamma"]["value"] == 1 else "",
                "result": "",
            })

        # append this loop’s rows to auto_filter.csv
        if lst_filters:
            filter_df = pd.DataFrame(lst_filters, columns=filter_columns)

            if os.path.exists(filter_file_path):
                filter_df.to_csv(filter_file_path, mode="a", index=False, header=False)
            else:
                filter_df.to_csv(filter_file_path, index=False, header=True)

    # Completed
    print(f"[automated_graph_generator] Done. Outputs → '{out_dir}', log → '{filter_file_path}'.")
    return


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
        print("Search space cannot be None")
        return None

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
                if best_individual is None or individual.std_cost < best_individual.std_cost:
                    best_individual = individual
                    temp_configs = new_configs.copy()

        # 1.1. Update the current best candidate
        if best_individual is None:
            print("No individual found.")
            break

        # 1.2. Check if fitness is valid
        if best_individual.std_cost is None:
            print("No cost found.")
            break

        # 1.3. Update the current best candidate
        if best_individual is not None and best_individual.std_cost < best_sol.std_cost:
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
        print("Search space or ImageObject cannot be None")
        return None

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

                if best_neighbor is None or neighbor.std_cost < best_neighbor.std_cost:
                    best_neighbor = neighbor
            elif isinstance(neighbor, FilterSearchSpace.Candidate):
                 new_configs = FilterSearchSpace.decode_filter_values(img_obj.configs.copy(), bright_candidate=neighbor)
                 neighbor.std_cost = FilterSearchSpace.cost_function(new_configs, img_obj)

                 if best_neighbor is None or neighbor.std_cost < best_neighbor.std_cost:
                     best_neighbor = neighbor

        # Update the current best candidate
        if best_neighbor is None:
            print("No neighbor found.")
            break

        if best_neighbor.std_cost is None:
            print("No cost found.")
            break

        if best_neighbor is not None and best_neighbor.std_cost < best_sol.std_cost:
            best_sol = best_neighbor
        else:
            # No improvement found, reached a local optimum
            print("Reached a local optimum.")
            break

    # 3. Update the current best candidate
    s_space.best_candidate = best_sol
    return None



def metaheuristic_image_configs(ntwk_obj: ImageProcessor, max_iters: int = 4, ga_init_pop: int = 8) -> dict|None:
    """
    A function that runs metaheuristic algorithms (Genetic Algorithm and Hill-climbing Algorithm) to find the best
    image configurations for extracting accurate graphs from SEM images.

    :param ntwk_obj: ImageProcessor object.
    :param max_iters: Maximum number of iterations to run the Genetic Algorithm and Hill-climbing Algorithm for.
    :param ga_init_pop: Initial size of the population for the Genetic Algorithm.

    :return: A dictionary containing the best candidate's image configuration settings.
    """

    if ntwk_obj is None:
        print("ImageProcessor object cannot be None")
        return None

    def _print_configs(title: str = ""):
        print(f"{title}")
        print(
            f"Configs: {filter_space.best_candidate.img_configs}\n"
            f"Cost: {filter_space.best_candidate.std_cost}\n"
            f"Graph Accuracy: {filter_space.best_candidate.graph_accuracy}\n"
            f"")

    options_model = {
        "find_filter_selections": {"id": "find_filter_selections", "type": "model-settings",
                                   "text": "Find Best Image Filter Combination", "value": 1},
        "find_filter_values": {"id": "find_filter_values", "type": "model-settings",
                               "text": "Estimate Image Filter Values", "value": 1},
        "find_brightness_contrast": {"id": "find_brightness_contrast", "type": "model-settings",
                                     "text": "Estimate Brightness and Contrast Values", "value": 1},
    }

    # 1. Create a search space
    filter_space = FilterSearchSpace.build_search_space(ntwk_obj.image_obj, initial_pop=ga_init_pop)
    _print_configs("Default Configs")

    if options_model["find_filter_selections"]["value"] == 1:
        # 2. Run the Hill-climbing algorithm to find the best "image config combination"
        # filter_space.ignore_candidates.append(filter_space.best_candidate.position)
        sgt_hill_climbing_algorithm(filter_space, ntwk_obj.image_obj, max_iters=max_iters)
        _print_configs("Selected Configs")

    if options_model["find_filter_values"]["value"] == 1:
        # 3. Run the Genetic Algorithm to find the best "image filter values"
        val_search_space = filter_space.best_candidate.value_space
        val_img_configs = sgt_genetic_algorithm(val_search_space, ntwk_obj.image_obj, generations=max_iters, pop_size=ga_init_pop)
        filter_space.best_candidate.std_cost = val_search_space.best_candidate.std_cost
        filter_space.img_configs = val_img_configs
        _print_configs("Best Image Configs")

    if options_model["find_brightness_contrast"]["value"] == 1:
        # 4. Run the Genetic Algorithm to find the best "brightness/contrast values" (only if 'val_search_space' fxn fails)
        bright_search_space = filter_space.best_candidate.brightness_space
        brt_img_configs = sgt_genetic_algorithm(bright_search_space, ntwk_obj.image_obj, generations=max_iters, pop_size=ga_init_pop)
        filter_space.best_candidate.std_cost = bright_search_space.best_candidate.std_cost
        filter_space.img_configs = brt_img_configs
        _print_configs("Best Brightness/Contrast Configs")
    return filter_space.best_candidate.img_configs


if __name__ == "__main__":
    print("Starting main...")
    # 1. Automatically generate graph images
    # auto_graph_generator(images_dir="../images", out_dir="../train_data/auto/auto_images", loops=10000)

    # 2. Run metaheuristic algorithms
    image_path = "../images/4_002.tif"
    res_dir = "../train_data/sgt_files"
    ntwk_p, _ = ImageProcessor.from_image_file(image_path, out_folder=res_dir)
    opt_img_configs = metaheuristic_image_configs(ntwk_p)
    ntwk_p.image_obj.configs = opt_img_configs
