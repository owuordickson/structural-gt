# SPDX-License-Identifier: GNU GPL v3
"""
Function to generate graph images after applying a combination of random image filters.
"""

import os, random, uuid
import pandas as pd
from sgtlib.modules import ALLOWED_IMG_EXTENSIONS, ImageProcessor
from matplotlib import pyplot as plt


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


def sgt_genetic_algorithm(max_iters: int = 1000, n_pop: int = 100, pct_crossover: float = 0.5, pct_mutate: float = 0.1, gamma: float = 0.9, sigma: float = 0.9):
    """Generate graph images using a genetic algorithm, cost function based on (1) Number of subgraphs and
     (2) if edges lie on "white" sections of Grayscale image."""

    def _crossover(parent1, parent2):
        """Cross over two parents to generate two children."""
        pass

    def _mutate(parent):
        """Mutate a parent to generate a new child."""
        pass

    # Have many tries to generate many good graphs
    pass


def sgt_hill_climbing_algorithm(max_iters: int = 10, step_size: float = 0.5):
    """Executes the hill climbing algorithm to find the best candidate from a small search space."""
    pass


def metaheuristic_image_configs():
    """A function that runs metaheuristic algorithms (Genetic Algorithm and Hill-climbing Algorithm) to find the best
    image configurations for extracting accurate graphs from SEM images."""
    pass


if __name__ == "__main__":
    print("Starting main...")
    # auto_graph_generator(images_dir="../images", out_dir="../train_data/auto/auto_images", loops=10000)
