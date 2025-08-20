# SPDX-License-Identifier: GNU GPL v3
"""
Function to generate graph images after applying a combination of random image filters.
"""

import random, uuid
import pandas as pd
from pathlib import Path
from sgtlib import modules as sgt
from matplotlib import pyplot as plt


def automated_graph_generator( images_dir: str, out_dir: str, loops: int = 1000, num_tries: int = 5 ):
    """
    Function to generate graph images after applying a combination of random image filters. Steps:
        A. identify an image folder
        B. Run 1, 2, 3 in a loop (if we want 10k, then the loop should repeat itself 10k times)
            1. randomly come with image filters
            2. apply them to all images
            3. save the graph images (with unique names) and the randomly selected filters
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
    img_paths = [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in sgt.ALLOWED_IMG_EXTENSIONS]
    if not img_paths:
        print(f"[automated_graph_generator] No images found in {images_dir}")
        return

    # 2. Make output directory and empty CSV file to store the generated filters
    filter_file = "auto_filter.csv"
    out_subdir = Path(out_dir)
    out_subdir.mkdir(parents=True, exist_ok=True)
    Path(Path(filter_file).parent).mkdir(parents=True, exist_ok=True)

    filter_columns = [
        "file_name","Adaptive Kernel","Global Threshold","OTSU","Dark FG","Autolevel",
        "Gaussian Kernel","Laplacian","Sobel","Median","Scharr","Lowpass Window","Gamma","result"
    ]

    for run_idx in range(loops):
        af_rows = []

        for img_path in img_paths:
            attempt = 0
            success = False
            img_file = None
            cfg = None  # store the config that succeeded

            while attempt < num_tries and not success:
                # Generate new random config each retry
                cfg_try = {
                    "threshold_type": random.choice([0, 1, 2]),
                    "global_threshold_value": random.randint(1, 255),
                    "adaptive_local_threshold_value": random.randrange(1, 1000, 2),

                    "lut_gamma": round(random.randint(1, 500) / 100.0, 2),
                    "gaussian_blur_size": random.choice([1, 3, 5, 7]),
                    "autolevel_blur_size": random.choice([1, 3, 5, 7]),
                    "lowpass_window_size": random.randint(0, 1000),
                    "laplacian_kernel_size": random.choice([1, 3, 5, 7]),
                    "sobel_kernel_size": random.choice([1, 3, 5, 7]),

                    "apply_gamma": random.randint(0, 1),
                    "apply_autolevel": random.randint(0, 1),
                    "apply_gaussian_blur": random.randint(0, 1),
                    "apply_lowpass_filter": random.randint(0, 1),
                    "apply_laplacian_gradient": random.randint(0, 1),
                    "apply_sobel_gradient": random.randint(0, 1),
                    "apply_median_filter": random.randint(0, 1),
                    "apply_scharr_gradient": random.randint(0, 1),
                    "apply_dark_foreground": random.randint(0, 1),
                }

                # build SGT object and apply config
                ntwk_obj, _ = sgt.ImageProcessor.create_imp_object(str(img_path))
                cfgs = ntwk_obj.image_obj.configs

                cfgs["threshold_type"]["value"] = cfg_try["threshold_type"]
                cfgs["global_threshold_value"]["value"] = cfg_try["global_threshold_value"]
                cfgs["adaptive_local_threshold_value"]["value"] = cfg_try["adaptive_local_threshold_value"]
                cfgs["otsu"]["value"] = 1 if cfg_try["threshold_type"] == 2 else 0

                cfgs["apply_gamma"]["value"] = cfg_try["apply_gamma"]
                cfgs["apply_gamma"]["dataValue"] = cfg_try["lut_gamma"]

                cfgs["apply_autolevel"]["value"] = cfg_try["apply_autolevel"]
                cfgs["apply_autolevel"]["dataValue"] = cfg_try["autolevel_blur_size"]

                cfgs["apply_gaussian_blur"]["value"] = cfg_try["apply_gaussian_blur"]
                cfgs["apply_gaussian_blur"]["dataValue"] = cfg_try["gaussian_blur_size"]

                cfgs["apply_lowpass_filter"]["value"] = cfg_try["apply_lowpass_filter"]
                cfgs["apply_lowpass_filter"]["dataValue"] = cfg_try["lowpass_window_size"]

                cfgs["apply_laplacian_gradient"]["value"] = cfg_try["apply_laplacian_gradient"]
                cfgs["apply_laplacian_gradient"]["dataValue"] = cfg_try["laplacian_kernel_size"]

                cfgs["apply_sobel_gradient"]["value"] = cfg_try["apply_sobel_gradient"]
                cfgs["apply_sobel_gradient"]["dataValue"] = cfg_try["sobel_kernel_size"]

                cfgs["apply_median_filter"]["value"] = cfg_try["apply_median_filter"]
                cfgs["apply_scharr_gradient"]["value"] = cfg_try["apply_scharr_gradient"]
                cfgs["apply_dark_foreground"]["value"] = cfg_try["apply_dark_foreground"]

                try:
                    ntwk_obj.apply_img_filters()
                    ntwk_obj.build_graph_network()

                    if getattr(ntwk_obj, "graph_image", None) is None:
                        attempt += 1
                        continue

                    uid = uuid.uuid4().hex[:8]
                    img_file = f"{img_path.stem}__run{run_idx:05d}__{uid}.png"
                    out_file = out_subdir / img_file

                    plt.figure()
                    plt.imshow(ntwk_obj.graph_image)
                    plt.axis("off")
                    plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    cfg = cfg_try  # save the successful config
                    success = True
                except Exception:
                    attempt += 1

            if not success:
                print(f"Skipping {img_path.name} after {max_iterations} failed attempts.")
                continue

            af_rows.append({
                "file_name": img_file,
                "Adaptive Kernel": str(cfg["adaptive_local_threshold_value"]) if cfg["threshold_type"] == 1 else "",
                "Global Threshold": str(cfg["global_threshold_value"]) if cfg["threshold_type"] == 0 else "",
                "OTSU": "TRUE" if cfg["threshold_type"] == 2 else "",
                "Dark FG": "TRUE" if cfg["apply_dark_foreground"] == 1 else "",
                "Autolevel": str(cfg["autolevel_blur_size"]) if cfg["apply_autolevel"] == 1 else "",
                "Gaussian Kernel": str(cfg["gaussian_blur_size"]) if cfg["apply_gaussian_blur"] == 1 else "",
                "Laplacian": str(cfg["laplacian_kernel_size"]) if cfg["apply_laplacian_gradient"] == 1 else "",
                "Sobel": str(cfg["sobel_kernel_size"]) if cfg["apply_sobel_gradient"] == 1 else "",
                "Median": "TRUE" if cfg["apply_median_filter"] == 1 else "",
                "Scharr": "TRUE" if cfg["apply_scharr_gradient"] == 1 else "",
                "Lowpass Window": str(cfg["lowpass_window_size"]) if cfg["apply_lowpass_filter"] == 1 else "",
                "Gamma": f"{cfg['lut_gamma']:.2f}" if cfg["apply_gamma"] == 1 else "",
                "result": "",
            })

        # append this loop’s rows to auto_filter.csv
        if af_rows:
            af_df = pd.DataFrame(af_rows, columns=filter_columns)
            if Path(filter_file).exists():
                af_df.to_csv(filter_file, mode="a", index=False, header=False)
            else:
                af_df.to_csv(filter_file, index=False, header=True)

    print(f"[automated_graph_generator] Done. Outputs → '{out_dir}', log → '{filter_file}'.")