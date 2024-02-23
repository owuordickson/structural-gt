# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Terminal interface implementations
"""

import time
import os
import multiprocessing as mp
from ..configs.config_loader import load_configs, get_num_cores
from ..modules.graph_struct import GraphStruct
from ..modules.graph_metrics import GraphMetrics


def terminal_app():
    config, options, options_img, options_gte, options_gtc = load_configs()
    alg = options.algChoice
    num_cores = options.numCores
    is_multi = options.multiImage
    img_path = options.filePath
    out_dir = options.outputDir
    filenames = []

    try:
        # 1. Get correct number of CPU cores
        if num_cores > 1:
            pass
        else:
            num_cores = get_num_cores()

        # 2. Verify image file or image-dir
        if is_multi == 1:
            # Process multiple images in one folder
            # getting the file names and directory
            files = os.listdir(img_path)
            files = sorted(files)
            out_path = img_path
            for a_file in files:
                if a_file.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                    print(a_file)
                    filenames.append(os.path.join(out_path, a_file))
            if len(filenames) <= 0:
                raise Exception("No workable images found! Files have to be either .tif, .png, or .jpg")
        else:
            # Process only a single image file
            # testing if file is a workable image
            if img_path.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                if os.path.isfile(img_path):
                    filenames.append(img_path)
                else:
                    raise Exception("File does not exist! Select valid file path.")
            else:
                raise Exception("File has to be a .tif, .png, or .jpg")

        # 3. Verify output directory
        if not os.path.isdir(out_dir):
            path, _ = os.path.split(filenames[0])
            out_dir = path

        # 4. Run GT program
        start = time.time()
        if alg == 0:
            file_count = len(filenames)
            for i in range(file_count):
                print(f'Analyzing Image: {i+1}/{len(filenames)}')
                im_path = filenames[i]
                produce_metrics(im_path, out_dir, options_img, options_gte, options_gtc)

                # updating the images completed
                print("Results generated for " + im_path)
                # print(f'Images Completed: {i+1}/{len(filenames)}')
                print("----------------\n\n")
        else:
            raise Exception("Wrong algorithm choice!")
        duration = time.time() - start
        out_line = "Run-time: " + str(duration) + " seconds\n"
        out_line += "Number of cores: " + str(num_cores) + '\n'
        print(out_line)
    except PermissionError as error:
        print(error)


def produce_metrics(img_path, out_dir, options_img, options_gte, options_gtc):
    graph_obj = GraphStruct(img_path, out_dir, options_img=options_img, options_gte=options_gte)
    graph_obj.add_listener(print_progress)
    graph_obj.fit()
    metrics_obj = GraphMetrics(graph_obj, options_gtc)
    metrics_obj.add_listener(print_progress)
    metrics_obj.compute_gt_metrics()
    if options_gte.weighted_by_diameter:
        metrics_obj.compute_weighted_gt_metrics()
    metrics_obj.generate_pdf_output()
    graph_obj.remove_listener(print_progress)
    metrics_obj.remove_listener(print_progress)


def print_progress(x, y):
    print(str(x) + ": " + y)
