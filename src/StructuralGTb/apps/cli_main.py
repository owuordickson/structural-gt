# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Terminal interface implementations
"""

import time
import os
from ypstruct import struct
# import multiprocessing as mp
from ..configs.config_loader import load_configs, get_num_cores
from ..SGT.image_processor import ImageProcessor
from ..SGT.graph_converter import GraphConverter
from ..SGT.graph_metrics import GraphMetrics
# from ..SGT.graph_metrics_clang import GraphMetricsClang


def terminal_app():
    """
    Initializes and executes StructuralGT functions.
    :return:
    """
    configs_data = load_configs()
    options = configs_data['main_options']
    options_img = configs_data['filter_options']
    options_gte = configs_data['extraction_options']
    options_gtc = configs_data['sgt_options']

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
                    logging.info(a_file, extra={'user': 'SGT Logs'})
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
                logging.info(f'Analyzing Image: {i+1}/{len(filenames)}', extra={'user': 'SGT Logs'})
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
        logging.info(out_line, extra={'user': 'SGT Logs'})
    except PermissionError as error:
        print(error)
        logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})


def produce_metrics(img_path, out_dir, options_img, options_gte, options_gtc):
    """
    Executes StructuralGT functions that compute all the user selected metrics.

    Args:
        img_path (str): input image path.
        out_dir (str): directory path for storing results.
        options_img (struct): image processing parameters and options.
        options_gte (struct): graph extraction parameters and options.
        options_gtc (struct): GT computation parameters and options.
    Returns:
        None:
    """
    imp_obj = ImageProcessor(img_path, out_dir, options_img=options_img)
    graph_obj = GraphConverter(imp_obj, options_gte=options_gte)
    graph_obj.add_listener(print_progress)
    graph_obj.fit()
    if options_gtc.compute_lang == 'C':
        metrics_obj = GraphMetricsClang(graph_obj, options_gtc)
    else:
        metrics_obj = GraphMetrics(graph_obj, options_gtc)
    metrics_obj.add_listener(print_progress)
    metrics_obj.compute_gt_metrics()
    if options_gte.weighted_by_diameter:
        metrics_obj.compute_weighted_gt_metrics()
    metrics_obj.generate_pdf_output()
    graph_obj.remove_listener(print_progress)
    metrics_obj.remove_listener(print_progress)


def print_progress(x, y):
    """
    Simple method to display progress updates.

    Args:
        x (int): progress value.
        y (str): progress message.
    Returns:
         None:
    """
    print(str(x) + "%: " + y)
    logging.info(str(x) + "%: " + y, extra={'user': 'SGT Logs'})
