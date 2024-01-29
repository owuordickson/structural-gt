# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs
"""

import os
import time
import multiprocessing as mp
from .graph_struct import GraphStruct
from .graph_metrics import GraphMetrics
from ._config_loader import load
from ._gui_main import ChaosGUI


def main_gui():
    app = ChaosGUI()
    app.mainloop()


def main_cli():
    config, options, options_img, options_gte, options_gtc = load()
    alg = options.algChoice
    num_cores = options.numCores
    is_multi = options.multiImage
    img_path = options.filePath
    out_dir = options.outputDir

    try:
        # 1. Get correct number of CPU cores
        if num_cores > 1:
            pass
        else:
            num_cores = __get_num_cores__()

        # 2. Verify image file or image-dir
        if is_multi == 1:
            # Process multiple images in one folder
            # getting the file names and directory
            files = os.listdir(img_path)
            files = sorted(files, key=str.lower)
            # out_path = img_path
            filenames = []
            for a_file in files:
                if a_file.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                    print(a_file)
                    filenames.append(a_file)
            if len(filenames) <= 0:
                raise "No workable images found! File have to be a .tif, .png, or .jpg"
        else:
            # Process only a single image file
            # testing if file is a workable image
            if img_path.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                pass
            else:
                raise "File has to be a .tif, .png, or .jpg"

        # 3. Verify output directory

        # 4. Run GT program
        start = time.time()
        if alg == 0:
            graph_obj = GraphStruct(img_path, out_dir, options_img, options_gte)
            graph_obj.fit()
            metrics_obj = GraphMetrics(graph_obj, options_gtc)
            metrics_obj.compute_gt_metrics()
            metrics_obj.generate_pdf_output()
        else:
            raise "Wrong algorithm choice!"
        duration = time.time() - start
        out_line = "Run-time: " + str(duration) + " seconds\n"
        out_line += "Number of cores: " + str(num_cores) + '\n'
        print(out_line)
    except PermissionError as error:
        print(error)


def __get_num_cores__():
    """
    Finds the count of CPU cores in a computer or a SLURM super-computer.
    :return: number of cpu cores (int)
    """
    num_cores = __get_slurm_cores__()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def __get_slurm_cores__():
    """
    Test computer to see if it is a SLURM environment, then gets number of CPU cores.
    :return: count of CPUs (int) or False
    """
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False
