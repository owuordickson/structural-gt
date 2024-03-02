# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Loads default configurations from 'configs.ini' file
"""

import os
import sys
import configparser
import multiprocessing as mp
from optparse import OptionParser
from ypstruct import struct


def load_configs():

    options_path = struct()
    options_img = struct()
    options_gte = struct()
    options_gtc = struct()

    # 1. Image Path
    options_path.is_multi_image = 0
    options_path.image_path = ""
    options_path.output_path = ""

    # 2. Image Detection settings
    options_img.threshold_type = 1
    options_img.threshold_global = 3
    options_img.threshold_adaptive = 3
    options_img.gamma = float(1)
    options_img.gaussian_blurring_size = 3
    options_img.autolevel_blurring_size = 3
    options_img.lowpass_window_size = 2
    options_img.laplacian_kernel_size = 3
    options_img.sobel_kernel_size = 3
    options_img.apply_autolevel = 0
    options_img.apply_laplacian = 0
    options_img.apply_scharr = 0
    options_img.apply_sobel = 0
    options_img.apply_median = 0
    options_img.apply_gaussian = 0
    options_img.apply_lowpass = 0
    options_img.apply_dark_foreground = 0
    options_img.brightness_level = 0
    options_img.contrast_level = 0

    # 3. Graph Extraction Settings
    options_gte.merge_nearby_nodes = 0
    options_gte.prune_dangling_edges = 0
    options_gte.remove_disconnected_segments = 0
    options_gte.remove_self_loops = 0
    options_gte.remove_object_size = 500
    options_gte.is_multigraph = 0
    options_gte.weighted_by_diameter = 0
    options_gte.display_node_id = 0
    options_gte.export_edge_list = 0
    options_gte.export_as_gexf = 0
    options_gte.export_adj_mat = 0
    options_gte.save_images = 0

    # 4. Networkx Calculation Settings
    options_gtc.display_heatmaps = 0
    options_gtc.display_degree_histogram = 0
    options_gtc.display_betweenness_histogram = 0
    options_gtc.display_currentflow_histogram = 0
    options_gtc.display_closeness_histogram = 0
    options_gtc.display_eigenvector_histogram = 0
    options_gtc.compute_nodal_connectivity = 0
    options_gtc.compute_graph_density = 0
    options_gtc.compute_graph_conductance = 0
    options_gtc.compute_global_efficiency = 0
    options_gtc.compute_clustering_coef = 0
    options_gtc.compute_assortativity_coef = 0
    options_gtc.compute_network_diameter = 0
    options_gtc.compute_wiener_index = 0

    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--inputImage',
                          dest='filePath',
                          help='path to image file/folder containing images',
                          default="",
                          type='string')
    opt_parser.add_option('-o', '--outputFolder',
                          dest='outputDir',
                          help='directory path for saving GT output',
                          default="",
                          type='string')
    opt_parser.add_option('-a', '--algorithmChoice',
                          dest='algChoice',
                          help='select GT algorithm',
                          default=0,
                          type='int')
    opt_parser.add_option('-m', '--isMultiImage',
                          dest='multiImage',
                          help='is it a multi-image (multiple images in a folder) analysis?',
                          default=0,
                          type='int')
    opt_parser.add_option('-c', '--cores',
                          dest='numCores',
                          help='number of cores',
                          default=1,
                          type='int')
    (options, args) = opt_parser.parse_args()

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)
        cpus = int(config.get('computation', 'cpu_cores'))
    except configparser.NoSectionError:
        return config, options, options_img, options_gte, options_gtc

    # 1. Image Path
    options_path.is_multi_image = int(config.get('image-dir', 'is_multi_image'))
    options_path.image_path = config.get('image-dir', 'image_path')
    options_path.output_path = config.get('image-dir', 'gt_output_path')

    # 2. Image Detection settings
    options_img.threshold_type = int(config.get('detection-settings', 'threshold'))
    options_img.threshold_global = int(config.get('detection-settings', 'global_threshold_value'))
    options_img.threshold_adaptive = int(config.get('detection-settings', 'adaptive_local_threshold_value'))
    options_img.gamma = float(config.get('detection-settings', 'adjust_gamma'))
    options_img.gaussian_blurring_size = int(config.get('detection-settings', 'blurring_window_size'))
    options_img.autolevel_blurring_size = int(config.get('detection-settings', 'blurring_window_size'))
    options_img.lowpass_window_size = int(config.get('detection-settings', 'filter_window_size'))
    options_img.laplacian_kernel_size = 3
    options_img.sobel_kernel_size = 3
    options_img.apply_autolevel = int(config.get('detection-settings', 'use_autolevel'))
    options_img.apply_laplacian = int(config.get('detection-settings', 'use_laplacian_gradient'))
    options_img.apply_scharr = int(config.get('detection-settings', 'use_scharr_gradient'))
    options_img.apply_sobel = int(config.get('detection-settings', 'use_sobel_gradient'))
    options_img.apply_median = int(config.get('detection-settings', 'apply_median_filter'))
    options_img.apply_gaussian = int(config.get('detection-settings', 'apply_gaussian_blur'))
    options_img.apply_lowpass = int(config.get('detection-settings', 'apply_lowpass_filter'))
    options_img.apply_dark_foreground = int(config.get('detection-settings', 'dark_foreground'))
    options_img.brightness_level = int(config.get('detection-settings', 'brightness_level'))
    options_img.contrast_level = int(config.get('detection-settings', 'contrast_level'))

    # 3. Graph Extraction Settings
    options_gte.merge_nearby_nodes = int(config.get('extraction-settings', 'merge_nearby_nodes'))
    options_gte.prune_dangling_edges = int(config.get('extraction-settings', 'prune_dangling_edges'))
    options_gte.remove_disconnected_segments = int(config.get('extraction-settings', 'remove_disconnected_segments'))
    options_gte.remove_self_loops = int(config.get('extraction-settings', 'remove_self_loops'))
    options_gte.remove_object_size = int(config.get('extraction-settings', 'remove_object_size'))
    options_gte.is_multigraph = int(config.get('extraction-settings', 'is_multigraph'))
    options_gte.weighted_by_diameter = int(config.get('extraction-settings', 'weighted_by_diameter'))
    options_gte.display_node_id = int(config.get('extraction-settings', 'display_node_id'))
    options_gte.export_edge_list = int(config.get('extraction-settings', 'export_edge_list'))
    options_gte.export_as_gexf = int(config.get('extraction-settings', 'export_as_gexf'))
    options_gte.export_adj_mat = int(config.get('extraction-settings', 'export_adj_mat'))
    options_gte.save_images = int(config.get('extraction-settings', 'save_images'))

    # 4. Networkx Calculation Settings
    options_gtc.display_heatmaps = int(config.get('computation-settings', 'display_heatmaps'))
    options_gtc.display_degree_histogram = int(config.get('computation-settings', 'display_degree_histogram'))
    options_gtc.display_betweenness_histogram = int(config.get('computation-settings',
                                                               'display_betweenness_centrality_histogram'))
    options_gtc.display_currentflow_histogram = int(config.get('computation-settings',
                                                               'display_current_flow_betweenness_centrality_histogram'))
    options_gtc.display_closeness_histogram = int(config.get('computation-settings',
                                                             'display_closeness_centrality_histogram'))
    options_gtc.display_eigenvector_histogram = int(config.get('computation-settings',
                                                               'display_eigenvector_centrality_histogram'))
    options_gtc.compute_nodal_connectivity = int(config.get('computation-settings',
                                                            'compute_avg_nodal_connectivity'))
    options_gtc.compute_graph_density = int(config.get('computation-settings',
                                                       'compute_graph_density'))
    options_gtc.compute_graph_conductance = int(config.get('computation-settings',
                                                           'compute_graph_conductance'))
    options_gtc.compute_global_efficiency = int(config.get('computation-settings',
                                                           'compute_global_efficiency'))
    options_gtc.compute_clustering_coef = int(config.get('computation-settings',
                                                         'compute_avg_clustering_coef'))
    options_gtc.compute_assortativity_coef = int(config.get('computation-settings',
                                                            'compute_assortativity_coef'))
    options_gtc.compute_network_diameter = int(config.get('computation-settings',
                                                          'compute_network_diameter'))
    options_gtc.compute_wiener_index = int(config.get('computation-settings',
                                                      'compute_wiener_index'))

    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--inputImage',
                          dest='filePath',
                          help='path to image file/folder containing images',
                          default=options_path.image_path,
                          type='string')
    opt_parser.add_option('-o', '--outputFolder',
                          dest='outputDir',
                          help='directory path for saving GT output',
                          default=options_path.output_path,
                          type='string')
    opt_parser.add_option('-a', '--algorithmChoice',
                          dest='algChoice',
                          help='select GT algorithm',
                          default=0,
                          type='int')
    opt_parser.add_option('-m', '--isMultiImage',
                          dest='multiImage',
                          help='is it a multi-image (multiple images in a folder) analysis?',
                          default=options_path.is_multi_image,
                          type='int')
    opt_parser.add_option('-c', '--cores',
                          dest='numCores',
                          help='number of cores',
                          default=cpus,
                          type='int')
    (options, args) = opt_parser.parse_args()

    return config, options, options_img, options_gte, options_gtc


def load_gui_configs():
    gui_txt = struct()

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs_gui.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)
        title = str(config.get('gui', 'title'))
    except configparser.NoSectionError:
        raise Exception("Unable to build UI! Add file config_gui.ini.")

    gui_txt = struct()
    gui_txt.title = title
    gui_txt.about = str(config.get('gui', 'about'))

    # 1. Extraction
    gui_txt.weighted = str(config.get('gui', 'weighted'))
    gui_txt.merge = str(config.get('gui', 'merge'))
    gui_txt.prune = str(config.get('gui', 'prune'))
    gui_txt.remove_disconnected = str(config.get('gui', 'remove_disconnected'))
    gui_txt.remove_loops = str(config.get('gui', 'remove_loops'))
    gui_txt.multigraph = str(config.get('gui', 'multigraph'))
    gui_txt.node_id = str(config.get('gui', 'node_id'))

    # 2. Computation
    gui_txt.heatmaps = str(config.get('gui', 'heatmaps'))
    gui_txt.degree = str(config.get('gui', 'degree'))
    gui_txt.diameter = str(config.get('gui', 'diameter'))
    gui_txt.connectivity = str(config.get('gui', 'connectivity'))
    gui_txt.clustering = str(config.get('gui', 'clustering'))
    gui_txt.assortativity = str(config.get('gui', 'assortativity'))
    gui_txt.betweenness = str(config.get('gui', 'betweenness'))
    gui_txt.current_flow = str(config.get('gui', 'current_flow'))
    gui_txt.closeness = str(config.get('gui', 'closeness'))
    gui_txt.eigenvector = str(config.get('gui', 'eigenvector'))
    gui_txt.density = str(config.get('gui', 'density'))
    gui_txt.conductance = str(config.get('gui', 'conductance'))
    gui_txt.efficiency = str(config.get('gui', 'efficiency'))
    gui_txt.wiener = str(config.get('gui', 'wiener'))

    # 3. Save Files
    gui_txt.gexf = str(config.get('gui', 'gexf'))
    gui_txt.edge_list = str(config.get('gui', 'edge_list'))
    gui_txt.adjacency = str(config.get('gui', 'adjacency'))
    gui_txt.save_images = str(config.get('gui', 'save_images'))

    return gui_txt


def get_num_cores():
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


def write_file(data, path, wr=True):
    """Description

    Writes data into a file
    :param data: information to be written
    :param path: name of file and storage path
    :param wr: writes data into file if True
    :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass
