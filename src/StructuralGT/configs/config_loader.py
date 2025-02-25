# SPDX-License-Identifier: GNU GPL v3

"""
Loads default configurations from 'configs.ini' file
"""

import os
import configparser
import multiprocessing as mp
from optparse import OptionParser
from ypstruct import struct


def load_project_configs():
    options_path = struct()

    # 1. Image Path
    options_path.is_multi_image = 0
    options_path.image_path = ""
    options_path.output_path = ""

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
    (options_main, args) = opt_parser.parse_args()

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
        return options_main

    # 1. Image Path
    options_path.is_multi_image = int(config.get('image-dir', 'is_multi_image'))
    options_path.image_path = config.get('image-dir', 'image_path')
    options_path.output_path = config.get('image-dir', 'gt_output_path')

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
    """opt_parser.add_option('-d', '--imageDim',
                          dest='imageDim',
                          help='is it a 2D or 3D image?',
                          default=
                          type='int')"""
    opt_parser.add_option('-c', '--cores',
                          dest='numCores',
                          help='number of cores',
                          default=cpus,
                          type='int')
    (options_main, args) = opt_parser.parse_args()

    return options_main

def load_img_configs():
    """Image Detection settings"""

    options_img = {
        "threshold_type": {"id": "threshold_type", "type": "binary-filter", "text": "", "visible": 1, "value": 1 },
        "global_threshold_value": {"id": "global_threshold_value", "type": "binary-filter", "text": "", "visible": 1, "value": 127 },
        "adaptive_local_threshold_value": {"id": "adaptive_local_threshold_value", "type": "binary-filter", "text": "", "visible": 1, "value": 11 },
        "otsu_value": {"id": "otsu_value", "type": "binary-filter", "text": "", "visible": 0, "value": 0},
        "apply_dark_foreground": {"id": "apply_dark_foreground", "type": "binary-filter", "text": "", "visible": 1, "value": 0},

        "apply_autolevel": {"id": "apply_autolevel", "type": "image-filter", "text": "Autolevel", "value": 0,
                            "dataId": "autolevel_blurring_size", "dataValue": 3, "minValue": 1, "maxValue": 7, "stepSize": 2},
        "apply_gaussian_blur": {"id": "apply_gaussian_blur", "type": "image-filter", "text": "Gaussian", "value": 0,
                                "dataId": "gaussian_blurring_size", "dataValue": 3, "minValue": 1, "maxValue": 7, "stepSize": 2 },
        "apply_laplacian_gradient": {"id": "apply_laplacian_gradient", "type": "image-filter", "text": "Laplacian",
                                     "value": 0, "dataId": "laplacian_kernel_size", "dataValue": 3, "minValue": 1, "maxValue": 7,  "stepSize": 2 },
        "apply_lowpass_filter": {"id": "apply_lowpass_filter", "type": "image-filter", "text": "Lowpass", "value": 0,
                                 "dataId": "lowpass_window_size", "dataValue": 10, "minValue": 0, "maxValue": 1000,  "stepSize": 1 },
        "apply_gamma": {"id": "apply_gamma", "type": "image-filter", "text": "LUT Gamma", "value": 1, "dataId": "adjust_gamma",
                        "dataValue": 1.0, "minValue": 0.01, "maxValue": 5.0, "stepSize": 0.01  },
        "apply_sobel_gradient": {"id": "apply_sobel_gradient", "type": "image-filter", "text": "Sobel", "value": 0,
                                 "dataId": "sobel_kernel_size", "dataValue": 3, "minValue": 1, "maxValue": 7,  "stepSize": 2 },
        "apply_median_filter": {"id": "apply_median_filter", "type": "image-filter", "text": "Median", "value": 0 },
        "apply_scharr_gradient": {"id": "apply_scharr_gradient", "type": "image-filter", "text": "Scharr", "value": 0},

        "brightness_level": {"id": "brightness_level", "type": "image-control", "text": "Brightness", "value": 0 },
        "contrast_level": {"id": "contrast_level", "type": "image-control", "text": "Contrast", "value": 0 },

        #"image_dim": {"id": "image_dim", "type": "image-property", "text": "", "value": 2},
        "scale_value_nanometers": {"id": "scale_value_nanometers", "type": "image-property", "text": "Scalebar (nm)", "value": 1.0 },
        "scalebar_pixel_count": {"id": "scalebar_pixel_count", "type": "image-property", "text": "Scalebar Pixel Count", "value": 0 },
        "resistivity": {"id": "resistivity", "type": "image-property", "text": "Resistivity (<html>&Omega;</html>m)", "value": 1.0 },
    }

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)

        #options_img["image_dim"]["value"] = int(config.get('filter-settings', 'image_dim'))
        options_img["threshold_type"]["value"] = int(config.get('filter-settings', 'threshold_type'))
        options_img["global_threshold_value"]["value"] = int(config.get('filter-settings', 'global_threshold_value'))
        options_img["adaptive_local_threshold_value"]["value"] = int(config.get('filter-settings', 'adaptive_local_threshold_value'))
        options_img["apply_dark_foreground"]["value"] = int(config.get('filter-settings', 'apply_dark_foreground'))

        options_img["apply_gamma"]["value"] = int(config.get('filter-settings', 'apply_gamma'))
        options_img["apply_gamma"]["dataValue"] = float(config.get('filter-settings', 'adjust_gamma'))
        options_img["apply_autolevel"]["value"] = int(config.get('filter-settings', 'apply_autolevel'))
        options_img["apply_autolevel"]["dataValue"] = int(config.get('filter-settings', 'blurring_window_size'))
        options_img["apply_laplacian_gradient"]["value"] = int(config.get('filter-settings', 'apply_laplacian_gradient'))
        options_img["apply_laplacian_gradient"]["dataValue"] = 3
        options_img["apply_sobel_gradient"]["value"] = int(config.get('filter-settings', 'apply_sobel_gradient'))
        options_img["apply_sobel_gradient"]["dataValue"] = 3
        options_img["apply_gaussian_blur"]["value"] = int(config.get('filter-settings', 'apply_gaussian_blur'))
        options_img["apply_gaussian_blur"]["dataValue"] = int(config.get('filter-settings', 'blurring_window_size'))
        options_img["apply_lowpass_filter"]["value"] = int(config.get('filter-settings', 'apply_lowpass_filter'))
        options_img["apply_lowpass_filter"]["dataValue"] = int(config.get('filter-settings', 'filter_window_size'))

        options_img["apply_scharr_gradient"]["value"] = int(config.get('filter-settings', 'apply_scharr_gradient'))
        options_img["apply_median_filter"]["value"] = int(config.get('filter-settings', 'apply_median_filter'))

        options_img["brightness_level"]["value"] = int(config.get('filter-settings', 'brightness_level'))
        options_img["contrast_level"]["value"] = int(config.get('filter-settings', 'contrast_level'))
        options_img["scale_value_nanometers"]["value"] = float(config.get('filter-settings', 'scale_value_nanometers'))
        options_img["scalebar_pixel_count"]["value"] = int(config.get('filter-settings', 'scalebar_pixel_count'))
        options_img["resistivity"]["value"] = float(config.get('filter-settings', 'resistivity'))

        return options_img
    except configparser.NoSectionError:
        return options_img

def load_gte_configs():
    """Graph Extraction Settings"""

    options_gte = {
        "has_weights": {"id": "has_weights", "type": "graph-extraction", "text": "Add Weights", "value": 0,
                        "items": [
                            {"id": "DIA", "text": "by diameter", "value": 1},
                            {"id": "AREA", "text": "by area", "value": 0},
                            {"id": "LEN", "text": "by length", "value": 0},
                            {"id": "ANGLE", "text": "by angle", "value": 0},
                            {"id": "INV-LEN", "text": "by inverse-length", "value": 0},
                            {"id": "FIX-CON", "text": "by conductance", "value": 0},
                            {"id": "RES", "text": "by resistance", "value": 0},
                        ]},
        "merge_nearby_nodes": {"id": "merge_nearby_nodes", "type": "graph-extraction", "text": "Merge Nearby Nodes", "value": 1},
        "prune_dangling_edges": {"id": "prune_dangling_edges", "type": "graph-extraction", "text": "Prune Dangling Edges", "value": 1},
        "remove_disconnected_segments": {"id": "remove_disconnected_segments", "type": "graph-extraction", "text": "Remove Disconnected Segments", "value": 1, "items": [{"id": "remove_object_size", "text": "", "value": 500}]},
        "remove_self_loops": {"id": "remove_self_loops", "type": "graph-extraction", "text": "Remove Self Loops", "value": 1},
        "is_multigraph": {"id": "is_multigraph", "type": "graph-extraction", "text": "Is Multigraph?", "value": 0},
        "display_node_id": {"id": "display_node_id", "type": "graph-extraction", "text": "Display Node ID", "value": 0},

        "export_edge_list": {"id": "export_edge_list", "type": "file-options", "text": "Export Edge List", "value": 0},
        "export_as_gexf": {"id": "export_as_gexf", "type": "file-options", "text": "Export as gexf", "value": 0},
        "export_adj_mat": {"id": "export_adj_mat", "type": "file-options", "text": "Export Adjacency Matrix", "value": 0},
        "save_images": {"id": "save_images", "type": "file-options", "text": "Save All Images", "value": 0},
    }

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)

        options_gte["merge_nearby_nodes"]["value"] = int(config.get('extraction-settings', 'merge_nearby_nodes'))
        options_gte["prune_dangling_edges"]["value"] = int(config.get('extraction-settings', 'prune_dangling_edges'))
        options_gte["remove_disconnected_segments"]["value"] = int(
            config.get('extraction-settings', 'remove_disconnected_segments'))
        options_gte["remove_self_loops"]["value"] = int(config.get('extraction-settings', 'remove_self_loops'))
        options_gte["remove_self_loops"]["items"][0]["value"] = int(config.get('extraction-settings', 'remove_object_size'))
        options_gte["is_multigraph"]["value"] = int(config.get('extraction-settings', 'is_multigraph'))
        options_gte["has_weights"]["value"] = int(config.get('extraction-settings', 'add_weights'))
        weight_type = str(config.get('extraction-settings', 'weight_type'))
        for i in range(len(options_gte["has_weights"]["items"])):
            options_gte["has_weights"]["items"][i]["value"] = 1 if options_gte["has_weights"]["items"][i]["id"] == weight_type else 0
        options_gte["display_node_id"]["value"] = int(config.get('extraction-settings', 'display_node_id'))
        options_gte["export_edge_list"]["value"] = int(config.get('extraction-settings', 'export_edge_list'))
        options_gte["export_as_gexf"]["value"] = int(config.get('extraction-settings', 'export_as_gexf'))
        options_gte["export_adj_mat"]["value"] = int(config.get('extraction-settings', 'export_adj_mat'))
        options_gte["save_images"]["value"] = int(config.get('extraction-settings', 'save_images'))

        return options_gte
    except configparser.NoSectionError:
        return options_gte

def load_gtc_configs():
    """Networkx Calculation Settings"""

    options_gtc = {
        "display_heatmaps": {"id": "display_heatmaps", "text": "Display Heatmaps", "value": 1},
        "display_degree_histogram": {"id": "display_degree_histogram", "text": "Average Degree", "value": 1},
        "compute_network_diameter": {"id": "compute_network_diameter", "text": "Network Diameter", "value": 1},
        "compute_graph_density": {"id": "compute_graph_density", "text": "Graph Density", "value": 1},
        "compute_wiener_index": {"id": "compute_wiener_index", "text": "Wiener Index", "value": 1},
        "compute_avg_node_connectivity": {"id": "compute_avg_node_connectivity", "text": "Average Node Connectivity", "value": 0},
        "compute_global_efficiency": {"id": "compute_global_efficiency", "text": "Global Coefficient", "value": 1},
        "compute_avg_clustering_coef": {"id": "compute_avg_clustering_coef", "text": "Average Clustering Coefficient", "value": 1},
        "compute_assortativity_coef": {"id": "compute_assortativity_coef", "text": "Assortativity Coefficient", "value": 1},
        "display_betweenness_centrality_histogram": {"id": "display_betweenness_centrality_histogram", "text": "Betweenness Centrality", "value": 1},
        "display_closeness_centrality_histogram": {"id": "display_closeness_centrality_histogram", "text": "Closenness Centrality", "value": 1},
        "display_eigenvector_centrality_histogram": {"id": "display_eigenvector_centrality_histogram", "text": "Eigenvector Centrality", "value": 1},
        "display_ohms_histogram": {"id": "display_ohms_histogram", "text": "Ohms Centrality", "value": 0},
        #"display_currentflow_histogram": {"id": "display_currentflow_histogram", "text": "Current Flow Betweenness Centrality", "value": 0},
        "display_edge_angle_centrality_histogram": {"id": "display_edge_angle_centrality_histogram", "text": "Edge Angle Centrality", "value": 0},
        #"compute_graph_conductance": {"id": "compute_graph_conductance", "text": "Graph Conductance", "value": 0},
        "display_percolation_histogram": {"id": "display_percolation_histogram", "text": "Percolation Centrality", "value": 0},
        #"compute_lang": {"id": "compute_lang", "text": "Programming Language", "value": 'Py'}
    }

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)

        options_gtc["display_heatmaps"]["value"] = int(config.get('sgt-settings', 'display_heatmaps'))
        options_gtc["display_degree_histogram"]["value"] = int(config.get('sgt-settings', 'display_degree_histogram'))
        options_gtc["display_betweenness_centrality_histogram"]["value"] = int(config.get('sgt-settings', 'display_betweenness_centrality_histogram'))
        # options_gtc["display_current_flow_betweenness_centrality_histogram"]["value"] = int(
        #    config.get('sgt-settings', 'display_current_flow_betweenness_centrality_histogram'))
        options_gtc["display_closeness_centrality_histogram"]["value"] = int(config.get('sgt-settings', 'display_closeness_centrality_histogram'))
        options_gtc["display_eigenvector_centrality_histogram"]["value"] = int(config.get('sgt-settings', 'display_eigenvector_centrality_histogram'))
        options_gtc["display_edge_angle_centrality_histogram"]["value"] = int(config.get('sgt-settings', 'display_edge_angle_centrality_histogram'))
        options_gtc["display_ohms_histogram"]["value"] = int(config.get('sgt-settings', 'display_ohms_histogram'))
        options_gtc["display_percolation_histogram"]["value"] = int(config.get('sgt-settings', 'display_percolation_histogram'))
        options_gtc["compute_avg_node_connectivity"]["value"] = int(config.get('sgt-settings', 'compute_avg_node_connectivity'))
        options_gtc["compute_graph_density"]["value"] = int(config.get('sgt-settings', 'compute_graph_density'))
        # options_gtc["compute_graph_conductance"]["value"] = int(config.get('sgt-settings', 'compute_graph_conductance'))
        options_gtc["compute_global_efficiency"]["value"] = int(config.get('sgt-settings', 'compute_global_efficiency'))
        options_gtc["compute_avg_clustering_coef"]["value"] = int(config.get('sgt-settings', 'compute_avg_clustering_coef'))
        options_gtc["compute_assortativity_coef"]["value"] = int(config.get('sgt-settings', 'compute_assortativity_coef'))
        options_gtc["compute_network_diameter"]["value"] = int(config.get('sgt-settings', 'compute_network_diameter'))
        options_gtc["compute_wiener_index"]["value"] = int(config.get('sgt-settings', 'compute_wiener_index'))
        # options_gtc["compute_lang"]["value"] = str(config.get('sgt-settings', 'compute_lang'))

        return options_gtc
    except configparser.NoSectionError:
        return options_gtc

def load_all_configs():

    # 5. Fractal Image Compression Settings
    options_fic = struct()
    options_fic.down_sampling_factor = 4
    options_fic.domain_block_size = 8
    options_fic.range_block_size = 4
    options_fic.block_step_size = 8
    options_fic.decompress_iteration_count = 8

    # 6. Graph Network Chaos Theory Settings
    options_gnct = struct()
    options_gnct.ml_model = 'MLP'

    """
    # 5. Fractal Image Compression Settings
    options_fic.down_sampling_factor = int(config.get('fic-settings', 'down_sampling_factor'))
    options_fic.domain_block_size = int(config.get('fic-settings', 'domain_block_size'))
    options_fic.range_block_size = int(config.get('fic-settings', 'range_block_size'))
    options_fic.block_step_size = int(config.get('fic-settings', 'block_step_size'))
    options_fic.decompress_iteration_count = int(config.get('fic-settings', 'decompress_iteration_count'))

    # 6. Graph Network Chaos Theory Settings
    options_gnct.ml_model = str(config.get('gnct-settings', 'ml_model'))
    """

    options_main = load_project_configs()
    options_img = load_img_configs()
    options_gte = load_gte_configs()
    options_gtc = load_gtc_configs()

    configs_data = {
        "main_options": options_main,
        "filter_options": options_img,
        "extraction_options": options_gte,
        "sgt_options": options_gtc,
        "fic_options": options_fic,
        "gnct_options": options_gnct
    }
    return configs_data


def load_gui_configs():
    gui_txt = struct()

    gui_txt.title = "Structural GT"
    gui_txt.about = "about Structural GT"

    # 1. Extraction
    gui_txt.weighted = "Add Weights"
    gui_txt.weight_by_dia = 'Diameter'
    gui_txt.weight_by_area = 'Area'
    gui_txt.weight_by_len = 'Length'
    gui_txt.weight_by_angle = 'Angle'
    gui_txt.weight_by_inv_len = 'Inverse Length'
    gui_txt.weight_by_var_con = 'Conductance'
    gui_txt.weight_by_res = 'Resistance'
    gui_txt.merge = "Merge Nearby Nodes"
    gui_txt.prune = "Prune Dangling Edges"
    gui_txt.remove_disconnected = "Remove Disconnected Segments (set size)"
    gui_txt.remove_loops = "Remove Self Loops"
    gui_txt.multigraph = "Is Multigraph?"
    gui_txt.node_id = "Display Node ID"

    # 2. Computation
    gui_txt.heatmaps = "Display Heatmaps"
    gui_txt.degree = "Average Degree"
    gui_txt.diameter = "Network Diameter"
    gui_txt.connectivity = "Average Node Connectivity"
    gui_txt.clustering = "Average Clustering Coefficient"
    gui_txt.assortativity = "Assortativity Coefficient"
    gui_txt.betweenness = "Betweenness Centrality"
    gui_txt.current_flow = "Current Flow Betweenness Centrality"
    gui_txt.closeness = "Closeness Centrality"
    gui_txt.eigenvector = "Eigenvector Centrality"
    gui_txt.edge_angle = "Edge Angle Centrality"
    gui_txt.ohms = "Ohms Centrality"
    gui_txt.percolation = "Percolation Centrality"
    gui_txt.density = "Graph Density"
    gui_txt.conductance = "Graph Conductance"
    gui_txt.efficiency = "Global Efficiency"
    gui_txt.wiener = "Wiener Index"

    # 3. Save Files
    gui_txt.gexf = "Export as gexf"
    gui_txt.edge_list = "Export Edge List"
    gui_txt.adjacency = "Export Adjacency Matrix"
    gui_txt.save_images = "Save All Images"

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
        return gui_txt

    gui_txt.title = title
    gui_txt.about = str(config.get('gui', 'about'))

    # 1. Extraction
    gui_txt.weighted = str(config.get('gui', 'weighted'))
    gui_txt.weight_by_dia = str(config.get('gui', 'weight_by_dia'))
    gui_txt.weight_by_area = str(config.get('gui', 'weight_by_area'))
    gui_txt.weight_by_len = str(config.get('gui', 'weight_by_len'))
    gui_txt.weight_by_angle = str(config.get('gui', 'weight_by_angle'))
    gui_txt.weight_by_inv_len = str(config.get('gui', 'weight_by_inv_len'))
    gui_txt.weight_by_var_con = str(config.get('gui', 'weight_by_var_con'))
    gui_txt.weight_by_res = str(config.get('gui', 'weight_by_res'))
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
    gui_txt.edge_angle = str(config.get('gui', 'edge_angle'))
    gui_txt.ohms = str(config.get('gui', 'ohms'))
    gui_txt.percolation = str(config.get('gui', 'percolation'))
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
