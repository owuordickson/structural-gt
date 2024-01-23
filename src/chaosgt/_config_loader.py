import configparser
# from os import path
import pathlib
from ypstruct import struct


def load():
    # Load configuration from file
    config_file = pathlib.Path(__file__).parent.absolute() / "configs.cfg"
    config = configparser.SafeConfigParser()
    config.read(config_file)
    # print(config.sections())
    options = struct()

    # 1. Image Path
    options.is_multi_image = int(config.get('image-dir', 'is_multi_image'))
    options.single_imagepath = config.get('image-dir', 'single_image_path')
    options.multi_imagepath = config.get('image-dir', 'multi_image_path')
    options.output_path = config.get('image-dir', 'gt_output_path')

    # 2. Image Detection settings
    var = int(config.get('detection-settings', 'threshold'))
    var = int(config.get('detection-settings', 'global_threshold_value'))
    var = int(config.get('detection-settings', 'adaptive_local_threshold_value'))
    var = float(config.get('detection-settings', 'adjust_gamma'))
    var = int(config.get('detection-settings', 'blurring_window_size'))
    var = int(config.get('detection-settings', 'filter_window_size'))
    var = int(config.get('detection-settings', 'use_autolevel'))
    var = int(config.get('detection-settings', 'use_laplacian_gradient'))
    var = int(config.get('detection-settings', 'use_scharr_gradient'))
    var = int(config.get('detection-settings', 'use_sobel_gradient'))
    var = int(config.get('detection-settings', 'apply_median_filter'))
    var = int(config.get('detection-settings', 'apply_gaussian_filter'))
    var = int(config.get('detection-settings', 'apply_lowpass_filter'))
    var = int(config.get('detection-settings', 'dark_foreground'))

    # 3. Graph Extraction Settings
    var = int(config.get('extraction-settings', 'merge_nearby_nodes'))
    var = int(config.get('extraction-settings', 'prune_dangling_edges'))
    var = int(config.get('extraction-settings', 'remove_disconnected_segments'))
    var = int(config.get('extraction-settings', 'remove_self_loops'))
    var = int(config.get('extraction-settings', 'remove_object_size'))
    var = int(config.get('extraction-settings', 'disable_multigraph'))
    var = int(config.get('extraction-settings', 'weighted_by_diameter'))
    var = int(config.get('extraction-settings', 'export_edge_list'))
    var = int(config.get('extraction-settings', 'export_as_gexf'))
    var = int(config.get('extraction-settings', 'display_node_id'))

    # 4. Networkx Calculation Settings
    var = int(config.get('computation-settings', 'display_heatmaps'))
    var = int(config.get('computation-settings', 'display_degree_histogram'))
    var = int(config.get('computation-settings', 'display_betweeness_centrality_histogram'))
    var = int(config.get('computation-settings', 'display_closeness_centrality_histogram'))
    var = int(config.get('computation-settings', 'display_eigenvector_centrality_histogram'))
    var = int(config.get('computation-settings', 'compute_avg_nodal_connectivity'))
    var = int(config.get('computation-settings', 'compute_graph_density'))
    var = int(config.get('computation-settings', 'compute_graph_conductance'))
    var = int(config.get('computation-settings', 'compute_global_efficiency'))
    var = int(config.get('computation-settings', 'compute_avg_clustering_coef'))
    var = int(config.get('computation-settings', 'compute_assortativity_coef'))
    var = int(config.get('computation-settings', 'compute_network_diameter'))
    var = int(config.get('computation-settings', 'compute_wiener_index'))

    return config, options
