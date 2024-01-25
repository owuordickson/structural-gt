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
    options_path = struct()
    options_img = struct()
    options_gte = struct()
    options_gtc = struct()

    # 1. Image Path
    options_path.is_multi_image = int(config.get('image-dir', 'is_multi_image'))
    options_path.single_imagepath = config.get('image-dir', 'single_image_path')
    options_path.multi_imagepath = config.get('image-dir', 'multi_image_path')
    options_path.output_path = config.get('image-dir', 'gt_output_path')

    # 2. Image Detection settings
    options_img.threshold_type = int(config.get('detection-settings', 'threshold'))
    options_img.threshold_global = int(config.get('detection-settings', 'global_threshold_value'))
    options_img.threshold_adaptive = int(config.get('detection-settings', 'adaptive_local_threshold_value'))
    options_img.gamma = float(config.get('detection-settings', 'adjust_gamma'))
    options_img.blurring_window_size = int(config.get('detection-settings', 'blurring_window_size'))
    options_img.filter_window_size = int(config.get('detection-settings', 'filter_window_size'))
    options_img.apply_autolevel = int(config.get('detection-settings', 'use_autolevel'))
    options_img.apply_laplacian = int(config.get('detection-settings', 'use_laplacian_gradient'))
    options_img.apply_scharr = int(config.get('detection-settings', 'use_scharr_gradient'))
    options_img.apply_sobel = int(config.get('detection-settings', 'use_sobel_gradient'))
    options_img.apply_median = int(config.get('detection-settings', 'apply_median_filter'))
    options_img.apply_gaussian = int(config.get('detection-settings', 'apply_gaussian_blur'))
    options_img.apply_lowpass = int(config.get('detection-settings', 'apply_lowpass_filter'))
    options_img.apply_dark_foreground = int(config.get('detection-settings', 'dark_foreground'))

    # 3. Graph Extraction Settings
    options_gte.merge_nearby_nodes = int(config.get('extraction-settings', 'merge_nearby_nodes'))
    options_gte.prune_dangling_edges = int(config.get('extraction-settings', 'prune_dangling_edges'))
    options_gte.remove_disconnected_segments = int(config.get('extraction-settings', 'remove_disconnected_segments'))
    options_gte.remove_self_loops = int(config.get('extraction-settings', 'remove_self_loops'))
    options_gte.remove_object_size = int(config.get('extraction-settings', 'remove_object_size'))
    options_gte.disable_multigraph = int(config.get('extraction-settings', 'disable_multigraph'))
    options_gte.weighted_by_diameter = int(config.get('extraction-settings', 'weighted_by_diameter'))
    options_gte.export_edge_list = int(config.get('extraction-settings', 'export_edge_list'))
    options_gte.export_as_gexf = int(config.get('extraction-settings', 'export_as_gexf'))
    options_gte.display_node_id = int(config.get('extraction-settings', 'display_node_id'))

    # 4. Networkx Calculation Settings
    options_gtc.display_heatmaps = int(config.get('computation-settings', 'display_heatmaps'))
    options_gtc.display_degree_histogram = int(config.get('computation-settings', 'display_degree_histogram'))
    options_gtc.display_betweeness_histogram = int(config.get('computation-settings',\
                                                              'display_betweeness_centrality_histogram'))
    options_gtc.display_closeness_histogram = int(config.get('computation-settings',\
                                                             'display_closeness_centrality_histogram'))
    options_gtc.display_eigenvector_histogram = int(config.get('computation-settings',\
                                                               'display_eigenvector_centrality_histogram'))
    options_gtc.compute_nodal_connectivity = int(config.get('computation-settings',\
                                                            'compute_avg_nodal_connectivity'))
    options_gtc.compute_graph_density = int(config.get('computation-settings',\
                                                       'compute_graph_density'))
    options_gtc.compute_graph_conductance = int(config.get('computation-settings',\
                                                           'compute_graph_conductance'))
    options_gtc.compute_global_efficiency = int(config.get('computation-settings',\
                                                           'compute_global_efficiency'))
    options_gtc.compute_clustering_coef = int(config.get('computation-settings',\
                                                         'compute_avg_clustering_coef'))
    options_gtc.compute_assortativity_coef = int(config.get('computation-settings',\
                                                            'compute_assortativity_coef'))
    options_gtc.compute_network_diameter = int(config.get('computation-settings',\
                                                          'compute_network_diameter'))
    options_gtc.compute_wiener_index = int(config.get('computation-settings',\
                                                      'compute_wiener_index'))

    return config, options_path, options_img, options_gte, options_gtc

