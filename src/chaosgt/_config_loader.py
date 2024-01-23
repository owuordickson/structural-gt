import configparser
# from os import path
import pathlib


def load():
    # Load configuration from file
    config_file = pathlib.Path(__file__).parent.absolute() / "configs.cfg"
    config = configparser.SafeConfigParser()
    config.read(config_file)
    # print(config.sections())

    # 1. Image Path
    imagepath = config.get('image-dir', 'single_image_path')
    multi_imagepath = config.get('image-dir', 'multi_image_path')
    output_path = config.get('image-dir', 'gt_output_path')

    # 2. Image Detection settings
    var = config.get('detection-settings', 'threshold')
    var = config.get('detection-settings', 'global_threshold_value')
    var = config.get('detection-settings', 'adaptive_local_threshold_value')
    var = config.get('detection-settings', 'adjust_gamma')
    var = config.get('detection-settings', 'blurring_window_size')
    var = config.get('detection-settings', 'filter_window_size')
    var = config.get('detection-settings', 'use_autolevel')
    var = config.get('detection-settings', 'use_laplacian_gradient')
    var = config.get('detection-settings', 'use_scharr_gradient')
    var = config.get('detection-settings', 'use_sobel_gradient')
    var = config.get('detection-settings', 'apply_median_filter')
    var = config.get('detection-settings', 'apply_gaussian_filter')
    var = config.get('detection-settings', 'apply_lowpass_filter')
    var = config.get('detection-settings', 'dark_foreground')

    # 3. Graph Extraction Settings
    var = config.get('extraction-settings', 'merge_nearby_nodes')
    var = config.get('extraction-settings', 'prune_dangling_edges')
    var = config.get('extraction-settings', 'remove_disconnected_segments')
    var = config.get('extraction-settings', 'remove_self_loops')
    var = config.get('extraction-settings', 'remove_object_size')
    var = config.get('extraction-settings', 'disable_multigraph')
    var = config.get('extraction-settings', 'weighted_by_diameter')
    var = config.get('extraction-settings', 'export_edge_list')
    var = config.get('extraction-settings', 'export_as_gexf')
    var = config.get('extraction-settings', 'display_node_id')

    # 4. Networkx Calculation Settings
    var = config.get('computation-settings', 'display_heatmaps')
    var = config.get('computation-settings', 'display_degree_histogram')
    var = config.get('computation-settings', 'display_betweeness_centrality_histogram')
    var = config.get('computation-settings', 'display_closeness_centrality_histogram')
    var = config.get('computation-settings', 'display_eigenvector_centrality_histogram')
    var = config.get('computation-settings', 'compute_avg_nodal_connectivity')
    var = config.get('computation-settings', 'compute_graph_density')
    var = config.get('computation-settings', 'compute_graph_conductance')
    var = config.get('computation-settings', 'compute_global_efficiency')
    var = config.get('computation-settings', 'compute_avg_clustering_coef')
    var = config.get('computation-settings', 'compute_assortativity_coef')
    var = config.get('computation-settings', 'compute_network_diameter')
    var = config.get('computation-settings', 'compute_wiener_index')

    return config
