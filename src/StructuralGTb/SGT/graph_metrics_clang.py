# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics using iGraph library implemented in C language.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import global_efficiency, clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.wiener import wiener_index

from .graph_metrics import GraphMetrics
from .graph_converter import GraphConverter
from .image_processor import ImageProcessor

import sgt
from ..configs.config_loader import get_num_cores


class GraphMetricsClang(GraphMetrics):
    """
        A class that computes all the user selected graph theory metrics and writes the results in a PDF file.
        This class uses C language and iGraph library to execute the long-running tasks.

        Args:
            g_obj: graph converter object.
            configs: graph theory computation parameters and options.
            allow_multiprocessing: a decision to allow multiprocessing computing.
    """

    def __init__(self, *args):
        """
        A class that computes all the user selected graph theory metrics and writes the results in a PDF file.

        :param g_obj: graph converter object.
        :param configs: graph theory computation parameters and options.
        :param allow_multiprocessing: allow multiprocessing computing.

        >>> from ypstruct import struct
        >>> opt_img = struct()
        >>> opt_img.threshold_type = 1
        >>> opt_img.threshold_global = 127
        >>> opt_img.threshold_adaptive = 11
        >>> opt_img.gamma = float(1)
        >>> opt_img.gaussian_blurring_size = 3
        >>> opt_img.autolevel_blurring_size = 3
        >>> opt_img.lowpass_window_size = 10
        >>> opt_img.laplacian_kernel_size = 3
        >>> opt_img.sobel_kernel_size = 3
        >>> opt_img.apply_autolevel = 0
        >>> opt_img.apply_laplacian = 0
        >>> opt_img.apply_scharr = 0
        >>> opt_img.apply_sobel = 0
        >>> opt_img.apply_median = 0
        >>> opt_img.apply_gaussian = 0
        >>> opt_img.apply_lowpass = 0
        >>> opt_img.apply_dark_foreground = 0
        >>> opt_img.brightness_level = 0
        >>> opt_img.contrast_level = 0
        >>>
        >>> opt_gte = struct()
        >>> opt_gte.merge_nearby_nodes = 1
        >>> opt_gte.prune_dangling_edges = 1
        >>> opt_gte.remove_disconnected_segments = 1
        >>> opt_gte.remove_self_loops = 1
        >>> opt_gte.remove_object_size = 500
        >>> opt_gte.is_multigraph = 0
        >>> opt_gte.weighted_by_diameter = 0
        >>> opt_gte.display_node_id = 0
        >>> opt_gte.export_edge_list = 0
        >>> opt_gte.export_as_gexf = 0
        >>> opt_gte.export_adj_mat = 0
        >>> opt_gte.save_images = 0
        >>>
        >>> opt_gtc = struct()
        >>> opt_gtc.display_heatmaps = 1
        >>> opt_gtc.display_degree_histogram = 1
        >>> opt_gtc.display_betweenness_histogram = 1
        >>> opt_gtc.display_currentflow_histogram = 1
        >>> opt_gtc.display_closeness_histogram = 1
        >>> opt_gtc.display_eigenvector_histogram = 1
        >>> opt_gtc.compute_nodal_connectivity = 1
        >>> opt_gtc.compute_graph_density = 1
        >>> opt_gtc.compute_graph_conductance = 0
        >>> opt_gtc.compute_global_efficiency = 1
        >>> opt_gtc.compute_clustering_coef = 1
        >>> opt_gtc.compute_assortativity_coef = 1
        >>> opt_gtc.compute_network_diameter = 1
        >>> opt_gtc.compute_wiener_index = 1
        >>>
        >>> i_path = "path/to/image"
        >>> o_dir = ""
        >>>
        >>> imp_obj = ImageProcessor(i_path, o_dir, options_img=opt_img)
        >>> graph_obj = GraphConverter(imp_obj, options_gte=opt_gte)
        >>> graph_obj.fit()
        >>> metrics_obj = GraphMetrics(graph_obj, opt_gtc)
        >>> metrics_obj.compute_gt_metrics()
        >>> if opt_gte.weighted_by_diameter:
        >>>     metrics_obj.compute_weighted_gt_metrics()
        >>> metrics_obj.generate_pdf_output()

        """
        super(GraphMetricsClang, self).__init__(*args)

    def compute_gt_metrics(self):
        """
        Compute un-weighted graph theory metrics using iGraph and NetworkX libraries.

        :return:
        """
        self.update_status([1, "Using iGraph to perform un-weighted analysis..."])

        graph = self.gc.nx_graph
        options = self.configs
        options_gte = self.gc.configs_graph
        data_dict = {"x": [], "y": []}

        node_count = int(nx.number_of_nodes(graph))
        edge_count = int(nx.number_of_edges(graph))

        data_dict["x"].append("Number of nodes")
        data_dict["y"].append(node_count)

        data_dict["x"].append("Number of edges")
        data_dict["y"].append(edge_count)

        if graph.number_of_nodes() <= 0:
            self.update_status([-1, "Problem with graph (change filter and graph options)."])
            return

        # creating degree histogram
        if options.display_degree_histogram == 1:
            self.update_status([5, "Computing graph degree..."])
            deg_distribution_1 = dict(nx.degree(graph))
            deg_distribution = np.array(list(deg_distribution_1.values()), dtype=float)
            deg_val = round(np.average(deg_distribution), 5)
            self.degree_distribution = deg_distribution
            data_dict["x"].append("Average degree")
            data_dict["y"].append(deg_val)

        if (options.compute_network_diameter == 1) or (options.compute_nodal_connectivity == 1):
            try:
                connected_graph = nx.is_connected(graph)
            except nx.exception.NetworkXPointlessConcept:
                connected_graph = None
        else:
            connected_graph = None

        # calculating network diameter
        if options.compute_network_diameter == 1:
            self.update_status([10, "Computing network diameter..."])
            if connected_graph:
                dia = int(diameter(graph))
            else:
                dia = 'NaN'
            data_dict["x"].append("Network diameter")
            data_dict["y"].append(dia)

        # calculating average nodal connectivity
        if options.compute_nodal_connectivity == 1:
            if self.abort:
                self.update_status([-1, "Task aborted."])
                return
            self.update_status([15, "Computing node connectivity..."])
            if connected_graph:
                avg_node_con = self.average_node_connectivity()
                avg_node_con = round(avg_node_con, 5)
            else:
                avg_node_con = 'NaN'
            data_dict["x"].append("Average node connectivity")
            data_dict["y"].append(avg_node_con)

        # calculating graph density
        if options.compute_graph_density == 1:
            self.update_status([20, "Computing graph density..."])
            g_density = nx.density(graph)
            g_density = round(g_density, 5)
            data_dict["x"].append("Graph density")
            data_dict["y"].append(g_density)

        # calculating global efficiency
        if options.compute_global_efficiency == 1:
            if self.abort:
                self.update_status([-1, "Task aborted."])
                return
            self.update_status([25, "Computing global efficiency..."])
            g_eff = global_efficiency(graph)
            g_eff = round(g_eff, 5)
            data_dict["x"].append("Global efficiency")
            data_dict["y"].append(g_eff)

        if options.compute_wiener_index == 1:
            self.update_status([30, "Computing wiener index..."])
            # settings.update_label("Calculating w_index...")
            w_index = wiener_index(graph)
            w_index = round(w_index, 1)
            data_dict["x"].append("Wiener Index")
            data_dict["y"].append(w_index)

        # calculating assortativity coefficient
        if options.compute_assortativity_coef == 1:
            self.update_status([35, "Computing assortativity coefficient..."])
            a_coef = degree_assortativity_coefficient(graph)
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Assortativity coefficient")
            data_dict["y"].append(a_coef)

        # calculating clustering coefficients
        if (options_gte.is_multigraph == 0) and (options.compute_clustering_coef == 1):
            self.update_status([40, "Computing clustering coefficients..."])
            coefficients_1 = clustering(graph)
            coefficients = np.array(list(coefficients_1.values()), dtype=float)
            avg_clust = round(np.average(coefficients), 5)  # average_clustering(graph)
            self.clustering_coefficients = coefficients
            data_dict["x"].append("Average clustering coefficient")
            data_dict["y"].append(avg_clust)

        # calculating betweenness centrality histogram
        if (options_gte.is_multigraph == 0) and (options.display_betweenness_histogram == 1):
            self.update_status([45, "Computing betweenness centrality..."])
            b_distribution_1 = betweenness_centrality(graph)
            b_distribution = np.array(list(b_distribution_1.values()), dtype=float)
            b_val = round(np.average(b_distribution), 5)
            self.betweenness_distribution = b_distribution
            data_dict["x"].append("Average betweenness centrality")
            data_dict["y"].append(b_val)

        # calculating eigenvector centrality
        if (options_gte.is_multigraph == 0) and (options.display_eigenvector_histogram == 1):
            self.update_status([50, "Computing eigenvector centrality..."])
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100)
            except nx.exception.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000)
            e_vecs = np.array(list(e_vecs_1.values()), dtype=float)
            e_val = round(np.average(e_vecs), 5)
            self.eigenvector_distribution = e_vecs
            data_dict["x"].append("Average eigenvector centrality")
            data_dict["y"].append(e_val)

        # calculating closeness centrality
        if options.display_closeness_histogram == 1:
            self.update_status([55, "Computing closeness centrality..."])
            close_distribution_1 = closeness_centrality(graph)
            close_distribution = np.array(list(close_distribution_1.values()), dtype=float)
            c_val = round(np.average(close_distribution), 5)
            self.closeness_distribution = close_distribution
            data_dict["x"].append("Average closeness centrality")
            data_dict["y"].append(c_val)

        # calculating graph conductance
        if options.compute_graph_conductance == 1:
            self.update_status([60, "Computing graph conductance..."])
            # res_items, sg_components = self.gc.approx_conductance_by_spectral()
            data_dict["x"].append("Largest-Entire graph ratio")
            data_dict["y"].append(str(round((self.gc.connect_ratio * 100), 5)) + "%")
            for item in self.gc.nx_info:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])

        # calculating current-flow betweenness
        if (options_gte.is_multigraph == 0) and (options.display_currentflow_histogram == 1):
            # We select source nodes and target nodes with highest degree-centrality

            gph = self.gc.nx_connected_graph
            all_nodes = list(gph.nodes())
            # source_nodes = random.sample(all_nodes, k=5)
            # rem_nodes = list(set(source_nodes) - set(source_nodes))
            # target_nodes = random.sample(rem_nodes, k=5)
            degree_centrality = nx.degree_centrality(gph)
            sorted_nodes = sorted(all_nodes, key=lambda x: degree_centrality[x], reverse=True)
            source_nodes = sorted_nodes[:5]
            target_nodes = sorted_nodes[-5:]

            self.update_status([65, "Computing current-flow betweenness centrality..."])
            cf_distribution_1 = nx.current_flow_betweenness_centrality_subset(gph, source_nodes, target_nodes)
            cf_distribution = np.array(list(cf_distribution_1.values()), dtype=float)
            cf_val = np.average(cf_distribution)
            cf_val = round(cf_val, 5)
            self.currentflow_distribution = cf_distribution
            data_dict["x"].append("Average current-flow betweenness centrality")
            data_dict["y"].append(cf_val)

        self.output_data = pd.DataFrame(data_dict)

    def average_node_connectivity(self, **kwargs):
        r"""Returns the average connectivity of a graph G.

        The average connectivity of a graph G is the average
        of local node connectivity over all pairs of nodes of G.
        """

        nx_graph = self.gc.nx_graph
        cpu_count = get_num_cores()
        anc = 0

        try:
            filename, output_location = self.gc.imp.create_filenames()
            g_filename = filename + "_graph.txt"
            graph_file = os.path.join(output_location, g_filename)
            nx.write_edgelist(nx_graph, graph_file, data=False)
            anc = sgt.compute_anc(graph_file, cpu_count, self.allow_mp)

        except Exception as err:
            print(err)
        return anc
