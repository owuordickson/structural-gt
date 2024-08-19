# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics
"""

import os
import math
import datetime
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ypstruct import struct
from statistics import stdev, StatisticsError
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.cluster import spectral_clustering
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality
from networkx.algorithms.centrality import eigenvector_centrality, percolation_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index

from .progress_update import ProgressUpdate
from .graph_converter import GraphConverter
from .image_processor import ImageProcessor


class GraphMetrics(ProgressUpdate):
    """
    A class that computes all the user selected graph theory metrics and writes the results in a PDF file.

    Args:
        g_obj: graph converter object.
        configs: graph theory computation parameters and options.
        allow_multiprocessing: a decision to allow multiprocessing computing.
    """

    def __init__(self, g_obj: GraphConverter, configs: struct, allow_multiprocessing: bool = True):
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
        >>> opt_gte.has_weights = 0
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
        >>> opt_gtc.compute_node_connectivity = 1
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
        >>> if opt_gte.has_weights:
        >>>     metrics_obj.compute_weighted_gt_metrics()
        >>> metrics_obj.generate_pdf_output()

        """
        super(GraphMetrics, self).__init__()
        self.allow_mp = allow_multiprocessing
        self.gc = g_obj
        self.configs = configs
        self.output_data = pd.DataFrame([])
        self.degree_distribution = [0]
        self.clustering_coefficients = [0]
        self.betweenness_distribution = [0]
        self.closeness_distribution = [0]
        self.eigenvector_distribution = [0]
        self.ohms_distribution = [0]
        self.percolation_distribution = []
        # self.nx_subgraph_components = []
        self.weighted_output_data = pd.DataFrame([])
        self.weighted_degree_distribution = [0]
        self.weighted_clustering_coefficients = [0]  # NOT USED
        self.weighted_betweenness_distribution = [0]
        self.currentflow_distribution = [0]
        self.weighted_closeness_distribution = [0]
        self.weighted_eigenvector_distribution = [0]
        self.weighted_percolation_distribution = [0]

    def compute_gt_metrics(self):
        """
        Compute un-weighted graph theory metrics.

        :return:
        """
        self.update_status([1, "Using NetworkX to perform un-weighted analysis..."])

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

        if (options.compute_network_diameter == 1) or (options.compute_node_connectivity == 1):
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
        if options.compute_node_connectivity == 1:
            if self.abort:
                self.update_status([-1, "Task aborted."])
                return
            self.update_status([15, "Computing node connectivity..."])
            if connected_graph:
                if self.allow_mp:
                    avg_node_con = self.average_node_connectivity()
                else:
                    avg_node_con = average_node_connectivity(graph)
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

        # calculating Ohms centrality
        if options.display_ohms_histogram == 1:
            self.update_status([60, "Computing Ohms centrality..."])
            ohms_distribution_1, res = self.compute_ohms_centrality()
            ohms_distribution = np.array(list(ohms_distribution_1.values()), dtype=float)
            ohms_val = round(np.average(ohms_distribution), 5)
            # ohms_std = round(np.std(ohms_distribution), 5)
            self.ohms_distribution = ohms_distribution
            data_dict["x"].append("Average Ohms centrality")
            data_dict["y"].append(ohms_val)
            data_dict["x"].append("Ohms centrality (avg. area)")
            data_dict["y"].append(f"{res['avg area']} " + r"$m^2$")
            data_dict["x"].append("Ohms centrality (avg. length)")
            data_dict["y"].append(f"{res['avg length']} m")

        # calculating current-flow betweenness centrality
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

            self.update_status([62, "Computing current-flow betweenness centrality..."])
            cf_distribution_1 = nx.current_flow_betweenness_centrality_subset(gph, source_nodes, target_nodes)
            cf_distribution = np.array(list(cf_distribution_1.values()), dtype=float)
            cf_val = np.average(cf_distribution)
            cf_val = round(cf_val, 5)
            self.currentflow_distribution = cf_distribution
            data_dict["x"].append("Average current-flow betweenness centrality")
            data_dict["y"].append(cf_val)

        # calculating percolation centrality
        if (options_gte.is_multigraph == 0) and (options.display_percolation_histogram == 1):
            self.update_status([65, "Computing percolation centrality..."])
            p_distribution_1 = percolation_centrality(graph, states=None)
            p_distribution = np.array(list(p_distribution_1.values()), dtype=float)
            p_val = round(np.average(p_distribution), 5)
            self.percolation_distribution = p_distribution
            data_dict["x"].append("Average percolation centrality")
            data_dict["y"].append(p_val)

        # calculating graph conductance
        if options.compute_graph_conductance == 1:
            self.update_status([66, "Computing graph conductance..."])
            # res_items, sg_components = self.gc.approx_conductance_by_spectral()
            data_dict["x"].append("Largest-Entire graph ratio")
            data_dict["y"].append(str(round((self.gc.connect_ratio * 100), 5)) + "%")
            for item in self.gc.nx_info:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])

        self.output_data = pd.DataFrame(data_dict)

    def compute_weighted_gt_metrics(self):
        """
        Compute weighted graph theory metrics.

        :return:
        """
        self.update_status([70, "Performing weighted analysis..."])

        graph = self.gc.nx_graph
        options = self.configs
        weight_type = GraphConverter.get_weight_options().get(self.gc.configs_graph.weight_type)
        data_dict = {"x": [], "y": []}

        if graph.number_of_nodes() <= 0:
            self.update_status([-1, "Problem with graph (change filter and graph options)."])
            return

        if options.display_degree_histogram == 1:
            self.update_status([72, "Compute weighted graph degree..."])
            deg_distribution_1 = dict(nx.degree(graph, weight='weight'))
            deg_distribution = np.array(list(deg_distribution_1.values()), dtype=float)
            deg_val = round(np.average(deg_distribution), 5)
            self.weighted_degree_distribution = deg_distribution
            data_dict["x"].append(f"{weight_type}-weighted average degree")
            data_dict["y"].append(deg_val)

        if options.compute_wiener_index == 1:
            self.update_status([74, "Compute weighted wiener index..."])
            w_index = wiener_index(graph, weight='length')
            w_index = round(w_index, 1)
            data_dict["x"].append("Length-weighted Wiener Index")
            data_dict["y"].append(w_index)

        if options.compute_node_connectivity == 1:
            self.update_status([76, "Compute weighted node connectivity..."])
            connected_graph = nx.is_connected(graph)
            if connected_graph:
                max_flow = float(0)
                p = periphery(graph)
                q = len(p) - 1
                for s in range(0, q - 1):
                    for t in range(s + 1, q):
                        flow_value = maximum_flow(graph, p[s], p[t], capacity='weight')[0]
                        if flow_value > max_flow:
                            max_flow = flow_value
                max_flow = round(max_flow, 5)
            else:
                max_flow = 'NaN'
            data_dict["x"].append("Max flow between periphery")
            data_dict["y"].append(max_flow)

        if options.compute_assortativity_coef == 1:
            self.update_status([78, "Compute weighted assortativity..."])
            a_coef = degree_assortativity_coefficient(graph, weight='width')
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Width-weighted assortativity coefficient")
            data_dict["y"].append(a_coef)

        if options.display_betweenness_histogram == 1:
            self.update_status([80, "Compute weighted betweenness centrality..."])
            b_distribution_1 = betweenness_centrality(graph, weight='weight')
            b_distribution = np.array(list(b_distribution_1.values()), dtype=float)
            b_val = round(np.average(b_distribution), 5)
            self.weighted_betweenness_distribution = b_distribution
            data_dict["x"].append(f"{weight_type}-weighted average betweenness centrality")
            data_dict["y"].append(b_val)

        if options.display_closeness_histogram == 1:
            self.update_status([82, "Compute weighted closeness centrality..."])
            close_distribution_1 = closeness_centrality(graph, distance='length')
            close_distribution = np.array(list(close_distribution_1.values()), dtype=float)
            c_val = round(np.average(close_distribution), 5)
            self.weighted_closeness_distribution = close_distribution
            data_dict["x"].append("Length-weighted average closeness centrality")
            data_dict["y"].append(c_val)

        if options.display_eigenvector_histogram == 1:
            if self.abort:
                self.update_status([-1, "Task aborted."])
                return
            self.update_status([84, "Compute weighted eigenvector centrality..."])
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100, weight='weight')
            except nx.exception.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000, weight='weight')
            e_vecs = np.array(list(e_vecs_1.values()), dtype=float)
            e_val = round(np.average(e_vecs), 5)
            self.weighted_eigenvector_distribution = e_vecs
            data_dict["x"].append(f"{weight_type}-weighted average eigenvector centrality")
            data_dict["y"].append(e_val)

        if options.display_percolation_histogram == 1:
            self.update_status([86, "Compute weighted percolation centrality..."])
            p_distribution_1 = percolation_centrality(graph, states=None, weight='weight')
            p_distribution = np.array(list(p_distribution_1.values()), dtype=float)
            p_val = round(np.average(p_distribution), 5)
            self.weighted_percolation_distribution = p_distribution
            data_dict["x"].append(f"{weight_type}-weighted average percolation centrality")
            data_dict["y"].append(p_val)

        # calculating graph conductance
        if options.compute_graph_conductance == 1:
            self.update_status([87, "Computing graph conductance..."])
            # res_items, sg_components = self.gc.approx_conductance_by_spectral(weighted=True)
            for item in self.gc.nx_info:
                data_dict["x"].append((str(item["name"])))
                data_dict["y"].append((str(item["value"])))

        # calculate cross-sectional area of edges
        if self.gc.configs_graph.weight_type == 'AREA':
            self.update_status([68, "Computing average (edge) cross-sectional area..."])
            temp_distribution = []
            for (s, e) in graph.edges():
                temp_distribution.append(graph[s][e]['weight'])
            a_distribution = np.array(temp_distribution, dtype=float)
            ae_val = np.average(a_distribution)
            ae_val = round(ae_val, 5)
            data_dict["x"].append(f"Average edge cross-sectional area (nm\u00b2)")
            data_dict["y"].append(ae_val)

        self.weighted_output_data = pd.DataFrame(data_dict)

    def compute_ohms_centrality(self):
        r"""
        Computes Ohms centrality value for each node.

        Returns: Ohms centrality distribution
        """
        ohms_dict = {}
        lst_area = []
        lst_len = []
        nx_graph = self.gc.nx_graph
        px_size = self.gc.imp.pixel_width
        rho_dim = self.gc.imp.configs_img.resistivity
        pixel_dim = px_size  # * (10 ** 9)  # Convert to nanometers

        b_dict = betweenness_centrality(nx_graph)
        lst_nodes = list(nx_graph.nodes())
        for n in lst_nodes:
            # compute Ohms centrality value for each node
            # print(n)
            b_val = float(b_dict[n])
            if b_val == 0:
                ohms_val = 0
            else:
                connected_nodes = dict(nx_graph[n])  # all nodes connected to node n
                arr_len = []
                arr_dia = []
                for idx, val in connected_nodes.items():
                    # print(f"{idx} -- {val['length']}")
                    arr_len.append(val['length'])
                    arr_dia.append(val['width'])
                arr_len = np.array(arr_len, dtype=float)
                arr_dia = np.array(arr_dia, dtype=float)
                # print(f"{n} -> {len(connected_nodes)}")
                # print(f"Lengths: {arr_len}; Diameters: {arr_dia}")

                pix_width = np.average(arr_dia)
                pix_length = np.sum(arr_len)
                length = pix_length * pixel_dim
                area = math.pi * 89.6 * (pix_width * pixel_dim * 0.5) ** 2
                ohms_val = ((b_val * length * rho_dim) / area)
                lst_len.append(length)
                lst_area.append(area)
                # if n < 5:
            #    print(f"Betweenness val: {b_val}")
            #    print(f"Ohms val: {ohms_val}")
            #    print("\n")
            ohms_dict[n] = ohms_val
        avg_area = np.average(np.array(lst_area, dtype=float))
        avg_len = np.average(np.array(lst_len, dtype=float))
        res = {'avg area': avg_area, 'avg length': avg_len}
        return ohms_dict, res

    def average_node_connectivity(self, flow_func=None):
        r"""Returns the average connectivity of a graph G.

        The average connectivity `\bar{\kappa}` of a graph G is the average
        of local node connectivity over all pairs of nodes of nx_graph.

        https://networkx.org/documentation/stable/_modules/networkx/algorithms/connectivity/connectivity.html#average_node_connectivity

        Parameters
        ----------

        flow_func : function
            A function for computing the maximum flow among a pair of nodes.
            The function has to accept at least three parameters: a Digraph,
            a source node, and a target node. And return a residual network
            that follows NetworkX conventions (see :meth:`maximum_flow` for
            details). If flow_func is None, the default maximum flow function
            (:meth:`edmonds_karp`) is used. See :meth:`local_node_connectivity`
            for details. The choice of the default function may change from
            version to version and should not be relied on. Default value: None.

        Returns
        -------
        K : float
            Average node connectivity

        References
        ----------
        [1]  Beineke, L., O. Oellermann, and r_network. Pippert (2002). The average
                connectivity of a graph. Discrete mathematics 252(1-3), 31-45.
                http://www.sciencedirect.com/science/article/pii/S0012365X01001807

        """

        nx_graph = self.gc.nx_graph
        if nx_graph.is_directed():
            iter_func = itertools.permutations
        else:
            iter_func = itertools.combinations

        # Reuse the auxiliary digraph and the residual network
        a_digraph = nx.algorithms.connectivity.build_auxiliary_node_connectivity(nx_graph)
        r_network = nx.algorithms.flow.build_residual_network(a_digraph, "capacity")
        # kwargs = {"flow_func": flow_func, "auxiliary": a_digraph, "residual": r_network}

        # for item in items:
        #    task(item)
        # with multiprocessing.Pool() as pool:
        #    call the function for each item in parallel
        #    for result in pool.map(task, items):
        #        print(result)
        num, den = 0, 0
        with multiprocessing.Pool() as pool:
            items = [(nx_graph, u, v, flow_func, a_digraph, r_network) for u, v in iter_func(nx_graph, 2)]
            for n in pool.starmap(nx.algorithms.connectivity.local_node_connectivity, items):
                num += n
                den += 1
                if self.abort:
                    self.update_status([-1, "Task aborted."])
                    return 0
        if den == 0:
            return 0
        return num / den

    def generate_pdf_output(self, gui_app: bool = False):
        """
        Generate and display results in a PDF file or list of Figures.

        :param gui_app: is it Graphical User Interface/Terminal app?
        :return:
        """

        if gui_app:
            return self.generate_output()
        else:
            self.generate_pdf()

    def generate_output(self):
        """
        Generate results as graphs and plots.
        :return: list of results.
        """

        opt_gtc = self.configs
        out_figs = []

        self.update_status([90, "Generating PDF GT Output..."])

        # 1. plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
        fig = self.gc.imp.display_images()
        out_figs.append(fig)

        # 2. plotting skeletal images
        fig = self.gc.display_skeletal_images()
        out_figs.append(fig)

        # 3. plotting sub-graph network
        fig = self.gc.draw_graph_network(a4_size=True)
        if fig:
            out_figs.append(fig)

        # 4. displaying all the GT calculations in Table  on entire page
        fig, fig_wt = self.display_gt_results()
        out_figs.append(fig)
        if fig_wt:
            out_figs.append(fig_wt)

        # 5. displaying histograms
        self.update_status([92, "Generating histograms..."])
        figs = self.display_histograms()
        for fig in figs:
            out_figs.append(fig)

        # 6. displaying heatmaps
        if opt_gtc.display_heatmaps == 1:
            self.update_status([95, "Generating heatmaps..."])
            figs = self.display_heatmaps()
            for fig in figs:
                out_figs.append(fig)

        # 8. displaying run information
        fig = self.display_info()
        out_figs.append(fig)
        return out_figs

    def generate_pdf(self):
        """
        Generate results and write them to a PDF file.

        :return:
        """

        opt_gtc = self.configs

        filename, output_location = self.gc.imp.create_filenames()
        pdf_filename = filename + "_SGT_results.pdf"
        pdf_file = os.path.join(output_location, pdf_filename)

        self.update_status([90, "Generating PDF GT Output..."])
        with (PdfPages(pdf_file) as pdf):

            # 1. plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
            fig = self.gc.imp.display_images()
            pdf.savefig(fig)

            # 2. plotting skeletal images
            fig = self.gc.display_skeletal_images()
            pdf.savefig(fig)  # causes PyQt5 to crash

            # 3. plotting sub-graph network
            fig = self.gc.draw_graph_network(a4_size=True)
            if fig:
                pdf.savefig(fig)

            # 4. displaying all the GT calculations in Table  on entire page
            fig, fig_wt = self.display_gt_results()
            pdf.savefig(fig)
            if fig_wt:
                pdf.savefig(fig_wt)

            # 5. displaying histograms
            self.update_status([92, "Generating histograms..."])
            figs = self.display_histograms()
            for fig in figs:
                pdf.savefig(fig)

            # 6. displaying heatmaps
            if opt_gtc.display_heatmaps == 1:
                self.update_status([95, "Generating heatmaps..."])
                figs = self.display_heatmaps()
                for fig in figs:
                    pdf.savefig(fig)

            # 8. displaying run information
            fig = self.display_info()
            pdf.savefig(fig)
        self.gc.save_files()

    def display_gt_results(self):
        """
        Create a table of weighted and un-weighted graph theory results.

        :return:
        """

        opt_gte = self.gc.configs_graph
        data = self.output_data
        w_data = self.weighted_output_data

        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title("Unweighted GT parameters")
        col_width = [2 / 3, 1 / 3]
        tab_1 = ax.table(cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
        tab_1.scale(1, 1.5)

        if opt_gte.has_weights == 1:
            fig_wt = plt.Figure(figsize=(8.5, 11), dpi=300)
            ax = fig_wt.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.set_title("Weighted GT parameters")
            tab_2 = ax.table(cellText=w_data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
            tab_2.scale(1, 1.5)
        else:
            fig_wt = None
        return fig, fig_wt

    def display_histograms(self):
        """
        Create plot figures of graph theory histograms selected by the user.

        :return:
        """

        opt_gte = self.gc.configs_graph
        opt_gtc = self.configs
        figs = []

        weight_type = GraphConverter.get_weight_options().get(opt_gte.weight_type)
        deg_distribution = self.degree_distribution
        w_deg_distribution = self.weighted_degree_distribution
        cluster_coefs = self.clustering_coefficients
        # w_cluster_coefs = self.weighted_clustering_coefficients
        bet_distribution = self.betweenness_distribution
        w_bet_distribution = self.weighted_betweenness_distribution
        clo_distribution = self.closeness_distribution
        w_clo_distribution = self.weighted_closeness_distribution
        eig_distribution = self.eigenvector_distribution
        w_eig_distribution = self.weighted_eigenvector_distribution
        cf_distribution = self.currentflow_distribution
        ohm_distribution = self.ohms_distribution
        per_distribution = self.percolation_distribution
        w_per_distribution = self.weighted_percolation_distribution

        # Degree and Closeness
        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        if opt_gtc.display_degree_histogram == 1:
            bins = np.arange(0.5, max(deg_distribution) + 1.5, 1)
            deg_title = r'Degree Distribution: $\sigma$='
            ax_1 = fig.add_subplot(2, 1, 1)
            GraphMetrics.plot_histogram(ax_1, deg_title, deg_distribution, 'Degree', bins=bins)

        if opt_gtc.display_closeness_histogram == 1:
            cc_title = r"Closeness Centrality: $\sigma$="
            ax_2 = fig.add_subplot(2, 1, 2)
            GraphMetrics.plot_histogram(ax_2, cc_title, clo_distribution, 'Closeness value')
        figs.append(fig)

        # Betweenness, Clustering, Eigenvector and Ohms
        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_betweenness_histogram == 1):
            bc_title = r"Betweenness Centrality: $\sigma$="
            ax_1 = fig.add_subplot(2, 2, 1)
            GraphMetrics.plot_histogram(ax_1, bc_title, bet_distribution, 'Betweenness value')

        if (opt_gte.is_multigraph == 0) and (opt_gtc.compute_clustering_coef == 1):
            clu_title = r"Clustering Coefficients: $\sigma$="
            ax_2 = fig.add_subplot(2, 2, 2)
            GraphMetrics.plot_histogram(ax_2, clu_title, cluster_coefs, 'Clust. Coeff.')

        if opt_gtc.display_ohms_histogram == 1:
            oh_title = r"Ohms Centrality: $\sigma$="
            ax_3 = fig.add_subplot(2, 2, 3)
            GraphMetrics.plot_histogram(ax_3, oh_title, ohm_distribution, 'Ohms value')

        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_eigenvector_histogram == 1):
            ec_title = r"Eigenvector Centrality: $\sigma$="
            ax_4 = fig.add_subplot(2, 2, 4)
            GraphMetrics.plot_histogram(ax_4, ec_title, eig_distribution, 'Eigenvector value')
        figs.append(fig)

        # Currentflow
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_currentflow_histogram == 1):
            fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            cf_title = r"Current-flow betweenness Centrality: $\sigma$="
            ax_1 = fig.add_subplot(2, 2, 1)
            GraphMetrics.plot_histogram(ax_1, cf_title, cf_distribution, 'Betweenness value')
            figs.append(fig)

        # percolation
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_percolation_histogram == 1):
            fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            pc_title = r"Percolation Centrality: $\sigma$="
            ax_1 = fig.add_subplot(2, 2, 1)
            GraphMetrics.plot_histogram(ax_1, pc_title, per_distribution, 'Percolation value')
            figs.append(fig)

        # weighted histograms
        if opt_gte.has_weights == 1:

            # degree, betweenness, closeness and eigenvector
            fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            if opt_gtc.display_degree_histogram == 1:
                bins = np.arange(0.5, max(w_deg_distribution) + 1.5, 1)
                w_deg_title = r"Weighted Degree: $\sigma$="
                ax_1 = fig.add_subplot(2, 2, 1)
                GraphMetrics.plot_histogram(ax_1, w_deg_title, w_deg_distribution, 'Degree', bins=bins)

            if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                w_bt_title = weight_type + r"-Weighted Betweenness: $\sigma$="
                ax_2 = fig.add_subplot(2, 2, 2)
                GraphMetrics.plot_histogram(ax_2, w_bt_title, w_bet_distribution, 'Betweenness value')

            if opt_gtc.display_closeness_histogram == 1:
                w_clo_title = r"Length-Weighted Closeness: $\sigma$="
                ax_3 = fig.add_subplot(2, 2, 3)
                GraphMetrics.plot_histogram(ax_3, w_clo_title, w_clo_distribution, 'Closeness value')

            if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                w_ec_title = weight_type + r"-Weighted Eigenvector Cent.: $\sigma$="
                ax_4 = fig.add_subplot(2, 2, 4)
                GraphMetrics.plot_histogram(ax_4, w_ec_title, w_eig_distribution, 'Eigenvector value')
            figs.append(fig)

            # percolation
            if (opt_gte.is_multigraph == 0) and (opt_gtc.display_percolation_histogram == 1):
                fig = plt.Figure(figsize=(8.5, 11), dpi=300)
                w_pc_title = weight_type + r"-Weighted Percolation Cent.: $\sigma$="
                ax_1 = fig.add_subplot(2, 2, 1)
                GraphMetrics.plot_histogram(ax_1, w_pc_title, w_per_distribution, 'Percolation value')
                figs.append(fig)

        return figs

    def display_heatmaps(self):
        """
        Create plot figures of graph theory heatmaps.

        :return:
        """

        opt_gte = self.gc.configs_graph
        opt_gtc = self.configs

        weight_type = GraphConverter.get_weight_options().get(opt_gte.weight_type)
        deg_distribution = self.degree_distribution
        w_deg_distribution = self.weighted_degree_distribution
        cluster_coefs = self.clustering_coefficients
        # w_cluster_coefs = self.weighted_clustering_coefficients
        bet_distribution = self.betweenness_distribution
        w_bet_distribution = self.weighted_betweenness_distribution
        clo_distribution = self.closeness_distribution
        w_clo_distribution = self.weighted_closeness_distribution
        eig_distribution = self.eigenvector_distribution
        w_eig_distribution = self.weighted_eigenvector_distribution
        # cf_distribution = self.currentflow_distribution
        ohm_distribution = self.ohms_distribution
        per_distribution = self.percolation_distribution
        w_per_distribution = self.weighted_percolation_distribution

        sz = 30
        lw = 1.5
        figs = []

        if opt_gtc.display_degree_histogram == 1:
            fig = self.plot_heatmap(deg_distribution, 'Degree Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_degree_histogram == 1) and (opt_gte.has_weights == 1):
            fig = self.plot_heatmap(w_deg_distribution, 'Weighted Degree Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.compute_clustering_coef == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(cluster_coefs, 'Clustering Coefficient Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(bet_distribution, 'Betweenness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.has_weights == 1) and \
                (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(w_bet_distribution,
                                    f'{weight_type}-Weighted Betweenness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if opt_gtc.display_closeness_histogram == 1:
            fig = self.plot_heatmap(clo_distribution, 'Closeness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_closeness_histogram == 1) and (opt_gte.has_weights == 1):
            fig = self.plot_heatmap(w_clo_distribution, 'Length-Weighted Closeness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(eig_distribution, 'Eigenvector Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.has_weights == 1) and \
                (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(w_eig_distribution,
                                    f'{weight_type}-Weighted Eigenvector Centrality Heatmap', sz, lw)
            figs.append(fig)
        if opt_gtc.display_ohms_histogram == 1:
            fig = self.plot_heatmap(ohm_distribution, 'Ohms Centrality Heatmap', sz, lw)
            figs.append(fig)

        if (opt_gtc.display_percolation_histogram == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(per_distribution, 'Percolation Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_percolation_histogram == 1) and (opt_gte.has_weights == 1) and \
                (opt_gte.is_multigraph == 0):
            fig = self.plot_heatmap(w_per_distribution,
                                    f'{weight_type}-Weighted Percolation Centrality Heatmap', sz, lw)
            figs.append(fig)
        return figs

    def display_info(self):
        """
        Create a page (as a figure) that show the user selected parameters and options.
        :return:
        """

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title("Run Info")

        # similar to the start of the csv file, this is just getting all the relevant settings to display in the pdf
        _, filename = os.path.split(self.gc.imp.img_path)
        now = datetime.datetime.now()

        run_info = ""
        run_info += filename + "\n"
        run_info += now.strftime("%Y-%m-%d %H:%M:%S") + "\n----------------------------\n\n"

        # Image Configs
        run_info += self.gc.imp.get_config_info()
        run_info += "\n\n"

        # Graph Configs
        run_info += self.gc.get_config_info()
        run_info += "\n\n"

        ax.text(0.5, 0.5, run_info, horizontalalignment='center', verticalalignment='center')
        return fig

    def plot_heatmap(self, distribution: list, title: str, size: float, line_width: float):
        """
        Create a heatmap from a distribution.

        :param distribution: dataset to be plotted.
        :param title: title of the plot figure.
        :param size: size of the scatter items.
        :param line_width: size of the plot line-width.
        :return: histogram plot figure.
        """
        nx_graph = self.gc.nx_graph
        opt_gte = self.gc.configs_graph
        img = self.gc.imp.img
        font_1 = {'fontsize': 9}

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=400)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontdict=font_1)
        ax.set_axis_off()

        ax.imshow(img, cmap='gray')
        nodes = nx_graph.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        c_set = ax.scatter(gn[:, 1], gn[:, 0], s=size, c=distribution, cmap='plasma')
        GraphMetrics.plot_graph_edges(nx_graph, ax, line_width, opt_gte.is_multigraph)
        fig.colorbar(c_set, ax=ax, orientation='vertical', label='Value')
        return fig

    @staticmethod
    def plot_histogram(ax: plt.axes, title: str, distribution: list, x_label: str, bins: np.ndarray = None,
                       y_label: str = 'Counts'):
        """
        Create a histogram from a distribution dataset.

        :param ax: plot axis.
        :param title: title text.
        :param distribution: dataset to be plotted.
        :param x_label: x-label title text.
        :param bins: bins dataset.
        :param y_label: y-label title text.
        :return:
        """
        font_1 = {'fontsize': 9}
        if bins is None:
            bins = np.linspace(min(distribution), max(distribution), 50)
        try:
            std_val = str(round(stdev(distribution), 3))
        except StatisticsError:
            std_val = "N/A"
        hist_title = title + std_val
        ax.set_title(hist_title, fontdict=font_1)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.hist(distribution, bins=bins)

    @staticmethod
    def plot_graph_edges(nx_graph: nx.Graph, ax: plt.axes, line_width: float, is_multigraph: bool = False):
        """
        Create a plot of graph edges and nodes.

        :param nx_graph: networkx graph.
        :param ax: plot axis.
        :param line_width: axis line-width parameter.
        :param is_multigraph: type of graph.
        :return:
        """
        if is_multigraph:
            for (s, e) in nx_graph.edges():
                for k in range(int(len(nx_graph[s][e]))):
                    ge = nx_graph[s][e][k]['pts']
                    ax.plot(ge[:, 1], ge[:, 0], 'black', linewidth=line_width)
        else:
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax.plot(ge[:, 1], ge[:, 0], 'black', linewidth=line_width)
