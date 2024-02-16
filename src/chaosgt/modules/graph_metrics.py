# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics
"""

import os
import cv2
import datetime
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
from statistics import stdev, StatisticsError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.cluster import spectral_clustering
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering, average_clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index


class GraphMetrics:

    def __init__(self, g_obj, configs, allow_multiprocessing=True):
        self.__listeners = []
        self.allow_mp = allow_multiprocessing
        self.g_struct = g_obj
        self.configs = configs
        self.output_data = pd.DataFrame([])
        self.degree_distribution = [0]
        self.clustering_coefficients = [0]
        self.betweenness_distribution = [0]
        self.closeness_distribution = [0]
        self.eigenvector_distribution = [0]
        # self.nx_subgraph_components = []
        self.weighted_output_data = pd.DataFrame([])
        self.weighted_degree_distribution = [0]
        self.weighted_clustering_coefficients = [0]  # NOT USED
        self.weighted_betweenness_distribution = [0]
        self.currentflow_distribution = [0]
        self.weighted_closeness_distribution = [0]
        self.weighted_eigenvector_distribution = [0]

    def add_listener(self, func):
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_listener(self, func):
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    # Trigger events.
    def update_status(self, args=None):
        # Run all the functions that are saved.
        if args is None:
            args = []
        for func in self.__listeners:
            func(*args)

    def compute_gt_metrics(self):
        """

        :return:
        """
        self.update_status([1, "Performing un-weighted analysis..."])

        graph = self.g_struct.nx_graph
        options = self.configs
        options_gte = self.g_struct.configs_graph
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
            avg_coefficients_1 = clustering(graph)
            avg_coefficients = np.array(list(avg_coefficients_1.values()), dtype=float)
            clust = average_clustering(graph)
            clust = round(clust, 5)
            self.clustering_coefficients = avg_coefficients
            data_dict["x"].append("Average clustering coefficient")
            data_dict["y"].append(clust)

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
            # res_items, sg_components = self.g_struct.approx_conductance_by_spectral()
            data_dict["x"].append("Largest-Entire graph ratio")
            data_dict["y"].append(str(round((self.g_struct.connect_ratio * 100), 5)) + "%")
            for item in self.g_struct.nx_info:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])

        # calculating current-flow betweenness
        if (options_gte.is_multigraph == 0) and (options.display_currentflow_histogram == 1):
            # We select source nodes and target nodes with highest degree-centrality

            gph = self.g_struct.nx_connected_graph
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

    def compute_weighted_gt_metrics(self):
        self.update_status([70, "Performing weighted analysis..."])

        graph = self.g_struct.nx_graph
        options = self.configs
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
            data_dict["x"].append("Weighted average degree")
            data_dict["y"].append(deg_val)

        if options.compute_wiener_index == 1:
            self.update_status([74, "Compute weighted wiener index..."])
            w_index = wiener_index(graph, weight='length')
            w_index = round(w_index, 1)
            data_dict["x"].append("Length-weighted Wiener Index")
            data_dict["y"].append(w_index)

        if options.compute_nodal_connectivity == 1:
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
            a_coef = degree_assortativity_coefficient(graph, weight='pixel width')
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Weighted assortativity coefficient")
            data_dict["y"].append(a_coef)

        if options.display_betweenness_histogram == 1:
            self.update_status([80, "Compute weighted betweenness centrality..."])
            b_distribution_1 = betweenness_centrality(graph, weight='weight')
            b_distribution = np.array(list(b_distribution_1.values()), dtype=float)
            b_val = round(np.average(b_distribution), 5)
            self.weighted_betweenness_distribution = b_distribution
            data_dict["x"].append("Width-weighted average betweenness centrality")
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
            self.update_status([84, "Compute weighted eigenvector centrality..."])
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100, weight='weight')
            except nx.exception.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000, weight='weight')
            e_vecs = np.array(list(e_vecs_1.values()), dtype=float)
            e_val = round(np.average(e_vecs), 5)
            self.weighted_eigenvector_distribution = e_vecs
            data_dict["x"].append("Width-weighted average eigenvector centrality")
            data_dict["y"].append(e_val)

        # calculating graph conductance
        if options.compute_graph_conductance == 1:
            self.update_status([86, "Computing graph conductance..."])
            # res_items, sg_components = self.g_struct.approx_conductance_by_spectral(weighted=True)
            for item in self.g_struct.nx_info:
                data_dict["x"].append((str("Weighted ") + str(item["name"])))
                data_dict["y"].append((str("Weighted ") + str(item["value"])))

        self.weighted_output_data = pd.DataFrame(data_dict)

    def generate_pdf_output(self):

        """

        :return:
        """

        opt_gtc = self.configs

        filename, output_location = self.g_struct.create_filenames(self.g_struct.img_path)
        pdf_filename = filename + "_SGT_results.pdf"
        pdf_file = os.path.join(output_location, pdf_filename)

        self.update_status([90, "Generating PDF GT Output..."])
        with (PdfPages(pdf_file) as pdf):

            # 1. plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
            fig = self.display_images()
            pdf.savefig(fig)

            # 2. plotting skeletal images
            fig = self.display_skeletal_images()
            pdf.savefig(fig)  # causes PyQt5 to crash

            # 3. plotting sub-graph network
            fig, _ = self.g_struct.draw_graph_network(a4_size=True)
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
        self.g_struct.save_files()

    def generate_list_output(self):

        """

        :return:
        """

        opt_gtc = self.configs
        lst_fig = []
        self.update_status([90, "Generating List GT Output..."])

        # 1. plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
        fig = self.display_images()
        lst_fig.append(fig)

        # 2. plotting skeletal images
        fig = self.display_skeletal_images()
        lst_fig.append(fig)

        # 3. plotting sub-graph network
        fig, _ = self.g_struct.draw_graph_network(a4_size=True)
        if fig:
            lst_fig.append(fig)

        # 4. displaying all the GT calculations in Table  on entire page
        fig, fig_wt = self.display_gt_results()
        lst_fig.append(fig)
        if fig_wt:
            lst_fig.append(fig_wt)

        # 5. displaying histograms
        self.update_status([92, "Generating histograms..."])
        figs = self.display_histograms()
        for fig in figs:
            lst_fig.append(fig)

        # 6. displaying heatmaps
        if opt_gtc.display_heatmaps == 1:
            self.update_status([95, "Generating heatmaps..."])
            figs = self.display_heatmaps()
            for fig in figs:
                lst_fig.append(fig)

        # 8. displaying run information
        fig = self.display_info()
        lst_fig.append(fig)
        # self.g_struct.save_files()
        return lst_fig

    def compute_betweenness_centrality(self):

        """
        Implements ideas proposed in: https://doi.org/10.1016/j.socnet.2004.11.009

        Computes betweenness centrality by also considering the edges that all paths (and not just the shortest path)\
        that passes through a vertex. The proposed idea is referred to as: 'random walk betweenness'.

        Random walk betweenness centrality is computed from a fully connected parts of the graph, because each \
        iteration of a random walk must move from source node to destination without disruption. Therefore, if a graph\
        is composed of isolated sub-graphs then betweenness centrality will be limited to only the fully connected\
        sections of the graph. An average is computed after an iteration of x random walks along edges.

        This measure is already implemented in 'networkx' package. Here is the link:\
        https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.current_flow_betweenness_centrality_subset.html

        :return:
        """

        # (NOT TRUE) Important Note: the graph CANNOT have isolated nodes or isolated sub-graphs
        # (NOT TRUE) Note: only works with fully-connected graphs
        # So, random betweenness centrality is computed between source node and destination node.

        graph = self.g_struct.nx_graph

        # 1. Compute laplacian matrix L = D - A
        lpl_mat = nx.laplacian_matrix(graph).toarray()
        print(lpl_mat)

        # 2. Remove any single row and corresponding column from L
        # 3. Invert the resulting matrix
        # 4. Add back the removed row and column to form matrix T
        # 5. Calculate betweenness from T

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

        nx_graph = self.g_struct.nx_graph
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
        if den == 0:
            return 0
        return num / den
        # cpus = get_num_cores()
        # num, den = 0, 0
        # for u, v in iter_func(nx_graph, 2):
        #   num += nx.algorithms.connectivity.local_node_connectivity(nx_graph, u, v, **kwargs)
        #   den += 1
        # if den == 0:  # Null Graph
        #    return 0
        # return num / den

    def display_images(self):

        opt_img = self.g_struct.configs_img
        raw_img = self.g_struct.img
        filtered_img = self.g_struct.img_filtered
        img_bin = self.g_struct.img_bin

        img_histogram = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=400)
        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        ax_3 = fig.add_subplot(2, 2, 3)
        ax_4 = fig.add_subplot(2, 2, 4)

        ax_1.set_title("Original Image")
        ax_1.set_axis_off()
        ax_1.imshow(raw_img, cmap='gray')

        ax_2.set_title("Processed Image")
        ax_2.set_axis_off()
        ax_2.imshow(filtered_img, cmap='gray')

        ax_3.set_title("Binary Image")
        ax_3.set_axis_off()
        ax_3.imshow(img_bin, cmap='gray')

        ax_4.set_title("Histogram of Processed Image")
        ax_4.set(yticks=[], xlabel='Pixel values', ylabel='Counts')
        ax_4.plot(img_histogram)
        if opt_img.threshold_type == 0:
            thresh_arr = np.array([[opt_img.threshold_global, opt_img.threshold_global],
                                   [0, max(img_histogram)]], dtype='object')
            ax_4.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
        elif opt_img.threshold_type == 2:
            thresh_arr = np.array([[self.g_struct.otsu_val, self.g_struct.otsu_val],
                                   [0, max(img_histogram)]], dtype='object')
            ax_4.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
        return fig

    def display_skeletal_images(self):

        opt_gte = self.g_struct.configs_graph
        nx_graph = self.g_struct.nx_graph
        g_skel = self.g_struct.graph_skeleton
        img = self.g_struct.img

        fig = plt.Figure(figsize=(8.5, 11), dpi=400)
        ax_1 = fig.add_subplot(2, 1, 1)
        ax_2 = fig.add_subplot(2, 1, 2)

        ax_1.set_title("Skeletal Image")
        ax_1.set_axis_off()
        ax_1.imshow(g_skel.skel_int, cmap='gray')
        ax_1.scatter(g_skel.bp_coord_x, g_skel.bp_coord_y, s=0.25, c='b')
        ax_1.scatter(g_skel.ep_coord_x, g_skel.ep_coord_y, s=0.25, c='r')

        ax_2.set_title("Final Graph")
        ax_2.set_axis_off()
        ax_2.imshow(img, cmap='gray')
        if opt_gte.is_multigraph:
            for (s, e) in nx_graph.edges():
                for k in range(int(len(nx_graph[s][e]))):
                    ge = nx_graph[s][e][k]['pts']
                    ax_2.plot(ge[:, 1], ge[:, 0], 'red')
        else:
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax_2.plot(ge[:, 1], ge[:, 0], 'red')
        nodes = nx_graph.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        if opt_gte.display_node_id == 1:
            i = 0
            for x, y in zip(gn[:, 1], gn[:, 0]):
                ax_2.annotate(str(i), (x, y), fontsize=5)
                i += 1
            ax_2.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
        else:
            ax_2.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
        return fig

    def display_gt_results(self):

        opt_gte = self.g_struct.configs_graph
        data = self.output_data
        w_data = self.weighted_output_data

        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title("Unweighted GT parameters")
        col_width = [2 / 3, 1 / 3]
        tab_1 = ax.table(cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
        tab_1.scale(1, 1.5)

        if opt_gte.weighted_by_diameter == 1:
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

        opt_gte = self.g_struct.configs_graph
        opt_gtc = self.configs
        font_1 = {'fontsize': 9}
        figs = []

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

        # Degree and Closeness
        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        if opt_gtc.display_degree_histogram == 1:
            bins = np.arange(0.5, max(deg_distribution) + 1.5, 1)
            try:
                deg_val = str(round(stdev(deg_distribution), 3))
            except StatisticsError:
                deg_val = "N/A"
            deg_title = r'Degree Distribution: $\sigma$=' + str(deg_val)
            ax_1 = fig.add_subplot(2, 1, 1)
            ax_1.set_title(deg_title)
            ax_1.set(xlabel='Degree', ylabel='Counts')
            ax_1.hist(deg_distribution, bins=bins)
        if opt_gtc.display_closeness_histogram == 1:
            bins = np.linspace(min(clo_distribution), max(clo_distribution), 50)
            try:
                cc_val = str(round(stdev(clo_distribution), 3))
            except StatisticsError:
                cc_val = "N/A"
            cc_title = r"Closeness Centrality: $\sigma$=" + str(cc_val)
            ax_2 = fig.add_subplot(2, 1, 2)
            ax_2.set_title(cc_title)
            ax_2.set(xlabel='Closeness value', ylabel='Counts')
            ax_2.hist(clo_distribution, bins=bins)
        figs.append(fig)

        # Betweenness, Clustering, Currentflow and Eigenvector
        fig = plt.Figure(figsize=(8.5, 11), dpi=300)
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_betweenness_histogram == 1):
            bins = np.linspace(min(bet_distribution), max(bet_distribution), 50)
            try:
                bt_val = str(round(stdev(bet_distribution), 3))
            except StatisticsError:
                bt_val = "N/A"
            bc_title = r"Betweenness Centrality: $\sigma$=" + str(bt_val)
            ax_1 = fig.add_subplot(2, 2, 1)
            ax_1.set_title(bc_title)
            ax_1.set(xlabel='Betweenness value', ylabel='Counts')
            ax_1.hist(bet_distribution, bins=bins)
        if (opt_gte.is_multigraph == 0) and (opt_gtc.compute_clustering_coef == 1):
            bins = np.linspace(min(cluster_coefs), max(cluster_coefs), 50)
            try:
                t_val = str(round(stdev(cluster_coefs), 3))
            except StatisticsError:
                t_val = "N/A"
            clu_title = r"Clustering Coefficients: $\sigma$=" + str(t_val)
            ax_2 = fig.add_subplot(2, 2, 2)
            ax_2.set_title(clu_title)
            ax_2.set(xlabel='Clust. Coeff.', ylabel='Counts')
            ax_2.hist(cluster_coefs, bins=bins)
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_currentflow_histogram == 1):
            bins = np.linspace(min(cf_distribution), max(cf_distribution), 50)
            try:
                cf_val = str(round(stdev(cf_distribution), 3))
            except StatisticsError:
                cf_val = "N/A"
            cf_title = r"Current-flow betweenness Centrality: $\sigma$=" + str(cf_val)
            ax_3 = fig.add_subplot(2, 2, 3)
            ax_3.set_title(cf_title)
            ax_3.set(xlabel='Betweenness value', ylabel='Counts')
            ax_3.hist(cf_distribution, bins=bins)
        if (opt_gte.is_multigraph == 0) and (opt_gtc.display_eigenvector_histogram == 1):
            bins = np.linspace(min(eig_distribution), max(eig_distribution), 50)
            try:
                ec_val = str(round(stdev(eig_distribution), 3))
            except StatisticsError:
                ec_val = "N/A"
            ec_title = r"Eigenvector Centrality: $\sigma$=" + str(ec_val)
            ax_4 = fig.add_subplot(2, 2, 4)
            ax_4.set_title(ec_title)
            ax_4.set(xlabel='Eigenvector value', ylabel='Counts')
            ax_4.hist(eig_distribution, bins=bins)
        figs.append(fig)

        # weighted histograms
        if opt_gte.weighted_by_diameter == 1:
            fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            if opt_gtc.display_degree_histogram == 1:
                bins = np.arange(0.5, max(w_deg_distribution) + 1.5, 1)
                try:
                    w_deg_val = str(round(stdev(w_deg_distribution), 3))
                except StatisticsError:
                    w_deg_val = "N/A"
                w_deg_title = r"Weighted Degree: $\sigma$=" + str(w_deg_val)
                ax_1 = fig.add_subplot(2, 2, 1)
                ax_1.set_title(w_deg_title, fontdict=font_1)
                ax_1.set(xlabel='Degree', ylabel='Counts')
                ax_1.hist(w_deg_distribution, bins=bins)

            if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                bins = np.linspace(min(w_bet_distribution), max(w_bet_distribution), 50)
                try:
                    w_bt_val = str(round(stdev(w_bet_distribution), 3))
                except StatisticsError:
                    w_bt_val = "N/A"
                w_bt_title = r"Width-Weighted Betweenness: $\sigma$=" + str(w_bt_val)
                ax_2 = fig.add_subplot(2, 2, 2)
                ax_2.set_title(w_bt_title, fontdict=font_1)
                ax_2.set(xlabel='Betweenness value', ylabel='Counts')
                ax_2.hist(w_bet_distribution, bins=bins)
            if opt_gtc.display_closeness_histogram == 1:
                bins = np.linspace(min(w_clo_distribution), max(w_clo_distribution), 50)
                try:
                    w_clo_val = str(round(stdev(w_clo_distribution), 3))
                except StatisticsError:
                    w_clo_val = "N/A"
                w_clo_title = r"Length-Weighted Closeness: $\sigma$=" + str(w_clo_val)
                ax_3 = fig.add_subplot(2, 2, 3)
                ax_3.set_title(w_clo_title, fontdict=font_1)
                ax_3.set(xlabel='Closeness value', ylabel='Counts')
                ax_3.hist(w_clo_distribution, bins=bins)
            if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                bins = np.linspace(min(w_eig_distribution), max(w_eig_distribution), 50)
                try:
                    w_ec_val = str(round(stdev(w_eig_distribution), 3))
                except StatisticsError:
                    w_ec_val = "N/A"
                w_ec_title = r"Width-Weighted Eigenvector Cent.: $\sigma$=" + str(w_ec_val)
                ax_4 = fig.add_subplot(2, 2, 4)
                ax_4.set_title(w_ec_title, fontdict=font_1)
                ax_4.set(xlabel='Eigenvector value', ylabel='Counts')
                ax_4.hist(w_eig_distribution, bins=bins)
            figs.append(fig)

        return figs

    def display_heatmaps(self):

        opt_gte = self.g_struct.configs_graph
        opt_gtc = self.configs

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

        sz = 30
        lw = 1.5
        figs = []
        # init_fig, init_ax = GraphMetrics.plot_histogram_bare(nx_graph, lw, opt_gte.is_multigraph)

        if opt_gtc.display_degree_histogram == 1:
            fig = self.plot_histogram(deg_distribution, 'Degree Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_degree_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
            fig = self.plot_histogram(w_deg_distribution, 'Weighted Degree Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.compute_clustering_coef == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_histogram(cluster_coefs, 'Clustering Coefficient Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_histogram(bet_distribution, 'Betweenness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and \
                (opt_gte.is_multigraph == 0):
            fig = self.plot_histogram(w_bet_distribution, 'Width-Weighted Betweenness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if opt_gtc.display_closeness_histogram == 1:
            fig = self.plot_histogram(clo_distribution, 'Closeness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_closeness_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
            fig = self.plot_histogram(w_clo_distribution, 'Length-Weighted Closeness Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
            fig = self.plot_histogram(eig_distribution, 'Eigenvector Centrality Heatmap', sz, lw)
            figs.append(fig)
        if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and \
                (opt_gte.is_multigraph == 0):
            fig = self.plot_histogram(w_eig_distribution, 'Width-Weighted Eigenvector Centrality Heatmap', sz, lw)
            figs.append(fig)
        return figs

    def display_info(self):

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title("Run Info")

        # similar to the start of the csv file, this is just getting all the relevant settings to display in the pdf
        opt_img = self.g_struct.configs_img
        opt_gte = self.g_struct.configs_graph
        _, filename = os.path.split(self.g_struct.img_path)
        run_info = ""
        run_info = run_info + filename + "\n"
        now = datetime.datetime.now()
        run_info = run_info + now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        if opt_img.threshold_type == 0:
            run_info = run_info + "Global Threshold (" + str(opt_img.threshold_global) + ")"
        elif opt_img.threshold_type == 1:
            run_info = run_info + " || Adaptive Threshold, " + str(opt_img.threshold_adaptive) + " bit kernel"
        elif opt_img.threshold_type == 2:
            run_info = run_info + " || OTSU Threshold"
        if opt_img.gamma != 1:
            run_info = run_info + "|| Gamma = " + str(opt_img.gamma)
        if opt_img.apply_median:
            run_info = run_info + " || Median Filter"
        if opt_img.apply_gaussian:
            run_info = run_info + " || Gaussian Blur, " + str(opt_img.blurring_window_size) + " bit kernel"
        if opt_img.apply_autolevel:
            run_info = run_info + " || Autolevel"
        if opt_img.apply_dark_foreground:
            run_info = run_info + " || Dark Foreground"
        if opt_img.apply_laplacian:
            run_info = run_info + " || Laplacian Gradient"
        if opt_img.apply_scharr:
            run_info = run_info + " || Scharr Gradient"
        if opt_img.apply_sobel:
            run_info = run_info + " || Sobel Gradient"
        if opt_img.apply_lowpass:
            run_info = run_info + " || Low-pass filter" + str(opt_img.filter_window_size)
        run_info = run_info + "\n"
        if opt_gte.merge_nearby_nodes:
            run_info = run_info + "Merge Nodes"
        if opt_gte.prune_dangling_edges:
            run_info = run_info + " || Prune Dangling Edges"
        if opt_gte.remove_disconnected_segments:
            run_info = run_info + " || Remove Objects of Size " + str(opt_gte.remove_object_size)
        if opt_gte.remove_self_loops:
            run_info = run_info + " || Remove Self Loops"
        if opt_gte.is_multigraph:
            run_info = run_info + " || Multi-graph allowed"

        ax.text(0.5, 0.5, run_info, horizontalalignment='center', verticalalignment='center')
        return fig

    def plot_histogram(self, distribution, title, size, line_width):

        nx_graph = self.g_struct.nx_graph
        opt_gte = self.g_struct.configs_graph
        img = self.g_struct.img
        font_1 = {'fontsize': 9}

        fig = plt.Figure(figsize=(8.5, 8.5), dpi=400)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontdict=font_1)

        ax.imshow(img, cmap='gray')
        nodes = nx_graph.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        ax.scatter(gn[:, 1], gn[:, 0], s=size, c=distribution, cmap='plasma')
        # ax.plot(ax_edges.lines[0].get_xdata(), ax_edges.lines[0].get_ydata())  # Copy the line plot
        # nx.draw_networkx_edges(nx_graph, pos=nx.spring_layout(nx_graph), ax=ax, edge_color='black', width=line_width)
        GraphMetrics.plot_graph_edges(nx_graph, ax, line_width, opt_gte.is_multigraph)
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Value')
        return fig

    @staticmethod
    def plot_graph_edges(nx_graph, ax, line_width, is_multigraph=False):
        if is_multigraph:
            for (s, e) in nx_graph.edges():
                for k in range(int(len(nx_graph[s][e]))):
                    ge = nx_graph[s][e][k]['pts']
                    ax.plot(ge[:, 1], ge[:, 0], 'black', linewidth=line_width)
        else:
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax.plot(ge[:, 1], ge[:, 0], 'black', linewidth=line_width)
        # return fig, ax
