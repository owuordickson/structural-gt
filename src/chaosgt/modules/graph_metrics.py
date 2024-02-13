# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics
"""

import cv2
import csv
import math
import time
import datetime
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import scipy as sp
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
from .graph_struct import GraphStruct


class GraphMetrics:

    def __init__(self, g_obj, configs):
        self.__listeners = []
        self.g_struct = g_obj
        self.configs = configs
        self.output_data = pd.DataFrame([])
        self.degree_distribution = [0]
        self.clustering_coefficients = [0]
        self.betweenness_distribution = [0]
        self.closeness_distribution = [0]
        self.eigenvector_distribution = [0]
        self.nx_subgraph_components = []
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

        data_dict["x"].append("Connectedness ratio")
        data_dict["y"].append(str(self.g_struct.connectedness_ratio * 100) + "%")

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
                # avg_node_con = average_node_connectivity(graph)
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
            res_items, sg_components = self.approx_conductance_by_spectral()
            for item in res_items:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])
            self.nx_subgraph_components = sg_components

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
            res_items, sg_components = self.approx_conductance_by_spectral(weighted=True)
            for item in res_items:
                data_dict["x"].append((str("Weighted ") + str(item["name"])))
                data_dict["y"].append((str("Weighted ") + str(item["value"])))
            self.nx_subgraph_components = sg_components

        self.weighted_output_data = pd.DataFrame(data_dict)

    def generate_pdf_output(self):
        """

        :return:
        """

        opt_img = self.g_struct.configs_img
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
        cf_distribution = self.currentflow_distribution

        data = self.output_data
        w_data = self.weighted_output_data
        nx_graph = self.g_struct.nx_graph

        raw_img = self.g_struct.img
        filtered_img = self.g_struct.img_filtered
        img_bin = self.g_struct.img_bin
        pdf_file, _, _ = self.g_struct.create_filenames(self.g_struct.img_path)

        self.update_status([90, "Generating PDF GT Output..."])
        with (PdfPages(pdf_file) as pdf):
            font_1 = {'fontsize': 12}
            font_2 = {'fontsize': 9}

            # 1. plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
            img_histogram = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])
            f1 = plt.figure(figsize=(8.5, 8.5), dpi=400)
            f1.add_subplot(2, 2, 1)
            plt.imshow(raw_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Original Image")
            f1.add_subplot(2, 2, 2)
            plt.imshow(filtered_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Processed Image")
            f1.add_subplot(2, 2, 3)
            plt.plot(img_histogram)
            if opt_img.threshold_type == 0:
                thresh_arr = np.array([[opt_img.threshold_global, opt_img.threshold_global],
                                       [0, max(img_histogram)]], dtype='object')
                plt.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
            elif opt_img.threshold_type == 2:
                thresh_arr = np.array([[self.g_struct.otsu_val, self.g_struct.otsu_val],
                                       [0, max(img_histogram)]], dtype='object')
                plt.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
            plt.yticks([])
            plt.title("Histogram of Processed Image")
            plt.xlabel("Pixel values")
            plt.ylabel("Counts")
            f1.add_subplot(2, 2, 4)
            plt.imshow(img_bin, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Binary Image")

            pdf.savefig()
            plt.close()

            # 2. plotting skeletal images
            g_skel = self.g_struct.graph_skeleton
            f2a = plt.figure(figsize=(8.5, 11), dpi=400)
            f2a.add_subplot(2, 1, 1)
            # g_skel.skel_int = -1*(g_skel.skel_int-1)
            plt.imshow(g_skel.skel_int, cmap='gray')
            plt.scatter(g_skel.bp_coord_x, g_skel.bp_coord_y, s=0.25, c='b')
            plt.scatter(g_skel.ep_coord_x, g_skel.ep_coord_y, s=0.25, c='r')
            plt.xticks([])
            plt.yticks([])
            plt.title("Skeletal Image")
            f2a.add_subplot(2, 1, 2)
            plt.imshow(raw_img, cmap='gray')
            if opt_gte.is_multigraph:
                for (s, e) in nx_graph.edges():
                    for k in range(int(len(nx_graph[s][e]))):
                        ge = nx_graph[s][e][k]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'red')
            else:
                for (s, e) in nx_graph.edges():
                    ge = nx_graph[s][e]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], 'red')

            # 3. plotting the final graph with the nodes
            nodes = nx_graph.nodes()
            gn = np.array([nodes[i]['o'] for i in nodes])
            if opt_gte.display_node_id == 1:
                i = 0
                for x, y in zip(gn[:, 1], gn[:, 0]):
                    plt.annotate(str(i), (x, y), fontsize=5)
                    i += 1
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
            else:
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
            plt.xticks([])
            plt.yticks([])
            plt.title("Final Graph")
            pdf.savefig()
            plt.close()

            # 4. plotting sub-graph network
            if len(self.nx_subgraph_components) > 0:
                f2b = plt.figure(figsize=(8.5, 11), dpi=400)
                f2b.add_subplot(1, 1, 1)
                plt.imshow(raw_img, cmap='gray')
                color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                color_cycle = itertools.cycle(color_list)
                for component in self.nx_subgraph_components:
                    sg = nx_graph.subgraph(component)
                    color = next(color_cycle)
                    for (s, e) in sg.edges():
                        ge = sg[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], color)
                plt.xticks([])
                plt.yticks([])
                plt.title("Sub Graphs")
                pdf.savefig()
                plt.close()

            # 5. displaying all the GT calculations in Table  on entire page
            f3a = plt.figure(figsize=(8.5, 11), dpi=300)
            f3a.add_subplot(1, 1, 1)
            f3a.patch.set_visible(False)
            plt.axis('off')
            col_width = [2 / 3, 1 / 3]
            table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
            table.scale(1, 1.5)
            plt.title("Unweighted GT parameters")
            pdf.savefig()
            plt.close()

            if opt_gte.weighted_by_diameter == 1:
                # try:  # generates exception is w_data is None
                f3b = plt.figure(figsize=(8.5, 11), dpi=300)
                f3b.add_subplot(1, 1, 1)
                f3b.patch.set_visible(False)
                plt.axis('off')
                col_width = [2 / 3, 1 / 3]
                tab_2 = plt.table(cellText=w_data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
                tab_2.scale(1, 1.5)
                plt.title("Weighted GT Parameters")
                pdf.savefig()
                plt.close()
                # except:
                #    pass

            # 6. displaying histograms
            f4a = plt.figure(figsize=(8.5, 11), dpi=300)
            if opt_gtc.display_degree_histogram == 1:
                f4a.add_subplot(2, 2, 1)
                bins_1 = np.arange(0.5, max(deg_distribution) + 1.5, 1)
                try:
                    deg_val = str(round(stdev(deg_distribution), 3))
                except StatisticsError:
                    deg_val = "N/A"
                deg_txt = r'Degree Distribution: $\sigma$=' + str(deg_val)
                plt.hist(deg_distribution, bins=bins_1)
                plt.title(deg_txt)
                plt.xlabel("Degree")
                plt.ylabel("Counts")
            if opt_gtc.display_closeness_histogram == 1:
                f4a.add_subplot(2, 2, 2)
                bins_3 = np.linspace(min(clo_distribution), max(clo_distribution), 50)
                try:
                    cc_val = str(round(stdev(clo_distribution), 3))
                except StatisticsError:
                    cc_val = "N/A"
                cc_txt = r"Closeness Centrality: $\sigma$=" + str(cc_val)
                plt.hist(clo_distribution, bins=bins_3)
                plt.title(cc_txt)
                plt.xlabel("Closeness value")
                plt.ylabel("Counts")
            pdf.savefig()
            plt.close()

            f4b = plt.figure(figsize=(8.5, 11), dpi=400)
            if (opt_gte.is_multigraph == 0) and (opt_gtc.display_betweenness_histogram == 1):
                f4b.add_subplot(2, 2, 1)
                bins_2 = np.linspace(min(bet_distribution), max(bet_distribution), 50)
                try:
                    bt_val = str(round(stdev(bet_distribution), 3))
                except StatisticsError:
                    bt_val = "N/A"
                bc_txt = r"Betweenness Centrality: $\sigma$=" + str(bt_val)
                plt.hist(bet_distribution, bins=bins_2)
                plt.title(bc_txt)
                plt.xlabel("Betweenness value")
                plt.ylabel("Counts")
            if (opt_gte.is_multigraph == 0) and (opt_gtc.compute_clustering_coef == 1):
                f4b.add_subplot(2, 2, 2)
                bins_t = np.linspace(min(cluster_coefs), max(cluster_coefs), 50)
                try:
                    t_val = str(round(stdev(cluster_coefs), 3))
                except StatisticsError:
                    t_val = "N/A"
                t_txt = r"Clustering Coefficients: $\sigma$=" + str(t_val)
                plt.hist(cluster_coefs, bins=bins_t)
                plt.title(t_txt)
                plt.xlabel("Clust. Coeff.")
                plt.ylabel("Counts")
            if (opt_gte.is_multigraph == 0) and (opt_gtc.display_currentflow_histogram == 1):
                f4b.add_subplot(2, 2, 3)
                bins_2 = np.linspace(min(cf_distribution), max(cf_distribution), 50)
                try:
                    cf_val = str(round(stdev(cf_distribution), 3))
                except StatisticsError:
                    cf_val = "N/A"
                cf_txt = r"Current-flow betweenness Centrality: $\sigma$=" + str(cf_val)
                plt.hist(cf_distribution, bins=bins_2)
                plt.title(cf_txt)
                plt.xlabel("Betweenness value")
                plt.ylabel("Counts")
            if (opt_gte.is_multigraph == 0) and (opt_gtc.display_eigenvector_histogram == 1):
                f4b.add_subplot(2, 2, 4)
                bins4 = np.linspace(min(eig_distribution), max(eig_distribution), 50)
                try:
                    ec_val = str(round(stdev(eig_distribution), 3))
                except StatisticsError:
                    ec_val = "N/A"
                ec_txt = r"Eigenvector Centrality: $\sigma$=" + str(ec_val)
                plt.hist(eig_distribution, bins=bins4)
                plt.title(ec_txt)
                plt.xlabel("Eigenvector value")
                plt.ylabel("Counts")
            pdf.savefig()
            plt.close()

            if opt_gte.weighted_by_diameter == 1:
                if opt_gte.is_multigraph == 1:
                    g_count_1 = opt_gtc.display_degree_histogram + opt_gtc.display_closeness_histogram
                else:
                    g_count_1 = opt_gtc.display_degree_histogram + opt_gtc.compute_clustering_coef + \
                                opt_gtc.display_betweenness_histogram + opt_gtc.display_closeness_histogram + \
                                opt_gtc.display_eigenvector_histogram
                g_count_2 = g_count_1 - opt_gtc.compute_clustering_coef + 1
                f5 = plt.figure(figsize=(8.5, 11), dpi=400)
                if g_count_2 > 2:
                    sy_2 = 2
                    fnt = font_2
                else:
                    sy_2 = 1
                    fnt = font_1
                index = 1
                if opt_gtc.display_degree_histogram == 1:
                    f5.add_subplot(sy_2, 2, index)
                    bins4 = np.arange(0.5, max(w_deg_distribution) + 1.5, 1)
                    try:
                        w_deg_val = str(round(stdev(w_deg_distribution), 3))
                    except StatisticsError:
                        w_deg_val = "N/A"
                    w_deg_txt = r"Weighted Degree: $\sigma$=" + str(w_deg_val)
                    plt.hist(w_deg_distribution, bins=bins4)
                    plt.title(w_deg_txt, fontdict=fnt)
                    plt.xlabel("Degree", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f5.add_subplot(sy_2, 2, index)
                    bins_5 = np.linspace(min(w_bet_distribution), max(w_bet_distribution), 50)
                    plt.hist(w_bet_distribution, bins=bins_5)
                    try:
                        w_bt_val = str(round(stdev(w_bet_distribution), 3))
                    except StatisticsError:
                        w_bt_val = "N/A"
                    w_bt_txt = r"Width-Weighted Betweenness: $\sigma$=" + str(w_bt_val)
                    plt.title(w_bt_txt, fontdict=fnt)
                    plt.xlabel("Betweenness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if opt_gtc.display_closeness_histogram == 1:
                    f5.add_subplot(sy_2, 2, index)
                    bins_6 = np.linspace(min(w_clo_distribution), max(w_clo_distribution), 50)
                    try:
                        w_clo_val = str(round(stdev(w_clo_distribution), 3))
                    except StatisticsError:
                        w_clo_val = "N/A"
                    w_clo_txt = r"Length-Weighted Closeness: $\sigma$=" + str(w_clo_val)
                    plt.hist(w_clo_distribution, bins=bins_6)
                    plt.title(w_clo_txt, fontdict=fnt)
                    plt.xlabel("Closeness value", fontdict=fnt)
                    plt.xticks(fontsize=8)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f5.add_subplot(sy_2, 2, index)
                    bins7 = np.linspace(min(w_eig_distribution), max(w_eig_distribution), 50)
                    plt.hist(w_eig_distribution, bins=bins7)
                    try:
                        w_ec_val = str(round(stdev(w_eig_distribution), 3))
                    except StatisticsError:
                        w_ec_val = "N/A"
                    w_ec_txt = r"Width-Weighted Eigenvector Cent.: $\sigma$=" + str(w_ec_val)
                    plt.title(w_ec_txt, fontdict=fnt)
                    plt.xlabel("Eigenvector value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                pdf.savefig()
                plt.close()

            # 7. displaying heatmaps
            if opt_gtc.display_heatmaps == 1:
                self.update_status([95, "Generating heatmaps..."])
                sz = 30
                lw = 1.5
                time.sleep(0.5)
                if opt_gtc.display_degree_histogram == 1:
                    f6a = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6a.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=deg_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Degree Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_degree_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
                    f6b = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6b.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_deg_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Weighted Degree Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.compute_clustering_coef == 1) and (opt_gte.is_multigraph == 0):
                    f6c = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6c.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=cluster_coefs, cmap='plasma')
                    for (s, e) in nx_graph.edges():
                        ge = nx_graph[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Clustering Coefficient Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f6d = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6d.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=bet_distribution, cmap='plasma')
                    for (s, e) in nx_graph.edges():
                        ge = nx_graph[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Betweenness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and \
                        (opt_gte.is_multigraph == 0):
                    f6e = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6e.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_bet_distribution, cmap='plasma')
                    for (s, e) in nx_graph.edges():
                        ge = nx_graph[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Betweenness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if opt_gtc.display_closeness_histogram == 1:
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=clo_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Closeness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_closeness_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_clo_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Length-Weighted Closeness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=eig_distribution, cmap='plasma')
                    for (s, e) in nx_graph.edges():
                        ge = nx_graph[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Eigenvector Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and \
                        (opt_gte.is_multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_eig_distribution, cmap='plasma')
                    for (s, e) in nx_graph.edges():
                        ge = nx_graph[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Eigenvector Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()

            # 8. displaying run information
            f8 = plt.figure(figsize=(8.5, 8.5), dpi=300)
            f8.add_subplot(1, 1, 1)
            plt.text(0.5, 0.5, self.get_info(), horizontalalignment='center', verticalalignment='center')
            plt.xticks([])
            plt.yticks([])
            pdf.savefig()
            plt.close()

        self.g_struct.save_files(opt_gte)

    def get_info(self):
        # similar to the start of the csv file, this is just getting all the relevant settings to display in the pdf
        opt_img = self.g_struct.configs_img
        opt_gte = self.g_struct.configs_graph
        run_info = "Run Info\n"
        run_info = run_info + self.g_struct.img_path
        now = datetime.datetime.now()
        run_info = run_info + " || " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        if opt_img.threshold_type == 0:
            run_info = run_info + " || Global Threshold (" + str(opt_img.threshold_global) + ")"
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
            run_info = run_info + " || Merge Nodes"
        if opt_gte.prune_dangling_edges:
            run_info = run_info + " || Prune Dangling Edges"
        if opt_gte.remove_disconnected_segments:
            run_info = run_info + " || Remove Objects of Size " + str(opt_gte.remove_object_size)
        if opt_gte.remove_self_loops:
            run_info = run_info + " || Remove Self Loops"
        if opt_gte.is_multigraph:
            run_info = run_info + " || Multi-graph allowed"
        return run_info

    def approx_conductance_by_spectral(self, weighted=False):
        """
        Implements ideas proposed in:    https://doi.org/10.1016/j.procs.2013.09.311

        Conductance can closely be approximated via eigenvalue computation,\
        a fact which has been well-known and well-used in the graph theory community.

        The Laplacian matrix of a directed graph is by definition generally non-symmetric,\
        while, e.g., traditional spectral clustering is primarily developed for undirected\
        graphs with symmetric adjacency and Laplacian matrices. A trivial approach to apply\
        techniques requiring the symmetry is to turn the original directed graph into an\
        undirected graph and build the Laplacian matrix for the latter.

        We need to remove isolated nodes (in order to avoid singular adjacency matrix).\
        The degree of a node is the number of edges incident to that node.\
        When a node has a degree of zero, it means that there are no edges\
        connected to that node. In other words, the node is isolated from\
        the rest of the graph.

        """

        # It is important to notice our graph is (mostly) a directed graph,
        # meaning that it is: (asymmetric) with self-looping nodes

        graph = self.g_struct.nx_graph
        data = []
        sub_components = []

        # 1. Make a copy of the graph
        # eig_graph = graph.copy()

        # 2a. Remove self-looping edges
        eig_graph = GraphMetrics.remove_self_loops(graph)

        # 2b. Identify isolated nodes
        isolated_nodes = list(nx.isolates(eig_graph))

        # 2c. Remove isolated nodes
        eig_graph.remove_nodes_from(isolated_nodes)

        # 3a. Check connectivity of graph
        try:
            # does not work if graphs has disconnected sub-graphs
            nx.fiedler_vector(eig_graph)
        except nx.NetworkXNotImplemented:
            # Graph is directed.
            non_directed_graph = GraphMetrics.make_graph_symmetrical(eig_graph)
            try:
                nx.fiedler_vector(non_directed_graph)
            except nx.NetworkXNotImplemented:
                print("Graph is directed. Cannot compute conductance")
                return None
            except nx.NetworkXError:
                # Graph has less than two nodes or is not connected.
                sub_graph_largest, sub_graph_smallest, size, sub_components = GraphStruct.graph_components(eig_graph)
                eig_graph = sub_graph_largest
                data.append({"name": "Subgraph Count", "value": size})
                data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
                data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
                data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
                data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})
        except nx.NetworkXError:
            # Graph has less than two nodes or is not connected.
            sub_graph_largest, sub_graph_smallest, size, sub_components = GraphStruct.graph_components(eig_graph)
            eig_graph = sub_graph_largest
            data.append({"name": "Subgraph Count", "value": size})
            data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
            data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
            data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
            data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})

        # 4. Compute normalized-laplacian matrix
        if weighted:
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph, weight='weight').toarray()
        else:
            # norm_laplacian_matrix = compute_norm_laplacian_matrix(eig_graph)
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph).toarray()

        # 5. Compute eigenvalues
        # e_vals, _ = np.linalg.eig(norm_laplacian_matrix)
        e_vals = sp.linalg.eigvals(norm_laplacian_matrix)

        # 6. Approximate conductance using the 2nd smallest eigenvalue
        eigenvalues = e_vals.real
        val_max, val_min = GraphMetrics.compute_conductance_range(eigenvalues)
        data.append({"name": "Graph Conductance (max)", "value": val_max})
        data.append({"name": "Graph Conductance (min)", "value": val_min})

        return data, sub_components

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

    @staticmethod
    def compute_norm_laplacian_matrix(graph):
        """
        Compute normalized-laplacian-matrix

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Compute Degree matrix
        deg_mat = np.diag(np.sum(adj_mat, axis=1))

        # 3. Compute Identity matrix
        id_mat = np.identity(adj_mat.shape[0])

        # 4. Compute (Degree inverse squared) D^{-1/2} matrix
        # Check for singular matrices
        if np.any(np.diag(deg_mat) == 0):
            # Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.
            # raise ValueError("Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.")
            print("Graph has nodes with zero degree. Cannot compute conductance")
            return None
        deg_inv_sqrt = np.linalg.inv(np.sqrt(deg_mat))

        # 5. Compute Laplacian matrix
        # lpl_mat = deg_mat - adj_mat

        # 6. Compute normalized-Laplacian matrix
        norm_lpl_mat = id_mat - np.dot(deg_inv_sqrt, np.dot(adj_mat, deg_inv_sqrt))
        # norm_lpl_mat = np.eye(sp_graph.number_of_nodes()) - np.dot(np.dot(deg_inv_sqrt, adj_mat), deg_inv_sqrt)

        # print(adj_mat)
        # print(adj_mat.shape)
        # print(deg_mat)
        # print(deg_inv_sqrt)
        # print(id_mat)
        # print(lpl_mat)
        # print(norm_lpl_mat)
        return norm_lpl_mat

    @staticmethod
    def remove_self_loops(graph):
        """
        Remove self-loops from graph, they cause zero values in Degree matrix.

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Symmetric-ize the Adjacency matrix
        # adj_mat = np.maximum(adj_mat, adj_mat.transpose())

        # 3. Remove (self-loops) non-zero diagonal values in Adjacency matrix
        np.fill_diagonal(adj_mat, 0)

        # 4. Create new graph
        new_graph = nx.from_numpy_array(adj_mat)

        return new_graph

    @staticmethod
    def make_graph_symmetrical(graph):
        """

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Symmetric-ize the Adjacency matrix
        adj_mat = np.maximum(adj_mat, adj_mat.transpose())

        # 3. Remove (self-loops) non-zero diagonal values in Adjacency matrix
        np.fill_diagonal(adj_mat, 0)

        # 4. Create new graph
        new_graph = nx.from_numpy_array(adj_mat)

        return new_graph

    @staticmethod
    def compute_conductance_range(eig_vals):
        """

        :param eig_vals:
        :return:
        """

        # Sort the eigenvalues in ascending order
        sorted_vals = np.array(eig_vals)
        sorted_vals.sort()

        # Sort the eigenvalues in descending order
        # eigenvalues[::-1].sort()

        # approximate conductance using the 2nd smallest eigenvalue
        try:
            conductance_max = math.sqrt((2 * sorted_vals[1]))
        except ValueError:
            conductance_max = None
        conductance_min = sorted_vals[1] / 2

        return conductance_max, conductance_min
